"""libgpiod-v2 (gpiod 2.2.0) shim presenting the small RPi.GPIO surface the
ported DW1000 driver uses, backed by the character-device GPIO API.

Why: RPi.GPIO does not work on the Pi 5 / CM5 (BCM2712 + RP1). This exposes
just enough of the old API (setmode/setup/output/add_event_detect/cleanup +
the BCM/IN/OUT/HIGH/LOW/RISING/PUD_* constants) so the driver imports it in
place of RPi.GPIO with no other changes to its GPIO call sites.

Chip-select is deliberately NOT handled here: CSn is wired to GPIO8/CE0 and is
driven in hardware by spidev (/dev/spidev0.0). The driver's readBytes/writeBytes
were rewritten to issue one batched xfer2() per transaction, so CE0 asserts for
the whole frame automatically. Only IRQ (GPIO24) and, if used, RST (GPIO25) are
managed here.
"""
from __future__ import annotations

import threading

import gpiod
from gpiod.line import Bias, Direction, Edge, Value

# --- RPi.GPIO-compatible constants the driver references ---
BCM = "BCM"
IN = "IN"
OUT = "OUT"
INPUT = "IN"      # driver uses GPIO.INPUT in begin()
HIGH = 1
LOW = 0
RISING = "RISING"
PUD_OFF = "PUD_OFF"
PUD_UP = "PUD_UP"
PUD_DOWN = "PUD_DOWN"

_CHIP_PATH = "/dev/gpiochip0"   # 40-pin header on Pi 5 / CM5 (RP1)


class _GPIO:
    # Constants accessed as GPIO.<NAME> by the driver (RPi.GPIO compatibility).
    # `output` is a real method (drives a pin); mode args use GPIO.OUT/GPIO.IN.
    BCM = BCM
    IN = IN
    OUT = OUT
    INPUT = INPUT
    HIGH = HIGH
    LOW = LOW
    RISING = RISING
    PUD_OFF = PUD_OFF
    PUD_UP = PUD_UP
    PUD_DOWN = PUD_DOWN

    def __init__(self):
        self._req = {}      # offset -> gpiod.LineRequest
        self._threads = {}  # offset -> Thread
        self._stop = {}     # offset -> threading.Event

    # -- no-ops kept for API compatibility --
    def setmode(self, mode):
        pass

    def setwarnings(self, flag):
        pass

    def _release(self, offset):
        """Stop+join the IRQ worker (if any), then release the line. Doing it in
        this order means a worker blocked in wait_edge_events is never left
        looping on a request we're about to release."""
        stop = self._stop.pop(offset, None)
        if stop is not None:
            stop.set()
        t = self._threads.pop(offset, None)
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)
        req = self._req.pop(offset, None)
        if req is not None:
            try:
                req.release()
            except Exception:
                pass

    def setup(self, offset, mode, pull_up_down=None):
        """Request `offset` as input or output (re-requesting if already held)."""
        self._release(offset)
        if mode == OUT:
            settings = gpiod.LineSettings(direction=Direction.OUTPUT,
                                          output_value=Value.INACTIVE)
        else:
            bias = Bias.AS_IS
            if pull_up_down == PUD_UP:
                bias = Bias.PULL_UP
            elif pull_up_down == PUD_DOWN:
                bias = Bias.PULL_DOWN
            settings = gpiod.LineSettings(direction=Direction.INPUT, bias=bias)
        self._req[offset] = gpiod.request_lines(
            _CHIP_PATH, consumer="dw1000", config={offset: settings})

    def output(self, offset, value):
        req = self._req.get(offset)
        if req is None:
            self.setup(offset, OUT)
            req = self._req[offset]
        req.set_value(offset, Value.ACTIVE if value else Value.INACTIVE)

    def add_event_detect(self, offset, edge, callback=None, bouncetime=None):
        """Watch `offset` for edges in a daemon thread, invoking callback(offset)
        per event — mirrors RPi.GPIO's threaded callback model."""
        self._release(offset)
        req = gpiod.request_lines(
            _CHIP_PATH, consumer="dw1000-irq",
            config={offset: gpiod.LineSettings(
                direction=Direction.INPUT,
                edge_detection=Edge.RISING if edge == RISING else Edge.BOTH,
                bias=Bias.DISABLED)})
        self._req[offset] = req
        stop = threading.Event()
        self._stop[offset] = stop

        def _worker():
            while not stop.is_set():
                # wake periodically so stop is honoured even with no edges
                if req.wait_edge_events(0.2):  # seconds; periodic wake for stop
                    for _ev in req.read_edge_events():
                        if callback is not None:
                            try:
                                callback(offset)
                            except Exception as exc:  # never kill the IRQ thread
                                print("DW1000 IRQ callback error:", exc)

        t = threading.Thread(target=_worker, name=f"dw1000-irq-{offset}",
                             daemon=True)
        self._threads[offset] = t
        t.start()

    def cleanup(self, offset=None):
        """Release one line (when offset given) or all of them. _release() does
        the per-line stop+join+release; the union of held offsets is the _req
        keys (add_event_detect always records a _req entry alongside the thread)."""
        if offset is not None:
            self._release(offset)
            return
        for off in list(self._req.keys()):
            self._release(off)


GPIO = _GPIO()
