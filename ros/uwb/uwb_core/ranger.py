"""Symmetric DS-TWR ranging core (ROS-free).

One `Dw1000Ranger` owns a single DW1000 radio and runs *both* sides of the
double-sided two-way-ranging exchange in one event loop:

  initiator:  POLL ──▶            timePollSent
                      ◀── POLL_ACK timePollAckReceived
              RANGE ──▶            timeRangeSent      (carries the 3 timestamps)
                      ◀── RANGE_REPORT                (carries the computed range)
  responder:          ◀── POLL     timePollReceived
              POLL_ACK ──▶         timePollAckSent
                      ◀── RANGE     timeRangeReceived
              RANGE_REPORT ──▶      (computes range, echoes it back)

The responder ends up with all four timestamps and computes the range (the
asymmetric formula cancels clock drift). The key extension over the upstream
RangingAnchor/Tag pair: the responder **echoes the computed distance in the
RANGE_REPORT payload**, so the initiator learns it too — *both* peers of a
single exchange emit a range and can publish it (`receiver=self, source=peer`).
That is how "every drone is both anchor and tag" is satisfied at the
localization layer without a fully-simultaneous symmetric radio protocol.

Who *initiates* is arbitrated, not simultaneous (one half-duplex radio): the
default scheduler has the **lower-index node initiate to any higher-index
peer**, which is collision-free for a single pair. `should_initiate()` is the
seam where GPS-disciplined 4-drone TDMA slotting (RFD §4.5) drops in later.

This module imports no ROS — it runs headless on the boards via
`python3 -m uwb_core.range_demo`, and the thin rclpy node wraps it.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable, Iterable

# Runtime gate for the WAIT4RESP hardware turnaround (default on). Set UWB_W4R=0
# to fall back to the software High-5 re-arm path, for A/B isolation on the bench.
_USE_W4R = os.environ.get("UWB_W4R", "1") != "0"

from .dw1000 import DW1000
from .dw1000 import DW1000Constants as C

# Reply turnaround for the delayed-TX legs (POLL_ACK and RANGE), matching the
# upstream examples. The radio hardware-times the actual departure to this slot.
REPLY_DELAY_US = 7000
LEN_DATA = 18

# Deferred high-rate hardening (codex, revisit when ramping to 50-100 Hz):
#  - High 1: snapshot RX frame+timestamp inside handleInterrupt (see _handle_received).
#  - High 5: the responder's delayed POLL_ACK/RANGE TX overlaps permanent-RX re-arm;
#    the upstream examples ran this pattern fine, but at high rate prefer arming TX,
#    waiting TXFRS, then re-enabling RX.
#  - Medium: DW1000.setDelay() folds in the hardcoded C.ANTENNA_DELAY rather than the
#    per-node configured antenna_delay; fold the calibrated value in at M4.

# RANGE_REPORT range-echo payload: distance in millimetres, uint32 LE at [3..6].
_DIST_OFF = 3


def _addr_str(idx: int) -> str:
    """Deterministic 8-byte EUI string for the chip, unique per node index."""
    return "7D:00:22:EA:82:60:3B:%02X" % (0x90 + (idx & 0x0F))


class Dw1000Ranger:
    def __init__(
        self,
        *,
        own_index: int,
        own_addr: tuple[int, int],
        peers: Iterable[tuple[int, tuple[int, int]]],
        on_range: Callable[[int, int, float, int], None],
        irq: int = 24,
        ss: int = 8,
        rst=None,
        bus: int = 0,
        device: int = 0,
        antenna_delay: int = C.ANTENNA_DELAY_RASPI,
        mode=None,
        poll_interval_s: float = 0.1,
        spi_hz: int | None = None,
    ):
        self.own_index = int(own_index)
        self.own_addr = (int(own_addr[0]) & 0xFF, int(own_addr[1]) & 0xFF)
        # peer short-address (lo, hi) -> peer index, for demuxing the sender.
        self.peer_by_addr = {(int(a[0]) & 0xFF, int(a[1]) & 0xFF): int(i) for i, a in peers}
        self.peer_indices = sorted(self.peer_by_addr.values())
        self.on_range = on_range
        self.mode = mode if mode is not None else C.MODE_LONGDATA_RANGE_ACCURACY
        self.poll_interval_s = float(poll_interval_s)
        self.antenna_delay = int(antenna_delay)
        self._ss = ss

        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._seq = 0
        self._stop = threading.Event()

        # Exchange state. One exchange at a time (single half-duplex radio).
        self.data = [0] * LEN_DATA
        self._last_tx_msg = None        # msg type of the in-flight TX
        self._expected = C.POLL         # next msg id we expect to receive
        self._init_peer = None          # peer we're polling (initiator leg)
        self._resp_peer = None          # peer we're answering (responder leg)
        self._rr = 0                    # round-robin cursor over initiate targets
        self._unknown = 0               # dropped unknown-source frames (diag)
        # FSM-level diagnostics. Watchdog resets keyed by the leg we were stalled
        # on: POLL_ACK = initiator lost POLL_ACK, RANGE = responder lost RANGE,
        # RANGE_REPORT = initiator lost the echo. Distinguishes early-leg losses
        # (invisible in the emit counters) from final-leg losses.
        self.stats = {"polls_sent": 0, "wd_POLL_ACK": 0, "wd_RANGE": 0,
                      "wd_RANGE_REPORT": 0, "wd_POLL": 0}
        self._last_activity = time.monotonic()
        self._last_poll = 0.0
        # initiator timestamps
        self.t_poll_sent = 0
        self.t_pollack_recv = 0
        self.t_range_sent = 0
        # responder timestamps
        self.t_poll_recv = 0
        self.t_pollack_sent = 0
        self.t_range_recv = 0
        self._protocol_failed = False

        self.dev = DW1000.DW1000(irq=irq, rst=rst, bus=bus, device=device)
        self.dev.setup(ss)
        self.dev.generalConfiguration(_addr_str(self.own_index), self.mode)
        # The 2 MHz init clock is only required until the PLL/XTAL are up; after
        # generalConfiguration the DW1000 tolerates fast SPI (≤20 MHz). Raising it
        # cuts per-exchange register-txn latency, the host-side rate wall.
        if spi_hz:
            self.dev.spi.max_speed_hz = int(spi_hz)
        self.dev.registerCallback("handleSent", self._on_sent)
        self.dev.registerCallback("handleReceived", self._on_received)
        self.dev.setAntennaDelay(self.antenna_delay)
        # Always listening; the IRQ handler re-arms RX in hardware (RXAUTR).
        self.dev.newReceive()
        self.dev.receivePermanently()
        self.dev.startReceive()

    # -- IRQ-thread callbacks: push a typed event, do NO SPI here (codex High 2).
    def _on_sent(self):
        self._events.put("sent")

    def _on_received(self):
        self._events.put("received")

    # -- role arbitration (the TDMA seam) ----------------------------------
    def should_initiate(self) -> int | None:
        """Return a peer index to poll now, or None. Default: lower-index node
        initiates to higher-index peers. Replace with GPS-TDMA slotting for the
        4-drone fleet (close pairs get more slots, RFD §4.5)."""
        targets = [p for p in self.peer_indices if p > self.own_index]
        if not targets:
            return None
        if (time.monotonic() - self._last_poll) < self.poll_interval_s:
            return None
        peer = targets[self._rr % len(targets)]   # round-robin over all targets
        self._rr += 1
        return peer

    # -- main loop ---------------------------------------------------------
    def run(self):
        while not self._stop.is_set():
            if self._expected == C.POLL and self._last_tx_msg is None:
                peer = self.should_initiate()
                if peer is not None:
                    self._transmit_poll(peer)
            try:
                ev = self._events.get(timeout=0.05)
            except queue.Empty:
                self._check_watchdog()
                continue
            if ev == "sent":
                self._handle_sent()
            elif ev == "received":
                self._handle_received()

    def stop(self):
        self._stop.set()

    def close(self):
        self.stop()
        try:
            self.dev.close()
        except Exception:
            pass

    # -- watchdog: recover a stalled exchange ------------------------------
    def _check_watchdog(self):
        if (time.monotonic() - self._last_activity) * C.MILLISECONDS > C.RESET_PERIOD:
            # drop any half-finished exchange and return to listening. Only a
            # mid-exchange wait (_expected != POLL) is a real stall/lost frame;
            # idling in POLL is normal housekeeping, not a loss.
            _leg = {C.POLL_ACK: "wd_POLL_ACK", C.RANGE: "wd_RANGE",
                    C.RANGE_REPORT: "wd_RANGE_REPORT"}.get(self._expected)
            if _leg:
                self.stats[_leg] += 1
            self._expected = C.POLL
            self._last_tx_msg = None
            self._init_peer = None
            self._resp_peer = None
            self._note_activity()

    def _note_activity(self):
        self._last_activity = time.monotonic()

    # -- TX helpers (compound SPI sequences held under the device lock so the
    #    IRQ handler can't interleave a re-arm mid-frame) -------------------
    def _zero(self):
        for i in range(LEN_DATA):
            self.data[i] = 0

    def _transmit_poll(self, peer: int):
        with self.dev._spi_lock:
            self.dev.newTransmit()
            self._zero()
            self.data[0] = C.POLL
            self.data[1], self.data[2] = self.own_addr
            self.dev.setData(self.data, LEN_DATA)
            self.dev.startTransmit(wait4resp=_USE_W4R)   # POLL expects POLL_ACK
        self._last_tx_msg = C.POLL
        self._init_peer = peer
        self._expected = C.POLL_ACK
        self._last_poll = time.monotonic()
        self.stats["polls_sent"] += 1
        self._note_activity()

    def _transmit_range(self):
        with self.dev._spi_lock:
            self.dev.newTransmit()
            self._zero()
            self.data[0] = C.RANGE
            self.data[1], self.data[2] = self.own_addr
            self.t_range_sent = self.dev.setDelay(REPLY_DELAY_US, C.MICROSECONDS)
            self.dev.setTimeStamp(self.data, self.t_poll_sent, 3)
            self.dev.setTimeStamp(self.data, self.t_pollack_recv, 8)
            self.dev.setTimeStamp(self.data, self.t_range_sent, 13)
            self.dev.setData(self.data, LEN_DATA)
            self.dev.startTransmit(wait4resp=_USE_W4R)   # RANGE expects RANGE_REPORT
        self._last_tx_msg = C.RANGE

    def _transmit_pollack(self):
        with self.dev._spi_lock:
            self.dev.newTransmit()
            self._zero()
            self.data[0] = C.POLL_ACK
            self.data[1], self.data[2] = self.own_addr
            self.dev.setDelay(REPLY_DELAY_US, C.MICROSECONDS)
            self.dev.setData(self.data, LEN_DATA)
            self.dev.startTransmit(wait4resp=_USE_W4R)   # POLL_ACK expects RANGE
        self._last_tx_msg = C.POLL_ACK

    def _transmit_range_report(self, distance_m: float):
        mm = max(0, int(round(distance_m * 1000.0)))
        with self.dev._spi_lock:
            self.dev.newTransmit()
            self._zero()
            self.data[0] = C.RANGE_REPORT
            self.data[1], self.data[2] = self.own_addr
            for k in range(4):                       # uint32 LE millimetres
                self.data[_DIST_OFF + k] = (mm >> (8 * k)) & 0xFF
            self.dev.setData(self.data, LEN_DATA)
            self.dev.startTransmit()
        self._last_tx_msg = C.RANGE_REPORT

    @staticmethod
    def _decode_distance(data) -> float:
        mm = (data[_DIST_OFF] | (data[_DIST_OFF + 1] << 8)
              | (data[_DIST_OFF + 2] << 16) | (data[_DIST_OFF + 3] << 24))
        return mm / 1000.0

    def _compute_range(self) -> float:
        round1 = self.dev.wrapTimestamp(self.t_pollack_recv - self.t_poll_sent)
        reply1 = self.dev.wrapTimestamp(self.t_pollack_sent - self.t_poll_recv)
        round2 = self.dev.wrapTimestamp(self.t_range_recv - self.t_pollack_sent)
        reply2 = self.dev.wrapTimestamp(self.t_range_sent - self.t_pollack_recv)
        tof = (round1 * round2 - reply1 * reply2) / (round1 + round2 + reply1 + reply2)
        return (tof % C.TIME_OVERFLOW) * C.DISTANCE_OF_RADIO

    def _emit(self, source_index: int, range_m: float):
        self.on_range(int(source_index), self.own_index, float(range_m), self._seq)
        self._seq += 1

    # -- event handlers ----------------------------------------------------
    def _handle_sent(self):
        msg = self._last_tx_msg
        self._last_tx_msg = None
        if msg == C.POLL:
            self.t_poll_sent = self.dev.getTransmitTimestamp()
        elif msg == C.RANGE:
            self.t_range_sent = self.dev.getTransmitTimestamp()
        elif msg == C.POLL_ACK:
            self.t_pollack_sent = self.dev.getTransmitTimestamp()
        # RANGE_REPORT: nothing to latch
        self._note_activity()

    def _handle_received(self):
        # Deferred (codex High 1): the RX frame + timestamp are read here in the
        # main loop, after the IRQ handler re-armed RX. DS-TWR is lock-step — the
        # peer never sends its next frame until we reply from here — so the RX
        # buffer can't be clobbered in between at these rates. For the high-rate /
        # retransmit regime, snapshot getData()+getReceiveTimestamp() inside
        # handleInterrupt (under _spi_lock, gated on LDEDONE) and pass them through
        # the event instead.
        data = self.dev.getData(LEN_DATA)
        msg = data[0]
        src = self.peer_by_addr.get((data[1], data[2]))
        if src is None:
            self._unknown += 1          # drop frames from unconfigured peers
            return

        # Each leg is guarded on (msg == _expected) and the matching peer, so a
        # stale/duplicate/cross-peer frame can't drive the FSM (codex High 4).
        # ---- responder side ----
        if msg == C.POLL:
            # A POLL starts a fresh responder exchange — accept unless we're
            # mid-initiation (initiator-only states). In the fleet, TDMA slots
            # keep a node from initiating and responding at the same instant.
            if self._expected in (C.POLL_ACK, C.RANGE_REPORT):
                return
            self._protocol_failed = False
            self.t_poll_recv = self.dev.getReceiveTimestamp()
            self._resp_peer = src
            self._expected = C.RANGE
            self._transmit_pollack()
            self._note_activity()
            return
        if msg == C.RANGE:
            if self._expected != C.RANGE or src != self._resp_peer:
                return
            self.t_range_recv = self.dev.getReceiveTimestamp()
            self._expected = C.POLL
            self._resp_peer = None
            if not self._protocol_failed:
                self.t_poll_sent = self.dev.getTimeStamp(data, 3)
                self.t_pollack_recv = self.dev.getTimeStamp(data, 8)
                self.t_range_sent = self.dev.getTimeStamp(data, 13)
                distance = self._compute_range()
                self._transmit_range_report(distance)
                self._emit(src, distance)        # responder publishes
            self._note_activity()
            return

        # ---- initiator side ----
        if msg == C.POLL_ACK:
            if self._expected != C.POLL_ACK or src != self._init_peer:
                return
            self.t_pollack_recv = self.dev.getReceiveTimestamp()
            self._expected = C.RANGE_REPORT
            self._transmit_range()
            self._note_activity()
            return
        if msg == C.RANGE_REPORT:
            if self._expected != C.RANGE_REPORT or src != self._init_peer:
                return
            distance = self._decode_distance(data)   # echoed by the responder
            self._expected = C.POLL
            self._init_peer = None
            self._emit(src, distance)            # initiator publishes too
            self._note_activity()
            return
