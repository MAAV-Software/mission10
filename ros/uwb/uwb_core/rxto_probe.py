"""Is ANY RF energy reaching the receiver? — preamble/SFD timeout instrumentation.

The deaf-link blindspot: with no RX timeouts configured, silent non-detection
latches NO status bit, so we cannot tell "no RF at all" from "RF arrived but the
frame failed". This probe turns on the receiver's own timeout reporters and lets
the chip tell us where reception dies:

  RXPTO  (bit 21)  preamble-detect timeout (DRX_PRETOC) : no preamble ever seen
  RXRFTO (bit 17)  frame-wait timeout      (RX_FWTO)    : armed, nothing complete
  RXSFDTO(bit 26)  SFD timeout                          : PREAMBLE seen, SFD failed
  RXPHE  (bit 12)  PHY-header error                     : preamble+SFD ok, PHR bad
  RXFCE  (bit 15)  CRC error                            : whole frame seen, CRC bad
  RXFCG  (bit 14)  good frame                           : link actually works

Diagnostic read:
  * Histogram dominated by RXPTO/RXRFTO  -> NO RF reaching RX  (TX-dead / supply
    brownout of the PA, or antenna/RF-path open). Power hypothesis supported.
  * Any RXSFDTO/RXPHE/RXFCE              -> RF IS arriving, front end senses it;
    the fault is SFD/PHY-layer, not radiate/sense. Power hypothesis weakened.

Run as a PAIR (half-duplex, one radio each):
  board A:  python3 -m uwb_core.rxto_probe tx     # continuous frame blaster
  board B:  python3 -m uwb_core.rxto_probe rx      # instrumented listener

Swap roles to test the other direction. Optional: rx N (cycles, default 300).
"""
import sys
import time
from .dw1000 import DW1000
from .dw1000 import DW1000Constants as C

ADDR = "7D:00:22:EA:82:60:3B:91"
MODE = C.MODE_LONGDATA_RANGE_ACCURACY

# SYS_CFG bit 28 = RXWTOE (RX wait-timeout enable). Not in Constants; use literal.
RXWTOE_BIT = 28
# RX_FWTO: RX frame wait-timeout period, reg 0x0C, 2 bytes, LSB ~= 1.0256 us.
RX_FWTO = 0x0C
# DRX_PRETOC: preamble-detect timeout, sub of DRX_CONF (0x27), 2 bytes, PAC units.
DRX_PRETOC_SUB = 0x24

# Terminal SYS_STATUS bits we tally, in priority order (first hit wins per cycle).
TERMINAL = [
    ("RXFCG",  C.RXFCG_BIT),
    ("RXFCE",  C.RXFCE_BIT),
    ("RXPHE",  C.RXPHE_BIT),
    ("RXSFDTO", C.RXSFDTO_BIT),
    ("RXRFSL", C.RXRFSL_BIT),
    ("LDEERR", C.LDEERR_BIT),
    ("RXPTO",  C.RXPTO_BIT),
    ("RXRFTO", C.RXRFTO_BIT),
]


def _open():
    d = DW1000.DW1000(irq=24, rst=None, bus=0, device=0)
    d.setup(8)
    d.generalConfiguration(ADDR, MODE)
    return d


def _bit_set(status5, bit):
    return (status5[bit // 8] >> (bit % 8)) & 1


def run_tx():
    d = _open()
    frame = [0xC5, 0, 0]  # minimal blink-ish payload; content is irrelevant here
    n = 0
    print("TX blaster up — transmitting continuously (Ctrl-C to stop)")
    try:
        while True:
            d.newTransmit()
            d.setData(frame, len(frame))
            d.startTransmit()  # plain TX, no WAIT4RESP — we just want RF on air
            # poll TXFRS so we pace to actual completion, then immediately re-fire
            for _ in range(200):
                ss = d.readBytes(C.SYS_STATUS, C.NO_SUB, [0] * 5, 5)
                if _bit_set(ss, C.TXFRS_BIT):
                    break
                time.sleep(0.0002)
            n += 1
            if n % 200 == 0:
                print("  ... %d frames sent" % n)
            time.sleep(0.003)
    except KeyboardInterrupt:
        print("\nstopped after %d frames" % n)
    finally:
        d.close()


def run_rx(cycles):
    d = _open()

    # Enable RX wait-timeout (RXWTOE) in the cached SYS_CFG, then write it.
    d.setBit(d._syscfg, 4, RXWTOE_BIT, True)
    d.writeBytes(C.SYS_CFG, C.NO_SUB, d._syscfg, 4)
    # Frame-wait timeout ~5 ms (5000us / 1.0256).
    fwto = 4875
    d.writeBytes(RX_FWTO, C.NO_SUB, [fwto & 0xFF, (fwto >> 8) & 0xFF], 2)
    # Preamble-detect timeout window (PAC symbols) — short, fires fast on no-preamble.
    pretoc = 16
    d.writeBytes(C.DRX_TUNE, DRX_PRETOC_SUB,
                 [pretoc & 0xFF, (pretoc >> 8) & 0xFF], 2)

    counts = {name: 0 for name, _ in TERMINAL}
    counts["NONE"] = 0
    print("RX instrumented listener — %d cycles, RX_FWTO~5ms, DRX_PRETOC=%d PAC"
          % (cycles, pretoc))
    print("(have the peer running `rxto_probe tx`)")
    try:
        for i in range(cycles):
            d.newReceive()
            d.startReceive()
            hit = None
            deadline = time.time() + 0.060   # generous backstop over the 5ms FWTO
            while time.time() < deadline:
                ss = d.readBytes(C.SYS_STATUS, C.NO_SUB, [0] * 5, 5)
                for name, bit in TERMINAL:
                    if _bit_set(ss, bit):
                        hit = name
                        break
                if hit:
                    break
                time.sleep(0.0003)
            counts[hit if hit else "NONE"] += 1
            if (i + 1) % 50 == 0:
                print("  %d/%d  %s" % (i + 1, cycles,
                      " ".join("%s=%d" % (k, v) for k, v in counts.items() if v)))
    except KeyboardInterrupt:
        pass
    finally:
        d.close()

    total = sum(counts.values())
    print("\n=== RX timeout histogram (%d cycles) ===" % total)
    for name in [n for n, _ in TERMINAL] + ["NONE"]:
        c = counts[name]
        if c:
            print("  %-8s %5d  %5.1f%%" % (name, c, 100.0 * c / total))
    seen_rf = sum(counts[k] for k in
                  ("RXFCG", "RXFCE", "RXPHE", "RXSFDTO", "RXRFSL", "LDEERR"))
    no_rf = counts["RXPTO"] + counts["RXRFTO"] + counts["NONE"]
    print("\n  RF-arrived (preamble+ detected): %d   no-RF (timeout/idle): %d"
          % (seen_rf, no_rf))
    if seen_rf == 0:
        print("  -> NO RF reaching the receiver. Supports TX-dead / supply / RF-path.")
    else:
        print("  -> RF IS arriving and being sensed. Fault is SFD/PHY, not radiate/sense.")


def main():
    role = sys.argv[1] if len(sys.argv) > 1 else "rx"
    if role == "tx":
        run_tx()
    elif role == "rx":
        cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        run_rx(cycles)
    else:
        print("usage: python3 -m uwb_core.rxto_probe {tx|rx [cycles]}")


if __name__ == "__main__":
    main()
