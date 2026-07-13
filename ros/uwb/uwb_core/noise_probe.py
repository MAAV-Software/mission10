"""Noise-floor / RX-liveness probe — does a *powered* RX front-end actually LISTEN?

codex caveat: bigrpi5's +185 mA when RX-armed proves the receiver is POWERED,
not that it's SENSITIVE. An EOS/ESD-degraded front-end can pull full bias and
still hear nothing. We can't tell those apart from current alone — but the chip
reports its own noise-floor estimate, which a live analog front-end produces
just from thermal + ambient energy at the antenna, with NO peer transmitting.

So: arm RX, let it hunt in ambient noise (no frame ever arrives), then read the
front-end's self-reported energy. A live RX -> non-trivial, board-comparable
STD_NOISE / CIR_PWR / AGC EDV2. A dead front-end -> stuck-low or railed.

Run on BOTH boards and compare:  python3 -m uwb_core.noise_probe
  - bigrpi5 (confirmed powered): establishes the 'live front-end' baseline.
  - drone0 (unmeasurable current): if its noise floor matches bigrpi5's, its RX
    is alive too -> drone0 is NOT a dead-RX; if it reads dead while bigrpi5 reads
    live, drone0's RX front-end is localized dead with no meter needed. If BOTH
    read dead, it's the EOS-hit-both story, not drone0-specifically.
"""
import time

from .dw1000 import DW1000
from .dw1000 import DW1000Constants as C

ADDR = "7D:00:22:EA:82:60:3B:91"
MODE = C.MODE_LONGDATA_RANGE_ACCURACY
SAMPLES = 10
DWELL = 0.25  # s armed-and-hunting before each read


def _val(d, reg, sub, n):
    b = d.readBytes(reg, sub, [0] * n, n)
    v = 0
    for i in range(n):
        v |= (b[i] & 0xFF) << (i * 8)
    return v


def _rx_state(d):
    # SYS_STATE 0x19, byte[2] = RX_STATE; 0x05 == preamble-hunting (RX armed)
    b = d.readBytes(0x19, C.NO_SUB, [0] * 5, 5)
    return b[2] & 0xFF


def main():
    d = DW1000.DW1000(irq=24, rst=None, bus=0, device=0)
    d.setup(8)
    d.generalConfiguration(ADDR, MODE)

    print("=== noise-floor / RX-liveness probe ===")
    print("  (RX armed, NO peer transmitting — reading ambient front-end energy)")
    print("  sample  RXST  STD_NOISE  CIR_PWR  AGC_STAT1   EDG1  EDV2   PLL(C/RF/CLK)")

    n_std = 0
    sum_std = 0
    for i in range(SAMPLES):
        d.newReceive()
        d.startReceive()
        time.sleep(DWELL)  # let it hunt in noise

        rxst = _rx_state(d)
        std = _val(d, C.RX_FQUAL, C.STD_NOISE_SUB, 2)
        cir = _val(d, C.RX_FQUAL, C.CIR_PWR_SUB, 2)
        agc = _val(d, C.AGC_TUNE, 0x1E, 3)  # AGC_STAT1
        edg1 = (agc >> 6) & 0x1F
        edv2 = (agc >> 11) & 0x1FF

        sys = _val(d, C.SYS_STATUS, C.NO_SUB, 5)
        cplock = (sys >> C.CPLOCK_BIT) & 1
        rfpll_ll = (sys >> C.RFPLL_LL_BIT) & 1   # latched RF PLL loss-of-lock
        clkpll_ll = (sys >> C.CLKPLL_LL_BIT) & 1  # latched CLK PLL loss-of-lock

        print("  %4d    0x%02X  %5d      %5d    0x%06X  %4d  %4d   %d/%d/%d"
              % (i, rxst, std, cir, agc, edg1, edv2, cplock, rfpll_ll, clkpll_ll))
        if std:
            n_std += 1
            sum_std += std

    print("  ----")
    print("  STD_NOISE nonzero: %d/%d   mean(nonzero)=%s"
          % (n_std, SAMPLES, ("%.1f" % (sum_std / n_std)) if n_std else "n/a"))
    print("  VERDICT: nonzero board-comparable STD_NOISE/EDV2 => RX front-end is")
    print("           LISTENING (alive). All-zero / railed => front-end deaf.")
    d.close()


if __name__ == "__main__":
    main()
