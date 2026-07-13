"""Throwaway RF/PLL health probe — read PLL-lock + RF config after init.

Distinguishes a wedged RF path (RF PLL not locking → digital OK but TX/RX dead)
from a healthy radio. Run: python3 -m uwb_core.pll_probe
"""
import time
from .dw1000 import DW1000
from .dw1000 import DW1000Constants as C


def main():
    d = DW1000.DW1000(irq=24, rst=None, bus=0, device=0)
    d.setup(8)
    d.generalConfiguration("7D:00:22:EA:82:60:3B:91", C.MODE_LONGDATA_RANGE_ACCURACY)
    time.sleep(0.2)

    ss = d.readBytes(C.SYS_STATUS, C.NO_SUB, [0] * 5, 5)
    cfg = d.readBytes(C.SYS_CFG, C.NO_SUB, [0] * 4, 4)
    chan = d.readBytes(C.CHAN_CTRL, C.NO_SUB, [0] * 4, 4)

    def b(arr, n, bit):
        return d.getBit(arr, n, bit)

    print("SYS_STATUS =", [hex(x) for x in ss])
    print("  CPLOCK(clk PLL locked) =", b(ss, 5, C.CPLOCK_BIT))
    print("  CLKPLL_LL(clk PLL lost lock) =", b(ss, 5, C.CLKPLL_LL_BIT))
    print("  RFPLL_LL(rf  PLL lost lock) =", b(ss, 5, C.RFPLL_LL_BIT))
    print("SYS_CFG =", [hex(x) for x in cfg], " RXAUTR =", b(cfg, 4, C.RXAUTR_BIT))
    print("CHAN_CTRL =", [hex(x) for x in chan])

    # TX-path check: fire one frame, see if TXFRS asserts.
    d.newTransmit()
    data = [0] * 4
    data[0] = 0xC5
    d.setData(data, 4)
    d.startTransmit()
    time.sleep(0.05)
    ss2 = d.readBytes(C.SYS_STATUS, C.NO_SUB, [0] * 5, 5)
    print("after TX: TXFRS =", b(ss2, 5, C.TXFRS_BIT),
          " RFPLL_LL =", b(ss2, 5, C.RFPLL_LL_BIT),
          " SYS_STATUS =", [hex(x) for x in ss2])
    d.close()


if __name__ == "__main__":
    main()
