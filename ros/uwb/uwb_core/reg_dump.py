"""Dump the live RF/PHY config registers governing radiate + detect.

codex part-A: the registers that can be wrong while TXFRS still fires, PLL still
locks, and RX_STATE still hunts — i.e. that produce total distance-independent
silence (which is what we now see: dead at 30 cm). Read them on BOTH boards and
diff. They must (a) match each other and (b) match the ch5/64MHz expected. A
mismatch is the fault; all-correct pushes us to physical RF-path / external.

Run on each board:  python3 -m uwb_core.reg_dump
"""
from .dw1000 import DW1000
from .dw1000 import DW1000Constants as C

ADDR = "7D:00:22:EA:82:60:3B:91"
MODE = C.MODE_LONGDATA_RANGE_ACCURACY


def _val(d, reg, sub, n):
    b = d.readBytes(reg, sub, [0] * n, n)
    v = 0
    for i in range(n):
        v |= b[i] << (i * 8)
    return v, b


def main():
    d = DW1000.DW1000(irq=24, rst=None, bus=0, device=0)
    d.setup(8)
    d.generalConfiguration(ADDR, MODE)

    # (name, reg, sub, nbytes, expected-or-None)  expected for ch5 / 64MHz PRF /
    # 110kbps / 2048-preamble / PAC16 / preamble-code 10.
    regs = [
        ("SYS_CFG",    C.SYS_CFG,  C.NO_SUB, 4, 0x20441200),
        ("TX_FCTRL",   C.TX_FCTRL, C.NO_SUB, 5, None),
        ("CHAN_CTRL",  C.CHAN_CTRL, C.NO_SUB, 4, 0x52BA0055),
        ("AGC_TUNE1",  C.AGC_TUNE, C.AGC_TUNE1_SUB, 2, 0x8870),
        ("AGC_TUNE2",  C.AGC_TUNE, C.AGC_TUNE2_SUB, 4, 0x2502A907),
        ("AGC_TUNE3",  C.AGC_TUNE, C.AGC_TUNE3_SUB, 2, 0x0035),
        ("DRX_TUNE0b", C.DRX_TUNE, C.DRX_TUNE0b_SUB, 2, 0x0016),
        ("DRX_TUNE1a", C.DRX_TUNE, C.DRX_TUNE1a_SUB, 2, 0x008D),
        ("DRX_TUNE1b", C.DRX_TUNE, 0x06, 2, None),
        ("DRX_TUNE2",  C.DRX_TUNE, C.DRX_TUNE2_SUB, 4, 0x333B00BE),
        ("DRX_TUNE4H", C.DRX_TUNE, C.DRX_TUNE4H_SUB, 2, None),
        ("RF_RXCTRLH", C.RF_CONF,  C.RF_RXCTRLH_SUB, 1, 0xD8),
        ("FS_XTALT",   C.FS_CTRL,  C.FS_XTALT_SUB, 1, None),
        ("LDE_CFG1",   C.LDE_IF,   C.LDE_CFG1_SUB, 1, None),
        ("LDE_CFG2",   C.LDE_IF,   C.LDE_CFG2_SUB, 2, 0x1607),
        ("LDE_REPC",   C.LDE_IF,   C.LDE_REPC_SUB, 2, None),
        ("PMSC_CTRL0", C.PMSC,     C.PMSC_CTRL0_SUB, 4, None),
    ]

    print("=== reg dump (ch5/64MHz/110k expected) ===")
    for name, reg, sub, n, exp in regs:
        v, _ = _val(d, reg, sub, n)
        width = n * 2
        if exp is None:
            tag = ""
        else:
            tag = "OK" if v == exp else ("*MISMATCH exp 0x%0*X*" % (width, exp))
        print("  %-11s = 0x%0*X  %s" % (name, width, v, tag))
    d.close()


if __name__ == "__main__":
    main()
