"""Dump the analog RF-TX config registers and compare to expected (ch5/64MHz).

If TX_POWER / RF_TXCTRL / PG_DELAY / FS_PLL read wrong, the transmitter completes
digitally (TXFRS) but radiates nothing — a config cause that fits 'both deaf'.
If they read correct, the digital+config path is perfect and the fault is purely
analog (PA/antenna/LNA) or electrical (RF supply).

Run: python3 -m uwb_core.tx_probe
"""
from .dw1000 import DW1000
from .dw1000 import DW1000Constants as C


def _u32(v):
    return (v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24))


def main():
    d = DW1000.DW1000(irq=24, rst=None, bus=0, device=0)
    d.setup(8)
    d.generalConfiguration("7D:00:22:EA:82:60:3B:91", C.MODE_LONGDATA_RANGE_ACCURACY)

    txpow = _u32(d.readBytes(C.TX_POWER, C.NO_SUB, [0] * 4, 4))
    txctrl = _u32(d.readBytes(C.RF_CONF, C.RF_TXCTRL_SUB, [0] * 4, 4))
    pgdelay = d.readBytes(C.TX_CAL, C.TC_PGDELAY_SUB, [0] * 1, 1)[0]
    pllcfg = _u32(d.readBytes(C.FS_CTRL, C.FS_PLLCFG_SUB, [0] * 4, 4))
    plltune = d.readBytes(C.FS_CTRL, C.FS_PLLTUNE_SUB, [0] * 1, 1)[0]

    def line(name, got, exp):
        ok = "OK " if got == exp else "*MISMATCH*"
        print("  %-10s = 0x%08X  expect 0x%08X  %s" % (name, got, exp, ok))

    line("TX_POWER", txpow, C.TX_POWER_5_64MHZ)
    line("RF_TXCTRL", txctrl, C.RF_TXCTRL_5)
    line("PG_DELAY", pgdelay, C.TC_PGDELAY_5)
    line("FS_PLLCFG", pllcfg, C.FS_PLLCFG_57)
    line("FS_PLLTUNE", plltune, C.FS_PLLTUNE_57)
    d.close()


if __name__ == "__main__":
    main()
