"""Dump key OTP words — is there an LDOTUNE calibration our init drops?

Our port loads only OTP 0x1E (XTAL trim). The canonical Decawave init also loads
LDOTUNE (0x04): the per-chip RF-LDO voltage trim. If 0x04 is non-zero here, we're
running the RF regulators at default and ignoring a baked-in calibration — a
band-independent RF suspect the 3.3V header reading can't see.

Run on each board:  python3 -m uwb_core.otp_probe
"""
from .dw1000 import DW1000
from .dw1000 import DW1000Constants as C

ADDR = "7D:00:22:EA:82:60:3B:91"

WORDS = [
    (0x00, "EUID_lo"),
    (0x04, "LDOTUNE"),     # <-- the one we never load
    (0x06, "ANT_DLY"),     # antenna delay cal (16/64MHz)
    (0x08, "XTRIM/volt"),
    (0x09, "volt/temp"),
    (0x1C, "VBAT/TEMP"),
    (0x1E, "XTAL_TRIM"),   # the one we DO load
]


def _u32(b):
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)


def main():
    d = DW1000.DW1000(irq=24, rst=None, bus=0, device=0)
    d.setup(8)
    d.generalConfiguration(ADDR, C.MODE_LONGDATA_RANGE_ACCURACY)
    print("=== OTP dump ===")
    for addr, name in WORDS:
        v = _u32(d.readBytesOTP(addr, [0] * 4))
        print("  OTP[0x%02X] %-10s = 0x%08X" % (addr, name, v))
    d.close()


if __name__ == "__main__":
    main()
