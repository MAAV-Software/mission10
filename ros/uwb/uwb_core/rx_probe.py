"""Is the receiver actually ARMED, or silently idle?

Arms permanent RX exactly as Dw1000Ranger does, then samples SYS_STATE (0x19)
and SYS_STATUS (0x0F) over ~2 s. SYS_STATE byte[2] is RX_STATE: 0x00 = IDLE
(receiver off), non-zero = receiver active (preamble hunt / decoding). If
RX_STATE is non-zero we know RX is genuinely listening and simply hears nothing
(RF/antenna/environment); if it's 0x00 the receiver never armed (software/config).

Run: python3 -m uwb_core.rx_probe
"""
import time
from .dw1000 import DW1000
from .dw1000 import DW1000Constants as C

SYS_STATE = 0x19


def main():
    d = DW1000.DW1000(irq=24, rst=None, bus=0, device=0)
    d.setup(8)
    d.generalConfiguration("7D:00:22:EA:82:60:3B:91", C.MODE_LONGDATA_RANGE_ACCURACY)
    d.newReceive()
    d.receivePermanently()
    d.startReceive()

    for i in range(10):
        st = d.readBytes(SYS_STATE, C.NO_SUB, [0] * 5, 5)
        ss = d.readBytes(C.SYS_STATUS, C.NO_SUB, [0] * 5, 5)
        tx_state = st[0] & 0x0F
        rx_state = st[2] & 0x1F
        pmsc = st[4] & 0x0F
        print("t=%4dms  RX_STATE=0x%02X TX_STATE=0x%02X PMSC=0x%01X  SYS_STATUS=%s"
              % (i * 200, rx_state, tx_state, pmsc, [hex(x) for x in ss]))
        time.sleep(0.2)
    d.close()


if __name__ == "__main__":
    main()
