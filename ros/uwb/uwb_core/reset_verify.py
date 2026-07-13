"""Verify GPIO25 actually drives the DW1000 RSTn line.

Hold RSTn low and read DEV_ID: a chip genuinely held in reset returns 0x0000
(or garbage), NOT 0xDECA. If DEV_ID is still DECA while we hold GPIO25 low,
GPIO25 is not wired to RSTn (and our 'hard reset' has been a no-op).

Run: python3 -m uwb_core.reset_verify
"""
import time
from .dw1000.linux_io import GPIO
from .dw1000 import DW1000
from .dw1000 import DW1000Constants as C

RST = 25


def _devid(d):
    v = d.readBytes(C.DEV_ID, C.NO_SUB, [0] * 4, 4)
    return (v[3] << 8) | v[2]


def main():
    d = DW1000.DW1000(irq=24, rst=None, bus=0, device=0)
    d.setup(8)
    print("DEV_ID normal               : %04X" % _devid(d))

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RST, GPIO.OUT)
    GPIO.output(RST, GPIO.LOW)
    time.sleep(0.05)
    print("DEV_ID while RSTn held LOW   : %04X  (expect NOT DECA if GPIO25==RSTn)"
          % _devid(d))

    GPIO.setup(RST, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    time.sleep(0.05)
    print("DEV_ID after release         : %04X" % _devid(d))
    GPIO.cleanup(RST)
    d.close()


if __name__ == "__main__":
    main()
