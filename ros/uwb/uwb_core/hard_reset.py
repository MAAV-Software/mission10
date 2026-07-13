"""Pulse the DW1000 RSTn line (GPIO25) low then release — a true hardware reset.

softReset() only resets the digital core over SPI; it cannot clear a wedged RF
front-end. RSTn does the full reset. The DW1000 RSTn is open-drain with an
internal pull-up: drive it LOW to reset, then RELEASE (never drive it high).
We release with a pull-up to defeat any board pulldown (the GPIO25 gotcha) so
the line rises and the chip comes out of reset.

Run: python3 -m uwb_core.hard_reset
"""
import time
from .dw1000.linux_io import GPIO

RST = 25


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RST, GPIO.OUT)      # output_value INACTIVE == drive low == reset
    GPIO.output(RST, GPIO.LOW)
    time.sleep(0.005)
    GPIO.setup(RST, GPIO.IN, pull_up_down=GPIO.PUD_UP)   # release, pull high
    time.sleep(0.020)
    GPIO.cleanup(RST)
    print("RSTn pulsed low->released on GPIO%d" % RST)


if __name__ == "__main__":
    main()
