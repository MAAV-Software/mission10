"""CIR accumulator probe — the free-running RX-front-end liveness test.

The DW1000 continuously accumulates the channel impulse response (CIR) into
ACC_MEM (reg 0x25) while the receiver is armed. A LIVE analog front-end feeds
thermal + ambient noise into the accumulator even with NO peer transmitting, so
the CIR samples read back small-but-nonzero. A DEAD front-end (EOS'd LNA/mixer,
collapsed RF supply, broken RF path) feeds nothing -> the CIR reads flat zero.

Unlike RX_FQUAL/AGC_STAT1 (latched only on a reception event), the accumulator
is free-running while RXENAB is set, so this works with no frame and no peer.

Reading ACC_MEM requires forcing the accumulator clocks on (PMSC_CTRL0:
byte0 |= 0x48 force-RX/ACC clock, byte1 |= 0x80 AMCE accumulator-mem clock),
reading, then restoring PMSC_CTRL0. The first octet read back is a dummy and is
discarded (Decawave dwt_readaccdata convention). Each CIR sample is a complex
pair: int16 real (LE) + int16 imag (LE) = 4 bytes.

Run on BOTH boards and compare:  python3 -m uwb_core.acc_probe
  live noise floor on bigrpi5 + flat zero on drone0 -> drone0 RX localized dead.
  noise on both -> both front-ends alive; the fault is elsewhere (not the radios).
  flat zero on both -> both deaf (EOS-hit-both), or the accumulator isn't being
  clocked (caveat: re-run interpreting against bigrpi5's known-powered baseline).
"""
import time

from .dw1000 import DW1000
from .dw1000 import DW1000Constants as C

ADDR = "7D:00:22:EA:82:60:3B:91"
MODE = C.MODE_LONGDATA_RANGE_ACCURACY
ACC_MEM = 0x25
NSAMP = 48          # CIR samples to read
ROUNDS = 6          # re-arm + re-read this many times
DWELL = 0.20        # s armed-and-accumulating before each read


def read_acc(d, nsamp):
    """Read nsamp complex CIR samples (real,imag int16) from ACC_MEM."""
    # save PMSC_CTRL0 low 2 bytes, force accumulator clocks on
    pmsc = d.readBytes(C.PMSC, C.PMSC_CTRL0_SUB, [0, 0], 2)
    on = [0x48 | (pmsc[0] & 0xB3), 0x80 | pmsc[1]]
    d.writeBytes(C.PMSC, C.PMSC_CTRL0_SUB, on, 2)

    nbytes = nsamp * 4 + 1                  # +1 dummy leading octet
    raw = d.readBytes(ACC_MEM, 0x00, [0] * nbytes, nbytes)

    d.writeBytes(C.PMSC, C.PMSC_CTRL0_SUB, pmsc, 2)  # restore clocks

    raw = raw[1:]                           # drop dummy octet
    samples = []
    for k in range(nsamp):
        b = raw[k * 4: k * 4 + 4]
        re = b[0] | (b[1] << 8)
        im = b[2] | (b[3] << 8)
        if re >= 32768:
            re -= 65536
        if im >= 32768:
            im -= 65536
        samples.append((re, im))
    return samples


def main():
    d = DW1000.DW1000(irq=24, rst=None, bus=0, device=0)
    d.setup(8)
    d.generalConfiguration(ADDR, MODE)

    print("=== CIR accumulator (ACC_MEM) noise-floor probe ===")
    print("  (RX armed, NO peer — reading free-running CIR noise accumulation)")
    print("  round  nonzero/N   |mag| mean   |mag| max   first 4 samples (re,im)")

    grand_nonzero = 0
    grand_total = 0
    for r in range(ROUNDS):
        d.newReceive()
        d.startReceive()
        time.sleep(DWELL)                   # accumulate noise
        s = read_acc(d, NSAMP)

        mags = [abs(re) + abs(im) for (re, im) in s]   # L1 magnitude
        nz = sum(1 for m in mags if m != 0)
        mean = sum(mags) / len(mags)
        mx = max(mags)
        grand_nonzero += nz
        grand_total += len(mags)
        head = " ".join("(%d,%d)" % s[k] for k in range(4))
        print("  %4d   %3d/%-3d     %8.2f    %6d     %s"
              % (r, nz, NSAMP, mean, mx, head))

    print("  ----")
    pct = 100.0 * grand_nonzero / grand_total if grand_total else 0
    print("  total nonzero CIR samples: %d/%d (%.1f%%)"
          % (grand_nonzero, grand_total, pct))
    print("  VERDICT: substantial nonzero noise CIR => RX analog front-end is")
    print("           LIVE and listening. Flat all-zero => front-end deaf (or")
    print("           accumulator unclocked — calibrate vs bigrpi5's reading).")
    d.close()


if __name__ == "__main__":
    main()
