"""Headless DS-TWR ranging demo — the ROS-free "test now" harness.

Runs one `Dw1000Ranger` and prints every range it emits. Replaces the upstream
RangingAnchor.py / RangingTag.py pair with a single symmetric program: run it
on both boards, lower index initiates, and **both** print the pair's range (the
responder computes it, the initiator reads the echoed value).

Usage (env or flags), e.g. a two-board bench:
    # on drone0 (index 0 — initiates):
    python3 -m uwb_core.range_demo --index 0 --peer 1
    # on bigrpi5 (index 1 — responds):
    python3 -m uwb_core.range_demo --index 1 --peer 0

Each node's 2-byte app address is derived from its index, so peers agree on the
mapping without extra config. Pins default to the bench wiring (IRQ=GPIO24,
CSn=CE0/GPIO8, RST floating).
"""
from __future__ import annotations

import argparse
import time

from .ranger import Dw1000Ranger
from .dw1000 import DW1000Constants as C

# Named radio modes (both peers MUST select the same one). The far/near regimes
# of RFD §4.5: LONGDATA_RANGE_ACCURACY for long range, SHORTDATA_FAST_ACCURACY
# for the close-in high-rate regime (6.8 Mb/s, short preamble, same 64 MHz PRF
# so ranging precision is preserved — only airtime shrinks).
_MODES = {
    "long_range": C.MODE_LONGDATA_RANGE_ACCURACY,
    "short_fast": C.MODE_SHORTDATA_FAST_ACCURACY,
    "long_fast": C.MODE_LONGDATA_FAST_ACCURACY,
}


def _addr_for(index: int) -> tuple[int, int]:
    """Deterministic 2-byte app address per node index (both peers compute the
    same map, so no address has to be configured by hand)."""
    return (0xA0 + (index & 0x0F), 0xC0 + (index & 0x0F))


def main(argv=None):
    ap = argparse.ArgumentParser(description="DW1000 DS-TWR headless ranging demo")
    ap.add_argument("--index", type=int, required=True, help="this node's index")
    ap.add_argument("--peer", type=int, action="append", default=[],
                    help="a peer node index (repeatable)")
    ap.add_argument("--irq", type=int, default=24)
    ap.add_argument("--ss", type=int, default=8)
    ap.add_argument("--bus", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--poll-interval", type=float, default=0.1,
                    help="seconds between polls when initiating")
    ap.add_argument("--mode", choices=sorted(_MODES), default="long_range",
                    help="radio mode (both peers must match)")
    ap.add_argument("--reply-delay-us", type=float, default=None,
                    help="override REPLY_DELAY_US for the delayed-TX legs")
    ap.add_argument("--spi-hz", type=int, default=None,
                    help="SPI clock after config (default 2 MHz; chip allows ≤20 MHz)")
    args = ap.parse_args(argv)

    if args.reply_delay_us is not None:
        import uwb_core.ranger as _r
        _r.REPLY_DELAY_US = args.reply_delay_us

    peers = args.peer or [1 - args.index]   # 2-board default: the other one
    peer_cfg = [(p, _addr_for(p)) for p in peers]

    def on_range(source_id, receiver_id, range_m, seq):
        print("range  recv=%d  src=%d  %.2f m  (seq %d)"
              % (receiver_id, source_id, range_m, seq), flush=True)

    ranger = Dw1000Ranger(
        own_index=args.index,
        own_addr=_addr_for(args.index),
        peers=peer_cfg,
        on_range=on_range,
        irq=args.irq, ss=args.ss, bus=args.bus, device=args.device,
        poll_interval_s=args.poll_interval,
        mode=_MODES[args.mode],
        spi_hz=args.spi_hz,
    )
    role = "initiator" if any(p > args.index for p in peers) else "responder"
    print("DW1000 ranger up: index=%d addr=%02X:%02X peers=%s role=%s"
          % (args.index, _addr_for(args.index)[0], _addr_for(args.index)[1],
             peers, role), flush=True)
    n = [0]

    def on_range_counting(source_id, receiver_id, range_m, seq):
        n[0] += 1
        on_range(source_id, receiver_id, range_m, seq)

    ranger.on_range = on_range_counting
    t_start = time.monotonic()
    try:
        ranger.run()
    except KeyboardInterrupt:
        pass
    finally:
        ranger.close()
        time.sleep(0.1)
        dt = max(1e-6, time.monotonic() - t_start)
        print("STATS rate=%.1f Hz emitted=%d (%.1fs)  fsm=%s  irq=%s"
              % (n[0] / dt, n[0], dt, ranger.stats, ranger.dev.irq_counts),
              flush=True)


if __name__ == "__main__":
    main()
