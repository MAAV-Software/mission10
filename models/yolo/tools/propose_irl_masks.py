#!/usr/bin/env python3
"""Generate SAM box-prompt cutout proposals for certified clear mines."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit.masks import propose_masks  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--sam-weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    try:
        review = propose_masks(
            args.labels, args.sam_weights, args.out, device=args.device
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"proposals={len(review['entries'])}")
    print(f"review={args.out / 'mask-review.json'}")


if __name__ == "__main__":
    main()
