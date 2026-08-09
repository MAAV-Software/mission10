#!/usr/bin/env python3
"""Build leakage-safe hard-negative and real-positive fold components."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit.real_positives import materialize_fold  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--mask-review", required=True, type=Path)
    parser.add_argument("--fold-lock", required=True, type=Path)
    parser.add_argument("--held-out-fold", required=True, type=int)
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument("--hardneg", required=True, type=Path)
    parser.add_argument("--filtered-hardneg-out", required=True, type=Path)
    parser.add_argument("--real-positive-out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        filtered, positives = materialize_fold(
            args.labels,
            args.mask_review,
            args.fold_lock,
            args.held_out_fold,
            args.production,
            args.hardneg,
            args.filtered_hardneg_out,
            args.real_positive_out,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"filtered_hardneg_tiles={filtered['counts']['train']['tiles']}")
    print(f"real_positive_tiles={positives['counts']['train']['tiles']}")


if __name__ == "__main__":
    main()
