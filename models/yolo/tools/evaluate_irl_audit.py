#!/usr/bin/env python3
"""Evaluate a saved IRL audit against certified labels without inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from audit.labels import ROLES  # noqa: E402
from audit.offline_evaluation import DEFAULT_THRESHOLD, run  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--fold-lock", type=Path)
    parser.add_argument("--held-out-fold", type=int)
    parser.add_argument(
        "--merge-overlap",
        type=float,
        help="override audit merge overlap (default: value saved in audit)",
    )
    args = parser.parse_args(argv)
    try:
        run(
            args.audit,
            args.labels,
            args.role,
            args.out,
            threshold=args.threshold,
            merge_overlap=args.merge_overlap,
            fold_lock_path=args.fold_lock,
            held_out_fold=args.held_out_fold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
