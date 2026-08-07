#!/usr/bin/env python3
"""Propose or materialize human-confirmed real-image hard negatives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from audit.hard_negatives import materialize, propose  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    proposal = commands.add_parser("propose")
    proposal.add_argument("--labels", required=True, type=Path)
    proposal.add_argument("--baseline", required=True, type=Path)
    proposal.add_argument("--review", required=True, type=Path)
    build = commands.add_parser("materialize")
    build.add_argument("--labels", required=True, type=Path)
    build.add_argument("--baseline", required=True, type=Path)
    build.add_argument("--review", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "propose":
        if args.review.exists():
            parser.error(f"refusing to overwrite review: {args.review}")
        review = propose(args.labels, args.baseline)
        args.review.parent.mkdir(parents=True, exist_ok=True)
        args.review.write_text(json.dumps(review, indent=2) + "\n")
        print(
            f"Wrote {len(review['entries'])} pending candidates to {args.review}; "
            "set each confirmation to confirmed or rejected after visual review."
        )
    else:
        lock = materialize(args.review, args.labels, args.baseline, args.out)
        print(f"Materialized {lock['counts']['train']['tiles']} confirmed train tiles")


if __name__ == "__main__":
    main()
