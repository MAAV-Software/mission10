#!/usr/bin/env python3
"""Review proposed real-image hard-negative crops in a local browser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from audit.hard_negative_review import serve  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--qa-size",
        type=int,
        default=32,
        help="representative QA sample size including saved decisions (default: 32)",
    )
    args = parser.parse_args(argv)
    if args.qa_size < 1:
        parser.error("--qa-size must be positive")
    for name in ("labels", "baseline", "review"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    serve(
        args.review,
        args.labels,
        args.baseline,
        host=args.host,
        port=args.port,
        qa_size=args.qa_size,
    )


if __name__ == "__main__":
    main()
