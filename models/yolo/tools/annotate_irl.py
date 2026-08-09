#!/usr/bin/env python3
"""Create and edit private Mission 10 real-image labels in a local browser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from audit.annotation import serve  # noqa: E402
from audit.labels import ROLES, freeze_roles, new_labels, write_labels  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--init", nargs="+", type=Path, metavar="IMAGE")
    parser.add_argument("--capture-group")
    parser.add_argument("--role", choices=sorted(ROLES))
    parser.add_argument("--freeze-by", help="human assigning capture-group roles")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if args.init:
        if args.labels.exists():
            parser.error(f"labels already exist: {args.labels}")
        if not args.capture_group or not args.role or not args.freeze_by:
            parser.error("--init requires --capture-group, --role, and --freeze-by")
        document = new_labels(
            args.init,
            base=args.labels.resolve().parent,
            capture_group=args.capture_group,
            role=args.role,
        )
        write_labels(args.labels, freeze_roles(document, args.freeze_by))
    elif any((args.capture_group, args.role, args.freeze_by)):
        parser.error("role assignment options are only valid with --init")
    if not args.labels.is_file():
        parser.error(f"missing labels: {args.labels}")
    serve(args.labels, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
