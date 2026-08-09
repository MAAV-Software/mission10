#!/usr/bin/env python3
"""Build an immutable exact-photo fold lock from certified real labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from audit.folds import DEFAULT_FOLDS, DEFAULT_ROLE, DEFAULT_SEED, build_fold_document  # noqa: E402
from audit.labels import load_labels, sha256  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite fold lock: {args.out}")
    try:
        labels = load_labels(args.labels, require_frozen=True, require_certified=True)
        document = build_fold_document(
            labels,
            sha256(args.labels),
            folds=args.folds,
            seed=args.seed,
            role=args.role,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"folds_sha256={sha256(args.out)}")
    for fold, counts in document["counts"].items():
        print(f"fold {fold}: {counts}")


if __name__ == "__main__":
    main()
