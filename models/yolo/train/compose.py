"""Compose locked prepared datasets into one deterministic training corpus.

Each preset specifies the fraction of one Ultralytics training epoch assigned
to each component.  The smallest exact epoch that contains every component's
training set is used; small components repeat in sorted order.  The production
validation and test sets are authoritative and are included in full exactly
once for every arm.

    python3 train/compose.py --preset combined --out /workspace/combined \
        --component production=/workspace/production/prepared \
        --component appearance=/workspace/appearance/prepared \
        --component hardneg=/workspace/hardneg/prepared
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Mapping

YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from audit.folds import load_fold_document, source_hashes  # noqa: E402


LOCK_SCHEMA = "mission10-yolo-composition/1"
SOURCE_SCHEMA = "mission10-yolo-dataset/1"
TRAINING_COMPONENT_SCHEMA = "mission10-yolo-training-component/1"
SPLIT_NAMES = ("train", "val", "test")
IMAGE_SUFFIXES = (
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)

# Fractions are strings so their decimal representation is exact and stable.
COMPOSITION_PRESETS = {
    "control": {"production": "1"},
    "appearance": {"production": "0.85", "appearance": "0.15"},
    "hardneg": {"production": "0.85", "hardneg": "0.15"},
    "combined": {
        "production": "0.70",
        "appearance": "0.15",
        "hardneg": "0.15",
    },
    "real_positive": {
        "production": "0.75",
        "hardneg": "0.15",
        "real_positive": "0.10",
    },
    "real_positive_appearance": {
        "production": "0.65",
        "appearance": "0.10",
        "hardneg": "0.15",
        "real_positive": "0.10",
    },
}
FOLD_SAFE_PRESETS = ("real_positive", "real_positive_appearance")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"invalid component name: {name!r}")
    return name


def _validate_label(path: Path) -> int:
    boxes = 0
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        try:
            cls = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric label") from exc
        if cls != 0:
            raise ValueError(f"{path}:{line_number}: expected class 0, got {cls}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            raise ValueError(f"{path}:{line_number}: center outside [0, 1]")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: invalid box size")
        boxes += 1
    return boxes


def _source_image(directory: Path, tile: str) -> Path:
    matches = [directory / f"{tile}{suffix}" for suffix in IMAGE_SUFFIXES]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one source image for {tile}, found {len(matches)}")
    return matches[0]


def _load_component(name: str, prepared: Path) -> dict:
    prepared = prepared.resolve()
    full_dataset = name in ("production", "appearance")
    lock_name = "split.lock.json" if full_dataset else "component.lock.json"
    lock_path = prepared / lock_name
    if not lock_path.is_file():
        raise ValueError(f"{name}: missing {lock_name} in {prepared}")
    lock = json.loads(lock_path.read_text())
    expected_schema = (
        SOURCE_SCHEMA if full_dataset else TRAINING_COMPONENT_SCHEMA
    )
    if lock.get("schema") != expected_schema:
        raise ValueError(f"{name}: expected source schema {expected_schema}")
    if full_dataset and not (prepared / "dataset.yaml").is_file():
        raise ValueError(f"{name}: missing dataset.yaml in {prepared}")
    entries = lock.get("entries")
    expected_splits = SPLIT_NAMES if full_dataset else ("train",)
    if not isinstance(entries, dict) or set(entries) != set(expected_splits):
        raise ValueError(
            f"{name}: lock entries must contain exactly {expected_splits}"
        )
    if not isinstance(lock.get("dataset_sha256"), str):
        raise ValueError(f"{name}: source lock has no dataset_sha256")

    normalized = {}
    all_tiles = set()
    for split in expected_splits:
        image_dir = prepared / "images" / split
        label_dir = prepared / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"{name}: missing {split} image or label directory")
        split_entries = entries[split]
        if not isinstance(split_entries, list) or not split_entries:
            raise ValueError(f"{name}: {split} entries must be a non-empty list")
        records = []
        split_tiles = set()
        for entry in split_entries:
            tile = entry.get("tile") if isinstance(entry, dict) else None
            if not isinstance(tile, str) or not tile or Path(tile).name != tile:
                raise ValueError(f"{name}: invalid tile identity in {split}")
            if tile in all_tiles:
                raise ValueError(f"{name}: duplicate tile identity: {tile}")
            image = _source_image(image_dir, tile)
            label = label_dir / f"{tile}.txt"
            if not label.is_file():
                raise ValueError(f"{name}: missing label for {tile}")
            image_hash = _sha256(image)
            label_hash = _sha256(label)
            if image_hash != entry.get("image_sha256"):
                raise ValueError(f"{name}: image hash mismatch for {tile}")
            if label_hash != entry.get("label_sha256"):
                raise ValueError(f"{name}: label hash mismatch for {tile}")
            boxes = _validate_label(label)
            if boxes != entry.get("boxes"):
                raise ValueError(f"{name}: box count mismatch for {tile}")
            records.append(
                {
                    "tile": tile,
                    "image": image,
                    "label": label,
                    "image_sha256": image_hash,
                    "label_sha256": label_hash,
                    "boxes": boxes,
                    "provenance": entry.get("provenance"),
                }
            )
            split_tiles.add(tile)
            all_tiles.add(tile)

        records.sort(key=lambda record: record["tile"])
        if len(records) != len(split_tiles):
            raise ValueError(f"{name}: duplicate {split} entries")
        actual_images = {
            path.stem for path in image_dir.iterdir() if path.is_file()
        }
        actual_labels = {
            path.stem
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix == ".txt"
        }
        if actual_images != split_tiles or actual_labels != split_tiles:
            raise ValueError(f"{name}: unindexed or missing files in {split}")
        normalized[split] = records

    return {
        "path": prepared,
        "lock_path": lock_path,
        "schema": lock["schema"],
        "lock_sha256": _sha256(lock_path),
        "dataset_sha256": lock["dataset_sha256"],
        "entries": normalized,
    }


def _epoch_counts(fractions: Mapping[str, Fraction], sizes: Mapping[str, int]) -> dict:
    quantum = math.lcm(*(fraction.denominator for fraction in fractions.values()))
    minimum = max(
        (size * fraction.denominator + fraction.numerator - 1)
        // fraction.numerator
        for name, fraction in fractions.items()
        for size in (sizes[name],)
    )
    epoch_size = ((minimum + quantum - 1) // quantum) * quantum
    counts = {
        name: epoch_size * fraction.numerator // fraction.denominator
        for name, fraction in fractions.items()
    }
    if sum(counts.values()) != epoch_size or any(
        counts[name] < sizes[name] for name in counts
    ):
        raise AssertionError("invalid exact epoch allocation")
    return counts


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _link_record(record: dict, image_dest: Path, label_dest: Path, stem: str) -> None:
    os.link(record["image"], image_dest / f"{stem}{record['image'].suffix.lower()}")
    os.link(record["label"], label_dest / f"{stem}.txt")


def _provenance_source_hashes(value: object) -> set[str]:
    hashes = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "source_sha256" and isinstance(child, str):
                hashes.add(child)
            else:
                hashes.update(_provenance_source_hashes(child))
    elif isinstance(value, list):
        for child in value:
            hashes.update(_provenance_source_hashes(child))
    return hashes


def compose(
    out: Path,
    preset: str,
    components: Mapping[str, Path],
    *,
    fold_lock_path: Path | None = None,
    held_out_fold: int | None = None,
) -> dict:
    """Validate inputs, hard-link a composition, and return its lock."""
    if preset not in COMPOSITION_PRESETS:
        raise ValueError(f"unknown composition preset: {preset!r}")
    requested = COMPOSITION_PRESETS[preset]
    if (fold_lock_path is None) != (held_out_fold is None):
        raise ValueError("fold lock and held-out fold must be provided together")
    if preset in FOLD_SAFE_PRESETS and fold_lock_path is None:
        raise ValueError(f"{preset} composition requires a held-out fold lock")
    if preset not in FOLD_SAFE_PRESETS and fold_lock_path is not None:
        raise ValueError(
            "fold selection is only valid for a real-positive composition"
        )
    if set(components) != set(requested):
        raise ValueError(
            f"{preset} components must be exactly {tuple(requested)}, "
            f"got {tuple(components)}"
        )
    sources = {
        _safe_name(name): _load_component(name, Path(components[name]))
        for name in requested
    }
    selection = None
    if fold_lock_path is not None:
        fold_lock_path = Path(fold_lock_path)
        fold_document = load_fold_document(fold_lock_path)
        held_out_hashes = source_hashes(fold_document, held_out_fold)
        for name in ("hardneg", "real_positive"):
            leaked = set()
            for record in sources[name]["entries"]["train"]:
                leaked.update(
                    _provenance_source_hashes(record.get("provenance"))
                    & held_out_hashes
                )
            if leaked:
                raise ValueError(
                    f"{name} contains {len(leaked)} held-out source photo(s)"
                )
        selection = {
            "fold_lock": str(fold_lock_path.resolve()),
            "fold_lock_sha256": _sha256(fold_lock_path),
            "held_out_fold": held_out_fold,
            "held_out_source_sha256": sorted(held_out_hashes),
        }
    fractions = {name: Fraction(value) for name, value in requested.items()}
    if sum(fractions.values()) != 1:
        raise AssertionError(f"{preset} fractions do not sum to one")
    source_sizes = {
        name: len(source["entries"]["train"]) for name, source in sources.items()
    }
    train_counts = _epoch_counts(fractions, source_sizes)

    out = out.resolve()
    _ensure_empty_output(out)
    linked_counts = {split: {} for split in SPLIT_NAMES}
    linked_boxes = {split: {} for split in SPLIT_NAMES}
    repeat_ranges = {}
    for split in SPLIT_NAMES:
        image_dest = out / "images" / split
        label_dest = out / "labels" / split
        image_dest.mkdir(parents=True)
        label_dest.mkdir(parents=True)
        split_sources = sources.items() if split == "train" else (
            ("production", sources["production"]),
        )
        for name, source in split_sources:
            records = source["entries"][split]
            count = train_counts[name] if split == "train" else len(records)
            boxes = 0
            copies = [0] * len(records)
            for ordinal in range(count):
                source_index = ordinal % len(records)
                record = records[source_index]
                copies[source_index] += 1
                boxes += record["boxes"]
                stem = f"{name}__{ordinal:08d}__{record['tile']}"
                _link_record(record, image_dest, label_dest, stem)
            linked_counts[split][name] = count
            linked_boxes[split][name] = boxes
            if split == "train":
                repeat_ranges[name] = {"min": min(copies), "max": max(copies)}

    component_lock = {
        name: {
            "source": str(source["path"]),
            "schema": source["schema"],
            "dataset_sha256": source["dataset_sha256"],
            "source_lock": source["lock_path"].name,
            "source_lock_sha256": source["lock_sha256"],
            "source_counts": {
                split: len(entries) for split, entries in source["entries"].items()
            },
        }
        for name, source in sources.items()
    }
    content = {
        "preset": preset,
        "requested_epoch_fractions": requested,
        "actual_epoch_fractions": {
            name: str(Fraction(count, sum(train_counts.values())))
            for name, count in train_counts.items()
        },
        "train_epoch_size": sum(train_counts.values()),
        "components": component_lock,
        "counts": linked_counts,
        "boxes": linked_boxes,
        "train_repeat_range": repeat_ranges,
        "ordering": "component preset order; source tile order; round-robin repetition",
        "selection": selection,
    }
    identity = {
        **content,
        "components": {
            name: {
                key: value
                for key, value in component.items()
                if key not in ("source", "source_lock_sha256")
            }
            for name, component in component_lock.items()
        },
    }
    lock = {
        "schema": LOCK_SCHEMA,
        "dataset_sha256": _canonical_sha256(identity),
        **content,
    }
    (out / "split.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    (out / "dataset.yaml").write_text(
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: mine\n"
    )
    return lock


def _component(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("component must be NAME=PREPARED_PATH")
    try:
        return _safe_name(name), Path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=tuple(COMPOSITION_PRESETS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--component",
        required=True,
        action="append",
        type=_component,
        metavar="NAME=PREPARED_PATH",
    )
    parser.add_argument("--fold-lock", type=Path)
    parser.add_argument("--held-out-fold", type=int)
    args = parser.parse_args(argv)
    components = dict(args.component)
    if len(components) != len(args.component):
        parser.error("component names must be unique")
    lock = compose(
        args.out,
        args.preset,
        components,
        fold_lock_path=args.fold_lock,
        held_out_fold=args.held_out_fold,
    )
    print(f"dataset_sha256={lock['dataset_sha256']}")
    print(f"train: {sum(lock['counts']['train'].values())} tiles")
    print(f"val: {sum(lock['counts']['val'].values())} tiles")
    print(f"test: {sum(lock['counts']['test'].values())} tiles")


if __name__ == "__main__":
    main()
