"""Build a leakage-free Ultralytics dataset view from materialized tiles.

The materializer stores all scenes together under ``raw/train``. This command
uses its authoritative tiles index plus an explicit scene split, validates the
dataset, and hard-links only indexed files into Ultralytics' conventional
images/labels tree. It also writes a content-addressed lock for the exact pilot
corpus.

    python3 train/prepare.py \
        --raw /workspace/dataset/pilot40-v1/raw \
        --out /workspace/dataset/pilot40-v1/prepared \
        --split train/pilot40-split.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


SPLIT_SCHEMA = "mission10-yolo-split/1"
LOCK_SCHEMA = "mission10-yolo-dataset/1"
SPLIT_NAMES = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_split(path: Path) -> Tuple[str, Dict[str, List[int]]]:
    raw = json.loads(path.read_text())
    if raw.get("schema") != SPLIT_SCHEMA:
        raise ValueError(f"expected {SPLIT_SCHEMA}, got {raw.get('schema')!r}")
    seed = raw.get("seed")
    if not isinstance(seed, str) or not seed:
        raise ValueError("split seed must be a non-empty string")
    scenes = raw.get("scenes")
    if not isinstance(scenes, dict) or set(scenes) != set(SPLIT_NAMES):
        raise ValueError(f"split scenes must contain exactly {SPLIT_NAMES}")

    normalized: Dict[str, List[int]] = {}
    owner: Dict[int, str] = {}
    for split in SPLIT_NAMES:
        values = scenes[split]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{split} scenes must be a non-empty list")
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError(f"{split} contains an invalid scene index")
        if values != sorted(set(values)):
            raise ValueError(f"{split} scenes must be sorted and unique")
        for scene in values:
            if scene in owner:
                raise ValueError(
                    f"scene {scene} appears in both {owner[scene]} and {split}"
                )
            owner[scene] = split
        normalized[split] = values
    return seed, normalized


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
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise ValueError(f"{path}:{line_number}: center outside [0, 1]")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{path}:{line_number}: invalid box size")
        boxes += 1
    return boxes


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _expected_scene_keys(seed: str, scenes: Mapping[str, Iterable[int]]) -> set[str]:
    return {
        f"{seed}_s{scene:04d}"
        for values in scenes.values()
        for scene in values
    }


def prepare(raw: Path, out: Path, split_file: Path) -> dict:
    """Validate and hard-link one prepared dataset; return its lock record."""
    raw = raw.resolve()
    out = out.resolve()
    split_file = split_file.resolve()
    seed, splits = _load_split(split_file)
    index_path = raw / "train" / "tiles.json"
    if not index_path.is_file():
        raise ValueError(f"missing tile index: {index_path}")
    index = json.loads(index_path.read_text())
    expected_keys = _expected_scene_keys(seed, splits)
    if set(index) != expected_keys:
        missing = sorted(expected_keys - set(index))
        extra = sorted(set(index) - expected_keys)
        raise ValueError(f"tile-index scene mismatch: missing={missing}, extra={extra}")

    _ensure_empty_output(out)
    split_for_scene = {
        f"{seed}_s{scene:04d}": split
        for split, values in splits.items()
        for scene in values
    }
    entries: Dict[str, List[dict]] = {split: [] for split in SPLIT_NAMES}
    seen_tiles = set()

    for scene_key in sorted(index):
        manifest = raw / f"{scene_key}.manifest.json"
        if not manifest.is_file():
            raise ValueError(f"missing scene manifest: {manifest}")
        manifest_data = json.loads(manifest.read_text())
        if manifest_data.get("seed") != seed or (
            manifest_data.get("scene") != int(scene_key.rsplit("s", 1)[1])
        ):
            raise ValueError(f"manifest identity mismatch: {manifest}")
        split = split_for_scene[scene_key]
        image_dest = out / "images" / split
        label_dest = out / "labels" / split
        image_dest.mkdir(parents=True, exist_ok=True)
        label_dest.mkdir(parents=True, exist_ok=True)

        scene_tiles = index[scene_key].get("tiles")
        if not isinstance(scene_tiles, list):
            raise ValueError(f"invalid tile list for {scene_key}")
        for record in scene_tiles:
            tile = record.get("tile")
            source = record.get("src")
            if not isinstance(tile, str) or not isinstance(source, str):
                raise ValueError(f"invalid tile record in {scene_key}")
            if tile in seen_tiles:
                raise ValueError(f"duplicate tile identity: {tile}")
            if not source.startswith(f"{scene_key}_k"):
                raise ValueError(f"tile {tile} points outside {scene_key}: {source}")
            seen_tiles.add(tile)

            image = raw / "train" / "images" / f"{tile}.png"
            label = raw / "train" / "labels" / f"{tile}.txt"
            if not image.is_file() or not label.is_file():
                raise ValueError(f"missing image/label pair for {tile}")
            boxes = _validate_label(label)
            os.link(image, image_dest / image.name)
            os.link(label, label_dest / label.name)
            entries[split].append(
                {
                    "scene": scene_key,
                    "tile": tile,
                    "boxes": boxes,
                    "image_sha256": _sha256(image),
                    "label_sha256": _sha256(label),
                }
            )

    for split in SPLIT_NAMES:
        entries[split].sort(key=lambda entry: entry["tile"])
    content = {"seed": seed, "scenes": splits, "entries": entries}
    lock = {
        "schema": LOCK_SCHEMA,
        "source": str(raw),
        "source_tiles_sha256": _sha256(index_path),
        "split_file_sha256": _sha256(split_file),
        "dataset_sha256": _canonical_sha256(content),
        **content,
        "counts": {
            split: {
                "tiles": len(entries[split]),
                "boxes": sum(entry["boxes"] for entry in entries[split]),
                "empty": sum(entry["boxes"] == 0 for entry in entries[split]),
            }
            for split in SPLIT_NAMES
        },
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    args = parser.parse_args(argv)
    lock = prepare(args.raw, args.out, args.split)
    print(f"dataset_sha256={lock['dataset_sha256']}")
    for split in SPLIT_NAMES:
        counts = lock["counts"][split]
        print(
            f"{split}: {counts['tiles']} tiles, {counts['boxes']} boxes, "
            f"{counts['empty']} empty"
        )


if __name__ == "__main__":
    main()
