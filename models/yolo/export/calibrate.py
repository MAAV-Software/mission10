"""Build a reproducible, train-only image set for Hailo quantization.

The selector first covers every observed value on the important synthetic-data
axes, including every training scene. It then fills the remaining capacity with
a deterministic uniform sample. Images are hard-linked into one flat directory
because that is the input layout accepted by ``hailomz optimize --calib-path``.

    python3 export/calibrate.py \
        --prepared /workspace/dataset/pilot40-v1/prepared \
        --raw /workspace/dataset/pilot40-v1/raw \
        --out /workspace/dataset/pilot40-v1/calibration \
        --count 1024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


CALIBRATION_SCHEMA = "mission10-yolo-calibration/1"
DATASET_SCHEMA = "mission10-yolo-dataset/1"
DEFAULT_SEED = "mission10-hailo-calibration-v1"


@dataclass(frozen=True)
class Candidate:
    tile: str
    scene: str
    image_sha256: str
    label_sha256: str
    features: Tuple[str, ...]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc


def _altitude_band(altitude_m: float) -> str:
    if altitude_m < 3.0:
        return "low_lt3m"
    if altitude_m < 5.0:
        return "mid_3to5m"
    return "high_ge5m"


def _content_band(label_path: Path, tile_px: int) -> str:
    max_side_px = 0.0
    boxes = 0
    for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        try:
            width, height = float(fields[3]), float(fields[4])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: invalid box") from exc
        boxes += 1
        max_side_px = max(max_side_px, width * tile_px, height * tile_px)
    if boxes == 0:
        return "empty"
    if max_side_px < 16.0:
        return "tiny_lt16px"
    if max_side_px < 32.0:
        return "small_16to32px"
    return "large_ge32px"


def _tag_band(fraction: float) -> str:
    if fraction < 0.34:
        return "low"
    if fraction < 0.67:
        return "mixed"
    return "high"


def _surface_name(surface: Mapping[str, object]) -> str:
    primary = str(surface.get("primary", "unknown"))
    secondary = surface.get("secondary")
    if secondary is None:
        return primary
    return "+".join(sorted((primary, str(secondary))))


def _scene_metadata(raw: Path, scene: str) -> Tuple[dict, Dict[str, dict]]:
    manifest_path = raw / f"{scene}.manifest.json"
    data = _load_json(manifest_path)
    if not isinstance(data, dict):
        raise ValueError(f"invalid manifest: {manifest_path}")
    stations = data.get("stations")
    if not isinstance(stations, list):
        raise ValueError(f"manifest has no station list: {manifest_path}")
    by_stem = {}
    for station in stations:
        if not isinstance(station, dict) or not isinstance(station.get("stem"), str):
            raise ValueError(f"invalid station in {manifest_path}")
        by_stem[station["stem"]] = station
    return data, by_stem


def _candidate_features(
    manifest: Mapping[str, object], station: Mapping[str, object], content: str
) -> Tuple[str, ...]:
    pos = station.get("pos")
    if not isinstance(pos, list) or len(pos) != 3:
        raise ValueError(f"invalid station position: {station!r}")
    altitude = abs(float(pos[2]))
    surface = manifest.get("surface")
    if not isinstance(surface, dict):
        raise ValueError("manifest has no surface metadata")
    grass = manifest.get("grass")
    grass_profile = "none"
    if isinstance(grass, dict):
        grass_profile = str(grass.get("profile", "unknown"))
    mines = manifest.get("mines")
    if not isinstance(mines, list):
        raise ValueError("manifest has no mine metadata")
    color_families = sorted(
        {
            str(mine.get("appearance", {}).get("color_family", "unknown"))
            for mine in mines
            if isinstance(mine, dict) and isinstance(mine.get("appearance"), dict)
        }
    )
    features = [
        f"surface:{_surface_name(surface)}",
        f"altitude:{_altitude_band(altitude)}",
        f"content:{content}",
        f"grass:{grass_profile}",
        f"tag:{_tag_band(float(manifest.get('tag_visible_fraction', 0.0)))}",
    ]
    features.extend(f"color:{family}" for family in color_families)
    return tuple(features)


def _load_candidates(prepared: Path, raw: Path) -> Tuple[dict, List[Candidate]]:
    lock_path = prepared / "split.lock.json"
    lock = _load_json(lock_path)
    if not isinstance(lock, dict) or lock.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"expected {DATASET_SCHEMA} in {lock_path}")
    entries = lock.get("entries")
    if not isinstance(entries, dict) or not isinstance(entries.get("train"), list):
        raise ValueError(f"invalid train entries in {lock_path}")

    metadata: Dict[str, Tuple[dict, Dict[str, dict]]] = {}
    candidates = []
    for entry in entries["train"]:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid train entry in {lock_path}")
        tile = entry.get("tile")
        scene = entry.get("scene")
        if not isinstance(tile, str) or not isinstance(scene, str):
            raise ValueError(f"invalid train identity in {lock_path}")
        image = prepared / "images" / "train" / f"{tile}.png"
        label = prepared / "labels" / "train" / f"{tile}.txt"
        if not image.is_file() or not label.is_file():
            raise ValueError(f"missing prepared train pair for {tile}")
        if scene not in metadata:
            metadata[scene] = _scene_metadata(raw, scene)
        manifest, stations = metadata[scene]
        source = tile.split("_x", 1)[0]
        station = stations.get(source)
        if station is None:
            raise ValueError(f"tile {tile} has no station in {scene}")
        content = _content_band(label, 640)
        features = (f"scene:{scene}",) + _candidate_features(
            manifest, station, content
        )
        candidates.append(
            Candidate(
                tile=tile,
                scene=scene,
                image_sha256=str(entry.get("image_sha256")),
                label_sha256=str(entry.get("label_sha256")),
                features=features,
            )
        )
    if not candidates:
        raise ValueError("training split contains no calibration candidates")
    return lock, candidates


def _select(
    candidates: Sequence[Candidate], count: int, seed: str
) -> Tuple[List[Candidate], int]:
    if count <= 0:
        raise ValueError("calibration count must be positive")
    if count > len(candidates):
        raise ValueError(
            f"requested {count} calibration images from only {len(candidates)} candidates"
        )

    feature_frequency = Counter(
        feature for candidate in candidates for feature in candidate.features
    )
    uncovered = set(feature_frequency)
    remaining = list(candidates)
    selected = []

    while uncovered and len(selected) < count:
        def coverage_key(candidate: Candidate) -> Tuple[float, int, str]:
            covered = uncovered.intersection(candidate.features)
            rarity = sum(1.0 / feature_frequency[value] for value in covered)
            return (-rarity, -len(covered), _stable_key(seed, candidate.tile))

        chosen = min(remaining, key=coverage_key)
        selected.append(chosen)
        remaining.remove(chosen)
        uncovered.difference_update(chosen.features)

    if uncovered:
        values = ", ".join(sorted(uncovered))
        raise ValueError(f"count {count} cannot cover calibration features: {values}")

    coverage_count = len(selected)
    remaining.sort(key=lambda candidate: _stable_key(seed, candidate.tile))
    selected.extend(remaining[: count - len(selected)])
    return selected, coverage_count


def _feature_counts(candidates: Iterable[Candidate]) -> Dict[str, int]:
    return dict(
        sorted(Counter(feature for item in candidates for feature in item.features).items())
    )


def build_calibration(
    prepared: Path, raw: Path, out: Path, count: int, seed: str = DEFAULT_SEED
) -> dict:
    """Select and hard-link calibration images, then return the lock record."""
    prepared = prepared.resolve()
    raw = raw.resolve()
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    dataset_lock, candidates = _load_candidates(prepared, raw)
    selected, coverage_count = _select(candidates, count, seed)
    images_dir = out / "images"
    images_dir.mkdir()
    records = []
    for index, candidate in enumerate(selected):
        source = prepared / "images" / "train" / f"{candidate.tile}.png"
        filename = f"{index:04d}_{source.name}"
        destination = images_dir / filename
        os.link(source, destination)
        records.append(
            {
                "file": filename,
                "tile": candidate.tile,
                "scene": candidate.scene,
                "image_sha256": candidate.image_sha256,
                "label_sha256": candidate.label_sha256,
                "features": list(candidate.features),
            }
        )

    content = {
        "dataset_sha256": dataset_lock.get("dataset_sha256"),
        "seed": seed,
        "images": records,
    }
    lock = {
        "schema": CALIBRATION_SCHEMA,
        "prepared": str(prepared),
        "raw": str(raw),
        "count": len(selected),
        "candidate_count": len(candidates),
        "coverage_count": coverage_count,
        "calibration_sha256": _canonical_sha256(content),
        **content,
        "candidate_feature_counts": _feature_counts(candidates),
        "selected_feature_counts": _feature_counts(selected),
    }
    (out / "calib.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    lock = build_calibration(
        args.prepared, args.raw, args.out, args.count, args.seed
    )
    print(f"calibration_sha256={lock['calibration_sha256']}")
    print(f"selected {lock['count']} of {lock['candidate_count']} train tiles")


if __name__ == "__main__":
    main()
