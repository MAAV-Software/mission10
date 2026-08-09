"""Leakage-safe real-positive contexts, composites, and hard negatives."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

from .evaluation import scale_probe
from .folds import load_fold_document, source_hashes
from .labels import canonical_sha256, load_labels, resolve_source, sha256
from .masks import load_mask_review


COMPONENT_SCHEMA = "mission10-yolo-training-component/1"
TILE_PX = 640
CONTEXT_SCALES = (30, 60, 120)
COMPOSITES_PER_CLEAR = 8
SEED = "mission10-real-positive-component-v1"


def _label_line(box) -> str:
    width = box.x1 - box.x0
    height = box.y1 - box.y0
    return (
        f"0 {(box.x0 + box.x1) / (2 * TILE_PX):.8f} "
        f"{(box.y0 + box.y1) / (2 * TILE_PX):.8f} "
        f"{width / TILE_PX:.8f} {height / TILE_PX:.8f}\n"
    )


def _write_component(out: Path, entries: list[dict], metadata: dict) -> dict:
    entries.sort(key=lambda entry: entry["tile"])
    content = {"entries": {"train": entries}}
    lock = {
        "schema": COMPONENT_SCHEMA,
        "scope": "train_only",
        **metadata,
        **content,
        "counts": {
            "train": {
                "tiles": len(entries),
                "boxes": sum(entry["boxes"] for entry in entries),
            }
        },
        "dataset_sha256": canonical_sha256(content),
    }
    (out / "component.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def filter_hard_negatives(
    source_component: Path,
    out: Path,
    fold_lock_path: Path,
    held_out_fold: int,
) -> dict:
    """Hard-link a train component after excluding held-out source photos."""
    from train.compose import _load_component

    source = _load_component("hardneg", Path(source_component))
    fold = load_fold_document(fold_lock_path)
    excluded = source_hashes(fold, held_out_fold)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    image_out, label_out = out / "images" / "train", out / "labels" / "train"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    entries = []
    removed = 0
    for record in source["entries"]["train"]:
        provenance = record.get("provenance") or {}
        if provenance.get("source_sha256") in excluded:
            removed += 1
            continue
        image_path = image_out / f"{record['tile']}{record['image'].suffix.lower()}"
        label_path = label_out / f"{record['tile']}.txt"
        os.link(record["image"], image_path)
        os.link(record["label"], label_path)
        entries.append(
            {
                "tile": record["tile"],
                "boxes": record["boxes"],
                "image_sha256": record["image_sha256"],
                "label_sha256": record["label_sha256"],
                "provenance": provenance,
            }
        )
    if not entries:
        raise ValueError("fold filtering removed every hard-negative tile")
    return _write_component(
        out,
        entries,
        {
            "component": "fold-filtered-real-hard-negatives",
            "source_component": str(Path(source_component).resolve()),
            "source_component_lock_sha256": source["lock_sha256"],
            "fold_lock": str(Path(fold_lock_path).resolve()),
            "fold_lock_sha256": sha256(fold_lock_path),
            "held_out_fold": held_out_fold,
            "removed_held_out_tiles": removed,
        },
    )


def _load_rgb(labels_path: Path, record: dict):
    from PIL import Image, ImageOps

    source_path = resolve_source(labels_path, record["source"])
    if sha256(source_path) != record["source_sha256"]:
        raise ValueError(f"source hash changed: {source_path}")
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if image.size != (record["width"], record["height"]):
        raise ValueError(f"oriented dimensions changed: {source_path}")
    return image


def _parse_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, width, height = map(float, line.split())
        boxes.append(
            (
                (cx - width / 2) * TILE_PX,
                (cy - height / 2) * TILE_PX,
                (cx + width / 2) * TILE_PX,
                (cy + height / 2) * TILE_PX,
            )
        )
    return boxes


def _intersects(box, others) -> bool:
    return any(
        min(box[2], other[2]) > max(box[0], other[0])
        and min(box[3], other[3]) > max(box[1], other[1])
        for other in others
    )


def _prepared_cutout(path: Path, target_px: int, angle: float, feather: float):
    from PIL import Image, ImageFilter

    with Image.open(path) as source:
        cutout = source.convert("RGBA")
    cutout = cutout.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    bbox = cutout.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError(f"empty confirmed cutout: {path}")
    cutout = cutout.crop(bbox)
    factor = target_px / max(cutout.size)
    size = tuple(max(1, round(value * factor)) for value in cutout.size)
    cutout = cutout.resize(size, Image.Resampling.LANCZOS)
    alpha = cutout.getchannel("A").filter(ImageFilter.GaussianBlur(feather))
    cutout.putalpha(alpha)
    return cutout


def _target_sizes() -> list[int]:
    low, high = math.log(20), math.log(130)
    return [
        round(math.exp(low + (index + 0.5) * (high - low) / 8)) for index in range(8)
    ]


def _composite(
    cutout_path: Path,
    target_record: dict,
    *,
    target_px: int,
    rng: random.Random,
):
    from PIL import Image

    with Image.open(target_record["image"]) as source:
        background = source.convert("RGBA")
    if background.size != (TILE_PX, TILE_PX):
        raise ValueError(f"composite target is not {TILE_PX}px")
    angle = rng.uniform(0.0, 360.0)
    feather = rng.uniform(1.0, 3.0)
    cutout = _prepared_cutout(cutout_path, target_px, angle, feather)
    existing = _parse_boxes(target_record["label"])
    position = None
    for _ in range(256):
        x = rng.randrange(0, TILE_PX - cutout.width + 1)
        y = rng.randrange(0, TILE_PX - cutout.height + 1)
        candidate = (x, y, x + cutout.width, y + cutout.height)
        if not _intersects(candidate, existing):
            position = (x, y)
            break
    if position is None:
        raise ValueError(f"could not place cutout on {target_record['tile']}")
    background.alpha_composite(cutout, position)
    box = (
        position[0],
        position[1],
        position[0] + cutout.width,
        position[1] + cutout.height,
    )
    label = target_record["label"].read_text()
    cx = (box[0] + box[2]) / (2 * TILE_PX)
    cy = (box[1] + box[3]) / (2 * TILE_PX)
    width = (box[2] - box[0]) / TILE_PX
    height = (box[3] - box[1]) / TILE_PX
    label += f"0 {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}\n"
    return (
        background.convert("RGB"),
        label,
        {
            "target_max_side_px": max(cutout.size),
            "rotation_degrees": angle,
            "feather_radius_px": feather,
            "pasted_xyxy": list(box),
        },
    )


def materialize_real_positives(
    labels_path: Path,
    mask_review_path: Path,
    fold_lock_path: Path,
    held_out_fold: int,
    production_component: Path,
    filtered_hardneg_component: Path,
    out: Path,
) -> dict:
    """Build training-fold contexts and confirmed clear-mask composites."""
    from train.compose import _load_component

    labels = load_labels(labels_path, require_frozen=True, require_certified=True)
    review = load_mask_review(mask_review_path, require_decided=True)
    if review["labels_sha256"] != sha256(labels_path):
        raise ValueError("mask review labels hash changed")
    fold = load_fold_document(fold_lock_path, labels_path)
    training_hashes = source_hashes(fold, held_out_fold, held_out=False)
    records = [
        record
        for record in labels["images"]
        if record["source_sha256"] in training_hashes
    ]
    clear_total = sum(
        annotation["visibility"] == "clear"
        for record in records
        for annotation in record["objects"]
    )
    confirmed = [
        entry
        for entry in review["entries"]
        if entry["source_sha256"] in training_hashes
        and entry["confirmation"] == "confirmed"
    ]
    required = math.ceil(clear_total * 0.75)
    if len(confirmed) < required:
        raise ValueError(
            f"only {len(confirmed)}/{clear_total} training clear masks confirmed; "
            f"need at least {required}"
        )
    production = _load_component("production", Path(production_component))
    hardneg = _load_component("hardneg", Path(filtered_hardneg_component))
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    image_out, label_out = out / "images" / "train", out / "labels" / "train"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    entries = []

    for record in records:
        image = None
        for object_index, annotation in enumerate(record["objects"]):
            if annotation["visibility"] not in {"clear", "partial"}:
                continue
            if image is None:
                image = _load_rgb(labels_path, record)
            for target_px in CONTEXT_SCALES:
                context, truth = scale_probe(image, annotation, target_px, TILE_PX)
                tile = f"rp_context_{record['source_sha256'][:12]}_{object_index}_{target_px}"
                image_path, label_path = (
                    image_out / f"{tile}.png",
                    label_out / f"{tile}.txt",
                )
                context.save(image_path, "PNG", optimize=False)
                label_path.write_text(_label_line(truth))
                entries.append(
                    {
                        "tile": tile,
                        "boxes": 1,
                        "image_sha256": sha256(image_path),
                        "label_sha256": sha256(label_path),
                        "provenance": {
                            "kind": "real_context",
                            "source": record["source"],
                            "source_sha256": record["source_sha256"],
                            "object_index": object_index,
                            "visibility": annotation["visibility"],
                            "target_max_side_px": target_px,
                            "partial_policy": "whole_context_only",
                        },
                    }
                )

    sizes = _target_sizes()
    for entry in confirmed:
        cutout_path = (mask_review_path.parent / entry["cutout"]).resolve()
        if sha256(cutout_path) != entry["cutout_sha256"]:
            raise ValueError(f"confirmed cutout hash changed: {cutout_path}")
        seed = int.from_bytes(f"{SEED}\0{entry['id']}".encode(), "little", signed=False)
        rng = random.Random(seed)
        real_targets = rng.sample(hardneg["entries"]["train"], 4)
        synthetic_targets = rng.sample(production["entries"]["train"], 4)
        targets = [("real_negative", target) for target in real_targets] + [
            ("production", target) for target in synthetic_targets
        ]
        shuffled_sizes = sizes.copy()
        rng.shuffle(shuffled_sizes)
        for ordinal, ((target_kind, target), target_px) in enumerate(
            zip(targets, shuffled_sizes, strict=True)
        ):
            composite, label, transform = _composite(
                cutout_path, target, target_px=target_px, rng=rng
            )
            tile = f"rp_paste_{entry['id']}_{ordinal}"
            image_path, label_path = (
                image_out / f"{tile}.png",
                label_out / f"{tile}.txt",
            )
            composite.save(image_path, "PNG", optimize=False)
            label_path.write_text(label)
            entries.append(
                {
                    "tile": tile,
                    "boxes": target["boxes"] + 1,
                    "image_sha256": sha256(image_path),
                    "label_sha256": sha256(label_path),
                    "provenance": {
                        "kind": "confirmed_clear_copy_paste",
                        "source": entry["source"],
                        "source_sha256": entry["source_sha256"],
                        "object_index": entry["object_index"],
                        "mask_review_id": entry["id"],
                        "mask_confirmation": entry["confirmation"],
                        "target_kind": target_kind,
                        "target_tile": target["tile"],
                        "target_provenance": target.get("provenance"),
                        **transform,
                    },
                }
            )
    metadata = {
        "component": "fold-filtered-real-positive-anchor",
        "labels_sha256": sha256(labels_path),
        "mask_review_sha256": sha256(mask_review_path),
        "fold_lock_sha256": sha256(fold_lock_path),
        "held_out_fold": held_out_fold,
        "confirmed_training_clear_masks": len(confirmed),
        "training_clear_masks": clear_total,
        "policy": {
            "clear_cutouts_only": True,
            "partial_definition": "any_occluding_grass_blade_is_partial",
            "contexts_per_visible_mine": len(CONTEXT_SCALES),
            "context_scales_px": list(CONTEXT_SCALES),
            "composites_per_confirmed_clear": COMPOSITES_PER_CLEAR,
            "composite_target_px": [20, 130],
            "composite_target_mix": {"real_negative": 4, "production": 4},
            "rotation_degrees": [0, 360],
            "feather_radius_px": [1, 3],
        },
    }
    return _write_component(out, entries, metadata)


def materialize_fold(
    labels_path: Path,
    mask_review_path: Path,
    fold_lock_path: Path,
    held_out_fold: int,
    production_component: Path,
    hardneg_component: Path,
    filtered_hardneg_out: Path,
    real_positive_out: Path,
) -> tuple[dict, dict]:
    filtered = filter_hard_negatives(
        hardneg_component, filtered_hardneg_out, fold_lock_path, held_out_fold
    )
    positives = materialize_real_positives(
        labels_path,
        mask_review_path,
        fold_lock_path,
        held_out_fold,
        production_component,
        filtered_hardneg_out,
        real_positive_out,
    )
    return filtered, positives
