"""Metrics for annotated deployment-tiled real-image evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .labels import EVALUATION_ROLES, validate_labels


IOU_THRESHOLD = 0.5
REPORT_THRESHOLDS = (0.001, 0.37)
SCALE_PROBE_PX = (30, 60, 120)


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0
    tile_x: int = 0
    tile_y: int = 0

    @classmethod
    def from_xyxy(
        cls,
        xyxy: Sequence[float],
        confidence: float = 1.0,
        tile_x: int = 0,
        tile_y: int = 0,
    ) -> "Box":
        return cls(*map(float, xyxy), float(confidence), tile_x, tile_y)


def area(box: Box) -> float:
    return max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)


def intersection(first: Box, second: Box) -> float:
    return max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0)) * max(
        0.0, min(first.y1, second.y1) - max(first.y0, second.y0)
    )


def iou(first: Box, second: Box) -> float:
    overlap = intersection(first, second)
    union = area(first) + area(second) - overlap
    return overlap / union if union > 0.0 else 0.0


def duplicate_overlap(first: Box, second: Box) -> float:
    overlap = intersection(first, second)
    smaller = min(area(first), area(second))
    containment = overlap / smaller if smaller > 0.0 else 0.0
    return max(iou(first, second), containment)


def merge_candidates(
    candidates: Sequence[Box], overlap_threshold: float = 0.5
) -> tuple[list[Box], int]:
    """Apply deployment merge and return kept boxes plus tile fragments."""
    kept: list[Box] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -item.confidence,
            item.tile_y,
            item.tile_x,
            item.y0,
            item.x0,
            item.y1,
            item.x1,
        ),
    ):
        if all(
            duplicate_overlap(candidate, accepted) < overlap_threshold
            for accepted in kept
        ):
            kept.append(candidate)
    return kept, len(candidates) - len(kept)


def _ignored(prediction: Box, ignore_regions: Sequence[Box]) -> bool:
    """Ignore a prediction whose center is inside a reviewed ignore region."""
    center_x = (prediction.x0 + prediction.x1) / 2.0
    center_y = (prediction.y0 + prediction.y1) / 2.0
    return any(
        region.x0 <= center_x <= region.x1 and region.y0 <= center_y <= region.y1
        for region in ignore_regions
    )


def evaluate_image(
    ground_truth: Sequence[Box],
    ignore_regions: Sequence[Box],
    candidates: Sequence[Box],
    *,
    threshold: float,
    merge_overlap: float = 0.5,
    ground_truth_visibility: Sequence[str] | None = None,
    empty_tile_origins: set[tuple[int, int]] | None = None,
) -> dict:
    if ground_truth_visibility is not None and len(ground_truth_visibility) != len(
        ground_truth
    ):
        raise ValueError("ground-truth visibility count does not match boxes")
    thresholded = [box for box in candidates if box.confidence >= threshold]
    merged, fragments = merge_candidates(thresholded, merge_overlap)
    scored = [box for box in merged if not _ignored(box, ignore_regions)]
    ignored = len(merged) - len(scored)

    unmatched = set(range(len(ground_truth)))
    matched: set[int] = set()
    false_positives: list[Box] = []
    for prediction in scored:
        choices = [(iou(prediction, ground_truth[index]), index) for index in unmatched]
        overlap, index = max(choices, default=(0.0, -1))
        if overlap >= IOU_THRESHOLD:
            unmatched.remove(index)
            matched.add(index)
        else:
            false_positives.append(prediction)
    false_positive_origins = {(box.tile_x, box.tile_y) for box in false_positives}
    empty_origins = empty_tile_origins or set()
    # The empty-real-tile promotion gate is deliberately tile-local: if an
    # actually empty deployment crop emits any non-ignored candidate, that
    # tile fails even when cross-tile merging would suppress the candidate.
    empty_false_positive_origins = {
        (box.tile_x, box.tile_y)
        for box in thresholded
        if (box.tile_x, box.tile_y) in empty_origins
        and not _ignored(box, ignore_regions)
    }
    visibility_counts: dict[str, dict[str, int | float]] = {}
    if ground_truth_visibility is not None:
        for visibility in sorted(set(ground_truth_visibility)):
            indices = {
                index
                for index, value in enumerate(ground_truth_visibility)
                if value == visibility
            }
            visible_tp = len(indices & matched)
            total = len(indices)
            visibility_counts[visibility] = {
                "tp": visible_tp,
                "fn": total - visible_tp,
                "total": total,
                "recall": visible_tp / total if total else 1.0,
            }
    return {
        "tp": len(matched),
        "fp": len(false_positives),
        "fn": len(unmatched),
        "tile_candidates": len(thresholded),
        "merged_candidates": len(merged),
        "tile_fragments": fragments,
        "ignored_candidates": ignored,
        "false_positive_tiles": len(false_positive_origins),
        "empty_real_tiles": len(empty_origins),
        "empty_real_tiles_with_false_positive": len(empty_false_positive_origins),
        "recall_by_visibility": visibility_counts,
    }


def aggregate_metrics(results: Iterable[dict], tiles: int) -> dict:
    rows = list(results)
    scalar_keys = (
        "tp",
        "fp",
        "fn",
        "tile_candidates",
        "merged_candidates",
        "tile_fragments",
        "ignored_candidates",
        "false_positive_tiles",
        "empty_real_tiles",
        "empty_real_tiles_with_false_positive",
    )
    totals = {key: sum(row[key] for row in rows) for key in scalar_keys}
    precision_denominator = totals["tp"] + totals["fp"]
    recall_denominator = totals["tp"] + totals["fn"]
    totals.update(
        {
            "images": len(rows),
            "tiles": tiles,
            "precision": (
                totals["tp"] / precision_denominator if precision_denominator else 1.0
            ),
            "recall": totals["tp"] / recall_denominator if recall_denominator else 1.0,
            "false_positives_per_tile": totals["fp"] / tiles if tiles else 0.0,
            "fragment_candidates_per_tile": (
                totals["tile_fragments"] / tiles if tiles else 0.0
            ),
            "false_positive_tile_rate": (
                totals["false_positive_tiles"] / tiles if tiles else 0.0
            ),
            "empty_real_tile_false_positive_rate": (
                totals["empty_real_tiles_with_false_positive"]
                / totals["empty_real_tiles"]
                if totals["empty_real_tiles"]
                else 0.0
            ),
        }
    )
    visibility_totals: dict[str, list[int]] = {}
    for row in rows:
        for visibility, counts in row["recall_by_visibility"].items():
            aggregate = visibility_totals.setdefault(visibility, [0, 0])
            aggregate[0] += int(counts["tp"])
            aggregate[1] += int(counts["total"])
    totals["recall_by_visibility"] = {
        visibility: {
            "tp": counts[0],
            "fn": counts[1] - counts[0],
            "total": counts[1],
            "recall": counts[0] / counts[1] if counts[1] else 1.0,
        }
        for visibility, counts in sorted(visibility_totals.items())
    }
    return totals


def annotation_boxes(record: dict) -> tuple[list[Box], list[Box]]:
    """Turn a label record into scorable objects and ignore regions."""
    ground_truth = []
    ignored = [Box.from_xyxy(region["xyxy"]) for region in record["ignore_regions"]]
    for annotation in record["objects"]:
        box = Box.from_xyxy(annotation["xyxy"])
        if annotation["visibility"] in {"clear", "partial"}:
            ground_truth.append(box)
        else:
            # A labeled but non-visible/unknown object is not a negative area.
            ignored.append(box)
    return ground_truth, ignored


def select_role(document: dict, role: str) -> list[dict]:
    """Select one frozen, certified evaluation role without cross-role mixing."""
    validate_labels(document, require_frozen=True, require_certified=True)
    if role not in EVALUATION_ROLES:
        raise ValueError(f"{role!r} is not an evaluation role")
    records = [image for image in document["images"] if image["role"] == role]
    if not records:
        raise ValueError(f"no images assigned to {role}")
    incomplete = [
        record["source"]
        for record in records
        if record["review_state"] != "certified"
    ]
    if incomplete:
        raise ValueError(f"evaluation labels are not certified: {incomplete}")
    return records


def find_empty_tiles(
    origins: Sequence[tuple[int, int]],
    tile_px: int,
    width: int,
    height: int,
    all_mines: Sequence[Box],
) -> set[tuple[int, int]]:
    """Return exact deployment tiles containing no labeled mine pixels."""
    empty = set()
    for x, y in origins:
        tile = Box(x, y, min(x + tile_px, width), min(y + tile_px, height))
        if all(intersection(tile, mine) <= 0.0 for mine in all_mines):
            empty.add((x, y))
    return empty


def _scale_probe_geometry(
    annotation: dict, target_px: int, tile_px: int
) -> tuple[tuple[float, float, float, float], float]:
    if target_px not in SCALE_PROBE_PX:
        raise ValueError(f"unsupported scale probe {target_px}")
    x0, y0, x1, y1 = map(float, annotation["xyxy"])
    object_side = max(x1 - x0, y1 - y0)
    if object_side <= 0.0:
        raise ValueError("scale probe object has empty box")
    scale = target_px / object_side
    source_side = tile_px / scale
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    crop_x0 = center_x - source_side / 2.0
    crop_y0 = center_y - source_side / 2.0
    crop_box = (
        crop_x0,
        crop_y0,
        crop_x0 + source_side,
        crop_y0 + source_side,
    )
    return crop_box, tile_px / source_side


def transform_scale_probe_box(
    annotation: dict,
    xyxy: Sequence[float],
    target_px: int,
    tile_px: int = 640,
) -> Box | None:
    """Transform and clip another annotation into a target's scale probe."""
    crop_box, scale = _scale_probe_geometry(annotation, target_px, tile_px)
    x0, y0, x1, y1 = map(float, xyxy)
    transformed = Box(
        max(0.0, min(tile_px, (x0 - crop_box[0]) * scale)),
        max(0.0, min(tile_px, (y0 - crop_box[1]) * scale)),
        max(0.0, min(tile_px, (x1 - crop_box[0]) * scale)),
        max(0.0, min(tile_px, (y1 - crop_box[1]) * scale)),
    )
    return transformed if area(transformed) > 0.0 else None


def scale_probe(image, annotation: dict, target_px: int, tile_px: int = 640):
    """Make a deterministic object-centered tile at an exact max-side scale.

    The probe uses a square source crop centered on the full-object box. Pillow
    deterministically pads crops crossing an image edge with black. This keeps
    every output exactly deployment-sized and makes 30/60/120 px results
    reproducible without random augmentation.
    """
    crop_box, _ = _scale_probe_geometry(annotation, target_px, tile_px)
    from PIL import Image

    probe = image.transform(
        (tile_px, tile_px),
        Image.Transform.EXTENT,
        crop_box,
        Image.Resampling.BILINEAR,
    )
    transformed = transform_scale_probe_box(
        annotation, annotation["xyxy"], target_px, tile_px
    )
    if transformed is None:
        raise ValueError("target vanished from its scale probe")
    return probe, transformed
