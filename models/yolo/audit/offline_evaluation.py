"""Re-score a frozen IRL inference audit against certified real labels.

This module deliberately consumes the tile-local candidates retained by
``tools/audit_irl.py``.  It never loads weights or source images and therefore
cannot silently turn a label review into a new inference run.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping

from .evaluation import (
    IOU_THRESHOLD,
    Box,
    aggregate_metrics,
    annotation_boxes,
    evaluate_image,
    find_empty_tiles,
)
from .labels import EVALUATION_ROLES, ROLES, validate_labels
from .folds import load_fold_document, source_hashes


MISSION_ENGINE_SRC = Path(__file__).resolve().parents[3] / "ros" / "mission_engine"
if str(MISSION_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(MISSION_ENGINE_SRC))

from mission_engine.core.tiles import tile_grid  # noqa: E402


AUDIT_SCHEMA = "mission10-yolo-irl-audit/1"
OFFLINE_EVALUATION_SCHEMA = "mission10-yolo-offline-evaluation/1"
DEFAULT_THRESHOLD = 0.37
DEFAULT_MERGE_OVERLAP = 0.5


def _number(value: object, where: str, *, minimum: float, maximum: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{where} must be a finite number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{where} must be in [{minimum:g}, {maximum:g}]")
    return result


def _sha256(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _audit_index(audit: object) -> tuple[dict[str, dict], int, int, float]:
    if not isinstance(audit, dict) or audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError(f"audit must use schema {AUDIT_SCHEMA}")
    tile = _positive_int(audit.get("tile_px"), "audit.tile_px")
    overlap = audit.get("overlap_px")
    if (
        not isinstance(overlap, int)
        or isinstance(overlap, bool)
        or not 0 <= overlap < tile
    ):
        raise ValueError("audit.overlap_px must satisfy 0 <= overlap < tile")
    candidate_floor = _number(
        audit.get("candidate_floor"),
        "audit.candidate_floor",
        minimum=0.0,
        maximum=1.0,
    )
    _sha256(audit.get("weights_sha256"), "audit.weights_sha256")
    if not isinstance(audit.get("weights"), str) or not audit["weights"]:
        raise ValueError("audit.weights must be a non-empty string")
    images = audit.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("audit.images must be a non-empty list")
    by_hash: dict[str, dict] = {}
    for index, record in enumerate(images):
        where = f"audit.images[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{where} must be an object")
        source_hash = _sha256(record.get("source_sha256"), f"{where}.source_sha256")
        if source_hash in by_hash:
            raise ValueError(f"audit has duplicate source hash {source_hash}")
        by_hash[source_hash] = record
    return by_hash, tile, overlap, candidate_floor


def _candidate_boxes(
    record: Mapping,
    *,
    source: str,
    width: int,
    height: int,
    origins: set[tuple[int, int]],
    tile: int,
) -> list[Box]:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(
            f"audit lacks retained candidates for {source}; rerun tools/audit_irl.py"
        )
    result = []
    for index, candidate in enumerate(candidates):
        where = f"audit candidate {index} for {source}"
        if not isinstance(candidate, dict):
            raise ValueError(f"{where} must be an object")
        tile_x, tile_y = candidate.get("tile_x"), candidate.get("tile_y")
        if (
            not isinstance(tile_x, int)
            or isinstance(tile_x, bool)
            or not isinstance(tile_y, int)
            or isinstance(tile_y, bool)
            or (tile_x, tile_y) not in origins
        ):
            raise ValueError(f"{where} has an invalid deployment tile origin")
        coordinates = []
        for name in ("x0", "y0", "x1", "y1"):
            value = candidate.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{where}.{name} must be a finite number")
            coordinates.append(float(value))
        x0, y0, x1, y1 = coordinates
        crop_x1, crop_y1 = min(tile_x + tile, width), min(tile_y + tile, height)
        if not (
            tile_x <= x0 < x1 <= crop_x1
            and tile_y <= y0 < y1 <= crop_y1
        ):
            raise ValueError(f"{where} is outside its deployment tile")
        confidence = _number(
            candidate.get("confidence"),
            f"{where}.confidence",
            minimum=0.0,
            maximum=1.0,
        )
        result.append(Box(x0, y0, x1, y1, confidence, tile_x, tile_y))
    return result


def _ensure_visibility_rows(metrics: dict) -> None:
    for visibility in ("clear", "partial"):
        metrics["recall_by_visibility"].setdefault(
            visibility,
            {"tp": 0, "fn": 0, "total": 0, "recall": 1.0},
        )


def evaluate_audit(
    audit: object,
    labels: dict,
    role: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    merge_overlap: float | None = None,
    selected_source_hashes: set[str] | None = None,
) -> dict:
    """Evaluate retained candidates, returning a deterministic report payload."""
    validate_labels(labels, require_frozen=True, require_certified=True)
    if role not in ROLES:
        raise ValueError(f"invalid role {role!r}")
    threshold = _number(threshold, "threshold", minimum=0.0, maximum=1.0)
    audit_by_hash, tile, overlap, candidate_floor = _audit_index(audit)
    if threshold < candidate_floor:
        raise ValueError(
            f"threshold {threshold:g} is below audit candidate floor "
            f"{candidate_floor:g}"
        )
    if merge_overlap is None:
        merge_overlap = _number(
            audit.get("merge_overlap", DEFAULT_MERGE_OVERLAP),
            "audit.merge_overlap",
            minimum=0.0,
            maximum=1.0,
        )
    else:
        merge_overlap = _number(
            merge_overlap, "merge_overlap", minimum=0.0, maximum=1.0
        )

    all_labels = {record["source_sha256"]: record for record in labels["images"]}
    selected = [record for record in labels["images"] if record["role"] == role]
    if selected_source_hashes is not None:
        selected = [
            record
            for record in selected
            if record["source_sha256"] in selected_source_hashes
        ]
        actual = {record["source_sha256"] for record in selected}
        if actual != selected_source_hashes:
            raise ValueError("fold selection is missing certified records")
    if not selected:
        raise ValueError(f"no labeled sources assigned to {role}")
    missing = [
        record["source"]
        for record in selected
        if record["source_sha256"] not in audit_by_hash
    ]
    if missing:
        raise ValueError(f"audit is missing labeled sources for {role}: {missing}")

    selected_hashes = {record["source_sha256"] for record in selected}
    extras = []
    for source_hash in sorted(set(audit_by_hash) - selected_hashes):
        audit_record = audit_by_hash[source_hash]
        entry = {
            "source": audit_record.get("source"),
            "source_sha256": source_hash,
        }
        if source_hash in all_labels:
            entry["label_role"] = all_labels[source_hash]["role"]
        extras.append(entry)
    rows = []
    image_reports = []
    total_tiles = 0
    for label_record in selected:
        source = label_record["source"]
        audit_record = audit_by_hash[label_record["source_sha256"]]
        width, height = label_record["width"], label_record["height"]
        if audit_record.get("width") != width or audit_record.get("height") != height:
            raise ValueError(f"audit dimensions disagree for {source}")
        origins = tile_grid(width, height, tile, overlap)
        if audit_record.get("tiles") != len(origins):
            raise ValueError(f"audit tile count disagrees for {source}")
        candidates = _candidate_boxes(
            audit_record,
            source=source,
            width=width,
            height=height,
            origins=set(origins),
            tile=tile,
        )
        ground_truth, ignore_regions = annotation_boxes(label_record)
        visibility = [
            annotation["visibility"]
            for annotation in label_record["objects"]
            if annotation["visibility"] in {"clear", "partial"}
        ]
        all_mines = [
            Box.from_xyxy(annotation["xyxy"])
            for annotation in label_record["objects"]
        ]
        empty_origins = find_empty_tiles(origins, tile, width, height, all_mines)
        metrics = evaluate_image(
            ground_truth,
            ignore_regions,
            candidates,
            threshold=threshold,
            merge_overlap=merge_overlap,
            ground_truth_visibility=visibility,
            empty_tile_origins=empty_origins,
        )
        _ensure_visibility_rows(metrics)
        rows.append(metrics)
        total_tiles += len(origins)
        image_reports.append(
            {
                "source": source,
                "source_sha256": label_record["source_sha256"],
                "width": width,
                "height": height,
                "capture_group": label_record["capture_group"],
                "role": role,
                "tiles": len(origins),
                "retained_candidates": len(candidates),
                "metrics": metrics,
            }
        )

    metrics = aggregate_metrics(rows, total_tiles)
    _ensure_visibility_rows(metrics)
    promotion_evaluation = role in EVALUATION_ROLES
    return {
        "schema": OFFLINE_EVALUATION_SCHEMA,
        "report_classification": (
            "promotion_evaluation" if promotion_evaluation else "diagnostic_only"
        ),
        "promotion_use_permitted": promotion_evaluation,
        "non_promotion_reason": (
            None
            if promotion_evaluation
            else f"role {role!r} is not an evaluation role"
        ),
        "role": role,
        "candidate_floor": candidate_floor,
        "threshold": threshold,
        "tile_px": tile,
        "overlap_px": overlap,
        "merge_overlap": merge_overlap,
        "iou_threshold": IOU_THRESHOLD,
        "weights": audit["weights"],
        "weights_sha256": audit["weights_sha256"],
        "metrics": metrics,
        "extra_unselected_audit_records": extras,
        "images": image_reports,
    }


def run(
    audit_path: Path,
    labels_path: Path,
    role: str,
    out: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    merge_overlap: float | None = None,
    fold_lock_path: Path | None = None,
    held_out_fold: int | None = None,
) -> dict:
    """Load immutable inputs, evaluate them, and write one new JSON report."""
    if out.exists():
        raise ValueError(f"refusing to overwrite evaluation report: {out}")
    labels_bytes = labels_path.read_bytes()
    audit_bytes = audit_path.read_bytes()
    labels = json.loads(labels_bytes)
    audit = json.loads(audit_bytes)
    if (fold_lock_path is None) != (held_out_fold is None):
        raise ValueError("fold lock and held-out fold must be provided together")
    selected_hashes = None
    selection = None
    if fold_lock_path is not None:
        fold_lock_path = Path(fold_lock_path)
        fold_document = load_fold_document(fold_lock_path, labels_path)
        selected_hashes = source_hashes(fold_document, held_out_fold)
        selection = {
            "fold_lock": str(fold_lock_path.resolve()),
            "fold_lock_sha256": hashlib.sha256(
                fold_lock_path.read_bytes()
            ).hexdigest(),
            "held_out_fold": held_out_fold,
            "source_sha256": sorted(selected_hashes),
        }
    report = evaluate_audit(
        audit,
        labels,
        role,
        threshold=threshold,
        merge_overlap=merge_overlap,
        selected_source_hashes=selected_hashes,
    )
    report.update(
        {
            "audit": str(audit_path.resolve()),
            "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "labels": str(labels_path.resolve()),
            "labels_sha256": hashlib.sha256(labels_bytes).hexdigest(),
            "selection": selection,
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2) + "\n"
    try:
        with out.open("x") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite evaluation report: {out}") from error
    return report
