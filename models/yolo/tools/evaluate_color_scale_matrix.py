#!/usr/bin/env python3
"""Evaluate the controlled mine color/scale diagnostic at deployment settings.

The renderer produces 15 single-mine cases and three mine-free background
plates.  This evaluator uses the same 640 px / 192 px-overlap tiled inference
as deployment, merges tile-local candidates in full-image coordinates, and
applies the frozen 0.37 operating threshold by default.

Pillow and Ultralytics are imported only by :func:`run`; the box matching and
acceptance helpers remain importable in the lightweight unit-test environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

INPUT_SCHEMA = "mine-color-scale-diagnostic/1"
REPORT_SCHEMA = "mine-color-scale-diagnostic-evaluation/1"
POSITIVE_CASES = 15
BACKGROUND_PLATES = 3
TILE_PX = 640
OVERLAP_PX = 192
DEFAULT_THRESHOLD = 0.37
IOU_THRESHOLD = 0.5
DEFAULT_MERGE_OVERLAP = 0.5


@dataclass(frozen=True)
class Box:
    """A global image-space box used by the dependency-free metric helpers."""

    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0

    @classmethod
    def from_xyxy(
        cls, xyxy: Sequence[float], confidence: float = 1.0
    ) -> "Box":
        if len(xyxy) != 4:
            raise ValueError("xyxy must have four coordinates")
        values = tuple(float(value) for value in xyxy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("xyxy coordinates must be finite")
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError("xyxy box must have positive area")
        return cls(*values, float(confidence))


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for an input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(first: Box, second: Box) -> float:
    """Return intersection over union for two valid image-space boxes."""
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    intersection = width * height
    first_area = (first.x1 - first.x0) * (first.y1 - first.y0)
    second_area = (second.x1 - second.x0) * (second.y1 - second.y0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def score_case(
    ground_truth: Sequence[Box],
    predictions: Sequence[Box],
    iou_threshold: float = IOU_THRESHOLD,
) -> dict:
    """Greedily score confidence-ordered predictions one-to-one against truth."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in [0, 1]")
    unmatched = set(range(len(ground_truth)))
    matches = []
    false_positive_indices = []
    ranked = sorted(
        enumerate(predictions),
        key=lambda item: (-item[1].confidence, item[0]),
    )
    for prediction_index, prediction in ranked:
        choices = [
            (iou(prediction, ground_truth[index]), index)
            for index in unmatched
        ]
        overlap, truth_index = max(choices, default=(0.0, -1))
        if overlap >= iou_threshold:
            unmatched.remove(truth_index)
            matches.append(
                {
                    "ground_truth_index": truth_index,
                    "prediction_index": prediction_index,
                    "iou": overlap,
                }
            )
        else:
            false_positive_indices.append(prediction_index)
    matches.sort(key=lambda item: item["ground_truth_index"])
    return {
        "tp": len(matches),
        "fp": len(false_positive_indices),
        "fn": len(unmatched),
        "matches": matches,
        "false_positive_prediction_indices": sorted(false_positive_indices),
    }


def acceptance_summary(
    positive_results: Sequence[Mapping[str, object]],
    plate_results: Sequence[Mapping[str, object]],
    *,
    required_positives: int = POSITIVE_CASES,
) -> dict:
    """Summarize the diagnostic's two independent promotion gates."""
    if len(positive_results) != required_positives:
        raise ValueError(
            f"need exactly {required_positives} positive results; "
            f"got {len(positive_results)}"
        )
    if len(plate_results) != BACKGROUND_PLATES:
        raise ValueError(
            f"need exactly {BACKGROUND_PLATES} plate results; "
            f"got {len(plate_results)}"
        )
    matched = sum(int(result["tp"]) for result in positive_results)
    positive_fp = sum(int(result["fp"]) for result in positive_results)
    positive_fn = sum(int(result["fn"]) for result in positive_results)
    plate_fp = sum(int(result["fp"]) for result in plate_results)
    all_positives_matched = matched == required_positives and positive_fn == 0
    background_plates_clean = plate_fp == 0
    return {
        "positive_cases_matched": matched,
        "positive_cases_required": required_positives,
        "positive_false_positives": positive_fp,
        "positive_false_negatives": positive_fn,
        "background_plate_false_positives": plate_fp,
        "all_positive_cases_matched": all_positives_matched,
        "background_plates_clean": background_plates_clean,
        "accepted": all_positives_matched and background_plates_clean,
    }


def _declared_image_sha256(manifest: Mapping, record: Mapping) -> str | None:
    """Read compatible per-image hash spellings from diagnostic manifests."""
    for key in ("image_sha256", "source_sha256", "sha256"):
        value = record.get(key)
        if value is not None:
            return str(value)
    record_hashes = record.get("hashes")
    if isinstance(record_hashes, Mapping):
        for key in ("image_sha256", "image", "sha256"):
            value = record_hashes.get(key)
            if value is not None:
                return str(value)
    manifest_hashes = manifest.get("hashes")
    image = record.get("image")
    if isinstance(manifest_hashes, Mapping) and image in manifest_hashes:
        value = manifest_hashes[image]
        if isinstance(value, Mapping):
            value = value.get("sha256") or value.get("image_sha256")
        return str(value) if value is not None else None
    return None


def _resolve_image(matrix_root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("diagnostic case has no image path")
    root = matrix_root.resolve()
    image = (root / relative).resolve()
    try:
        image.relative_to(root)
    except ValueError as error:
        raise ValueError(f"image escapes diagnostic directory: {relative}") from error
    if not image.is_file():
        raise ValueError(f"diagnostic image does not exist: {image}")
    return image


def _load_manifest(manifest_path: Path) -> dict:
    document = json.loads(manifest_path.read_text())
    if document.get("schema") != INPUT_SCHEMA:
        raise ValueError(
            f"expected manifest schema {INPUT_SCHEMA!r}; "
            f"got {document.get('schema')!r}"
        )
    positives = document.get("positive_cases")
    plates = document.get("background_plates")
    if not isinstance(positives, list) or len(positives) != POSITIVE_CASES:
        raise ValueError(f"manifest must contain exactly {POSITIVE_CASES} positives")
    if not isinstance(plates, list) or len(plates) != BACKGROUND_PLATES:
        raise ValueError(f"manifest must contain exactly {BACKGROUND_PLATES} plates")
    case_ids = [record.get("case_id") for record in positives + plates]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("every diagnostic case needs a case_id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("diagnostic case_id values must be unique")
    for record in positives:
        truth = record.get("ground_truth")
        if not isinstance(truth, list) or len(truth) != 1:
            raise ValueError(
                f"positive case {record['case_id']} must have one ground-truth box"
            )
        Box.from_xyxy(truth[0].get("xyxy_px", ()))
    for record in plates:
        if record.get("ground_truth") != []:
            raise ValueError(
                f"background plate {record['case_id']} must have empty ground truth"
            )
    return document


def _prediction_boxes(detections: Sequence) -> list[Box]:
    return [
        Box(
            float(detection.x0),
            float(detection.y0),
            float(detection.x1),
            float(detection.y1),
            float(detection.confidence),
        )
        for detection in detections
    ]


def run(
    weights: Path,
    matrix: Path,
    out: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    batch: int = 16,
    merge_overlap: float = DEFAULT_MERGE_OVERLAP,
    device: str | None = None,
) -> dict:
    """Run the full diagnostic and write one immutable JSON report."""
    if out.exists():
        raise ValueError(f"refusing to overwrite evaluation report: {out}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if batch < 1:
        raise ValueError("batch must be positive")
    if not 0.0 <= merge_overlap <= 1.0:
        raise ValueError("merge overlap must be in [0, 1]")

    manifest_path = matrix / "manifest.json"
    manifest = _load_manifest(manifest_path)

    from PIL import Image, ImageOps
    from ultralytics import YOLO

    from tools.audit_irl import CANDIDATE_FLOOR, merge_detections, predict_tiled

    model = YOLO(str(weights))

    def evaluate(record: Mapping, expected_kind: str) -> dict:
        image_path = _resolve_image(matrix, record.get("image"))
        image_sha256 = sha256(image_path)
        declared_sha256 = _declared_image_sha256(manifest, record)
        if declared_sha256 is not None and declared_sha256 != image_sha256:
            raise ValueError(f"image hash changed: {image_path}")
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        origins, candidates = predict_tiled(
            model,
            image,
            tile=TILE_PX,
            overlap=OVERLAP_PX,
            batch=batch,
            device=device,
        )
        thresholded = [
            candidate
            for candidate in candidates
            if candidate.confidence >= threshold
        ]
        merged = merge_detections(thresholded, merge_overlap)
        ground_truth = [
            Box.from_xyxy(annotation["xyxy_px"])
            for annotation in record["ground_truth"]
        ]
        metrics = score_case(ground_truth, _prediction_boxes(merged))
        return {
            "case_id": record["case_id"],
            "kind": expected_kind,
            "color_family": record.get("color_family"),
            "target_width_px": record.get("target_width_px"),
            "background": record.get("background"),
            "image": record["image"],
            "image_sha256": image_sha256,
            "declared_image_sha256": declared_sha256,
            "width": image.width,
            "height": image.height,
            "tiles": len(origins),
            "candidate_tile_detections": len(candidates),
            "threshold_tile_detections": len(thresholded),
            "detections": [asdict(detection) for detection in merged],
            **metrics,
        }

    positive_results = [
        evaluate(record, "positive") for record in manifest["positive_cases"]
    ]
    plate_results = [
        evaluate(record, "background_plate")
        for record in manifest["background_plates"]
    ]
    summary = acceptance_summary(positive_results, plate_results)
    report = {
        "schema": REPORT_SCHEMA,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "weights": str(weights.resolve()),
        "weights_sha256": sha256(weights),
        "threshold": threshold,
        "candidate_floor": CANDIDATE_FLOOR,
        "tile_px": TILE_PX,
        "overlap_px": OVERLAP_PX,
        "merge_overlap": merge_overlap,
        "iou_threshold": IOU_THRESHOLD,
        "positive_cases": positive_results,
        "background_plates": plate_results,
        "summary": summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--matrix",
        type=Path,
        required=True,
        help="directory containing manifest.json and the rendered images",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--merge-overlap", type=float, default=DEFAULT_MERGE_OVERLAP)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    try:
        report = run(
            args.weights,
            args.matrix,
            args.out,
            threshold=args.threshold,
            batch=args.batch,
            merge_overlap=args.merge_overlap,
            device=args.device,
        )
    except ValueError as error:
        parser.error(str(error))
    summary = report["summary"]
    print(
        f"positives {summary['positive_cases_matched']}/"
        f"{summary['positive_cases_required']}; "
        f"plate false positives {summary['background_plate_false_positives']}; "
        f"accepted={summary['accepted']}"
    )
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
