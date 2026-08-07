#!/usr/bin/env python3
"""Run object-scale diagnostics against frozen, certified real labels.

This runner intentionally does not perform full-image evaluation or change the
training status of any input.  It makes one deterministic 640 px scale probe
for every clear/partial annotation at each standard probe size, then submits
all probes through one batched Ultralytics prediction call.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence


YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from audit.evaluation import (  # noqa: E402
    IOU_THRESHOLD,
    REPORT_THRESHOLDS,
    SCALE_PROBE_PX,
    Box,
    _scale_probe_geometry,
    aggregate_metrics,
    evaluate_image,
    scale_probe,
    transform_scale_probe_box,
)
from audit.labels import (  # noqa: E402
    ROLES,
    load_labels,
    resolve_source,
    sha256,
    validate_labels,
)


SCHEMA = "mission10-yolo-certified-scale-probe/1"
TILE_PX = 640
CANDIDATE_FLOOR = min(REPORT_THRESHOLDS)
SCORABLE_VISIBILITIES = frozenset({"clear", "partial"})


def select_certified_role(document: dict, role: str | None) -> list[dict]:
    """Select one explicit role from a frozen, certified label document."""
    if not role:
        raise ValueError("role must be explicitly requested")
    validate_labels(document, require_frozen=True, require_certified=True)
    if role not in ROLES:
        raise ValueError(f"invalid role {role!r}")
    records = [record for record in document["images"] if record["role"] == role]
    if not records:
        raise ValueError(f"no images assigned to {role}")
    return records


def probe_transform(annotation: dict, target_px: int) -> dict:
    """Describe the deterministic source-to-probe affine transform."""
    crop_box, scale = _scale_probe_geometry(annotation, target_px, TILE_PX)
    return {
        "source_crop_xyxy": list(crop_box),
        "source_to_probe_scale": scale,
        "output_width": TILE_PX,
        "output_height": TILE_PX,
    }


def candidates_from_result(result, tile_px: int = TILE_PX) -> list[Box]:
    """Convert one Ultralytics result into class-0 audit candidate boxes."""
    candidates = []
    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
        strict=True,
    ):
        confidence = float(confidence)
        if int(class_id) != 0 or confidence < CANDIDATE_FLOOR:
            continue
        x0, y0, x1, y1 = map(float, xyxy)
        candidates.append(
            Box(
                max(0.0, min(tile_px, x0)),
                max(0.0, min(tile_px, y0)),
                max(0.0, min(tile_px, x1)),
                max(0.0, min(tile_px, y1)),
                confidence,
            )
        )
    return candidates


def predict_probes(
    model,
    probes: Sequence,
    *,
    batch: int,
    device: str | None = None,
) -> list[list[Box]]:
    """Run all probes through a single internally batched model invocation."""
    if not probes:
        return []
    results = list(
        model.predict(
            source=list(probes),
            imgsz=TILE_PX,
            conf=CANDIDATE_FLOOR,
            iou=0.7,
            max_det=100,
            batch=batch,
            device=device,
            verbose=False,
            stream=False,
        )
    )
    if len(results) != len(probes):
        raise ValueError(
            f"model returned {len(results)} results for {len(probes)} probes"
        )
    return [candidates_from_result(result) for result in results]


def _aggregate(rows: Sequence[dict]) -> dict:
    return aggregate_metrics(rows, tiles=len(rows))


def summarize_probes(probes: Sequence[dict]) -> dict:
    """Aggregate probe metrics overall and by visibility/target scale."""

    def summary(rows: Sequence[dict]) -> dict:
        return {
            "probes": len(rows),
            "thresholds": {
                f"{threshold:g}": {
                    "threshold": threshold,
                    **_aggregate(
                        [row["thresholds"][f"{threshold:g}"] for row in rows]
                    ),
                }
                for threshold in REPORT_THRESHOLDS
            },
        }

    return {
        "overall": summary(probes),
        "by_visibility": {
            visibility: summary(
                [row for row in probes if row["visibility"] == visibility]
            )
            for visibility in sorted(SCORABLE_VISIBILITIES)
        },
        "by_target_max_side_px": {
            str(target_px): summary(
                [row for row in probes if row["target_max_side_px"] == target_px]
            )
            for target_px in SCALE_PROBE_PX
        },
        "by_visibility_and_target_max_side_px": {
            visibility: {
                str(target_px): summary(
                    [
                        row
                        for row in probes
                        if row["visibility"] == visibility
                        and row["target_max_side_px"] == target_px
                    ]
                )
                for target_px in SCALE_PROBE_PX
            }
            for visibility in sorted(SCORABLE_VISIBILITIES)
        },
    }


def _load_rgb(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def run(
    weights: Path,
    labels_path: Path,
    role: str | None,
    out: Path,
    *,
    batch: int = 16,
    device: str | None = None,
    model=None,
    image_loader: Callable[[Path], object] = _load_rgb,
) -> dict:
    """Run and write an immutable certified-label scale-probe report."""
    weights = Path(weights)
    labels_path = Path(labels_path)
    out = Path(out)
    if out.exists():
        raise ValueError(f"refusing to overwrite scale-probe report: {out}")
    if not role:
        raise ValueError("role must be explicitly requested")
    if batch < 1:
        raise ValueError("batch must be positive")

    document = load_labels(
        labels_path, require_frozen=True, require_certified=True
    )
    records = select_certified_role(document, role)

    pending = []
    probe_images = []
    for record in records:
        source_path = resolve_source(labels_path, record["source"])
        if sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"source hash changed: {source_path}")
        image = image_loader(source_path)
        if image.size != (record["width"], record["height"]):
            raise ValueError(f"oriented dimensions changed: {source_path}")

        for object_index, annotation in enumerate(record["objects"]):
            visibility = annotation["visibility"]
            if visibility not in SCORABLE_VISIBILITIES:
                continue
            for target_px in SCALE_PROBE_PX:
                probe, truth = scale_probe(image, annotation, target_px, TILE_PX)
                ignored = []
                for other_index, other in enumerate(record["objects"]):
                    if other_index == object_index:
                        continue
                    transformed = transform_scale_probe_box(
                        annotation, other["xyxy"], target_px, TILE_PX
                    )
                    if transformed is not None:
                        ignored.append(transformed)
                for region in record["ignore_regions"]:
                    transformed = transform_scale_probe_box(
                        annotation, region["xyxy"], target_px, TILE_PX
                    )
                    if transformed is not None:
                        ignored.append(transformed)
                pending.append(
                    {
                        "source": record["source"],
                        "source_sha256": record["source_sha256"],
                        "object_index": object_index,
                        "visibility": visibility,
                        "source_object_xyxy": list(annotation["xyxy"]),
                        "target_max_side_px": target_px,
                        "probe_transform": probe_transform(annotation, target_px),
                        "target_xyxy": [truth.x0, truth.y0, truth.x1, truth.y1],
                        "ignored_xyxy": [
                            [box.x0, box.y0, box.x1, box.y1] for box in ignored
                        ],
                        "_truth": truth,
                        "_ignored": ignored,
                    }
                )
                probe_images.append(probe)

    if model is None:
        from ultralytics import YOLO

        model = YOLO(str(weights))
    candidate_sets = predict_probes(
        model, probe_images, batch=batch, device=device
    )

    probes = []
    for pending_row, candidates in zip(pending, candidate_sets, strict=True):
        truth = pending_row.pop("_truth")
        ignored = pending_row.pop("_ignored")
        threshold_results = {
            f"{threshold:g}": evaluate_image(
                [truth],
                ignored,
                candidates,
                threshold=threshold,
                ground_truth_visibility=[pending_row["visibility"]],
            )
            for threshold in REPORT_THRESHOLDS
        }
        probes.append(
            {
                **pending_row,
                "candidates": [asdict(candidate) for candidate in candidates],
                "thresholds": threshold_results,
            }
        )

    report = {
        "schema": SCHEMA,
        "labels": str(labels_path.resolve()),
        "labels_sha256": sha256(labels_path),
        "weights": str(weights.resolve()),
        "weights_sha256": sha256(weights),
        "role": role,
        "purpose": "diagnostic",
        "automatic_training_promotion": False,
        "training_candidate_policy": (
            "diagnostic_only_no_promotion"
            if role == "training_candidate"
            else "not_applicable"
        ),
        "tile_px": TILE_PX,
        "candidate_floor": CANDIDATE_FLOOR,
        "iou_threshold": IOU_THRESHOLD,
        "thresholds": list(REPORT_THRESHOLDS),
        "target_max_side_px": list(SCALE_PROBE_PX),
        "summary": summarize_probes(probes),
        "probes": probes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        help="Ultralytics device (for example 0 or cpu); defaults to auto",
    )
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("batch must be positive")
    try:
        run(
            args.weights,
            args.labels,
            args.role,
            args.out,
            batch=args.batch,
            device=args.device,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
