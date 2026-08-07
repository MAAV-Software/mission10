#!/usr/bin/env python3
"""Evaluate certified real labels with exact deployment tiling and scale probes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

MISSION_ENGINE_SRC = Path(__file__).resolve().parents[3] / "ros" / "mission_engine"
if str(MISSION_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(MISSION_ENGINE_SRC))

from audit.evaluation import (  # noqa: E402
    REPORT_THRESHOLDS,
    SCALE_PROBE_PX,
    Box,
    aggregate_metrics,
    annotation_boxes,
    evaluate_image,
    find_empty_tiles,
    scale_probe,
    select_role,
    transform_scale_probe_box,
)
from audit.labels import (  # noqa: E402
    EVALUATION_ROLES,
    load_labels,
    resolve_source,
    sha256,
)
from tools.audit_irl import CANDIDATE_FLOOR, predict_tiled  # noqa: E402


EVALUATION_SCHEMA = "mission10-yolo-real-evaluation/1"


def _boxes(candidates) -> list[Box]:
    return [
        Box(
            candidate.x0,
            candidate.y0,
            candidate.x1,
            candidate.y1,
            candidate.confidence,
            candidate.tile_x,
            candidate.tile_y,
        )
        for candidate in candidates
    ]


def run(
    weights: Path,
    labels_path: Path,
    role: str,
    out: Path,
    *,
    tile: int = 640,
    overlap: int = 192,
    batch: int = 16,
    merge_overlap: float = 0.5,
    device: str | None = None,
) -> dict:
    if out.exists():
        raise ValueError(f"refusing to overwrite evaluation report: {out}")
    document = load_labels(
        labels_path, require_frozen=True, require_certified=True
    )
    records = select_role(document, role)

    from PIL import Image, ImageOps
    from ultralytics import YOLO

    model = YOLO(str(weights))
    image_reports = []
    metric_rows = {threshold: [] for threshold in REPORT_THRESHOLDS}
    total_tiles = 0
    probe_rows = {
        size: {threshold: [] for threshold in REPORT_THRESHOLDS}
        for size in SCALE_PROBE_PX
    }
    probe_candidates = {size: 0 for size in SCALE_PROBE_PX}

    for record in records:
        source_path = resolve_source(labels_path, record["source"])
        if sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"source hash changed: {source_path}")
        with Image.open(source_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise ValueError(f"oriented dimensions changed: {source_path}")

        origins, raw_candidates = predict_tiled(
            model,
            image,
            tile=tile,
            overlap=overlap,
            batch=batch,
            device=device,
        )
        candidates = _boxes(raw_candidates)
        ground_truth, ignore_regions = annotation_boxes(record)
        ground_truth_visibility = [
            annotation["visibility"]
            for annotation in record["objects"]
            if annotation["visibility"] in {"clear", "partial"}
        ]
        all_mines = [
            Box.from_xyxy(annotation["xyxy"]) for annotation in record["objects"]
        ]
        empty_origins = find_empty_tiles(
            origins, tile, record["width"], record["height"], all_mines
        )
        total_tiles += len(origins)
        threshold_results = {}
        for threshold in REPORT_THRESHOLDS:
            metrics = evaluate_image(
                ground_truth,
                ignore_regions,
                candidates,
                threshold=threshold,
                merge_overlap=merge_overlap,
                ground_truth_visibility=ground_truth_visibility,
                empty_tile_origins=empty_origins,
            )
            metric_rows[threshold].append(metrics)
            threshold_results[f"{threshold:g}"] = metrics
        image_reports.append(
            {
                "source": record["source"],
                "source_sha256": record["source_sha256"],
                "width": record["width"],
                "height": record["height"],
                "capture_group": record["capture_group"],
                "role": record["role"],
                "tiles": len(origins),
                # All candidates are retained, including those below 0.37.
                "candidates": [asdict(candidate) for candidate in candidates],
                "thresholds": threshold_results,
            }
        )

        for annotation in record["objects"]:
            if annotation["visibility"] not in {"clear", "partial"}:
                continue
            for target_px in SCALE_PROBE_PX:
                probe, probe_truth = scale_probe(image, annotation, target_px, tile)
                probe_ignored = []
                for other in record["objects"]:
                    if other is annotation:
                        continue
                    transformed = transform_scale_probe_box(
                        annotation, other["xyxy"], target_px, tile
                    )
                    if transformed is not None:
                        probe_ignored.append(transformed)
                for region in record["ignore_regions"]:
                    transformed = transform_scale_probe_box(
                        annotation, region["xyxy"], target_px, tile
                    )
                    if transformed is not None:
                        probe_ignored.append(transformed)
                _, raw_probe_candidates = predict_tiled(
                    model,
                    probe,
                    tile=tile,
                    overlap=overlap,
                    batch=batch,
                    device=device,
                )
                candidates_for_probe = _boxes(raw_probe_candidates)
                probe_candidates[target_px] += len(candidates_for_probe)
                for threshold in REPORT_THRESHOLDS:
                    probe_rows[target_px][threshold].append(
                        evaluate_image(
                            [probe_truth],
                            probe_ignored,
                            candidates_for_probe,
                            threshold=threshold,
                            merge_overlap=merge_overlap,
                            ground_truth_visibility=[annotation["visibility"]],
                        )
                    )

    report = {
        "schema": EVALUATION_SCHEMA,
        "labels": str(labels_path.resolve()),
        "labels_sha256": sha256(labels_path),
        "weights": str(weights.resolve()),
        "weights_sha256": sha256(weights),
        "role": role,
        "candidate_floor": CANDIDATE_FLOOR,
        "tile_px": tile,
        "overlap_px": overlap,
        "merge_overlap": merge_overlap,
        "iou_threshold": 0.5,
        "thresholds": {
            f"{threshold:g}": {
                "threshold": threshold,
                **aggregate_metrics(metric_rows[threshold], total_tiles),
            }
            for threshold in REPORT_THRESHOLDS
        },
        "scale_probes": {
            str(size): {
                "target_max_side_px": size,
                "probes": len(probe_rows[size][REPORT_THRESHOLDS[0]]),
                "candidate_count_at_0.001": probe_candidates[size],
                "thresholds": {
                    f"{threshold:g}": {
                        "threshold": threshold,
                        **aggregate_metrics(
                            probe_rows[size][threshold],
                            len(probe_rows[size][threshold]),
                        ),
                    }
                    for threshold in REPORT_THRESHOLDS
                },
            }
            for size in SCALE_PROBE_PX
        },
        "images": image_reports,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(EVALUATION_ROLES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--merge-overlap", type=float, default=0.5)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    if args.tile < 1 or not 0 <= args.overlap < args.tile:
        parser.error("need tile > 0 and 0 <= overlap < tile")
    if args.batch < 1:
        parser.error("batch must be positive")
    if not 0.0 <= args.merge_overlap <= 1.0:
        parser.error("merge overlap must be in [0, 1]")
    run(
        args.weights,
        args.labels,
        args.role,
        args.out,
        tile=args.tile,
        overlap=args.overlap,
        batch=args.batch,
        merge_overlap=args.merge_overlap,
        device=args.device,
    )


if __name__ == "__main__":
    main()
