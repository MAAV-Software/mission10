"""Score retained paired-model CM2 candidates against certified exact frames."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

YOLO_ROOT = Path(__file__).resolve().parents[1]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

MISSION_ENGINE_SRC = Path(__file__).resolve().parents[3] / "ros" / "mission_engine"
if str(MISSION_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(MISSION_ENGINE_SRC))

from audit.evaluation import (
    Box,
    aggregate_metrics,
    annotation_boxes,
    evaluate_image,
    find_empty_tiles,
)
from audit.labels import load_labels, sha256
from mission_engine.core.tiles import tile_grid

MODELS = ("appearance", "production")
BAGS = ("manual", "return", "petal")
THRESHOLDS = (
    0.30,
    0.35,
    0.37,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.82,
    0.84,
    0.85,
    0.86,
    0.88,
    0.90,
    0.92,
    0.95,
)
TILE_PX = 640
OVERLAP_PX = 192


def retained_frames(path: Path, wanted: set[int]) -> dict[int, list[Box]]:
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for expected, line in enumerate(stream):
            document = json.loads(line)
            if int(document["frame"]) != expected:
                raise ValueError(f"non-contiguous candidate report: {path}")
            if expected in wanted:
                records[expected] = [
                    Box(
                        float(item["x0"]),
                        float(item["y0"]),
                        float(item["x1"]),
                        float(item["y1"]),
                        float(item["confidence"]),
                        int(item["tile_x"]),
                        int(item["tile_y"]),
                    )
                    for item in document["candidates"]
                ]
    if set(records) != wanted:
        raise ValueError(f"candidate report is missing selected frames: {path}")
    return records


def range_band(range_m: float) -> str:
    if range_m < 1.0:
        return "below_1m"
    if range_m < 3.0:
        return "1_to_3m"
    return "3m_and_above"


def report_group(
    records: list[dict[str, Any]],
    candidates: dict[tuple[str, int], list[Box]],
    threshold: float,
) -> dict[str, Any]:
    rows = []
    images = []
    for record in records:
        label = record["label"]
        ground_truth, ignored = annotation_boxes(label)
        origins = tile_grid(label["width"], label["height"], TILE_PX, OVERLAP_PX)
        empty = find_empty_tiles(
            origins,
            TILE_PX,
            label["width"],
            label["height"],
            [Box.from_xyxy(item["xyxy"]) for item in label["objects"]],
        )
        result = evaluate_image(
            ground_truth,
            ignored,
            candidates[(record["bag"], record["frame"])],
            threshold=threshold,
            merge_overlap=0.5,
            ground_truth_visibility=[
                item["visibility"]
                for item in label["objects"]
                if item["visibility"] in {"clear", "partial"}
            ],
            empty_tile_origins=empty,
        )
        rows.append(result)
        images.append(
            {
                "id": record["id"],
                "bag": record["bag"],
                "frame": record["frame"],
                "range_m": record["range_m"],
                **result,
            }
        )
    metrics = aggregate_metrics(rows, sum(len(tile_grid(
        record["label"]["width"],
        record["label"]["height"],
        TILE_PX,
        OVERLAP_PX,
    )) for record in records))
    metrics["false_positives_per_image"] = metrics["fp"] / len(records) if records else 0.0
    metrics["images_with_false_positive"] = sum(row["fp"] > 0 for row in rows)
    metrics["false_positive_image_rate"] = (
        metrics["images_with_false_positive"] / len(records) if records else 0.0
    )
    return {"metrics": metrics, "images": images}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("certification", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--labels", default="labels.json")
    args = parser.parse_args()

    labels_path = args.certification / args.labels
    labels = load_labels(labels_path, require_frozen=True, require_certified=True)
    exact_manifest = json.loads((args.certification / "manifest.json").read_text())
    by_id = {record["id"]: record for record in exact_manifest["records"]}
    records = []
    for label in labels["images"]:
        sample_id = Path(label["source"]).stem
        exact = by_id[sample_id]
        if sha256(args.certification / label["source"]) != label["source_sha256"]:
            raise ValueError(f"certified source changed: {label['source']}")
        records.append({**exact, "label": label})

    report: dict[str, Any] = {
        "schema": "mission10-cm2-certified-candidate-evaluation/1",
        "labels": str(labels_path.resolve()),
        "labels_sha256": sha256(labels_path),
        "exact_manifest_sha256": sha256(args.certification / "manifest.json"),
        "thresholds": THRESHOLDS,
        "tile_px": TILE_PX,
        "overlap_px": OVERLAP_PX,
        "models": {},
    }
    wanted = {bag: {record["frame"] for record in records if record["bag"] == bag} for bag in BAGS}
    for model in MODELS:
        candidates: dict[tuple[str, int], list[Box]] = {}
        checkpoint_hashes = set()
        for bag in BAGS:
            path = args.archive / "model-evaluations" / f"{model}_{bag}.jsonl.gz"
            meta = json.loads(path.with_suffix("").with_suffix(".meta.json").read_text())
            if sha256(path) != meta["output_sha256"]:
                raise ValueError(f"candidate report hash changed: {path}")
            checkpoint_hashes.add(meta["weights_sha256"])
            candidates.update(
                {
                    (bag, frame): boxes
                    for frame, boxes in retained_frames(path, wanted[bag]).items()
                }
            )
        if len(checkpoint_hashes) != 1:
            raise ValueError(f"{model} reports use different checkpoints")
        threshold_reports = {}
        for threshold in THRESHOLDS:
            groups = {"all": records, "competition_altitude": [record for record in records if record["range_m"] >= 1.0]}
            groups.update(
                {
                    f"bag_{bag}": [record for record in records if record["bag"] == bag]
                    for bag in BAGS
                }
            )
            groups.update(
                {
                    f"range_{band}": [record for record in records if range_band(record["range_m"]) == band]
                    for band in ("below_1m", "1_to_3m", "3m_and_above")
                }
            )
            threshold_reports[f"{threshold:.2f}"] = {
                name: report_group(group, candidates, threshold)
                for name, group in groups.items()
                if group
            }
        report["models"][model] = {
            "weights_sha256": checkpoint_hashes.pop(),
            "thresholds": threshold_reports,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
