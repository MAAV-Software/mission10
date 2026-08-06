"""Choose a recall-first operating threshold and evaluate held-out tiles.

The checkpoint is selected by Ultralytics on the validation split. This tool
then selects only the confidence threshold on validation (F-beta with a
precision floor) and applies that frozen threshold to test.

    python3 train/evaluate.py \
        --weights /workspace/runs/mission10-yolo/run/weights/best.pt \
        --prepared /workspace/dataset/production300-v1/prepared \
        --raw /workspace/dataset/production300-v1/raw \
        --out /workspace/runs/mission10-yolo/run/operational-evaluation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


EVALUATION_SCHEMA = "mission10-yolo-evaluation/1"
IMAGE_PX = 640


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0


@dataclass(frozen=True)
class ImageRecord:
    tile: str
    ground_truth: tuple[Box, ...]
    predictions: tuple[Box, ...]
    image_groups: tuple[tuple[str, str], ...] = ()
    ground_truth_size_bands: tuple[str, ...] = ()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(first: Box, second: Box) -> float:
    intersection = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0)) * max(
        0.0, min(first.y1, second.y1) - max(first.y0, second.y0)
    )
    first_area = max(0.0, first.x1 - first.x0) * max(0.0, first.y1 - first.y0)
    second_area = max(0.0, second.x1 - second.x0) * max(0.0, second.y1 - second.y0)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def match_image(
    ground_truth: Sequence[Box],
    predictions: Sequence[Box],
    threshold: float,
    iou_threshold: float = 0.5,
) -> tuple[set[int], int]:
    """Return matched ground-truth indices and the false-positive count."""
    unmatched = set(range(len(ground_truth)))
    matched = set()
    false_positives = 0
    for prediction in sorted(
        (box for box in predictions if box.confidence >= threshold),
        key=lambda box: box.confidence,
        reverse=True,
    ):
        candidates = [
            (iou(prediction, ground_truth[index]), index) for index in unmatched
        ]
        overlap, index = max(candidates, default=(0.0, -1))
        if overlap >= iou_threshold:
            unmatched.remove(index)
            matched.add(index)
        else:
            false_positives += 1
    return matched, false_positives


def _metrics(tp: int, fp: int, fn: int, beta: float) -> dict:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    beta2 = beta * beta
    denominator = beta2 * precision + recall
    fbeta = (
        (1.0 + beta2) * precision * recall / denominator
        if denominator
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
    }


def evaluate_records(
    records: Sequence[ImageRecord], threshold: float, beta: float = 2.0
) -> dict:
    tp = fp = fn = empty_tiles = empty_with_predictions = 0
    recall_groups: dict[str, dict[str, list[int]]] = {}
    for record in records:
        matched, image_fp = match_image(
            record.ground_truth, record.predictions, threshold
        )
        tp += len(matched)
        fp += image_fp
        fn += len(record.ground_truth) - len(matched)
        predictions_kept = sum(
            prediction.confidence >= threshold for prediction in record.predictions
        )
        if not record.ground_truth:
            empty_tiles += 1
            empty_with_predictions += predictions_kept > 0
        for gt_index in range(len(record.ground_truth)):
            groups = list(record.image_groups)
            if record.ground_truth_size_bands:
                groups.append(("box_size", record.ground_truth_size_bands[gt_index]))
            for axis, value in groups:
                counts = recall_groups.setdefault(axis, {}).setdefault(value, [0, 0])
                counts[0] += gt_index in matched
                counts[1] += 1
    result = _metrics(tp, fp, fn, beta)
    result.update(
        {
            "images": len(records),
            "empty_tiles": empty_tiles,
            "empty_tiles_with_predictions": empty_with_predictions,
            "empty_tile_false_positive_rate": (
                empty_with_predictions / empty_tiles if empty_tiles else 0.0
            ),
            "recall_by": {
                axis: {
                    value: {
                        "matched": counts[0],
                        "total": counts[1],
                        "recall": counts[0] / counts[1] if counts[1] else 1.0,
                    }
                    for value, counts in sorted(values.items())
                }
                for axis, values in sorted(recall_groups.items())
            },
        }
    )
    return result


def threshold_sweep(
    records: Sequence[ImageRecord], beta: float = 2.0
) -> list[dict]:
    return [
        {"threshold": step / 100.0, **evaluate_records(records, step / 100.0, beta)}
        for step in range(1, 100)
    ]


def choose_threshold(sweep: Sequence[dict], precision_floor: float) -> dict:
    eligible = [row for row in sweep if row["precision"] >= precision_floor]
    floor_met = bool(eligible)
    candidates = eligible if eligible else list(sweep)
    if not candidates:
        raise ValueError("threshold sweep is empty")
    chosen = max(
        candidates,
        key=lambda row: (
            row["fbeta"],
            row["recall"],
            row["precision"],
            -row["threshold"],
        ),
    )
    return {**chosen, "precision_floor_met": floor_met}


def _size_band(width: float, height: float) -> str:
    side = max(width, height) * IMAGE_PX
    if side < 16.0:
        return "tiny_lt16px"
    if side < 32.0:
        return "small_16to32px"
    return "large_ge32px"


def _altitude_band(altitude: float) -> str:
    if altitude < 3.0:
        return "low_lt3m"
    if altitude < 5.0:
        return "mid_3to5m"
    return "high_ge5m"


def _ground_truth(path: Path) -> tuple[tuple[Box, ...], tuple[str, ...]]:
    boxes = []
    size_bands = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"{path}:{line_number}: expected one-class YOLO label")
        x, y, width, height = map(float, fields[1:])
        boxes.append(
            Box(
                (x - width / 2) * IMAGE_PX,
                (y - height / 2) * IMAGE_PX,
                (x + width / 2) * IMAGE_PX,
                (y + height / 2) * IMAGE_PX,
            )
        )
        size_bands.append(_size_band(width, height))
    return tuple(boxes), tuple(size_bands)


def _scene_metadata(raw: Path, scenes: Iterable[str]) -> tuple[dict, dict]:
    manifests = {}
    stations = {}
    for scene in sorted(set(scenes)):
        manifest = json.loads((raw / f"{scene}.manifest.json").read_text())
        manifests[scene] = manifest
        stations.update(
            {station["stem"]: station for station in manifest["stations"]}
        )
    return manifests, stations


def _image_groups(manifest: dict, station: dict) -> tuple[tuple[str, str], ...]:
    grass = (manifest.get("grass") or {}).get("profile", "none")
    colors = sorted(
        {mine["appearance"]["color_family"] for mine in manifest["mines"]}
    )
    if len(colors) != 1:
        raise ValueError("expected one filament color family per scene")
    altitude = abs(float(station["pos"][2]))
    return (
        ("altitude", _altitude_band(altitude)),
        ("surface", manifest["surface"]["primary"]),
        ("grass", grass),
        ("color", colors[0]),
    )


def _predict_split(model, prepared: Path, raw: Path, lock: dict, split: str, batch: int):
    entries = lock["entries"][split]
    manifests, stations = _scene_metadata(raw, (entry["scene"] for entry in entries))
    image_paths = [prepared / "images" / split / f"{entry['tile']}.png" for entry in entries]
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=IMAGE_PX,
        conf=0.001,
        iou=0.7,
        max_det=100,
        batch=batch,
        device=0,
        verbose=False,
        stream=True,
    )
    records = []
    for entry, result in zip(entries, results, strict=True):
        tile = entry["tile"]
        ground_truth, size_bands = _ground_truth(
            prepared / "labels" / split / f"{tile}.txt"
        )
        predictions = tuple(
            Box(*map(float, xyxy), float(confidence))
            for xyxy, confidence in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                strict=True,
            )
        )
        source = tile.rsplit("_x", 1)[0]
        records.append(
            ImageRecord(
                tile=tile,
                ground_truth=ground_truth,
                predictions=predictions,
                image_groups=_image_groups(manifests[entry["scene"]], stations[source]),
                ground_truth_size_bands=size_bands,
            )
        )
    if len(records) != len(entries):
        raise RuntimeError(f"prediction count changed for {split}")
    return records


def _json_metrics(metrics) -> dict:
    return {key: float(value) for key, value in metrics.results_dict.items()}


def run(
    weights: Path,
    prepared: Path,
    raw: Path,
    out: Path,
    beta: float,
    precision_floor: float,
    batch: int,
) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    lock_path = prepared / "split.lock.json"
    lock = json.loads(lock_path.read_text())
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    validation = _predict_split(model, prepared, raw, lock, "val", batch)
    sweep = threshold_sweep(validation, beta)
    selected = choose_threshold(sweep, precision_floor)
    test = _predict_split(model, prepared, raw, lock, "test", batch)
    test_operational = evaluate_records(test, selected["threshold"], beta)
    test_metrics = model.val(
        data=str((prepared / "dataset.yaml").resolve()),
        split="test",
        imgsz=IMAGE_PX,
        batch=batch,
        device=0,
        workers=8,
        plots=True,
        project=str(out.resolve()),
        name="ultralytics-test",
        exist_ok=False,
    )
    test_ultralytics = _json_metrics(test_metrics)
    with (out / "thresholds.csv").open("w", newline="") as stream:
        fields = [
            "threshold",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "fbeta",
            "empty_tile_false_positive_rate",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sweep)
    acceptance = {
        "precision_at_least_0_90": test_operational["precision"] >= 0.90,
        "recall_at_least_0_95": test_operational["recall"] >= 0.95,
        "map50_95_at_least_0_85": (
            test_ultralytics.get("metrics/mAP50-95(B)", 0.0) >= 0.85
        ),
        "empty_tile_false_positive_rate_at_most_0_05": (
            test_operational["empty_tile_false_positive_rate"] <= 0.05
        ),
    }
    acceptance["numeric_gates_pass"] = all(acceptance.values())
    report = {
        "schema": EVALUATION_SCHEMA,
        "weights": str(weights.resolve()),
        "weights_sha256": sha256(weights),
        "dataset_sha256": lock["dataset_sha256"],
        "dataset_lock_sha256": sha256(lock_path),
        "matching_iou": 0.5,
        "nms_iou": 0.7,
        "beta": beta,
        "precision_floor": precision_floor,
        "validation_selection": selected,
        "test_operational": test_operational,
        "test_ultralytics": test_ultralytics,
        "acceptance": acceptance,
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--precision-floor", type=float, default=0.90)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)
    if args.beta <= 0.0 or not 0.0 <= args.precision_floor <= 1.0:
        parser.error("beta must be positive and precision floor must be in [0, 1]")
    if args.batch < 1:
        parser.error("batch must be positive")
    report = run(
        args.weights,
        args.prepared,
        args.raw,
        args.out,
        args.beta,
        args.precision_floor,
        args.batch,
    )
    selected = report["validation_selection"]
    test = report["test_operational"]
    print(
        f"threshold={selected['threshold']:.2f} "
        f"test_precision={test['precision']:.6f} "
        f"test_recall={test['recall']:.6f} "
        f"empty_fpr={test['empty_tile_false_positive_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
