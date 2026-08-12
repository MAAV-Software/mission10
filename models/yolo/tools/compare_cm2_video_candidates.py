"""Compare paired full-video candidate reports across CM2 flight bags."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

MISSION_ENGINE_SRC = Path(__file__).resolve().parents[3] / "ros" / "mission_engine"
if str(MISSION_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(MISSION_ENGINE_SRC))

from audit_irl import Detection, merge_detections
from mission_engine.core.backproject import AboveHorizon, ground_point
from mission_engine.core.config import CameraModel

DEVELOPMENT_BAGS = ("manual", "return", "petal")
BAG_DIRECTORIES = {
    "manual": "manual_survey",
    "return": "return_failure",
    "petal": "petal_qual",
    "survey": "survey",
}
MODELS = ("appearance", "production")
THRESHOLDS = (0.30, 0.35, 0.37, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
TRACK_THRESHOLDS = (0.37, 0.60)
CAMERA = CameraModel(width_px=1640, height_px=1232)


def truth(value: str) -> bool:
    return value.lower() == "true"


def operational(row: dict[str, str]) -> bool:
    try:
        return (
            truth(row["flags_cs_in_air"])
            and int(row["status_arming_state"]) == 2
            and float(row["range_current_distance"]) >= 0.4
        )
    except (KeyError, TypeError, ValueError):
        return False


def detection(document: dict[str, Any]) -> Detection:
    return Detection(**document)


def project(row: dict[str, str], box: Detection) -> tuple[float, float] | None:
    try:
        height = float(row["range_current_distance"])
        position = (float(row["local_x"]), float(row["local_y"]), -height)
        attitude = tuple(float(row[f"attitude_q{index}"]) for index in range(4))
        pixel = ((box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0)
        return ground_point(CAMERA, position, attitude, pixel)
    except (AboveHorizon, KeyError, TypeError, ValueError):
        return None


def load_frames(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows):
        if int(row["frame"]) != index:
            raise ValueError(f"non-contiguous frame CSV: {path}")
    return rows


def load_candidates(path: Path, expected: int) -> list[list[Detection]]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            document = json.loads(line)
            if document["frame"] != index:
                raise ValueError(f"non-contiguous candidates: {path}")
            records.append([detection(item) for item in document["candidates"]])
    if len(records) != expected:
        raise ValueError(f"candidate/frame mismatch for {path}: {len(records)}/{expected}")
    return records


def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def build_tracks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    next_id = 0
    for record in records:
        frame = record["frame"]
        if not record["operational"]:
            tracks.extend(active)
            active = []
            continue
        expired = [track for track in active if frame - track["last_frame"] > 2]
        tracks.extend(expired)
        active = [track for track in active if track not in expired]
        used = set()
        for item in record["detections"]:
            point = item["ground_ne"]
            if point is None:
                continue
            choices = [
                (distance(point, track["last_ne"]), index, track)
                for index, track in enumerate(active)
                if index not in used
            ]
            choices = [choice for choice in choices if choice[0] <= 0.75]
            if choices:
                _, index, track = min(choices)
                used.add(index)
            else:
                track = {
                    "track": next_id,
                    "start_frame": frame,
                    "last_frame": frame,
                    "last_ne": point,
                    "observations": [],
                }
                next_id += 1
                active.append(track)
                used.add(len(active) - 1)
            track["last_frame"] = frame
            track["last_ne"] = point
            track["observations"].append(
                {
                    "frame": frame,
                    "confidence": item["box"]["confidence"],
                    "ground_ne": point,
                }
            )
        expired = [track for track in active if frame - track["last_frame"] >= 2]
        tracks.extend(expired)
        active = [track for track in active if track not in expired]
    tracks.extend(active)
    for track in tracks:
        observations = track.pop("observations")
        peak = max(observations, key=lambda item: item["confidence"])
        track.update(
            {
                "detections": len(observations),
                "end_frame": observations[-1]["frame"],
                "peak_frame": peak["frame"],
                "peak_confidence": peak["confidence"],
                "start_ne": observations[0]["ground_ne"],
                "end_ne": observations[-1]["ground_ne"],
            }
        )
        track.pop("last_frame", None)
        track.pop("last_ne", None)
    return tracks


def summarize(records: list[dict[str, Any]], tracks: list[dict[str, Any]]) -> dict[str, Any]:
    operational_records = [record for record in records if record["operational"]]
    detections = [item for record in operational_records for item in record["detections"]]
    persistent = [track for track in tracks if track["detections"] >= 2]
    return {
        "operational_frames": len(operational_records),
        "operational_minutes": len(operational_records) / 600.0,
        "frames_with_detections": sum(bool(record["detections"]) for record in operational_records),
        "detections": len(detections),
        "maximum_confidence": max(
            (item["box"]["confidence"] for item in detections), default=None
        ),
        "tracks": len(tracks),
        "persistent_tracks": len(persistent),
        "persistent_tracks_per_minute": len(persistent) / (len(operational_records) / 600.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--bags",
        nargs="+",
        choices=tuple(BAG_DIRECTORIES),
        default=DEVELOPMENT_BAGS,
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    parser.add_argument(
        "--track-thresholds", nargs="+", type=float, default=TRACK_THRESHOLDS
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "mission10-cm2-candidate-comparison/1",
        "thresholds": args.thresholds,
        "track_policy": {"maximum_gap_frames": 2, "maximum_ground_distance_m": 0.75},
        "bags": {},
    }
    review_sources: dict[str, dict[int, dict[str, Any]]] = {}
    for bag in args.bags:
        frame_path = args.archive / "bags" / BAG_DIRECTORIES[bag] / "frames.csv"
        frames = load_frames(frame_path)
        review_sources[bag] = {}
        report["bags"][bag] = {
            "frames": len(frames),
            "operational_frames": sum(operational(row) for row in frames),
            "models": {},
        }
        for model in MODELS:
            candidate_path = args.archive / "model-evaluations" / f"{model}_{bag}.jsonl.gz"
            candidates = load_candidates(candidate_path, len(frames))
            threshold_report = {}
            for threshold in args.thresholds:
                records = []
                for index, (row, raw) in enumerate(zip(frames, candidates, strict=True)):
                    selected = [item for item in raw if item.confidence >= threshold]
                    merged = merge_detections(selected, 0.5)
                    items = [
                        {"box": asdict(box), "ground_ne": project(row, box)}
                        for box in merged
                    ]
                    records.append(
                        {"frame": index, "operational": operational(row), "detections": items}
                    )
                tracks = build_tracks(records)
                threshold_report[f"{threshold:.2f}"] = {
                    **summarize(records, tracks),
                    "persistent_track_records": [
                        track for track in tracks if track["detections"] >= 2
                    ],
                }
                if threshold in args.track_thresholds:
                    for record in records:
                        if record["detections"]:
                            source = review_sources[bag].setdefault(record["frame"], {})
                            source[f"{model}_{threshold:.2f}"] = record["detections"]
            report["bags"][bag]["models"][model] = threshold_report

    review = {
        bag: [
            {"frame": frame, "detections": detections}
            for frame, detections in sorted(records.items())
        ]
        for bag, records in review_sources.items()
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "review_sources.json").write_text(json.dumps(review, indent=2) + "\n")
    print(f"wrote {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
