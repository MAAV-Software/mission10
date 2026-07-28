"""Run and score the focused SVO replay matrix."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from common import median, percentile, write_csv, write_json


RUN_IDS = (
    "A_static_allan",
    "B_hover_psd_floor",
    "C_hover_midpoint",
    "G_psd_solver_relaxed",
)


def make_dataset(
    source: Path, destination: Path, calibration: Path
) -> None:
    image_destination = destination / "data" / "img"
    image_destination.mkdir(parents=True, exist_ok=True)
    for filename in ("images.txt", "imu.txt"):
        shutil.copy2(source / "data" / filename, destination / "data" / filename)
    for image in (source / "data" / "img").iterdir():
        linked = image_destination / image.name
        if not linked.exists():
            os.link(image, linked)
    shutil.copy2(calibration, destination / "calib.yaml")


def run_matrix(
    dataset: Path,
    output: Path,
    svo_repo: Path,
    generated: Path,
    *,
    timeout_seconds: int = 900,
) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    run_records = []
    for run_id in RUN_IDS:
        run_output = output / "runs" / run_id
        run_dataset = output / "datasets" / run_id
        calibration = generated / "calibrations" / f"{run_id}.yaml"
        parameters = generated / "params" / f"{run_id}.yaml"
        record = {
            "run_id": run_id,
            "calibration": str(calibration.resolve()),
            "parameters": str(parameters.resolve()),
        }
        if (run_output / "init_events.txt").stat().st_size > 0 if (
            run_output / "init_events.txt"
        ).exists() else False:
            record.update({"returncode": 0, "status": "reused"})
            run_records.append(record)
            continue
        make_dataset(dataset, run_dataset, calibration)
        run_output.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [
                    str(svo_repo / "scripts" / "run_benchmark.sh"),
                    str(run_dataset),
                    str(parameters),
                    str(run_output),
                    "60",
                ],
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            record.update(
                {
                    "returncode": completed.returncode,
                    "status": (
                        "complete"
                        if completed.returncode == 0
                        else "failed"
                    ),
                }
            )
        except subprocess.TimeoutExpired:
            record.update({"returncode": 124, "status": "timeout"})
        record["wall_seconds"] = time.monotonic() - started
        run_records.append(record)
    write_json(output / "run_records.json", run_records)
    return run_records


def load_numeric(path: Path) -> list[list[float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            rows.append([float(value) for value in line.split()])
        except ValueError:
            continue
    return rows


def norm3(row) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in row))


def sustained_first(times, values, threshold: float, seconds: float):
    start = None
    for timestamp, value in zip(times, values, strict=True):
        if value > threshold:
            start = timestamp if start is None else start
            if timestamp - start >= seconds:
                return start
        else:
            start = None
    return None


def segment_score(rows, start: float, end: float) -> dict:
    chosen = [row for row in rows if start <= row[1] < end and len(row) >= 11]
    if not chosen:
        return {"samples": 0}
    speed = [norm3(row[2:5]) for row in chosen]
    gyro_bias = [norm3(row[5:8]) for row in chosen]
    accel_bias = [norm3(row[8:11]) for row in chosen]
    duration = chosen[-1][1] - chosen[0][1]
    accel_slope = 0.0
    if duration > 1.0:
        accel_slope = float(
            np.polyfit(
                [row[1] - chosen[0][1] for row in chosen],
                accel_bias,
                1,
            )[0]
        )
    return {
        "samples": len(chosen),
        "speed_median_m_s": median(speed),
        "speed_p95_m_s": percentile(speed, 95),
        "speed_max_m_s": max(speed),
        "gyro_bias_max_rad_s": max(gyro_bias),
        "accel_bias_max_m_s2": max(accel_bias),
        "accel_bias_slope_m_s3": accel_slope,
    }


def parse_initialization(path: Path) -> dict:
    result = {
        "visual_success_count": 0,
        "metric_init_count": 0,
        "first_visual_success_s": None,
        "first_metric_init_s": None,
    }
    if not path.exists():
        return result
    for line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0].startswith("#"):
            continue
        if fields[0] == "visual_init" and "Success" in fields:
            result["visual_success_count"] += 1
            if result["first_visual_success_s"] is None:
                result["first_visual_success_s"] = float(fields[1])
        elif fields[0] == "metric_init":
            result["metric_init_count"] += 1
            if result["first_metric_init_s"] is None:
                result["first_metric_init_s"] = float(fields[1])
    return result


def score_matrix(
    run_root: Path, segments_path: Path, output: Path
) -> dict:
    segments = json.loads(segments_path.read_text())["segments"]
    summaries = []
    segment_rows = []
    for run_id in RUN_IDS:
        run = run_root / run_id
        state = load_numeric(run / "speed_bias_estimate.txt")
        status_lines = (
            (run / "status.txt")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
            if (run / "status.txt").exists()
            else []
        )
        init = parse_initialization(run / "init_events.txt")
        reinitializations = max(0, init["visual_success_count"] - 1)
        speed = [norm3(row[2:5]) for row in state if len(row) >= 11]
        gyro_bias = [norm3(row[5:8]) for row in state if len(row) >= 11]
        accel_bias = [norm3(row[8:11]) for row in state if len(row) >= 11]
        times = [row[1] for row in state if len(row) >= 11]
        per_segment = {}
        for segment in segments:
            score = segment_score(
                state, segment["start_s"], segment["end_s"]
            )
            per_segment[segment["name"]] = score
            segment_rows.append(
                {"run_id": run_id, "segment": segment["name"], **score}
            )
        preflight = per_segment.get("preflight", {})
        sustained = sustained_first(times, speed, 3.0, 1.0) if speed else None
        physically_sane = bool(
            state
            and preflight.get("speed_median_m_s", math.inf) < 0.05
            and preflight.get("speed_p95_m_s", math.inf) < 0.15
            and sustained is None
            and max(gyro_bias, default=math.inf) < 0.02
            and max(accel_bias, default=math.inf) < 0.5
            and reinitializations == 0
            and not any(
                line.split()[-1] == "Failure"
                for line in status_lines
                if line.split()
            )
            and not any(
                abs(score.get("accel_bias_slope_m_s3", 0.0)) > 0.02
                and score.get("accel_bias_max_m_s2", 0.0) > 0.3
                for score in per_segment.values()
            )
        )
        summary = {
            "run_id": run_id,
            "state_samples": len(state),
            "status_samples": len(status_lines),
            "failure_status_samples": sum(
                line.split()[-1] == "Failure"
                for line in status_lines
                if line.split()
            ),
            "reinitialization_count": reinitializations,
            **init,
            "first_metric_init_after_takeoff_s": (
                init["first_metric_init_s"] - 1784937714.727742
                if init["first_metric_init_s"] is not None
                else None
            ),
            "final_speed_m_s": speed[-1] if speed else None,
            "max_speed_m_s": max(speed) if speed else None,
            "first_sustained_speed_over_3_s": sustained,
            "final_gyro_bias_rad_s": gyro_bias[-1] if gyro_bias else None,
            "max_gyro_bias_rad_s": max(gyro_bias) if gyro_bias else None,
            "final_accel_bias_m_s2": accel_bias[-1] if accel_bias else None,
            "max_accel_bias_m_s2": max(accel_bias) if accel_bias else None,
            "physically_sane": physically_sane,
        }
        summaries.append(summary)

    sane = {row["run_id"] for row in summaries if row["physically_sane"]}
    adjacent = any(
        pair <= sane
        for pair in (
            {"A_static_allan", "B_hover_psd_floor"},
            {"B_hover_psd_floor", "C_hover_midpoint"},
        )
    )
    decision = {
        "runs": summaries,
        "two_adjacent_noise_settings_sane": adjacent,
        "svo_advances": adjacent,
        "interpretation": (
            "SVO merits the next extrinsic-calibration stage."
            if adjacent
            else "SVO does not pass the bounded physical-sanity campaign."
        ),
    }
    write_csv(output / "svo_summary.csv", summaries)
    write_csv(output / "svo_segments.csv", segment_rows)
    write_json(output / "svo_decision.json", decision)
    return decision
