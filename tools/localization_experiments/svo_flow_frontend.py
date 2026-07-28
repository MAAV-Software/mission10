#!/usr/bin/env python3
"""Prepare, replay, and score the rl_vo SVO frontend on the Drone4 CM2 bag.

This experiment consumes SVO's adjacent-frame pixel correspondences. It never
uses SVO pose or scale and never publishes a ROS or PX4 message.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import (
    image_to_gray,
    iter_messages,
    median,
    percentile,
    split_mcaps,
    stamp_ns,
    write_csv,
    write_json,
    write_pgm,
)
from flow import (
    _collect_motion,
    _evaluate_hypothesis,
    _load_calibration,
    _rotation_hypothesis,
    _score_quality,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DEFAULT_BAG = (
    WORKSPACE
    / "reference/20260727_drone4_cm2_flow_handheld_validation_uncompressed"
)
DEFAULT_WORK = Path("/tmp/maav_svo_cm2_flow")
DEFAULT_CALIBRATION = (
    WORKSPACE / "mission10/tools/flight_recorder/config/cm2_intrinsics_rs.yaml"
)
DEFAULT_SVO_PARAMS = HERE / "config/svo_flow_params.yaml"
DEFAULT_SVO_CALIBRATION = HERE / "config/svo_flow_cm2_820.yaml"
DEFAULT_KLT_ANALYSIS = (
    DEFAULT_BAG / "analysis/localization_experiments"
)
WIDTH = 820
HEIGHT = 616
NATIVE_WIDTH = 1640
NATIVE_HEIGHT = 1232
LINE_DELAY_S = 0.00000968869339923
HALF_READOUT_NS = int(
    round(0.5 * (NATIVE_HEIGHT - 1) * LINE_DELAY_S * 1e9)
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def prepare_dataset(
    bag: Path,
    output: Path,
    *,
    max_frames: int | None = None,
) -> dict:
    """Extract lossless half-resolution luma and preserve physical timestamps."""
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and max_frames is None:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    frame_rows = []
    for topic, _, message in iter_messages(bag, ["/camera_down/image_raw"]):
        if topic != "/camera_down/image_raw":
            continue
        if max_frames is not None and len(frame_rows) >= max_frames:
            break
        first_row_ns = stamp_ns(message.header)
        gray = image_to_gray(message)
        if gray.shape != (NATIVE_HEIGHT, NATIVE_WIDTH):
            raise RuntimeError(
                f"unexpected CM2 shape {gray.shape}; expected "
                f"{(NATIVE_HEIGHT, NATIVE_WIDTH)}"
            )
        working = cv2.resize(
            gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
        )
        filename = f"{len(frame_rows):06d}_{first_row_ns}.pgm"
        write_pgm(image_dir / filename, working)
        frame_rows.append(
            {
                "frame": len(frame_rows),
                "first_row_timestamp_ns": first_row_ns,
                "center_row_timestamp_ns": first_row_ns + HALF_READOUT_NS,
                "image": f"images/{filename}",
            }
        )
    if len(frame_rows) < 2:
        raise RuntimeError(f"only extracted {len(frame_rows)} CM2 frames")
    write_csv(output / "frames.csv", frame_rows)
    timestamps = np.asarray(
        [row["center_row_timestamp_ns"] for row in frame_rows],
        dtype=np.int64,
    )
    source_rows = [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in split_mcaps(bag)
    ]
    result = {
        "bag": str(bag.resolve()),
        "sources": source_rows,
        "frames": len(frame_rows),
        "width": WIDTH,
        "height": HEIGHT,
        "native_width": NATIVE_WIDTH,
        "native_height": NATIVE_HEIGHT,
        "line_delay_s": LINE_DELAY_S,
        "timestamp_reference": "physical center row",
        "strict_timestamps": bool(np.all(np.diff(timestamps) > 0)),
        "effective_rate_hz": float(
            (len(timestamps) - 1) * 1e9
            / (int(timestamps[-1]) - int(timestamps[0]))
        ),
        "svo_params": str(DEFAULT_SVO_PARAMS.resolve()),
        "svo_calibration": str(DEFAULT_SVO_CALIBRATION.resolve()),
    }
    if max_frames is None:
        write_json(manifest_path, result)
    else:
        write_json(output / "manifest_smoke.json", result)
    return result


def _tag_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    parameters.aprilTagQuadDecimate = 2.0
    parameters.minMarkerPerimeterRate = 0.01
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.errorCorrectionRate = 0.8
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def replay_svo(
    dataset: Path,
    svo_build: Path,
    output: Path,
    *,
    svo_params: Path = DEFAULT_SVO_PARAMS,
    svo_calibration: Path = DEFAULT_SVO_CALIBRATION,
    param_overrides: list[str] | None = None,
    frontend: str = "tracker",
    start_frame: int = 0,
    max_frames: int | None = None,
    pace_hz: float = 0.0,
    tag_load: bool = False,
    seed: int = 7,
) -> dict:
    """Run the patched one-environment SVO binding and export all track pairs."""
    sys.path.insert(0, str(svo_build.resolve()))
    try:
        import svo_env
    except ImportError as exc:
        raise RuntimeError(
            f"cannot import svo_env from {svo_build}; use the patched build"
        ) from exc

    output.mkdir(parents=True, exist_ok=True)
    frames = [
        row for row in _read_csv(dataset / "frames.csv")
        if int(row["frame"]) >= start_frame
    ]
    if max_frames is not None:
        frames = frames[:max_frames]
    if len(frames) < 2:
        raise RuntimeError(
            f"only {len(frames)} frames remain after start-frame={start_frame}"
        )
    effective_params = svo_params
    parsed_overrides = {}
    if param_overrides:
        parameters = yaml.safe_load(svo_params.read_text(encoding="utf-8"))
        for override in param_overrides:
            key, separator, raw_value = override.partition("=")
            if not separator or not key:
                raise ValueError(
                    f"invalid --param {override!r}; expected KEY=YAML_VALUE"
                )
            value = yaml.safe_load(raw_value)
            parameters[key] = value
            parsed_overrides[key] = value
        effective_params = output / "effective_svo_params.yaml"
        effective_params.write_text(
            yaml.safe_dump(parameters, sort_keys=False), encoding="utf-8"
        )
    environment = svo_env.SVOEnv(
        str(effective_params),
        str(svo_calibration),
        1,
        True,
    )
    environment.setSeed(seed)
    cv2.setRNGSeed(seed)
    action = np.zeros((1, 2), dtype=np.float64)
    use_rl = np.zeros(1, dtype=np.float64)
    poses = np.zeros((1, 16), dtype=np.float64)
    observations = np.zeros((1, 180 * 3 + 24), dtype=np.float64)
    dones = np.zeros(1, dtype=np.float64)
    stages = np.zeros(1, dtype=np.float64)
    runtime = np.zeros(1, dtype=np.float64)
    use_gt = np.zeros(1, dtype=np.float64)
    gt_init = -np.ones((1, 7), dtype=np.float64)

    detector = _tag_detector() if tag_load else None
    detector_pool = ThreadPoolExecutor(max_workers=1) if tag_load else None
    detector_future = None
    detector_submitted = detector_processed = detector_dropped = 0
    detector_latencies = []

    frame_output = output / "frames.csv"
    track_output = output / "tracks.csv"
    frame_fields = [
        "frame", "timestamp_ns", "stage", "done", "runtime_ms",
        "wall_ms", "track_count", "schedule_backlog_ms", "reset_after_frame",
    ]
    track_fields = [
        "frame", "previous_timestamp_ns", "timestamp_ns", "track_id",
        "u_previous", "v_previous", "u_current", "v_current",
    ]
    wall_start = time.perf_counter()
    first_sensor_ns = int(frames[0]["center_row_timestamp_ns"])
    previous_sensor_ns = None
    latency_values = []
    backlog_values = []
    valid_frames = 0
    reset_count = 0
    with frame_output.open("w", newline="", encoding="utf-8") as frame_stream, \
            track_output.open("w", newline="", encoding="utf-8") as track_stream:
        frame_writer = csv.DictWriter(frame_stream, fieldnames=frame_fields)
        track_writer = csv.DictWriter(track_stream, fieldnames=track_fields)
        frame_writer.writeheader()
        track_writer.writeheader()
        for replay_index, row in enumerate(frames):
            frame = int(row["frame"])
            sensor_ns = int(row["center_row_timestamp_ns"])
            if pace_hz > 0:
                target = wall_start + replay_index / pace_hz
                remaining = target - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            else:
                target = time.perf_counter()
            image = cv2.imread(
                str(dataset / row["image"]), cv2.IMREAD_GRAYSCALE
            )
            if image is None or image.shape != (HEIGHT, WIDTH):
                raise RuntimeError(f"failed to load frame {row['image']}")

            if detector_pool is not None:
                if detector_future is not None and detector_future.done():
                    detector_latencies.append(detector_future.result())
                    detector_processed += 1
                    detector_future = None
                if detector_future is None:
                    # Live detection sees native CM2 frames and internally
                    # decimates by two. Upscaling preserves approximately the
                    # same amount of detector work without a 5.1 GiB replay.
                    tag_image = cv2.resize(
                        image, (NATIVE_WIDTH, NATIVE_HEIGHT),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    def detect_timed(payload=tag_image):
                        started = time.perf_counter()
                        detector.detectMarkers(payload)
                        return (time.perf_counter() - started) * 1e3

                    detector_future = detector_pool.submit(detect_timed)
                    detector_submitted += 1
                else:
                    detector_dropped += 1

            started = time.perf_counter()
            if frontend == "tracker":
                pairs = np.asarray(
                    environment.env_flow_step(0, image, sensor_ns),
                    dtype=np.float64,
                ).reshape(-1, 5)
                stages[0] = 1.0 if replay_index == 0 else 2.0
                dones[0] = 0.0
            else:
                timestamp = np.asarray([sensor_ns], dtype=np.float64)
                batch = image[None, :, :, None]
                environment.step(
                    batch, timestamp, action, use_rl, poses, observations,
                    dones, stages, runtime, use_gt, gt_init,
                )
                pairs = np.asarray(
                    environment.env_track_pairs(0), dtype=np.float64
                ).reshape(-1, 5)
            wall_ms = (time.perf_counter() - started) * 1e3
            reset_after_frame = frontend == "map" and bool(dones[0])
            if len(pairs) >= 8:
                valid_frames += 1
            for pair in pairs:
                track_writer.writerow(
                    {
                        "frame": frame,
                        "previous_timestamp_ns": previous_sensor_ns,
                        "timestamp_ns": sensor_ns,
                        "track_id": int(pair[0]),
                        "u_previous": pair[1],
                        "v_previous": pair[2],
                        "u_current": pair[3],
                        "v_current": pair[4],
                    }
                )
            backlog_ms = max(
                0.0, (time.perf_counter() - target) * 1e3
            )
            latency_values.append(wall_ms)
            backlog_values.append(backlog_ms)
            frame_writer.writerow(
                {
                    "frame": frame,
                    "timestamp_ns": sensor_ns,
                    "stage": int(stages[0]),
                    "done": int(dones[0]),
                    "runtime_ms": (
                        float(runtime[0]) * 1e3
                        if frontend == "map" else wall_ms
                    ),
                    "wall_ms": wall_ms,
                    "track_count": len(pairs),
                    "schedule_backlog_ms": backlog_ms,
                    "reset_after_frame": int(reset_after_frame),
                }
            )
            if reset_after_frame:
                # SVOEnv follows the vectorized RL environment contract:
                # `done` asks the caller to reset that environment. Without
                # this, it remains unable to initialize after a lost map.
                environment.reset(np.asarray([0.0], dtype=np.float64))
                dones[0] = 0.0
                reset_count += 1
            previous_sensor_ns = sensor_ns

    if detector_pool is not None:
        if detector_future is not None:
            detector_latencies.append(detector_future.result())
            detector_processed += 1
        detector_pool.shutdown()
    sensor_duration_s = (
        int(frames[-1]["center_row_timestamp_ns"]) - first_sensor_ns
    ) / 1e9
    warmup_cutoff = first_sensor_ns + 1_000_000_000
    after_warmup = [
        row for row in _read_csv(frame_output)
        if int(row["timestamp_ns"]) >= warmup_cutoff
    ]
    valid_after_warmup = sum(
        int(row["track_count"]) >= 8 for row in after_warmup
    )
    result = {
        "frames": len(frames),
        "frontend": frontend,
        "svo_params": str(effective_params.resolve()),
        "svo_calibration": str(svo_calibration.resolve()),
        "param_overrides": parsed_overrides,
        "start_frame": start_frame,
        "first_timestamp_ns": first_sensor_ns,
        "sensor_duration_s": sensor_duration_s,
        "wall_duration_s": time.perf_counter() - wall_start,
        "pace_hz": pace_hz,
        "tag_load": tag_load,
        "valid_frames": valid_frames,
        "resets": reset_count,
        "valid_rate_hz": valid_frames / max(sensor_duration_s, 1e-9),
        "valid_availability_after_1s": (
            valid_after_warmup / max(1, len(after_warmup))
        ),
        "latency_ms": {
            "median": median(latency_values),
            "p95": percentile(latency_values, 95),
            "max": max(latency_values),
        },
        "schedule_backlog_ms": {
            "p95": percentile(backlog_values, 95),
            "max": max(backlog_values),
        },
        "tag_detector": {
            "submitted": detector_submitted,
            "processed": detector_processed,
            "dropped": detector_dropped,
            "latency_median_ms": median(detector_latencies),
            "latency_p95_ms": percentile(detector_latencies, 95),
        } if tag_load else None,
        "seed": seed,
    }
    write_json(output / "summary.json", result)
    return result


def _tile_coverage(points: np.ndarray) -> float:
    occupied = {
        (
            min(7, max(0, int(x * 8 / WIDTH))),
            min(5, max(0, int(y * 6 / HEIGHT))),
        )
        for x, y in points
    }
    return len(occupied) / 48.0


def _symmetric_transfer_median(
    homography: np.ndarray, p0: np.ndarray, p1: np.ndarray
) -> float:
    forward = cv2.perspectiveTransform(
        p0.reshape(-1, 1, 2), homography
    )[:, 0]
    inverse = np.linalg.inv(homography)
    backward = cv2.perspectiveTransform(
        p1.reshape(-1, 1, 2), inverse
    )[:, 0]
    return float(np.median(
        0.5 * (
            np.linalg.norm(forward - p1, axis=1)
            + np.linalg.norm(backward - p0, axis=1)
        )
    ))


def tracks_to_flow(
    bag: Path,
    replay: Path,
    output: Path,
    calibration: Path = DEFAULT_CALIBRATION,
    ransac_threshold_px: float = 2.0,
) -> dict:
    """Apply the existing CM2/PX4 angular-flow geometry to SVO correspondences."""
    output.mkdir(parents=True, exist_ok=True)
    frame_rows = {
        int(row["frame"]): row for row in _read_csv(replay / "frames.csv")
    }
    grouped: dict[int, list[dict]] = {}
    for row in _read_csv(replay / "tracks.csv"):
        grouped.setdefault(int(row["frame"]), []).append(row)
    motion = _collect_motion(bag)
    native_k, distortion, line_delay, native_resolution = _load_calibration(
        calibration
    )
    if native_resolution != (NATIVE_WIDTH, NATIVE_HEIGHT):
        raise RuntimeError(f"unexpected calibration resolution {native_resolution}")
    working_k = native_k.copy()
    working_k[:2] *= 0.5
    r_b_c = _rotation_hypothesis(0)
    pair_rows = []
    flow_rows = []
    for frame in sorted(grouped):
        rows = grouped[frame]
        if frame <= 0 or len(rows) < 8:
            continue
        p0 = np.asarray(
            [[float(row["u_previous"]), float(row["v_previous"])] for row in rows]
        )
        p1 = np.asarray(
            [[float(row["u_current"]), float(row["v_current"])] for row in rows]
        )
        homography, mask = cv2.findHomography(
            p0, p1, cv2.RANSAC, ransac_threshold_px
        )
        if homography is None or mask is None:
            continue
        mask = mask[:, 0].astype(bool)
        tracked = len(p0)
        p0 = p0[mask]
        p1 = p1[mask]
        if len(p0) < 8:
            continue
        previous = frame_rows[frame - 1]
        current = frame_rows[frame]
        t0 = int(previous["timestamp_ns"]) / 1e9
        t1 = int(current["timestamp_ns"]) / 1e9
        if t1 <= t0:
            continue
        p0_norm = cv2.undistortPoints(
            p0.reshape(-1, 1, 2), working_k, distortion
        )[:, 0]
        p1_norm = cv2.undistortPoints(
            p1.reshape(-1, 1, 2), working_k, distortion
        )[:, 0]
        evaluated = _evaluate_hypothesis(
            p0_norm, p1_norm, p0[:, 1], p1[:, 1], t0, t1, motion,
            r_b_c, NATIVE_HEIGHT, HEIGHT, line_delay, 16, 1,
        )
        transfer = _symmetric_transfer_median(homography, p0, p1)
        coverage = _tile_coverage(p0)
        inlier_fraction = len(p0) / tracked
        quality = _score_quality(
            coverage,
            inlier_fraction,
            transfer,
            evaluated["compensated_residual_p95_rad"],
        )
        pair_id = frame - 1
        pair_rows.append(
            {
                "pair": pair_id,
                "frame": frame,
                "timestamp_sample_s": t1,
                "integration_time_s": t1 - t0,
                "detected_features": tracked,
                "fb_good_features": tracked,
                "homography_inliers": len(p0),
                "tile_coverage": coverage,
                "homography_inlier_fraction": inlier_fraction,
                # Compatibility name for the existing quality/report tooling.
                "fb_error_median_px": transfer,
                "tracker_error_kind": "symmetric_homography_transfer",
                "distance_m": None,
                "distance_quality": None,
                "local_x_m": None,
                "local_y_m": None,
                "local_vx_m_s": None,
                "local_vy_m_s": None,
                "heading_rad": None,
                "setpoint_x_m": None,
                "setpoint_y_m": None,
            }
        )
        flow_rows.append(
            {
                "pair": pair_id,
                "timestamp_sample_s": t1,
                "yaw_quadrants": 0,
                "rs_bands": 16,
                "readout_sign": 1,
                "quality": quality,
                **evaluated,
            }
        )
    write_csv(output / "flow_pairs.csv", pair_rows)
    write_csv(output / "flow_selected.csv", flow_rows)
    total_pairs = max(1, len(frame_rows) - 1)
    result = {
        "input_pairs": total_pairs,
        "accepted_pairs": len(flow_rows),
        "availability": len(flow_rows) / total_pairs,
        "quality_median": median(row["quality"] for row in flow_rows),
        "quality_p05": percentile([row["quality"] for row in flow_rows], 5),
        "residual_p95_rad": percentile(
            [row["compensated_residual_p95_rad"] for row in flow_rows], 95
        ),
        "geometry": {
            "yaw_quadrants": 0,
            "rolling_shutter_bands": 16,
            "readout_sign": 1,
            "line_delay_s": line_delay,
            "ransac_threshold_px": ransac_threshold_px,
        },
    }
    write_json(output / "flow_summary.json", result)
    return result


def decision(
    replay_summary: Path,
    flow_summary: Path,
    svo_aprilgrid: Path,
    klt_analysis: Path,
    output: Path,
) -> dict:
    replay = json.loads(replay_summary.read_text())
    flow = json.loads(flow_summary.read_text())
    svo_angular = json.loads(
        (svo_aprilgrid / "angular_flow_validation.json").read_text()
    )
    svo_loop = json.loads(
        (svo_aprilgrid / "square_loop_validation.json").read_text()
    )
    klt_angular = json.loads(
        (klt_analysis / "angular_flow_validation.json").read_text()
    )
    klt_loop = json.loads(
        (klt_analysis / "square_loop_validation.json").read_text()
    )
    svo_window = svo_angular["half_second_level_windows"]
    klt_window = klt_angular["half_second_level_windows"]
    checks = {
        "valid_availability_after_1s": (
            replay["valid_availability_after_1s"] >= 0.95
        ),
        "accepted_flow_availability": flow["availability"] >= 0.95,
        "valid_rate_hz": replay["valid_rate_hz"] >= 20.0,
        "latency_p95_ms": replay["latency_ms"]["p95"] <= 50.0,
        "backlog_max_ms": replay["schedule_backlog_ms"]["max"] <= 100.0,
        "window_median_error": (
            svo_window["vector_error_median_m"]
            <= 1.10 * klt_window["vector_error_median_m"]
        ),
        "window_p95_error": (
            svo_window["vector_error_p95_m"]
            <= 1.10 * klt_window["vector_error_p95_m"]
        ),
        "loop_endpoint_error": (
            svo_loop["endpoint_vector_error_m"]
            <= 1.10 * klt_loop["endpoint_vector_error_m"]
        ),
        "window_scale": 0.90 <= svo_window["scale_median"] <= 1.10,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "svo": {
            "replay": replay,
            "flow": flow,
            "half_second_windows": svo_window,
            "loop": {
                "path_length_scale": svo_loop["path_length_scale"],
                "endpoint_vector_error_m": svo_loop[
                    "endpoint_vector_error_m"
                ],
            },
        },
        "klt_baseline": {
            "half_second_windows": klt_window,
            "loop": {
                "path_length_scale": klt_loop["path_length_scale"],
                "endpoint_vector_error_m": klt_loop[
                    "endpoint_vector_error_m"
                ],
            },
        },
        "recommendation": (
            "advance to a selectable live SVO flow backend"
            if all(checks.values())
            else "do not replace KLT; inspect failed gates"
        ),
    }
    write_json(output, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    prepare.add_argument("--output", type=Path, default=DEFAULT_WORK / "dataset")
    prepare.add_argument("--max-frames", type=int)
    replay = commands.add_parser("replay")
    replay.add_argument("--dataset", type=Path, default=DEFAULT_WORK / "dataset")
    replay.add_argument("--svo-build", type=Path, required=True)
    replay.add_argument("--output", type=Path, default=DEFAULT_WORK / "replay")
    replay.add_argument("--svo-params", type=Path, default=DEFAULT_SVO_PARAMS)
    replay.add_argument(
        "--svo-calibration", type=Path, default=DEFAULT_SVO_CALIBRATION
    )
    replay.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one SVO YAML parameter; repeat for a sweep point",
    )
    replay.add_argument(
        "--frontend", choices=("tracker", "map"), default="tracker"
    )
    replay.add_argument("--start-frame", type=int, default=0)
    replay.add_argument("--max-frames", type=int)
    replay.add_argument("--pace-hz", type=float, default=0.0)
    replay.add_argument("--tag-load", action="store_true")
    replay.add_argument("--seed", type=int, default=7)
    convert = commands.add_parser("convert")
    convert.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    convert.add_argument("--replay", type=Path, default=DEFAULT_WORK / "replay")
    convert.add_argument("--output", type=Path, default=DEFAULT_WORK / "flow")
    convert.add_argument("--ransac-threshold-px", type=float, default=2.0)
    decide = commands.add_parser("decision")
    decide.add_argument("--replay-summary", type=Path, required=True)
    decide.add_argument("--flow-summary", type=Path, required=True)
    decide.add_argument("--svo-aprilgrid", type=Path, required=True)
    decide.add_argument("--klt-analysis", type=Path, default=DEFAULT_KLT_ANALYSIS)
    decide.add_argument("--output", type=Path, default=DEFAULT_WORK / "decision.json")
    return result


def main() -> int:
    arguments = parser().parse_args()
    if arguments.command == "prepare":
        value = prepare_dataset(
            arguments.bag, arguments.output, max_frames=arguments.max_frames
        )
    elif arguments.command == "replay":
        value = replay_svo(
            arguments.dataset,
            arguments.svo_build,
            arguments.output,
            svo_params=arguments.svo_params,
            svo_calibration=arguments.svo_calibration,
            param_overrides=arguments.param,
            frontend=arguments.frontend,
            start_frame=arguments.start_frame,
            max_frames=arguments.max_frames,
            pace_hz=arguments.pace_hz,
            tag_load=arguments.tag_load,
            seed=arguments.seed,
        )
    elif arguments.command == "convert":
        value = tracks_to_flow(
            arguments.bag,
            arguments.replay,
            arguments.output,
            ransac_threshold_px=arguments.ransac_threshold_px,
        )
    else:
        value = decision(
            arguments.replay_summary,
            arguments.flow_summary,
            arguments.svo_aprilgrid,
            arguments.klt_analysis,
            arguments.output,
        )
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
