"""Offline CM2 angular-flow replay with gyro and rolling-shutter ablations."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import (
    image_to_gray,
    iter_messages,
    median,
    percentile,
    stamp_ns,
    write_csv,
    write_json,
)


MOTION_TOPICS = [
    "/imu",
    "/fmu/out/sensor_combined",
    "/fmu/out/distance_sensor",
    "/fmu/out/vehicle_local_position_v1",
    "/fmu/out/trajectory_setpoint",
]


@dataclass
class Motion:
    imu_t: np.ndarray
    gyro: np.ndarray
    gyro_integral: np.ndarray
    range_t: np.ndarray
    ranges: np.ndarray
    local_t: np.ndarray
    local: np.ndarray
    setpoint_t: np.ndarray
    setpoint: np.ndarray
    clock: dict


def _load_calibration(path: Path):
    data = yaml.safe_load(path.read_text())
    camera = data["cam0"]
    intrinsics = camera["intrinsics"]
    distortion = camera["distortion_coeffs"]
    line_delay = float(camera["line_delay"])
    resolution = camera["resolution"]
    return (
        np.array(
            [
                [intrinsics[0], 0.0, intrinsics[2]],
                [0.0, intrinsics[1], intrinsics[3]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(distortion, dtype=np.float64),
        line_delay,
        tuple(int(value) for value in resolution),
    )


def _collect_motion(bag: Path) -> Motion:
    imu_t_ns = []
    gyro = []
    sensor_hrt = []
    ranges = []
    local = []
    setpoints = []
    for topic, _, message in iter_messages(bag, MOTION_TOPICS):
        if topic == "/imu":
            imu_t_ns.append(stamp_ns(message.header))
            angular = message.angular_velocity
            gyro.append((angular.x, angular.y, angular.z))
        elif topic == "/fmu/out/sensor_combined":
            sensor_hrt.append(int(message.timestamp) * 1000)
        elif topic == "/fmu/out/distance_sensor":
            ranges.append(
                (
                    int(message.timestamp) * 1000,
                    float(message.current_distance),
                    int(message.signal_quality),
                    float(message.min_distance),
                    float(message.max_distance),
                )
            )
        elif topic == "/fmu/out/vehicle_local_position_v1":
            local.append(
                (
                    int(message.timestamp_sample or message.timestamp) * 1000,
                    float(message.x),
                    float(message.y),
                    float(message.vx),
                    float(message.vy),
                    float(message.heading),
                )
            )
        elif topic == "/fmu/out/trajectory_setpoint":
            setpoints.append(
                (
                    int(message.timestamp) * 1000,
                    *[float(value) for value in message.position[:2]],
                    *[float(value) for value in message.velocity[:2]],
                )
            )
    imu_t_ns = np.asarray(imu_t_ns, dtype=np.int64)
    order = np.argsort(imu_t_ns)
    imu_t_ns = imu_t_ns[order]
    imu_t = imu_t_ns.astype(np.float64) / 1e9
    gyro = np.asarray(gyro, dtype=np.float64)[order]
    sensor_set = set(sensor_hrt)
    exact_matches = sum(
        int(timestamp) in sensor_set for timestamp in imu_t_ns
    )

    def map_rows(rows):
        if not rows:
            return np.empty(0), np.empty((0, 0))
        values = np.asarray(rows, dtype=np.float64)
        # PX4 timestamps have already been translated by DDS into the same ROS
        # realtime epoch used by camera headers and /imu.
        timestamps = values[:, 0] / 1e9
        order = np.argsort(timestamps)
        return timestamps[order], values[order, 1:]

    range_t, range_values = map_rows(ranges)
    local_t, local_values = map_rows(local)
    setpoint_t, setpoint_values = map_rows(setpoints)
    dt = np.diff(imu_t)
    increments = 0.5 * (gyro[:-1] + gyro[1:]) * dt[:, None]
    integral = np.vstack(
        [np.zeros(3), np.cumsum(increments, axis=0)]
    )
    return Motion(
        imu_t=imu_t,
        gyro=gyro,
        gyro_integral=integral,
        range_t=range_t,
        ranges=range_values,
        local_t=local_t,
        local=local_values,
        setpoint_t=setpoint_t,
        setpoint=setpoint_values,
        clock={
            "px4_timestamp_domain": "DDS-adjusted ROS realtime",
            "sensor_combined_samples": len(sensor_hrt),
            "derived_imu_samples": len(imu_t),
            "exact_timestamp_matches": exact_matches,
        },
    )


def _interp_integral(motion: Motion, query: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    tolerance = 100e-6
    if (
        query.min() < motion.imu_t[0] - tolerance
        or query.max() > motion.imu_t[-1] + tolerance
    ):
        raise ValueError(
            "camera interval is outside IMU coverage: "
            f"query=[{query.min():.9f},{query.max():.9f}], "
            f"imu=[{motion.imu_t[0]:.9f},{motion.imu_t[-1]:.9f}]"
        )
    # Guard only floating-point loss from representing epoch seconds. The
    # tolerance is far below one IMU sample and never fills a missing sample.
    query = np.clip(query, motion.imu_t[0], motion.imu_t[-1])
    return np.column_stack(
        [
            np.interp(query, motion.imu_t, motion.gyro_integral[:, axis])
            for axis in range(3)
        ]
    )


def _nearest(timestamps, values, query, max_age=0.1):
    if len(timestamps) == 0:
        return None
    index = int(np.searchsorted(timestamps, query))
    candidates = [
        chosen
        for chosen in (index - 1, index)
        if 0 <= chosen < len(timestamps)
    ]
    chosen = min(candidates, key=lambda i: abs(timestamps[i] - query))
    if abs(timestamps[chosen] - query) > max_age:
        return None
    return values[chosen]


def _rotation_hypothesis(yaw_quadrants: int) -> np.ndarray:
    # Base: image-right -> body-right, image-down -> body-back, optical axis
    # -> body-down. Rotating around body Z enumerates all physical cable/mount
    # yaw possibilities without introducing a reflection.
    base = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    angle = yaw_quadrants * math.pi / 2.0
    yaw = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return yaw @ base


def _rotate_vectors(vectors: np.ndarray, rotation_vectors: np.ndarray):
    angles = np.linalg.norm(rotation_vectors, axis=1)
    result = vectors.copy()
    moving = angles > 1e-12
    if not np.any(moving):
        return result
    axes = rotation_vectors[moving] / angles[moving, None]
    selected = vectors[moving]
    cosine = np.cos(angles[moving])[:, None]
    sine = np.sin(angles[moving])[:, None]
    result[moving] = (
        selected * cosine
        + np.cross(axes, selected) * sine
        + axes * np.sum(axes * selected, axis=1)[:, None] * (1.0 - cosine)
    )
    return result


def _features(image, tiles_x=8, tiles_y=6, per_tile=8):
    height, width = image.shape
    selected = []
    occupied = 0
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            x0 = round(tile_x * width / tiles_x)
            x1 = round((tile_x + 1) * width / tiles_x)
            y0 = round(tile_y * height / tiles_y)
            y1 = round((tile_y + 1) * height / tiles_y)
            points = cv2.goodFeaturesToTrack(
                image[y0:y1, x0:x1],
                maxCorners=per_tile,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=7,
            )
            if points is not None:
                points[:, 0, 0] += x0
                points[:, 0, 1] += y0
                selected.append(points)
                occupied += 1
    if not selected:
        return np.empty((0, 1, 2), dtype=np.float32), 0
    return np.concatenate(selected), occupied


def _quantized_native_rows(rows, height, native_height, bands):
    rows = np.asarray(rows, dtype=np.float64)
    if bands <= 0:
        return np.full(
            rows.shape, (native_height - 1) / 2.0, dtype=np.float64
        )
    band = np.clip(
        (rows * bands / height).astype(int), 0, bands - 1
    )
    centers = (band + 0.5) * height / bands
    return centers * native_height / height


def _evaluate_hypothesis(
    p0_norm,
    p1_norm,
    rows0,
    rows1,
    t0,
    t1,
    motion,
    r_b_c,
    native_height,
    working_height,
    line_delay,
    bands,
    readout_sign,
):
    native0 = _quantized_native_rows(
        rows0, working_height, native_height, bands
    )
    native1 = _quantized_native_rows(
        rows1, working_height, native_height, bands
    )
    center = (native_height - 1) / 2.0
    time0 = t0 + readout_sign * (native0 - center) * line_delay
    time1 = t1 + readout_sign * (native1 - center) * line_delay
    delta_body = _interp_integral(motion, time1) - _interp_integral(
        motion, time0
    )
    rays_c0 = np.column_stack([p0_norm, np.ones(len(p0_norm))])
    rays_b0 = rays_c0 @ r_b_c.T
    rays_b1 = _rotate_vectors(rays_b0, -delta_body)
    rays_c1 = rays_b1 @ r_b_c
    predicted = rays_c1[:, :2] / rays_c1[:, 2:3]
    residual = p1_norm - predicted
    angular_flow_camera = np.median(residual, axis=0)
    # Feature motion is opposite camera translation. Convert the camera-center
    # displacement direction into PX4's body-aligned angular-flow convention.
    camera_translation = np.array(
        [-angular_flow_camera[0], -angular_flow_camera[1], 0.0]
    )
    body_translation = r_b_c @ camera_translation
    pixel_flow = np.array(
        [-body_translation[1], body_translation[0]]
    )
    residual_centered = residual - angular_flow_camera
    delta_angle = np.median(delta_body, axis=0)
    # The historical `pixel_flow_*` fields below are gyro-compensated
    # translation and remain unchanged for metric replay. SensorOpticalFlow
    # must instead contain raw image motion because PX4 subtracts the supplied
    # delta angle itself.
    raw_pixel_flow = pixel_flow + delta_angle[:2]
    return {
        "pixel_flow_x_rad": float(pixel_flow[0]),
        "pixel_flow_y_rad": float(pixel_flow[1]),
        "raw_pixel_flow_x_rad": float(raw_pixel_flow[0]),
        "raw_pixel_flow_y_rad": float(raw_pixel_flow[1]),
        "flow_camera_x_rad": float(angular_flow_camera[0]),
        "flow_camera_y_rad": float(angular_flow_camera[1]),
        "compensated_residual_median_rad": float(
            np.median(np.linalg.norm(residual_centered, axis=1))
        ),
        "compensated_residual_p95_rad": float(
            np.percentile(np.linalg.norm(residual_centered, axis=1), 95)
        ),
        "delta_angle_x_rad": float(delta_angle[0]),
        "delta_angle_y_rad": float(delta_angle[1]),
        "delta_angle_z_rad": float(delta_angle[2]),
    }


def _score_quality(coverage, inlier_fraction, fb_median, residual_p95):
    factors = (
        min(1.0, coverage / 0.65),
        min(1.0, inlier_fraction / 0.75),
        max(0.0, 1.0 - fb_median / 1.0),
        max(0.0, 1.0 - residual_p95 / 0.015),
    )
    return int(round(255.0 * math.prod(factors) ** 0.25))


def run_flow(
    bag: Path,
    output: Path,
    calibration: Path,
    *,
    max_pairs: int | None = None,
    start_s: float | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    samples = output / "annotated"
    samples.mkdir(exist_ok=True)
    motion = _collect_motion(bag)
    native_k, distortion, line_delay, native_resolution = _load_calibration(
        calibration
    )
    native_width, native_height = native_resolution
    width, height = native_width // 2, native_height // 2
    working_k = native_k.copy()
    working_k[:2] *= 0.5
    map_x, map_y = cv2.initUndistortRectifyMap(
        working_k,
        distortion,
        None,
        working_k,
        (width, height),
        cv2.CV_32FC1,
    )

    variants = [(0, 1)]
    variants.extend(
        (bands, sign) for bands in (8, 16) for sign in (-1, 1)
    )
    hypothesis_rows = []
    pair_rows = []
    previous = None
    previous_time = None
    frame_index = -1
    pair_index = 0
    for topic, _, message in iter_messages(
        bag, ["/camera_down/image_raw"]
    ):
        if topic != "/camera_down/image_raw":
            continue
        frame_index += 1
        header_time = stamp_ns(message.header) / 1e9
        # libcamera SensorTimestamp is a frame-start reference. Convert it to
        # the center-row reference used by the PX4 flow contract.
        center_time = header_time + 0.5 * (native_height - 1) * line_delay
        if start_s is not None and center_time < start_s:
            previous = None
            previous_time = None
            continue
        gray_native = image_to_gray(message)
        gray = cv2.resize(
            gray_native, (width, height), interpolation=cv2.INTER_AREA
        )
        gray = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)
        if previous is None:
            previous = gray
            previous_time = center_time
            continue
        half_readout = 0.5 * (native_height - 1) * line_delay
        if (
            previous_time - half_readout < motion.imu_t[0]
            or center_time + half_readout > motion.imu_t[-1]
        ):
            previous = gray
            previous_time = center_time
            continue
        points0, occupied = _features(previous)
        detected = len(points0)
        if detected < 8:
            previous = gray
            previous_time = center_time
            continue
        points1, status1, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            gray,
            points0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        points0_back, status2, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            previous,
            points1,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        fb = np.linalg.norm(
            points0[:, 0] - points0_back[:, 0], axis=1
        )
        good = (
            (status1[:, 0] == 1)
            & (status2[:, 0] == 1)
            & (fb < 1.0)
            & (points1[:, 0, 0] >= 0)
            & (points1[:, 0, 0] < width)
            & (points1[:, 0, 1] >= 0)
            & (points1[:, 0, 1] < height)
        )
        p0 = points0[good, 0]
        p1 = points1[good, 0]
        fb_good = fb[good]
        if len(p0) < 8:
            previous = gray
            previous_time = center_time
            continue
        _, homography_mask = cv2.findHomography(
            p0, p1, cv2.RANSAC, 2.0
        )
        if homography_mask is None:
            previous = gray
            previous_time = center_time
            continue
        homography_mask = homography_mask[:, 0].astype(bool)
        p0 = p0[homography_mask]
        p1 = p1[homography_mask]
        fb_good = fb_good[homography_mask]
        if len(p0) < 8:
            previous = gray
            previous_time = center_time
            continue
        p0_norm = cv2.undistortPoints(
            p0.reshape(-1, 1, 2), working_k, None
        )[:, 0]
        p1_norm = cv2.undistortPoints(
            p1.reshape(-1, 1, 2), working_k, None
        )[:, 0]
        range_row = _nearest(
            motion.range_t, motion.ranges, center_time, max_age=0.05
        )
        distance = None
        range_quality = None
        if range_row is not None:
            candidate, signal, minimum, maximum = range_row
            if minimum <= candidate <= maximum and signal != 0:
                distance = float(candidate)
                range_quality = int(signal)
        local = _nearest(
            motion.local_t, motion.local, center_time, max_age=0.1
        )
        setpoint = _nearest(
            motion.setpoint_t, motion.setpoint, center_time, max_age=0.1
        )
        coverage = occupied / 48.0
        inlier_fraction = len(p0) / max(1, int(good.sum()))
        pair = {
            "pair": pair_index,
            "frame": frame_index,
            "timestamp_sample_s": center_time,
            "integration_time_s": center_time - previous_time,
            "detected_features": detected,
            "fb_good_features": int(good.sum()),
            "homography_inliers": len(p0),
            "tile_coverage": coverage,
            "homography_inlier_fraction": inlier_fraction,
            "fb_error_median_px": float(np.median(fb_good)),
            "distance_m": distance,
            "distance_quality": range_quality,
            "local_x_m": float(local[0]) if local is not None else None,
            "local_y_m": float(local[1]) if local is not None else None,
            "local_vx_m_s": float(local[2]) if local is not None else None,
            "local_vy_m_s": float(local[3]) if local is not None else None,
            "heading_rad": float(local[4]) if local is not None else None,
            "setpoint_x_m": (
                float(setpoint[0]) if setpoint is not None else None
            ),
            "setpoint_y_m": (
                float(setpoint[1]) if setpoint is not None else None
            ),
        }
        pair_rows.append(pair)
        for yaw_quadrants in range(4):
            r_b_c = _rotation_hypothesis(yaw_quadrants)
            for bands, sign in variants:
                evaluated = _evaluate_hypothesis(
                    p0_norm,
                    p1_norm,
                    p0[:, 1],
                    p1[:, 1],
                    previous_time,
                    center_time,
                    motion,
                    r_b_c,
                    native_height,
                    height,
                    line_delay,
                    bands,
                    sign,
                )
                quality = _score_quality(
                    coverage,
                    inlier_fraction,
                    float(np.median(fb_good)),
                    evaluated["compensated_residual_p95_rad"],
                )
                hypothesis_rows.append(
                    {
                        "pair": pair_index,
                        "timestamp_sample_s": center_time,
                        "yaw_quadrants": yaw_quadrants,
                        "rs_bands": bands,
                        "readout_sign": sign,
                        "quality": quality,
                        **evaluated,
                    }
                )
        if pair_index % 250 == 0:
            annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for begin, end in zip(p0, p1, strict=True):
                cv2.arrowedLine(
                    annotated,
                    tuple(np.round(begin).astype(int)),
                    tuple(np.round(end).astype(int)),
                    (0, 220, 255),
                    1,
                    tipLength=0.25,
                )
            cv2.imwrite(
                str(samples / f"pair_{pair_index:05d}.jpg"), annotated
            )
        pair_index += 1
        previous = gray
        previous_time = center_time
        if max_pairs is not None and pair_index >= max_pairs:
            break

    if not hypothesis_rows:
        raise RuntimeError("no valid CM2 flow pairs")
    write_csv(output / "flow_pairs.csv", pair_rows)
    write_csv(output / "flow_hypotheses.csv", hypothesis_rows)

    grouped = {}
    for row in hypothesis_rows:
        key = (
            row["yaw_quadrants"],
            row["rs_bands"],
            row["readout_sign"],
        )
        grouped.setdefault(key, []).append(row)
    scores = []
    for key, rows in grouped.items():
        scores.append(
            {
                "yaw_quadrants": key[0],
                "rs_bands": key[1],
                "readout_sign": key[2],
                "pairs": len(rows),
                "quality_median": median(row["quality"] for row in rows),
                "quality_p05": percentile(
                    [row["quality"] for row in rows], 5
                ),
                "residual_median_rad": median(
                    row["compensated_residual_median_rad"] for row in rows
                ),
                "residual_p95_rad": percentile(
                    [row["compensated_residual_p95_rad"] for row in rows], 95
                ),
            }
        )
    scores.sort(
        key=lambda row: (
            row["residual_p95_rad"],
            row["residual_median_rad"],
        )
    )
    write_csv(output / "flow_hypothesis_summary.csv", scores)
    best = scores[0]
    selected_key = (
        best["yaw_quadrants"],
        best["rs_bands"],
        best["readout_sign"],
    )
    selected_rows = grouped[selected_key]
    selected = []
    path_north = 0.0
    path_east = 0.0
    contextual_velocity = []
    pairs_by_id = {row["pair"]: row for row in pair_rows}
    for row in selected_rows:
        pair = pairs_by_id[row["pair"]]
        dt = pair["integration_time_s"]
        distance = pair["distance_m"]
        vx = vy = None
        north = east = None
        if distance is not None and dt > 0:
            vx = row["pixel_flow_y_rad"] * distance / dt
            vy = -row["pixel_flow_x_rad"] * distance / dt
            heading = pair["heading_rad"]
            if heading is not None:
                north = math.cos(heading) * vx - math.sin(heading) * vy
                east = math.sin(heading) * vx + math.cos(heading) * vy
                path_north += north * dt
                path_east += east * dt
                if (
                    pair["local_vx_m_s"] is not None
                    and pair["local_vy_m_s"] is not None
                ):
                    contextual_velocity.append(
                        (
                            north,
                            east,
                            pair["local_vx_m_s"],
                            pair["local_vy_m_s"],
                        )
                    )
        selected.append(
            {
                **row,
                "distance_m": distance,
                "velocity_body_x_m_s": vx,
                "velocity_body_y_m_s": vy,
                "velocity_north_m_s": north,
                "velocity_east_m_s": east,
                "path_north_m": path_north,
                "path_east_m": path_east,
                "local_x_m": pair["local_x_m"],
                "local_y_m": pair["local_y_m"],
                "local_vx_m_s": pair["local_vx_m_s"],
                "local_vy_m_s": pair["local_vy_m_s"],
            }
        )
    write_csv(output / "flow_selected.csv", selected)
    availability = len(pair_rows) / max(1, frame_index)
    yaw_best = []
    for yaw_quadrants in range(4):
        yaw_best.append(
            min(
                (
                    score
                    for score in scores
                    if score["yaw_quadrants"] == yaw_quadrants
                ),
                key=lambda score: score["residual_p95_rad"],
            )
        )
    yaw_best.sort(key=lambda score: score["residual_p95_rad"])
    second_yaw = yaw_best[1]
    same_yaw = [
        score
        for score in scores
        if score["yaw_quadrants"] == best["yaw_quadrants"]
        and score is not best
    ]
    second_timing = min(
        same_yaw, key=lambda score: score["residual_p95_rad"]
    )
    velocity_comparison = None
    if contextual_velocity:
        values = np.asarray(contextual_velocity, dtype=np.float64)
        flow_velocity = values[:, :2]
        local_velocity = values[:, 2:]
        local_speed = np.linalg.norm(local_velocity, axis=1)
        moving = local_speed > 0.3
        if np.any(moving):
            flow_moving = flow_velocity[moving]
            local_moving = local_velocity[moving]
            flow_speed = np.linalg.norm(flow_moving, axis=1)
            local_speed = np.linalg.norm(local_moving, axis=1)
            dot = np.sum(flow_moving * local_moving, axis=1)
            velocity_comparison = {
                "moving_samples": int(moving.sum()),
                "direction_agreement_fraction": float(
                    np.mean(dot > 0)
                ),
                "median_speed_ratio_to_px4": float(
                    np.median(flow_speed / local_speed)
                ),
                "vector_rmse_to_px4_m_s": float(
                    np.sqrt(np.mean((flow_moving - local_moving) ** 2))
                ),
                "north_correlation": float(
                    np.corrcoef(flow_moving[:, 0], local_moving[:, 0])[0, 1]
                ),
                "east_correlation": float(
                    np.corrcoef(flow_moving[:, 1], local_moving[:, 1])[0, 1]
                ),
                "warning": "PX4 fused GNSS in this flight and is contextual, not truth.",
            }
    local_positions = [
        (row["local_x_m"], row["local_y_m"])
        for row in pair_rows
        if row["local_x_m"] is not None and row["local_y_m"] is not None
    ]
    local_endpoint = None
    if local_positions:
        local_endpoint = {
            "north_m": local_positions[-1][0] - local_positions[0][0],
            "east_m": local_positions[-1][1] - local_positions[0][1],
        }
    tag_context = None
    tag_csv = bag / "analysis" / "flight1_cm2_tags.csv"
    if tag_csv.exists():
        with tag_csv.open(newline="", encoding="utf-8") as stream:
            tag_times = sorted(
                {
                    int(row["timestamp_ns"]) / 1e9
                    for row in csv.DictReader(stream)
                }
            )
        selected_times = np.asarray(
            [row["timestamp_sample_s"] for row in selected],
            dtype=np.float64,
        )
        matched = []
        for timestamp in tag_times:
            index = int(np.searchsorted(selected_times, timestamp))
            candidates = [
                chosen
                for chosen in (index - 1, index)
                if 0 <= chosen < len(selected)
            ]
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda i: abs(selected_times[i] - timestamp),
                )
                if abs(selected_times[chosen] - timestamp) <= 0.025:
                    matched.append(selected[chosen])
        if matched:
            tag_context = {
                "unique_tag_frames": len(tag_times),
                "matched_flow_pairs": len(matched),
                "quality_median": median(
                    row["quality"] for row in matched
                ),
                "residual_p95_rad": percentile(
                    [
                        row["compensated_residual_p95_rad"]
                        for row in matched
                    ],
                    95,
                ),
                "warning": "Tags mark sighting epochs only; their positions are not surveyed truth.",
            }
    result = {
        "input_frames_seen": frame_index + 1,
        "accepted_pairs": len(pair_rows),
        "accepted_pair_availability": availability,
        "recorded_rate_hz": 1.0
        / median(row["integration_time_s"] for row in pair_rows),
        "calibration": str(calibration.resolve()),
        "line_delay_s_per_row": line_delay,
        "clock_mapping": motion.clock,
        "selected_hypothesis": best,
        "second_timing_hypothesis_same_yaw": second_timing,
        "timing_selection_margin_p95_fraction": (
            second_timing["residual_p95_rad"]
            / best["residual_p95_rad"]
            - 1.0
        ),
        "rolling_shutter_variant_is_unique": (
            second_timing["residual_p95_rad"]
            > 1.05 * best["residual_p95_rad"]
        ),
        "second_axis_hypothesis": second_yaw,
        "axis_selection_margin_p95_fraction": (
            second_yaw["residual_p95_rad"] / best["residual_p95_rad"] - 1.0
            if best["residual_p95_rad"] > 0
            else None
        ),
        "axis_mapping_is_unique": (
            second_yaw["residual_p95_rad"]
            > 1.05 * best["residual_p95_rad"]
        ),
        "diagnostic_endpoint_north_m": path_north,
        "diagnostic_endpoint_east_m": path_east,
        "px4_contextual_endpoint": local_endpoint,
        "px4_contextual_velocity_comparison": velocity_comparison,
        "tag_sighting_context": tag_context,
        "limitations": [
            "The recorded CM2 rate is about 30 Hz, not the intended 41 Hz.",
            "CM2-to-Pixhawk extrinsics are discrete hypotheses, not a joint calibration.",
            "dToF, mission geometry, tags, PX4, and closure are proxies rather than independent truth.",
            "Per-frame exposure and gain were not serialized in this bag.",
        ],
    }
    write_json(output / "flow_decision.json", result)
    return result


def synthetic_self_check() -> dict:
    distance = 2.0
    dt = 0.04
    body_displacements = {
        "forward": np.array([0.08, 0.0]),
        "right": np.array([0.0, 0.08]),
    }
    observed = {
        "forward": np.array([0.0, body_displacements["forward"][0] / distance]),
        "right": np.array([-body_displacements["right"][1] / distance, 0.0]),
    }
    checks = {
        "forward_pixel_flow_y_positive": bool(observed["forward"][1] > 0),
        "right_pixel_flow_x_negative": bool(observed["right"][0] < 0),
        "forward_velocity_recovered": bool(abs(
            observed["forward"][1] * distance / dt - 2.0
        )
        < 1e-9),
        "right_velocity_recovered": bool(abs(
            -observed["right"][0] * distance / dt - 2.0
        )
        < 1e-9),
    }
    return {"passed": all(checks.values()), "checks": checks}
