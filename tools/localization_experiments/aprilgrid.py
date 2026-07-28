"""Validate integrated CM2+dToF flow against a metric AprilGrid pose track."""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import image_to_gray, iter_messages, median, percentile, stamp_ns


CAMERA_TOPIC = "/camera_down/image_raw"
ROWS = 1232
LINE_DELAY_S = 9.689e-6
TAG_ROWS = 6
TAG_COLUMNS = 6
TAG_SIZE_M = 0.025
TAG_SPACING = 0.3
TAG_PITCH_M = TAG_SIZE_M * (1.0 + TAG_SPACING)

CAMERA_MATRIX = np.array(
    [
        [1298.69385194, 0.0, 827.70242273],
        [0.0, 1299.56328818, 617.60425847],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DISTORTION = np.array(
    [0.15046410, -0.23368367, 0.00000042, -0.00209991],
    dtype=np.float64,
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _detector():
    parameters = cv2.aruco.DetectorParameters()
    # OpenCV's AprilTag-specific refinement rejects the contiguous Kalibr
    # checkerboard layout. Detect the tags normally, then refine their corners.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.aprilTagQuadDecimate = 2.0
    parameters.minMarkerPerimeterRate = 0.01
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.errorCorrectionRate = 0.8
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def _tag_object_center(tag_id: int) -> np.ndarray:
    """Return a tag center in the printed target's metric coordinate system."""
    row, column = divmod(tag_id, TAG_COLUMNS)
    return np.array(
        [
            column * TAG_PITCH_M + 0.5 * TAG_SIZE_M,
            row * TAG_PITCH_M + 0.5 * TAG_SIZE_M,
            0.0,
        ],
        dtype=np.float32,
    )


def _pose(gray: np.ndarray, detector) -> dict | None:
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    object_points = []
    image_points = []
    used_ids = []
    for marker_corners, raw_id in zip(corners, ids.reshape(-1)):
        tag_id = int(raw_id)
        if not 0 <= tag_id < TAG_ROWS * TAG_COLUMNS:
            continue
        object_points.append(_tag_object_center(tag_id))
        image_points.append(
            np.asarray(marker_corners, dtype=np.float32)
            .reshape(4, 2)
            .mean(axis=0)
        )
        used_ids.append(tag_id)
    if len(used_ids) < 4:
        return None
    object_points_array = np.asarray(object_points, dtype=np.float32)
    image_points_array = np.asarray(image_points, dtype=np.float32)
    _, homography_inliers = cv2.findHomography(
        object_points_array[:, :2],
        image_points_array,
        cv2.RANSAC,
        2.0,
    )
    if homography_inliers is None:
        return None
    inlier_indices = np.flatnonzero(homography_inliers.reshape(-1))
    if len(inlier_indices) < 4:
        return None
    inlier_object = object_points_array[inlier_indices]
    inlier_image = image_points_array[inlier_indices]
    ok, rvec, tvec = cv2.solvePnP(
        inlier_object,
        inlier_image,
        CAMERA_MATRIX,
        DISTORTION,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        inlier_object,
        inlier_image,
        CAMERA_MATRIX,
        DISTORTION,
        rvec,
        tvec,
    )
    projected, _ = cv2.projectPoints(
        inlier_object,
        rvec,
        tvec,
        CAMERA_MATRIX,
        DISTORTION,
    )
    error = projected.reshape(-1, 2) - inlier_image
    rotation_grid_to_camera, _ = cv2.Rodrigues(rvec)
    camera_position_grid = (
        -rotation_grid_to_camera.T @ np.asarray(tvec).reshape(3)
    )
    return {
        "tag_ids": used_ids,
        "inlier_tags": len(inlier_indices),
        "reprojection_rmse_px": float(
            np.sqrt(np.mean(np.sum(error * error, axis=1)))
        ),
        "position": camera_position_grid,
        "rvec": np.asarray(rvec).reshape(3),
        "tvec": np.asarray(tvec).reshape(3),
    }


def extract_poses(bag: Path, output: Path) -> list[dict]:
    detector = _detector()
    poses = []
    frame_count = 0
    for topic, _, message in iter_messages(bag, [CAMERA_TOPIC]):
        if topic != CAMERA_TOPIC:
            continue
        frame_count += 1
        result = _pose(image_to_gray(message), detector)
        if result is not None:
            # Picamera2 SensorTimestamp is the first-row readout time. Match
            # the center-row convention used by the flow replay.
            timestamp_s = (
                stamp_ns(message.header) / 1e9
                + 0.5 * (ROWS - 1) * LINE_DELAY_S
            )
            position = result["position"]
            rvec = result["rvec"]
            tvec = result["tvec"]
            poses.append(
                {
                    "frame": frame_count - 1,
                    "timestamp_s": timestamp_s,
                    "tag_count": len(result["tag_ids"]),
                    "tag_ids": " ".join(
                        str(value) for value in sorted(result["tag_ids"])
                    ),
                    "inlier_tags": result["inlier_tags"],
                    "reprojection_rmse_px": result[
                        "reprojection_rmse_px"
                    ],
                    "grid_x_m": float(position[0]),
                    "grid_y_m": float(position[1]),
                    "grid_z_m": float(position[2]),
                    "rvec_x": float(rvec[0]),
                    "rvec_y": float(rvec[1]),
                    "rvec_z": float(rvec[2]),
                    "tvec_x_m": float(tvec[0]),
                    "tvec_y_m": float(tvec[1]),
                    "tvec_z_m": float(tvec[2]),
                }
            )
        if frame_count % 100 == 0:
            print(
                f"AprilGrid: {frame_count} frames, {len(poses)} valid poses",
                flush=True,
            )
    print(
        f"AprilGrid: {frame_count} frames, {len(poses)} valid poses",
        flush=True,
    )
    _write_csv(output / "aprilgrid_poses.csv", poses)
    return poses


class _FlowTrack:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.t = np.asarray(
            [float(row["timestamp_sample_s"]) for row in rows],
            dtype=np.float64,
        )
        self.xy = np.asarray(
            [
                [float(row["path_north_m"]), float(row["path_east_m"])]
                for row in rows
            ],
            dtype=np.float64,
        )

    def at(self, timestamp_s: float) -> np.ndarray | None:
        if timestamp_s < self.t[0] or timestamp_s > self.t[-1]:
            return None
        return np.array(
            [
                np.interp(timestamp_s, self.t, self.xy[:, 0]),
                np.interp(timestamp_s, self.t, self.xy[:, 1]),
            ]
        )


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit target = source @ rotation + translation, with no scale."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_center - source_center @ rotation
    return rotation, translation


def _orthogonal_fit(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Choose the coordinate convention, but never fit metric scale."""
    proper_rotation, proper_translation = _rigid_fit(source, target)
    proper_residual = (
        source @ proper_rotation + proper_translation - target
    )
    reflected = source.copy()
    reflected[:, 1] *= -1.0
    reflected_rotation, reflected_translation = _rigid_fit(reflected, target)
    reflected_residual = (
        reflected @ reflected_rotation + reflected_translation - target
    )
    if np.mean(reflected_residual**2) < np.mean(proper_residual**2):
        reflection = np.diag([1.0, -1.0])
        return (
            reflection @ reflected_rotation,
            reflected_translation,
            True,
        )
    return proper_rotation, proper_translation, False


def _nearest_index(timestamps: np.ndarray, target: float) -> int | None:
    index = bisect_left(timestamps, target)
    candidates = [
        candidate
        for candidate in (index - 1, index)
        if 0 <= candidate < len(timestamps)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda value: abs(timestamps[value] - target))


def _window_displacements(
    timestamps: np.ndarray,
    grid_xy: np.ndarray,
    flow_xy: np.ndarray,
    rotation: np.ndarray,
    seconds: float,
) -> list[dict]:
    rows = []
    for first in range(len(timestamps)):
        second = _nearest_index(timestamps, timestamps[first] + seconds)
        if second is None:
            continue
        actual_dt = timestamps[second] - timestamps[first]
        if not 0.8 * seconds <= actual_dt <= 1.2 * seconds:
            continue
        grid_delta = (grid_xy[second] - grid_xy[first]) @ rotation
        flow_delta = flow_xy[second] - flow_xy[first]
        grid_distance = float(np.linalg.norm(grid_delta))
        if grid_distance < 0.03:
            continue
        flow_distance = float(np.linalg.norm(flow_delta))
        rows.append(
            {
                "timestamp_s": float(timestamps[first]),
                "dt_s": float(actual_dt),
                "grid_dx_m": float(grid_delta[0]),
                "grid_dy_m": float(grid_delta[1]),
                "flow_dx_m": float(flow_delta[0]),
                "flow_dy_m": float(flow_delta[1]),
                "grid_distance_m": grid_distance,
                "flow_distance_m": flow_distance,
                "flow_to_grid_scale": flow_distance / grid_distance,
                "vector_error_m": float(
                    np.linalg.norm(flow_delta - grid_delta)
                ),
            }
        )
    return rows


def _window_center(
    timestamps: np.ndarray, values: np.ndarray, start: float, stop: float
) -> np.ndarray:
    selected = (timestamps >= start) & (timestamps <= stop)
    if not np.any(selected):
        raise ValueError(f"no samples in [{start}, {stop}]")
    return np.median(values[selected], axis=0)


def _rotation_from_pose(row: dict) -> np.ndarray:
    vector = np.array(
        [float(row["rvec_x"]), float(row["rvec_y"]), float(row["rvec_z"])]
    )
    return cv2.Rodrigues(vector)[0]


def _position_from_pose(row: dict) -> np.ndarray:
    return np.array(
        [
            float(row["grid_x_m"]),
            float(row["grid_y_m"]),
            float(row["grid_z_m"]),
        ]
    )


def angular_validation(
    poses: list[dict], flow_csv: Path, output: Path
) -> dict:
    """Compare the measured angular flow with motion from adjacent PnP poses."""
    pose_by_frame = {
        int(row["frame"]): row
        for row in poses
        if float(row["reprojection_rmse_px"]) <= 1.5
    }
    pair_by_id = {
        int(row["pair"]): row
        for row in _read_csv(flow_csv.parent / "flow_pairs.csv")
    }
    samples = []
    for flow_row in _read_csv(flow_csv):
        pair = pair_by_id[int(flow_row["pair"])]
        frame = int(pair["frame"])
        before = pose_by_frame.get(frame - 1)
        after = pose_by_frame.get(frame)
        if before is None or after is None:
            continue
        dt = float(after["timestamp_s"]) - float(before["timestamp_s"])
        if not 0.02 <= dt <= 0.05:
            continue
        common_ids = sorted(
            set(int(value) for value in before["tag_ids"].split())
            & set(int(value) for value in after["tag_ids"].split())
        )
        if len(common_ids) < 4:
            continue
        points_grid = np.asarray(
            [_tag_object_center(tag_id) for tag_id in common_ids]
        )
        position0 = _position_from_pose(before)
        position1 = _position_from_pose(after)
        rotation0 = _rotation_from_pose(before)
        rotation1 = _rotation_from_pose(after)
        camera0 = (points_grid - position0) @ rotation0.T
        camera1 = (points_grid - position1) @ rotation1.T
        visible = (camera0[:, 2] > 0.05) & (camera1[:, 2] > 0.05)
        if int(visible.sum()) < 4:
            continue
        camera0 = camera0[visible]
        camera1 = camera1[visible]
        ray0 = camera0 / camera0[:, 2:3]
        ray1 = camera1 / camera1[:, 2:3]
        rotation_1_0 = rotation1 @ rotation0.T
        rotated = ray0 @ rotation_1_0.T
        predicted = rotated[:, :2] / rotated[:, 2:3]
        truth = np.median(ray1[:, :2] - predicted, axis=0)
        measured = np.array(
            [
                float(flow_row["flow_camera_x_rad"]),
                float(flow_row["flow_camera_y_rad"]),
            ]
        )

        plane_normal_camera = rotation0[:, 2]
        plane_origin_camera = -rotation0 @ position0
        axial_distance = abs(
            float(
                np.dot(plane_normal_camera, plane_origin_camera)
                / plane_normal_camera[2]
            )
        )
        incidence_deg = math.degrees(
            math.acos(
                float(np.clip(abs(plane_normal_camera[2]), 0.0, 1.0))
            )
        )
        estimated_camera_delta = np.array(
            [
                -measured[0] * axial_distance,
                -measured[1] * axial_distance,
                0.0,
            ]
        )
        estimated_grid_delta = rotation0.T @ estimated_camera_delta
        true_grid_delta = position1 - position0
        samples.append(
            {
                "frame": frame,
                "timestamp_s": float(after["timestamp_s"]),
                "dt_s": dt,
                "common_tags": len(common_ids),
                "incidence_deg": incidence_deg,
                "axial_plane_distance_m": axial_distance,
                "measured_flow_x_rad": float(measured[0]),
                "measured_flow_y_rad": float(measured[1]),
                "pnp_flow_x_rad": float(truth[0]),
                "pnp_flow_y_rad": float(truth[1]),
                "angular_error_rad": float(np.linalg.norm(measured - truth)),
                "measured_magnitude_rad": float(np.linalg.norm(measured)),
                "pnp_magnitude_rad": float(np.linalg.norm(truth)),
                "estimated_grid_dx_m": float(estimated_grid_delta[0]),
                "estimated_grid_dy_m": float(estimated_grid_delta[1]),
                "true_grid_dx_m": float(true_grid_delta[0]),
                "true_grid_dy_m": float(true_grid_delta[1]),
                "true_grid_dz_m": float(true_grid_delta[2]),
            }
        )
    _write_csv(output / "angular_flow_aprilgrid.csv", samples)
    if len(samples) < 20:
        raise RuntimeError(f"only {len(samples)} adjacent PnP/flow samples")

    measured = np.asarray(
        [
            [row["measured_flow_x_rad"], row["measured_flow_y_rad"]]
            for row in samples
        ]
    )
    truth = np.asarray(
        [
            [row["pnp_flow_x_rad"], row["pnp_flow_y_rad"]]
            for row in samples
        ]
    )
    energetic = np.linalg.norm(truth, axis=1) >= 0.001
    measured_energetic = measured[energetic]
    truth_energetic = truth[energetic]
    scale = float(
        np.sum(measured_energetic * truth_energetic)
        / np.sum(truth_energetic * truth_energetic)
    )
    correlation = float(
        np.corrcoef(measured_energetic.reshape(-1), truth_energetic.reshape(-1))[
            0, 1
        ]
    )
    magnitude_ratio = np.linalg.norm(
        measured_energetic, axis=1
    ) / np.linalg.norm(truth_energetic, axis=1)
    cosine = np.sum(measured_energetic * truth_energetic, axis=1) / (
        np.linalg.norm(measured_energetic, axis=1)
        * np.linalg.norm(truth_energetic, axis=1)
        + 1e-12
    )
    direction_error = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    # Evaluate 0.5 s displacement windows only where the board stays visible,
    # the camera remains nearly normal to the table, and vertical motion is
    # small. This tests scale without asking 2-D flow to explain raise/lower.
    window_rows = []
    for begin in range(len(samples)):
        total_measured = np.zeros(2)
        total_truth = np.zeros(2)
        stop = begin
        while stop < len(samples):
            row = samples[stop]
            if (
                (stop > begin and row["frame"] != samples[stop - 1]["frame"] + 1)
                or row["incidence_deg"] > 15.0
                or abs(row["true_grid_dz_m"]) > 0.004
            ):
                break
            total_measured += np.array(
                [row["estimated_grid_dx_m"], row["estimated_grid_dy_m"]]
            )
            total_truth += np.array(
                [row["true_grid_dx_m"], row["true_grid_dy_m"]]
            )
            elapsed = (
                row["timestamp_s"] - samples[begin]["timestamp_s"]
                + row["dt_s"]
            )
            stop += 1
            if elapsed >= 0.5:
                truth_distance = float(np.linalg.norm(total_truth))
                if truth_distance >= 0.02:
                    measured_distance = float(np.linalg.norm(total_measured))
                    window_rows.append(
                        {
                            "start_frame": samples[begin]["frame"],
                            "end_frame": row["frame"],
                            "elapsed_s": elapsed,
                            "truth_dx_m": float(total_truth[0]),
                            "truth_dy_m": float(total_truth[1]),
                            "estimated_dx_m": float(total_measured[0]),
                            "estimated_dy_m": float(total_measured[1]),
                            "truth_distance_m": truth_distance,
                            "estimated_distance_m": measured_distance,
                            "flow_to_pnp_scale": measured_distance
                            / truth_distance,
                            "vector_error_m": float(
                                np.linalg.norm(total_measured - total_truth)
                            ),
                        }
                    )
                break
    _write_csv(output / "angular_flow_half_second_windows.csv", window_rows)

    metrics = {
        "adjacent_pose_pairs": len(samples),
        "energetic_pairs": int(energetic.sum()),
        "angular_vector_scale_through_origin": scale,
        "angular_component_correlation": correlation,
        "angular_rmse_rad": float(
            np.sqrt(np.mean((measured_energetic - truth_energetic) ** 2))
        ),
        "magnitude_ratio_median": float(np.median(magnitude_ratio)),
        "magnitude_ratio_p05": float(np.percentile(magnitude_ratio, 5)),
        "magnitude_ratio_p95": float(np.percentile(magnitude_ratio, 95)),
        "direction_error_deg_median": float(np.median(direction_error)),
        "direction_error_deg_p95": float(np.percentile(direction_error, 95)),
        "half_second_level_windows": {
            "samples": len(window_rows),
            "scale_median": median(
                row["flow_to_pnp_scale"] for row in window_rows
            ),
            "scale_p05": percentile(
                (row["flow_to_pnp_scale"] for row in window_rows), 5
            ),
            "scale_p95": percentile(
                (row["flow_to_pnp_scale"] for row in window_rows), 95
            ),
            "vector_error_median_m": median(
                row["vector_error_m"] for row in window_rows
            ),
            "vector_error_p95_m": percentile(
                (row["vector_error_m"] for row in window_rows), 95
            ),
        },
    }
    (output / "angular_flow_validation.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def loop_validation(
    poses: list[dict],
    flow_csv: Path,
    output: Path,
    anchor_windows: list[tuple[float, float]],
) -> dict:
    """Score a hand-choreographed loop from settled AprilGrid anchor windows.

    Window times are seconds relative to the first accepted AprilGrid pose.
    AprilGrid supplies camera orientation and plane distance between anchors;
    its horizontal position is used only to score the resulting flow legs.
    """
    usable = [
        row for row in poses if float(row["reprojection_rmse_px"]) <= 1.5
    ]
    timestamps = np.asarray(
        [float(row["timestamp_s"]) for row in usable]
    )
    positions = np.asarray([_position_from_pose(row) for row in usable])
    origin = float(timestamps[0])
    anchors = []
    for start_s, stop_s in anchor_windows:
        selected = (
            (timestamps >= origin + start_s)
            & (timestamps <= origin + stop_s)
        )
        if int(selected.sum()) < 3:
            raise ValueError(
                f"loop anchor {start_s}:{stop_s} has "
                f"{int(selected.sum())} poses"
            )
        anchors.append(
            {
                "window_start_s": start_s,
                "window_stop_s": stop_s,
                "timestamp_s": float(np.median(timestamps[selected])),
                "position": np.median(positions[selected], axis=0),
                "pose_samples": int(selected.sum()),
            }
        )

    increments = []
    for row in _read_csv(flow_csv):
        timestamp_s = float(row["timestamp_sample_s"])
        if not anchors[0]["timestamp_s"] < timestamp_s <= anchors[-1][
            "timestamp_s"
        ]:
            continue
        index = int(np.argmin(np.abs(timestamps - timestamp_s)))
        support_age_s = abs(float(timestamps[index]) - timestamp_s)
        if support_age_s > 0.6:
            continue
        rotation = _rotation_from_pose(usable[index])
        position = positions[index]
        normal = rotation[:, 2]
        plane_origin_camera = -rotation @ position
        axial_distance = abs(
            float(np.dot(normal, plane_origin_camera) / normal[2])
        )
        measured = np.array(
            [
                float(row["flow_camera_x_rad"]),
                float(row["flow_camera_y_rad"]),
            ]
        )
        camera_delta = np.array(
            [
                -measured[0] * axial_distance,
                -measured[1] * axial_distance,
                0.0,
            ]
        )
        grid_delta = rotation.T @ camera_delta
        increments.append(
            {
                "timestamp_s": timestamp_s,
                "support_age_s": support_age_s,
                "grid_delta": grid_delta,
            }
        )

    legs = []
    total_flow = np.zeros(2)
    truth_path_length = 0.0
    flow_path_length = 0.0
    for leg_number, (begin, end) in enumerate(
        zip(anchors, anchors[1:], strict=False), start=1
    ):
        truth_delta = (end["position"] - begin["position"])[:2]
        flow_delta = sum(
            (
                row["grid_delta"][:2]
                for row in increments
                if begin["timestamp_s"]
                < row["timestamp_s"]
                <= end["timestamp_s"]
            ),
            np.zeros(2),
        )
        truth_distance = float(np.linalg.norm(truth_delta))
        flow_distance = float(np.linalg.norm(flow_delta))
        truth_path_length += truth_distance
        flow_path_length += flow_distance
        total_flow += flow_delta
        legs.append(
            {
                "leg": leg_number,
                "truth_dx_m": float(truth_delta[0]),
                "truth_dy_m": float(truth_delta[1]),
                "truth_distance_m": truth_distance,
                "flow_dx_m": float(flow_delta[0]),
                "flow_dy_m": float(flow_delta[1]),
                "flow_distance_m": flow_distance,
                "flow_to_truth_scale": flow_distance / truth_distance,
                "vector_error_m": float(
                    np.linalg.norm(flow_delta - truth_delta)
                ),
            }
        )
    _write_csv(output / "square_loop_legs.csv", legs)

    truth_closure = (anchors[-1]["position"] - anchors[0]["position"])[:2]
    support_ages = [row["support_age_s"] for row in increments]
    metrics = {
        "range_and_orientation_source": "AprilGrid PnP",
        "horizontal_truth_source": "AprilGrid PnP anchor windows",
        "dtof_used": False,
        "anchors": [
            {
                "window_start_s": row["window_start_s"],
                "window_stop_s": row["window_stop_s"],
                "timestamp_s_relative": row["timestamp_s"] - origin,
                "position_grid_m": row["position"].tolist(),
                "pose_samples": row["pose_samples"],
            }
            for row in anchors
        ],
        "flow_pairs_integrated": len(increments),
        "orientation_range_support_age_p95_s": percentile(
            support_ages, 95
        ),
        "orientation_range_support_age_max_s": max(support_ages),
        "truth_path_length_m": truth_path_length,
        "flow_path_length_m": flow_path_length,
        "path_length_scale": flow_path_length / truth_path_length,
        "truth_endpoint_displacement_m": truth_closure.tolist(),
        "truth_endpoint_displacement_norm_m": float(
            np.linalg.norm(truth_closure)
        ),
        "flow_endpoint_displacement_m": total_flow.tolist(),
        "flow_endpoint_displacement_norm_m": float(
            np.linalg.norm(total_flow)
        ),
        "endpoint_vector_error_m": float(
            np.linalg.norm(total_flow - truth_closure)
        ),
        "legs": legs,
    }
    (output / "square_loop_validation.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def compare(poses: list[dict], flow_csv: Path, output: Path) -> dict:
    flow = _FlowTrack(_read_csv(flow_csv))
    matched = []
    for row in poses:
        flow_xy = flow.at(float(row["timestamp_s"]))
        if flow_xy is None or float(row["reprojection_rmse_px"]) > 1.5:
            continue
        matched.append((row, flow_xy))
    if len(matched) < 20:
        raise RuntimeError(f"only {len(matched)} matched AprilGrid poses")

    timestamps = np.asarray(
        [float(row["timestamp_s"]) for row, _ in matched]
    )
    grid_xyz = np.asarray(
        [
            [float(row["grid_x_m"]), float(row["grid_y_m"]), float(row["grid_z_m"])]
            for row, _ in matched
        ]
    )
    grid_xy = grid_xyz[:, :2]
    flow_xy = np.asarray([value for _, value in matched])
    rotation, translation, reflected = _orthogonal_fit(grid_xy, flow_xy)
    aligned_grid = grid_xy @ rotation + translation
    residual = flow_xy - aligned_grid
    residual_norm = np.linalg.norm(residual, axis=1)

    comparison_rows = []
    for (pose, raw_flow), aligned, error in zip(
        matched, aligned_grid, residual
    ):
        comparison_rows.append(
            {
                "timestamp_s": pose["timestamp_s"],
                "tag_count": pose["tag_count"],
                "reprojection_rmse_px": pose["reprojection_rmse_px"],
                "grid_x_m": pose["grid_x_m"],
                "grid_y_m": pose["grid_y_m"],
                "grid_z_m": pose["grid_z_m"],
                "aligned_grid_x_m": float(aligned[0]),
                "aligned_grid_y_m": float(aligned[1]),
                "flow_north_m": float(raw_flow[0]),
                "flow_east_m": float(raw_flow[1]),
                "error_x_m": float(error[0]),
                "error_y_m": float(error[1]),
                "error_norm_m": float(np.linalg.norm(error)),
            }
        )
    _write_csv(output / "flow_aprilgrid_alignment.csv", comparison_rows)

    windows = {}
    window_rows = []
    for seconds in (0.5, 1.0, 2.0):
        rows = _window_displacements(
            timestamps, grid_xy, flow_xy, rotation, seconds
        )
        for row in rows:
            row["window_s"] = seconds
            window_rows.append(row)
        windows[str(seconds)] = {
            "samples": len(rows),
            "scale_median": median(
                row["flow_to_grid_scale"] for row in rows
            ),
            "scale_p05": percentile(
                (row["flow_to_grid_scale"] for row in rows), 5
            ),
            "scale_p95": percentile(
                (row["flow_to_grid_scale"] for row in rows), 95
            ),
            "vector_error_median_m": median(
                row["vector_error_m"] for row in rows
            ),
            "vector_error_p95_m": percentile(
                (row["vector_error_m"] for row in rows), 95
            ),
        }
    _write_csv(output / "flow_aprilgrid_windows.csv", window_rows)

    start_time = timestamps[0]
    stop_time = timestamps[-1]
    start_stop_window_s = min(3.0, 0.1 * (stop_time - start_time))
    grid_start = _window_center(
        timestamps,
        grid_xy,
        start_time,
        start_time + start_stop_window_s,
    )
    grid_stop = _window_center(
        timestamps,
        grid_xy,
        stop_time - start_stop_window_s,
        stop_time,
    )
    flow_start = _window_center(
        timestamps,
        flow_xy,
        start_time,
        start_time + start_stop_window_s,
    )
    flow_stop = _window_center(
        timestamps,
        flow_xy,
        stop_time - start_stop_window_s,
        stop_time,
    )
    grid_closure = (grid_stop - grid_start) @ rotation
    flow_closure = flow_stop - flow_start

    metrics = {
        "frames_with_pose": len(poses),
        "matched_quality_poses": len(matched),
        "pose_availability_fraction": len(poses) / 2603.0,
        "tag_count_median": median(row["tag_count"] for row, _ in matched),
        "tag_count_p05": percentile(
            (row["tag_count"] for row, _ in matched), 5
        ),
        "reprojection_rmse_px_median": median(
            row["reprojection_rmse_px"] for row, _ in matched
        ),
        "reprojection_rmse_px_p95": percentile(
            (row["reprojection_rmse_px"] for row, _ in matched), 95
        ),
        "alignment": {
            "scale_fitted": False,
            "coordinate_reflection_used": reflected,
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
            "point_error_rmse_m": float(
                np.sqrt(np.mean(residual_norm**2))
            ),
            "point_error_median_m": float(np.median(residual_norm)),
            "point_error_p95_m": float(np.percentile(residual_norm, 95)),
        },
        "local_displacement_windows": windows,
        "start_stop": {
            "window_s": start_stop_window_s,
            "grid_displacement_aligned_m": grid_closure.tolist(),
            "grid_displacement_norm_m": float(np.linalg.norm(grid_closure)),
            "flow_displacement_m": flow_closure.tolist(),
            "flow_displacement_norm_m": float(np.linalg.norm(flow_closure)),
            "closure_vector_error_m": float(
                np.linalg.norm(flow_closure - grid_closure)
            ),
        },
    }
    (output / "aprilgrid_validation.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(
        aligned_grid[:, 0],
        aligned_grid[:, 1],
        label="AprilGrid PnP (metric)",
        linewidth=2,
    )
    axes[0].plot(
        flow_xy[:, 0],
        flow_xy[:, 1],
        label="CM2+dToF flow",
        linewidth=1.2,
    )
    axes[0].scatter(*aligned_grid[0], marker="o", label="start")
    axes[0].scatter(*aligned_grid[-1], marker="x", label="end")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("aligned axis 1 (m)")
    axes[0].set_ylabel("aligned axis 2 (m)")
    axes[0].set_title("Trajectory (rigid alignment; no scale fit)")
    axes[0].legend()

    elapsed = timestamps - timestamps[0]
    axes[1].plot(elapsed, residual_norm, linewidth=1)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("elapsed time (s)")
    axes[1].set_ylabel("flow − AprilGrid position error (m)")
    axes[1].set_title("Accumulated horizontal error")
    figure.tight_layout()
    figure.savefig(output / "flow_aprilgrid_validation.png", dpi=180)
    plt.close(figure)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-path-comparison",
        action="store_true",
        help=(
            "score angular flow and optional loop anchors without requiring "
            "a dToF-integrated path in flow_selected.csv"
        ),
    )
    parser.add_argument(
        "--loop-anchor-window",
        action="append",
        default=[],
        metavar="START:STOP",
        help=(
            "settled loop anchor in seconds relative to the first accepted "
            "AprilGrid pose; repeat for each corner"
        ),
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    poses = extract_poses(arguments.bag, arguments.output)
    metrics = {
        "angular_flow": angular_validation(
            poses, arguments.flow, arguments.output
        ),
    }
    if not arguments.skip_path_comparison:
        metrics["dtof_metric_path"] = compare(
            poses, arguments.flow, arguments.output
        )
    if arguments.loop_anchor_window:
        windows = [
            tuple(float(value) for value in item.split(":", maxsplit=1))
            for item in arguments.loop_anchor_window
        ]
        if len(windows) < 2:
            parser.error("at least two --loop-anchor-window values are required")
        metrics["loop"] = loop_validation(
            poses, arguments.flow, arguments.output, windows
        )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
