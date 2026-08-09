"""Mission-owned CM2 angular flow using the July 27 KLT model.

The frontend estimates gyro-compensated translational image motion for its
quality checks, then adds the center-row gyro integral back to form PX4's raw
SensorOpticalFlow measurement. PX4 removes that rotation exactly once.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import threading

import cv2
import numpy as np
import yaml


STATUS_INITIALIZING = 0
STATUS_VALID = 1
STATUS_NO_IMU = 2
STATUS_BAD_TIMING = 3
STATUS_TOO_FEW_FEATURES = 4
STATUS_BAD_HOMOGRAPHY = 5
STATUS_FRONTEND_ERROR = 6


class ImuHistory:
    """Thread-safe trapezoidal gyro integral in ROS/DDS realtime."""

    def __init__(self, horizon_s: float = 5.0):
        self._horizon_ns = int(horizon_s * 1e9)
        self._t = deque()
        self._gyro = deque()
        self._lock = threading.Lock()

    def note(self, timestamp_ns: int, gyro) -> None:
        sample = np.asarray(gyro, dtype=np.float64)
        with self._lock:
            if self._t and timestamp_ns <= self._t[-1]:
                return
            self._t.append(int(timestamp_ns))
            self._gyro.append(sample)
            cutoff = timestamp_ns - self._horizon_ns
            while len(self._t) > 2 and self._t[1] < cutoff:
                self._t.popleft()
                self._gyro.popleft()

    def integrals(self, query_ns) -> np.ndarray | None:
        query = np.asarray(query_ns, dtype=np.int64)
        with self._lock:
            if len(self._t) < 2:
                return None
            t_ns = np.asarray(self._t, dtype=np.int64)
            gyro = np.asarray(self._gyro, dtype=np.float64)
        if query.min() < t_ns[0] or query.max() > t_ns[-1]:
            return None
        t = (t_ns - t_ns[0]).astype(np.float64) * 1e-9
        dt = np.diff(t)
        integral = np.vstack(
            [np.zeros(3), np.cumsum(0.5 * (gyro[:-1] + gyro[1:]) * dt[:, None], axis=0)]
        )
        q = (query - t_ns[0]).astype(np.float64) * 1e-9
        return np.column_stack(
            [np.interp(q, t, integral[:, axis]) for axis in range(3)]
        )


class RangeHistory:
    def __init__(self, horizon_s: float = 2.0):
        self._horizon_ns = int(horizon_s * 1e9)
        self._rows = deque()
        self._lock = threading.Lock()

    def note(self, timestamp_ns: int, distance_m: float, quality: int) -> None:
        with self._lock:
            self._rows.append((int(timestamp_ns), float(distance_m), int(quality)))
            cutoff = timestamp_ns - self._horizon_ns
            while len(self._rows) > 1 and self._rows[1][0] < cutoff:
                self._rows.popleft()

    def nearest(self, timestamp_ns: int):
        with self._lock:
            if not self._rows:
                return None
            row = min(self._rows, key=lambda value: abs(value[0] - timestamp_ns))
        return row


@dataclass
class FlowResult:
    status: int
    timestamp_sample_ns: int
    integration_timespan_us: int
    quality: int
    pixel_flow_raw: np.ndarray
    pixel_flow_compensated: np.ndarray
    delta_angle: np.ndarray
    detected: int = 0
    tracked: int = 0
    inliers: int = 0
    coverage: float = 0.0
    inlier_fraction: float = 0.0
    fb_median_px: float = math.nan
    residual_p95_rad: float = math.nan
    tracks_xyxy: list[float] | None = None
    track_fb: list[float] | None = None


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


class Cm2FlowFrontend:
    """One stateful 30 Hz frontend. Calls are serialized by its worker."""

    name = "klt"

    def __init__(
        self, calibration: str | Path, imu: ImuHistory, bands: int = 8,
        downsample: int = 4,
    ):
        data = yaml.safe_load(Path(calibration).read_text())["cam0"]
        fx, fy, cx, cy = (float(value) for value in data["intrinsics"])
        self.native_width, self.native_height = (
            int(value) for value in data["resolution"]
        )
        self.line_delay_s = float(data["line_delay"])
        self.imu = imu
        self.bands = int(bands)
        self.downsample = int(downsample)
        self.width = self.native_width // self.downsample
        self.height = self.native_height // self.downsample
        self.k = np.array(
            [
                [fx / self.downsample, 0.0, cx / self.downsample],
                [0.0, fy / self.downsample, cy / self.downsample],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(data["distortion_coeffs"], dtype=np.float64)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.k, distortion, None, self.k, (self.width, self.height), cv2.CV_32FC1
        )
        # image-right -> body-right, image-down -> body-back, optical -> down.
        self.r_b_c = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.previous = None
        self.previous_center_ns = 0

    def _gray(self, img) -> np.ndarray:
        payload = np.frombuffer(img.data, dtype=np.uint8)
        if img.encoding in ("yuv422_yuy2", "yuyv"):
            native = payload.reshape(img.height, img.step)[:, : img.width * 2 : 2]
        elif img.encoding == "mono8":
            native = payload.reshape(img.height, img.step)[:, : img.width]
        else:
            raise ValueError(f"unsupported CM2 encoding {img.encoding!r}")
        small = cv2.resize(native, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return cv2.remap(small, self.map_x, self.map_y, cv2.INTER_LINEAR)

    def _features(self, image):
        selected = []
        occupied = 0
        for tile_y in range(6):
            for tile_x in range(8):
                x0 = round(tile_x * self.width / 8)
                x1 = round((tile_x + 1) * self.width / 8)
                y0 = round(tile_y * self.height / 6)
                y1 = round((tile_y + 1) * self.height / 6)
                points = cv2.goodFeaturesToTrack(
                    image[y0:y1, x0:x1],
                    maxCorners=6,
                    qualityLevel=0.01,
                    minDistance=5,
                    blockSize=5,
                )
                if points is not None:
                    points[:, 0, 0] += x0
                    points[:, 0, 1] += y0
                    selected.append(points)
                    occupied += 1
        if not selected:
            return np.empty((0, 1, 2), dtype=np.float32), 0
        return np.concatenate(selected), occupied

    def _invalid(self, status, center_ns, dt_us=0, detected=0, tracked=0):
        return FlowResult(
            status=status,
            timestamp_sample_ns=center_ns,
            integration_timespan_us=dt_us,
            quality=0,
            pixel_flow_raw=np.zeros(2),
            pixel_flow_compensated=np.zeros(2),
            delta_angle=np.full(3, np.nan),
            detected=detected,
            tracked=tracked,
        )

    def process(self, img, first_row_ns: int) -> FlowResult:
        half_readout_ns = int(
            0.5 * (self.native_height - 1) * self.line_delay_s * 1e9
        )
        center_ns = int(first_row_ns) + half_readout_ns
        gray = self._gray(img)
        if self.previous is None:
            self.previous = gray
            self.previous_center_ns = center_ns
            return self._invalid(STATUS_INITIALIZING, center_ns)
        dt_us = int(round((center_ns - self.previous_center_ns) / 1000))
        if dt_us <= 0 or dt_us > 100_000:
            self.previous = gray
            self.previous_center_ns = center_ns
            return self._invalid(STATUS_BAD_TIMING, center_ns, max(0, dt_us))

        points0, occupied = self._features(self.previous)
        detected = len(points0)
        if detected < 8:
            self.previous = gray
            self.previous_center_ns = center_ns
            return self._invalid(STATUS_TOO_FEW_FEATURES, center_ns, dt_us, detected)
        points1, status1, _ = cv2.calcOpticalFlowPyrLK(
            self.previous, gray, points0, None, winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        points0_back, status2, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous, points1, None, winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        fb = np.linalg.norm(points0[:, 0] - points0_back[:, 0], axis=1)
        good = (
            (status1[:, 0] == 1) & (status2[:, 0] == 1) & (fb < 1.0)
            & (points1[:, 0, 0] >= 0) & (points1[:, 0, 0] < self.width)
            & (points1[:, 0, 1] >= 0) & (points1[:, 0, 1] < self.height)
        )
        p0 = points0[good, 0]
        p1 = points1[good, 0]
        fb_good = fb[good]
        tracked = len(p0)
        if tracked < 8:
            self.previous = gray
            self.previous_center_ns = center_ns
            return self._invalid(
                STATUS_TOO_FEW_FEATURES, center_ns, dt_us, detected, tracked
            )
        _, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 2.0)
        if mask is None:
            self.previous = gray
            self.previous_center_ns = center_ns
            return self._invalid(STATUS_BAD_HOMOGRAPHY, center_ns, dt_us, detected, tracked)
        mask = mask[:, 0].astype(bool)
        p0 = p0[mask]
        p1 = p1[mask]
        fb_good = fb_good[mask]
        if len(p0) < 8:
            self.previous = gray
            self.previous_center_ns = center_ns
            return self._invalid(STATUS_BAD_HOMOGRAPHY, center_ns, dt_us, detected, tracked)

        p0_norm = cv2.undistortPoints(p0.reshape(-1, 1, 2), self.k, None)[:, 0]
        p1_norm = cv2.undistortPoints(p1.reshape(-1, 1, 2), self.k, None)[:, 0]
        rows0 = np.clip((p0[:, 1] * self.bands / self.height).astype(int), 0, self.bands - 1)
        rows1 = np.clip((p1[:, 1] * self.bands / self.height).astype(int), 0, self.bands - 1)
        native0 = (rows0 + 0.5) * self.height / self.bands * self.native_height / self.height
        native1 = (rows1 + 0.5) * self.height / self.bands * self.native_height / self.height
        row_center = 0.5 * (self.native_height - 1)
        t0_ns = self.previous_center_ns + (
            (native0 - row_center) * self.line_delay_s * 1e9
        ).astype(np.int64)
        t1_ns = center_ns + (
            (native1 - row_center) * self.line_delay_s * 1e9
        ).astype(np.int64)
        queries = np.concatenate(
            [t0_ns, t1_ns, [self.previous_center_ns, center_ns]]
        )
        integrals = self.imu.integrals(queries)
        if integrals is None:
            self.previous = gray
            self.previous_center_ns = center_ns
            return self._invalid(STATUS_NO_IMU, center_ns, dt_us, detected, tracked)
        n = len(p0)
        row_delta = integrals[n:2 * n] - integrals[:n]
        center_delta = integrals[-1] - integrals[-2]
        rays_c0 = np.column_stack([p0_norm, np.ones(n)])
        rays_b0 = rays_c0 @ self.r_b_c.T
        rays_b1 = _rotate_vectors(rays_b0, -row_delta)
        rays_c1 = rays_b1 @ self.r_b_c
        predicted = rays_c1[:, :2] / rays_c1[:, 2:3]
        residual = p1_norm - predicted
        angular_camera = np.median(residual, axis=0)
        camera_translation = np.array([-angular_camera[0], -angular_camera[1], 0.0])
        body_translation = self.r_b_c @ camera_translation
        compensated = np.array([-body_translation[1], body_translation[0]])
        # SensorOpticalFlow is raw image motion. PX4 later computes
        # (-pixel_flow + delta_angle.xy) / dt, so restore the rotation removed
        # above instead of making PX4 compensate the same gyro twice.
        raw = compensated + center_delta[:2]
        centered = residual - angular_camera
        residual_p95 = float(np.percentile(np.linalg.norm(centered, axis=1), 95))
        coverage = occupied / 48.0
        inlier_fraction = len(p0) / max(1, tracked)
        fb_median = float(np.median(fb_good))
        factors = (
            min(1.0, coverage / 0.65),
            min(1.0, inlier_fraction / 0.75),
            max(0.0, 1.0 - fb_median / 1.0),
            # The July 27 capture's 95th percentile is 0.0114 rad. Use
            # 0.015 rad as the rejection edge so normal hand/flight texture
            # remains available while grossly non-planar fits still go to 0.
            max(0.0, 1.0 - residual_p95 / 0.015),
        )
        quality = int(round(255.0 * math.prod(factors) ** 0.25))
        self.previous = gray
        self.previous_center_ns = center_ns
        return FlowResult(
            status=STATUS_VALID if quality > 0 else STATUS_BAD_HOMOGRAPHY,
            timestamp_sample_ns=center_ns,
            integration_timespan_us=dt_us,
            quality=quality,
            pixel_flow_raw=raw,
            pixel_flow_compensated=compensated,
            delta_angle=center_delta,
            detected=detected,
            tracked=tracked,
            inliers=len(p0),
            coverage=coverage,
            inlier_fraction=inlier_fraction,
            fb_median_px=fb_median,
            residual_p95_rad=residual_p95,
            tracks_xyxy=np.column_stack([p0, p1]).astype(float).reshape(-1).tolist(),
            track_fb=fb_good.astype(float).tolist(),
        )
