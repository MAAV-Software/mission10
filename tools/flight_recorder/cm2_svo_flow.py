"""Live CM2 angular flow from rl_vo's standalone SVO feature tracker."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

from cm2_flow import (
    FlowResult,
    STATUS_BAD_HOMOGRAPHY,
    STATUS_BAD_TIMING,
    STATUS_INITIALIZING,
    STATUS_NO_IMU,
    STATUS_TOO_FEW_FEATURES,
    STATUS_VALID,
    _rotate_vectors,
)


TRACK_COLUMNS = 16


class Cm2SvoFlowFrontend:
    """Track CM2 features with SVO and reduce them to PX4 angular flow."""

    name = "svo"

    def __init__(
        self,
        calibration: str | Path,
        imu,
        svo_build: str | Path,
        svo_params: str | Path,
        svo_calibration: str | Path,
        bands: int = 16,
    ):
        data = yaml.safe_load(Path(calibration).read_text())["cam0"]
        fx, fy, cx, cy = (float(value) for value in data["intrinsics"])
        self.native_width, self.native_height = (
            int(value) for value in data["resolution"]
        )
        self.line_delay_s = float(data["line_delay"])
        self.width = self.native_width // 2
        self.height = self.native_height // 2
        self.k = np.array(
            [
                [fx / 2.0, 0.0, cx / 2.0],
                [0.0, fy / 2.0, cy / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.distortion = np.asarray(
            data["distortion_coeffs"], dtype=np.float64
        )
        # image-right -> body-right, image-down -> body-back, optical -> down.
        self.r_b_c = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.imu = imu
        self.bands = int(bands)
        self.previous_center_ns = 0
        self.previous_tracks = np.empty((0, 3), dtype=np.float64)

        sys.path.insert(0, str(Path(svo_build).resolve()))
        import svo_env

        self.environment = svo_env.SVOEnv(
            str(Path(svo_params).resolve()),
            str(Path(svo_calibration).resolve()),
            1,
            True,
        )
        self.environment.setSeed(7)
        cv2.setRNGSeed(7)

    def _gray(self, img) -> np.ndarray:
        payload = np.frombuffer(img.data, dtype=np.uint8)
        if img.encoding in ("yuv422_yuy2", "yuyv"):
            native = payload.reshape(img.height, img.step)[:, : img.width * 2 : 2]
        elif img.encoding == "mono8":
            native = payload.reshape(img.height, img.step)[:, : img.width]
        else:
            raise ValueError(f"unsupported CM2 encoding {img.encoding!r}")
        return cv2.resize(
            native, (self.width, self.height), interpolation=cv2.INTER_AREA
        )

    def _invalid(
        self, status, center_ns, dt_us=0, detected=0, tracked=0
    ) -> FlowResult:
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

    def _native_rows(self, rows: np.ndarray) -> np.ndarray:
        band = np.clip(
            (rows * self.bands / self.height).astype(int),
            0,
            self.bands - 1,
        )
        return (
            (band + 0.5)
            * self.height
            / self.bands
            * self.native_height
            / self.height
        )

    def _project(self, rays: np.ndarray) -> np.ndarray:
        pixels, _ = cv2.projectPoints(
            rays.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            self.k,
            self.distortion,
        )
        return pixels[:, 0]

    def _predictions(self, center_ns: int) -> np.ndarray | None:
        if not len(self.previous_tracks):
            return np.empty((0, 3), dtype=np.float64)
        center_integrals = self.imu.integrals(
            [self.previous_center_ns, center_ns]
        )
        if center_integrals is None:
            return None

        track_ids = self.previous_tracks[:, 0]
        pixels0 = self.previous_tracks[:, 1:3]
        normalized0 = cv2.undistortPoints(
            pixels0.reshape(-1, 1, 2), self.k, self.distortion
        )[:, 0]
        rays_c0 = np.column_stack([normalized0, np.ones(len(normalized0))])
        rays_b0 = rays_c0 @ self.r_b_c.T
        center_delta = center_integrals[1] - center_integrals[0]
        rays_b1 = _rotate_vectors(
            rays_b0,
            np.repeat((-center_delta)[None, :], len(rays_b0), axis=0),
        )
        center_pixels1 = self._project(rays_b1 @ self.r_b_c)

        center_row = 0.5 * (self.native_height - 1)
        native0 = self._native_rows(pixels0[:, 1])
        native1 = self._native_rows(center_pixels1[:, 1])
        t0_ns = self.previous_center_ns + (
            (native0 - center_row) * self.line_delay_s * 1e9
        ).astype(np.int64)
        t1_ns = center_ns + (
            (native1 - center_row) * self.line_delay_s * 1e9
        ).astype(np.int64)
        integrals = self.imu.integrals(np.concatenate([t0_ns, t1_ns]))
        if integrals is None:
            return None
        count = len(pixels0)
        row_delta = integrals[count:] - integrals[:count]
        rays_b1 = _rotate_vectors(rays_b0, -row_delta)
        pixels1 = self._project(rays_b1 @ self.r_b_c)
        finite = np.isfinite(pixels1).all(axis=1)
        return np.column_stack([track_ids[finite], pixels1[finite]])

    def _tile_coverage(self, points: np.ndarray) -> float:
        occupied = {
            (
                min(7, max(0, int(x * 8 / self.width))),
                min(5, max(0, int(y * 6 / self.height))),
            )
            for x, y in points
        }
        return len(occupied) / 48.0

    @staticmethod
    def _transfer_error(
        homography: np.ndarray, p0: np.ndarray, p1: np.ndarray
    ) -> float:
        forward = cv2.perspectiveTransform(
            p0.reshape(-1, 1, 2), homography
        )[:, 0]
        backward = cv2.perspectiveTransform(
            p1.reshape(-1, 1, 2), np.linalg.inv(homography)
        )[:, 0]
        return float(
            np.median(
                0.5
                * (
                    np.linalg.norm(forward - p1, axis=1)
                    + np.linalg.norm(backward - p0, axis=1)
                )
            )
        )

    def _reduce(
        self, residual: np.ndarray, pairs: np.ndarray, points
    ) -> np.ndarray:
        minimum_eigenvalue = np.maximum(pairs[:, 14], 1e-6)
        condition = np.maximum(pairs[:, 15], 1.0)
        photometric_rmse = np.maximum(pairs[:, 13], 1e-3)
        weights = (
            np.sqrt(minimum_eigenvalue)
            / (photometric_rmse * np.sqrt(condition))
        )
        middle = float(np.median(weights))
        if np.isfinite(middle) and middle > 0:
            weights = np.clip(weights / middle, 0.1, 10.0)
        else:
            weights = np.ones(len(pairs))

        tiles = np.column_stack(
            [
                np.clip(
                    (points[:, 0] * 8 / self.width).astype(int), 0, 7
                ),
                np.clip(
                    (points[:, 1] * 6 / self.height).astype(int), 0, 5
                ),
            ]
        )
        for tile in np.unique(tiles, axis=0):
            selected = np.all(tiles == tile, axis=1)
            total = float(np.sum(weights[selected]))
            if total > 0:
                weights[selected] /= total

        location = np.median(residual, axis=0)
        for _ in range(3):
            distances = np.linalg.norm(residual - location, axis=1)
            scale = 1.4826 * float(np.median(distances))
            if not np.isfinite(scale) or scale <= 1e-12:
                break
            threshold = 1.345 * scale
            robust = np.ones(len(distances))
            outside = distances > threshold
            robust[outside] = threshold / distances[outside]
            combined = weights * robust
            location = np.sum(
                residual * combined[:, None], axis=0
            ) / np.sum(combined)
        return location

    def process(self, img, first_row_ns: int) -> FlowResult:
        half_readout_ns = int(
            0.5 * (self.native_height - 1) * self.line_delay_s * 1e9
        )
        center_ns = int(first_row_ns) + half_readout_ns
        gray = self._gray(img)
        if not self.previous_center_ns:
            self.environment.env_flow_step_detailed(
                0, gray, center_ns, np.empty((0, 3), dtype=np.float64)
            )
            self.previous_center_ns = center_ns
            return self._invalid(STATUS_INITIALIZING, center_ns)

        dt_us = int(round((center_ns - self.previous_center_ns) / 1000))
        if dt_us <= 0 or dt_us > 100_000:
            self.environment.reset(np.asarray([0.0], dtype=np.float64))
            self.previous_center_ns = 0
            self.previous_tracks = np.empty((0, 3), dtype=np.float64)
            return self._invalid(
                STATUS_BAD_TIMING, center_ns, max(0, dt_us)
            )

        predictions = self._predictions(center_ns)
        have_imu = predictions is not None
        if predictions is None:
            predictions = np.empty((0, 3), dtype=np.float64)
        attempts = np.asarray(
            self.environment.env_flow_step_detailed(
                0, gray, center_ns, predictions
            ),
            dtype=np.float64,
        ).reshape(-1, TRACK_COLUMNS)
        successful = attempts[attempts[:, 9] > 0.5]
        self.previous_tracks = (
            successful[:, [0, 5, 6]]
            if len(successful)
            else np.empty((0, 3), dtype=np.float64)
        )
        previous_center_ns = self.previous_center_ns
        self.previous_center_ns = center_ns
        detected = len(attempts)
        tracked = len(successful)
        if not have_imu:
            return self._invalid(
                STATUS_NO_IMU, center_ns, dt_us, detected, tracked
            )
        if tracked < 8:
            return self._invalid(
                STATUS_TOO_FEW_FEATURES,
                center_ns,
                dt_us,
                detected,
                tracked,
            )

        p0 = successful[:, 1:3]
        p1 = successful[:, 5:7]
        homography, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 2.0)
        if homography is None or mask is None:
            return self._invalid(
                STATUS_BAD_HOMOGRAPHY,
                center_ns,
                dt_us,
                detected,
                tracked,
            )
        inliers = mask[:, 0].astype(bool)
        if int(np.sum(inliers)) < 8:
            return self._invalid(
                STATUS_BAD_HOMOGRAPHY,
                center_ns,
                dt_us,
                detected,
                tracked,
            )

        center_row = 0.5 * (self.native_height - 1)
        native0 = self._native_rows(p0[:, 1])
        native1 = self._native_rows(p1[:, 1])
        t0_ns = previous_center_ns + (
            (native0 - center_row) * self.line_delay_s * 1e9
        ).astype(np.int64)
        t1_ns = center_ns + (
            (native1 - center_row) * self.line_delay_s * 1e9
        ).astype(np.int64)
        queries = np.concatenate(
            [t0_ns, t1_ns, [previous_center_ns, center_ns]]
        )
        integrals = self.imu.integrals(queries)
        if integrals is None:
            return self._invalid(
                STATUS_NO_IMU, center_ns, dt_us, detected, tracked
            )
        count = len(p0)
        row_delta = integrals[count : 2 * count] - integrals[:count]
        center_delta = integrals[-1] - integrals[-2]

        p0_norm = cv2.undistortPoints(
            p0.reshape(-1, 1, 2), self.k, self.distortion
        )[:, 0]
        p1_norm = cv2.undistortPoints(
            p1.reshape(-1, 1, 2), self.k, self.distortion
        )[:, 0]
        rays_c0 = np.column_stack([p0_norm, np.ones(count)])
        rays_b0 = rays_c0 @ self.r_b_c.T
        rays_b1 = _rotate_vectors(rays_b0, -row_delta)
        rays_c1 = rays_b1 @ self.r_b_c
        predicted = rays_c1[:, :2] / rays_c1[:, 2:3]
        residual = p1_norm - predicted
        angular_camera = self._reduce(residual, successful, p0)
        centered = residual - angular_camera

        camera_translation = np.array(
            [-angular_camera[0], -angular_camera[1], 0.0]
        )
        body_translation = self.r_b_c @ camera_translation
        compensated = np.array(
            [-body_translation[1], body_translation[0]]
        )
        raw = compensated + center_delta[:2]

        coverage = self._tile_coverage(p0)
        inlier_fraction = float(np.mean(inliers))
        transfer = self._transfer_error(
            homography, p0[inliers], p1[inliers]
        )
        residual_p95 = float(
            np.percentile(np.linalg.norm(centered, axis=1), 95)
        )
        factors = (
            min(1.0, coverage / 0.65),
            min(1.0, inlier_fraction / 0.75),
            max(0.0, 1.0 - transfer / 1.0),
            max(0.0, 1.0 - residual_p95 / 0.015),
        )
        quality = int(round(255.0 * math.prod(factors) ** 0.25))
        prediction_error = np.linalg.norm(
            successful[:, 5:7] - successful[:, 3:5], axis=1
        )
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
            inliers=int(np.sum(inliers)),
            coverage=coverage,
            inlier_fraction=inlier_fraction,
            fb_median_px=transfer,
            residual_p95_rad=residual_p95,
            tracks_xyxy=np.column_stack([p0, p1]).reshape(-1).tolist(),
            track_fb=prediction_error.tolist(),
        )
