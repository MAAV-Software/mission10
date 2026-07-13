"""Relative-position estimation from range-only UWB + differential GNSS + velocity.

The real airframes carry no AoA: a UWB packet gives only a scalar range to the peer.
Each packet does carry the peer's own GNSS estimate and velocity vector. Inside the 8 m
GNSS circle the *absolute* GNSS is useless (metre-level), but the **difference** of two
GNSS estimates over that short baseline cancels the common-mode error, leaving a usable
(if loose) relative-position vector. UWB range then sharpens that vector along the
line-of-sight only (it is a 1-D measurement). The peer's velocity dead-reckons the estimate
between round-robin range packets.

`RelativePositionEKF` fuses these into the relative position (peer - own, ENU 2-D) with a
covariance. The covariance comes out **anisotropic** — tight radial (range), loose
tangential (diff-GNSS) — which is exactly what the B-UAVC clip ([[buac-transit-validated]])
consumes; Phase A injected this shape by hand, Phase B measures it.

Pure NumPy, no ROS. Frame: shared horizontal ENU (x east, y north).
"""
from __future__ import annotations

import math

import numpy as np


def gnss_common_mode(t, amp=2.5, omega=0.15):
    """Deterministic common-mode GNSS bias (ENU) shared by all receivers at time ``t``.

    Both drones evaluate the identical bias, so differencing their GNSS estimates cancels
    it — the whole point of differential GNSS. Sim-side only (a stand-in for slowly drifting
    ionospheric/ephemeris error); the estimator never sees it directly."""
    return np.array([amp * math.sin(omega * t), amp * math.cos(omega * t)])


class RelativePositionEKF:
    """EKF on the relative position r = p_peer - p_own (ENU 2-D).

    State x = [r_e, r_n]. Predict integrates the measured relative velocity (own/peer EKF +
    packet); GNSS update is linear in both axes (loose); range update is nonlinear along the
    current line-of-sight (tight, 1-D)."""

    def __init__(self, vel_noise_std=0.3):
        self.x = None
        self.P = None
        self.vel_noise_std = float(vel_noise_std)

    @property
    def initialized(self):
        return self.x is not None

    @property
    def mean(self):
        return None if self.x is None else self.x.copy()

    @property
    def cov(self):
        return None if self.P is None else self.P.copy()

    def init_from_gnss(self, diff_gnss, R):
        self.x = np.asarray(diff_gnss, float).copy()
        self.P = np.asarray(R, float).copy()

    def predict(self, rel_vel, dt):
        """Dead-reckon by the measured relative velocity ``rel_vel`` = v_peer - v_own."""
        if self.x is None:
            return
        self.x = self.x + np.asarray(rel_vel, float) * dt
        # velocity-error random walk: position variance grows like (sigma_v dt)^2
        q = (self.vel_noise_std * dt) ** 2
        self.P = self.P + q * np.eye(2)

    def update_gnss(self, diff_gnss, R):
        """Linear update from the differential GNSS relative-position measurement (both axes)."""
        if self.x is None:
            self.init_from_gnss(diff_gnss, R)
            return
        z = np.asarray(diff_gnss, float)
        R = np.asarray(R, float)
        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(2) - K) @ self.P

    def update_range(self, range_m, R):
        """Nonlinear update from the scalar UWB range; tightens the line-of-sight axis only."""
        if self.x is None:
            return
        dist = float(np.linalg.norm(self.x))
        if dist < 1e-6:
            return
        H = (self.x / dist).reshape(1, 2)
        S = float((H @ self.P @ H.T)[0, 0]) + float(R)
        K = (self.P @ H.T) / S  # 2x1
        innovation = float(range_m) - dist
        self.x = self.x + (K.flatten() * innovation)
        self.P = (np.eye(2) - K @ H) @ self.P
