"""Relative-position estimation from range-only UWB and shared-frame velocity.

The range-history estimator deliberately has no GNSS input. It recovers the
relative position (peer - own) from scalar ranges and integrated relative motion,
retaining both mirror hypotheses when the motion is only one-dimensional. Once
the geometry is observable, a small EKF carries the estimate between batch solves.

Pure NumPy, no ROS. Frame: shared horizontal ENU (x east, y north).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


STATUS_UNOBSERVABLE = 0
STATUS_AMBIGUOUS = 1
STATUS_TRACKING = 2


@dataclass(frozen=True)
class RelativeEstimate:
    """Snapshot returned by :class:`RangeHistoryRelativeEstimator`.

    Positions and velocities use the observer's shared-north ENU axes.  The
    alternate fields are populated only while a straight-line history leaves a
    mirror ambiguity.  ``position`` is deliberately ``None`` while no bearing
    information exists; range alone is not silently turned into a direction.
    """

    status: int
    position: np.ndarray | None
    covariance: np.ndarray | None
    alternate_position: np.ndarray | None
    alternate_covariance: np.ndarray | None
    relative_velocity: np.ndarray
    range_m: float
    range_rate_mps: float
    residual_rms_m: float
    observability: float
    confidence_radius_95_m: float
    sample_count: int


@dataclass(frozen=True)
class _RangeMotionSample:
    time_s: float
    range_m: float
    displacement: np.ndarray
    relative_velocity: np.ndarray


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


@dataclass(frozen=True)
class FusedRelativeEstimate:
    position: np.ndarray
    covariance: np.ndarray
    relative_velocity: np.ndarray
    range_m: float
    range_rate_mps: float
    residual_m: float
    confidence_radius_95_m: float
    sample_count: int
    reseeded: bool


class FusedRelativeTracker:
    """Pairwise relative position from shared pose, velocity, and UWB range.

    Shared position supplies the bearing branch.  UWB range is an ordinary EKF
    measurement that tightens the radial axis; it is not a mode switch.
    """

    def __init__(self, *, position_noise_std=0.6, range_noise_std=0.1,
                 velocity_noise_std=0.3):
        self.position_variance = float(position_noise_std) ** 2
        self.range_variance = float(range_noise_std) ** 2
        self.ekf = RelativePositionEKF(velocity_noise_std)
        self.last_time_s = None
        self.last_range_time_s = None
        self.last_range_m = math.nan
        self.epochs = None
        self.sample_count = 0

    def update(self, time_s, own_position, peer_position, own_velocity,
               peer_velocity, *, range_m=math.nan, own_epoch=0, peer_epoch=0):
        now = float(time_s)
        own_position = np.asarray(own_position, float)[:2]
        peer_position = np.asarray(peer_position, float)[:2]
        own_velocity = np.asarray(own_velocity, float)[:2]
        peer_velocity = np.asarray(peer_velocity, float)[:2]
        values = np.concatenate((own_position, peer_position, own_velocity, peer_velocity))
        if not math.isfinite(now) or not np.all(np.isfinite(values)):
            raise ValueError("fused relative state requires finite time, pose, and velocity")
        if self.last_time_s is not None and now < self.last_time_s:
            raise ValueError("fused relative-state timestamps must not move backwards")

        relative_position = peer_position - own_position
        relative_velocity = peer_velocity - own_velocity
        epochs = (int(own_epoch), int(peer_epoch))
        reseeded = not self.ekf.initialized or self.epochs != epochs
        if reseeded:
            self.ekf.init_from_gnss(
                relative_position, self.position_variance * np.eye(2))
            self.epochs = epochs
        else:
            dt = now - self.last_time_s
            if dt > 0.0:
                self.ekf.predict(relative_velocity, dt)
            self.ekf.update_gnss(
                relative_position, self.position_variance * np.eye(2))
        self.last_time_s = now

        if math.isfinite(float(range_m)) and float(range_m) > 0.0:
            measured_range = float(range_m)
            self.ekf.update_range(measured_range, self.range_variance)
            self.last_range_time_s = now
            self.last_range_m = measured_range
            self.sample_count += 1
        else:
            measured_range = self.last_range_m

        position = self.ekf.mean
        covariance = self.ekf.cov
        distance = float(np.linalg.norm(position))
        range_rate = (
            float(position @ relative_velocity) / distance
            if distance > 1e-6 else math.nan
        )
        residual = (
            abs(float(np.linalg.norm(position)) - measured_range)
            if math.isfinite(measured_range) else math.nan
        )
        radius = float(math.sqrt(
            5.991 * max(0.0, np.linalg.eigvalsh(covariance)[-1])))
        return FusedRelativeEstimate(
            position=position,
            covariance=covariance,
            relative_velocity=relative_velocity.copy(),
            range_m=measured_range,
            range_rate_mps=range_rate,
            residual_m=residual,
            confidence_radius_95_m=radius,
            sample_count=self.sample_count,
            reseeded=reseeded,
        )


class RangeHistoryRelativeEstimator:
    """Relative bearing from scalar ranges and shared-frame velocity history.

    For a window anchored at ``t0``, relative motion gives a known displacement
    ``d_k`` and UWB supplies ``rho_k``::

        rho_k^2 - rho_0^2 - ||d_k||^2 = 2 r_0^T d_k

    A two-dimensional motion history determines ``r_0``.  A rank-one history
    leaves the two mirror points where that line constraint intersects the
    initial range circle; both are retained.  No-motion histories remain
    explicitly unobservable.  Once a unique estimate is sufficiently tight,
    the inexpensive EKF carries it between range samples.
    """

    def __init__(self, *, window_s=5.0, range_noise_std=0.1,
                 velocity_noise_std=0.3, min_samples=6,
                 min_baseline_m=0.5, min_observability=0.10,
                 tracking_radius_95_m=0.5, likelihood_ratio=10.0):
        self.window_s = float(window_s)
        self.range_noise_std = float(range_noise_std)
        self.velocity_noise_std = float(velocity_noise_std)
        self.min_samples = int(min_samples)
        self.min_baseline_m = float(min_baseline_m)
        self.min_observability = float(min_observability)
        self.tracking_radius_95_m = float(tracking_radius_95_m)
        self.likelihood_ratio = float(likelihood_ratio)
        self.samples: deque[_RangeMotionSample] = deque()
        self._displacement = np.zeros(2)
        self._last_time = None
        self._last_rel_velocity = np.zeros(2)
        self._ekf: RelativePositionEKF | None = None
        self._tracking_acquired = False
        self._last_estimate = self._empty_estimate()

    def _empty_estimate(self):
        return RelativeEstimate(
            status=STATUS_UNOBSERVABLE,
            position=None,
            covariance=None,
            alternate_position=None,
            alternate_covariance=None,
            relative_velocity=np.zeros(2),
            range_m=math.nan,
            range_rate_mps=math.nan,
            residual_rms_m=math.inf,
            observability=0.0,
            confidence_radius_95_m=math.inf,
            sample_count=0,
        )

    @property
    def estimate(self):
        return self._last_estimate

    def reset(self):
        self.samples.clear()
        self._displacement = np.zeros(2)
        self._last_time = None
        self._last_rel_velocity = np.zeros(2)
        self._ekf = None
        self._tracking_acquired = False
        self._last_estimate = self._empty_estimate()

    def add_sample(self, time_s, range_m, own_velocity_enu, peer_velocity_enu):
        """Add one time-aligned UWB/velocity observation and return a snapshot."""
        now = float(time_s)
        rho = float(range_m)
        rel_velocity = (np.asarray(peer_velocity_enu, float)[:2]
                        - np.asarray(own_velocity_enu, float)[:2])
        if not math.isfinite(now) or not math.isfinite(rho) or rho <= 0.0:
            raise ValueError("range samples require a finite timestamp and positive range")
        if not np.all(np.isfinite(rel_velocity)):
            raise ValueError("velocity samples must be finite")
        if self._last_time is not None:
            dt = now - self._last_time
            if dt <= 0.0:
                raise ValueError("range sample timestamps must increase")
            self._displacement += 0.5 * (self._last_rel_velocity + rel_velocity) * dt
            if self._ekf is not None:
                self._ekf.predict(0.5 * (self._last_rel_velocity + rel_velocity), dt)
                self._ekf.update_range(rho, self.range_noise_std ** 2)
        self._last_time = now
        self._last_rel_velocity = rel_velocity.copy()
        self.samples.append(_RangeMotionSample(
            now, rho, self._displacement.copy(), rel_velocity.copy()))
        while self.samples and now - self.samples[0].time_s > self.window_s:
            self.samples.popleft()

        # Continue evaluating the window until it becomes uniquely observable.
        # Once tracking, the EKF is the primary fast path; the batch solve still
        # supplies observability and residual diagnostics.
        batch = self._solve_window()
        # A rank-two solve can select a unique bearing branch before its absolute
        # error is small enough for ORCA.  Keep that coarse branch in the EKF so
        # later motion and range updates can refine it; throwing it away here
        # makes a subsequent straight transit ambiguous all over again.
        batch_unique = (
            batch.position is not None
            and batch.alternate_position is None
            and batch.covariance is not None
            and np.all(np.isfinite(batch.position))
            and np.all(np.isfinite(batch.covariance))
        )
        if batch_unique:
            batch_radius = self._confidence_radius(batch.covariance)
            ekf_radius = (
                math.inf if self._ekf is None
                else self._confidence_radius(self._ekf.cov)
            )
            if batch_radius < ekf_radius:
                self._ekf = RelativePositionEKF(self.velocity_noise_std)
                self._ekf.x = batch.position.copy()
                self._ekf.P = batch.covariance.copy()

        # Use separate acquire/drop thresholds so a harmless covariance tick at
        # a rate or observability boundary does not chatter the bearing on and
        # off. Once motion has resolved the mirror branch, a later straight-line
        # window does not make that history unknowable; the EKF carries the
        # selected branch until its own covariance says it is no longer useful.
        if self._ekf is not None:
            position = self._ekf.mean
            covariance = self._ekf.cov
            radius = self._confidence_radius(covariance)
            if radius <= self.tracking_radius_95_m:
                self._tracking_acquired = True
            if (self._tracking_acquired
                    and radius <= 2.0 * self.tracking_radius_95_m):
                self._last_estimate = RelativeEstimate(
                    status=STATUS_TRACKING,
                    position=position,
                    covariance=covariance,
                    alternate_position=None,
                    alternate_covariance=None,
                    relative_velocity=rel_velocity.copy(),
                    range_m=rho,
                    range_rate_mps=self._range_rate(),
                    residual_rms_m=batch.residual_rms_m,
                    observability=batch.observability,
                    confidence_radius_95_m=radius,
                    sample_count=len(self.samples),
                )
                return self._last_estimate

        # Retain a uniquely selected but coarse branch while it refines.  The
        # published status remains AMBIGUOUS until the normal confidence gate is
        # met, so ORCA never consumes this loose estimate.  A genuinely diverged
        # branch is discarded and must be reacquired.
        if self._ekf is not None:
            radius = self._confidence_radius(self._ekf.cov)
            if radius <= 10.0 * self.tracking_radius_95_m:
                self._last_estimate = batch
                return self._last_estimate

        self._ekf = None
        self._tracking_acquired = False
        self._last_estimate = batch
        return self._last_estimate

    def _solve_window(self):
        count = len(self.samples)
        if count < self.min_samples:
            return self._status_only(STATUS_UNOBSERVABLE, 0.0)
        samples = list(self.samples)
        origin = samples[0].displacement
        deltas = np.asarray([sample.displacement - origin for sample in samples])
        ranges = np.asarray([sample.range_m for sample in samples])
        baseline = float(np.max(np.linalg.norm(deltas, axis=1)))
        if baseline < self.min_baseline_m:
            return self._status_only(STATUS_UNOBSERVABLE, 0.0)

        a_mat = deltas[1:]
        b_vec = 0.5 * (ranges[1:] ** 2 - ranges[0] ** 2
                       - np.sum(a_mat * a_mat, axis=1))
        _, singular, vh = np.linalg.svd(a_mat, full_matrices=False)
        if singular.size == 0 or singular[0] <= 1e-9:
            return self._status_only(STATUS_UNOBSERVABLE, 0.0)
        observability = float(singular[-1] / singular[0]) if singular.size >= 2 else 0.0
        current_delta = deltas[-1]

        if observability < self.min_observability:
            direction = vh[0]
            projected = a_mat @ direction
            denom = float(projected @ projected)
            if denom <= 1e-9:
                return self._status_only(STATUS_UNOBSERVABLE, observability)
            along = float(projected @ b_vec) / denom
            perpendicular = np.array([-direction[1], direction[0]])
            cross_sq = max(0.0, ranges[0] ** 2 - along ** 2)
            cross = math.sqrt(cross_sq)
            first0 = along * direction + cross * perpendicular
            second0 = along * direction - cross * perpendicular
            first = first0 + current_delta
            second = second0 + current_delta
            residual_first = self._residual_rms(first0, deltas, ranges)
            residual_second = self._residual_rms(second0, deltas, ranges)
            radial = max(self.range_noise_std ** 2, 1e-6)
            tangent = max(0.25, ranges[0] ** 2)
            cov0 = radial * np.outer(direction, direction) + tangent * np.outer(
                perpendicular, perpendicular)
            q = (self.velocity_noise_std * (samples[-1].time_s - samples[0].time_s)) ** 2
            cov = cov0 + q * np.eye(2)
            return RelativeEstimate(
                status=STATUS_AMBIGUOUS,
                position=first,
                covariance=cov,
                alternate_position=second,
                alternate_covariance=cov.copy(),
                relative_velocity=samples[-1].relative_velocity.copy(),
                range_m=float(ranges[-1]),
                range_rate_mps=self._range_rate(),
                residual_rms_m=min(residual_first, residual_second),
                observability=observability,
                confidence_radius_95_m=self._confidence_radius(cov),
                sample_count=count,
            )

        initial, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        refined = self._gauss_newton(initial, deltas, ranges)
        residual = self._residual_rms(refined, deltas, ranges)
        jacobian = self._range_jacobian(refined, deltas)
        information = jacobian.T @ jacobian / max(self.range_noise_std ** 2, 1e-9)
        cov0 = np.linalg.pinv(information)
        elapsed = samples[-1].time_s - samples[0].time_s
        cov0 += (self.velocity_noise_std * elapsed / max(1.0, math.sqrt(count))) ** 2 * np.eye(2)
        current = refined + current_delta
        radius = self._confidence_radius(cov0)
        status = STATUS_TRACKING if radius <= self.tracking_radius_95_m else STATUS_AMBIGUOUS
        return RelativeEstimate(
            status=status,
            position=current,
            covariance=cov0,
            alternate_position=None,
            alternate_covariance=None,
            relative_velocity=samples[-1].relative_velocity.copy(),
            range_m=float(ranges[-1]),
            range_rate_mps=self._range_rate(),
            residual_rms_m=residual,
            observability=observability,
            confidence_radius_95_m=radius,
            sample_count=count,
        )

    def _status_only(self, status, observability):
        last = self.samples[-1] if self.samples else None
        return RelativeEstimate(
            status=status,
            position=None,
            covariance=None,
            alternate_position=None,
            alternate_covariance=None,
            relative_velocity=(last.relative_velocity.copy() if last else np.zeros(2)),
            range_m=(last.range_m if last else math.nan),
            range_rate_mps=self._range_rate(),
            residual_rms_m=math.inf,
            observability=float(observability),
            confidence_radius_95_m=math.inf,
            sample_count=len(self.samples),
        )

    def _gauss_newton(self, initial, deltas, ranges):
        x = np.asarray(initial, float).copy()
        for _ in range(12):
            predicted = np.linalg.norm(x + deltas, axis=1)
            valid = predicted > 1e-6
            if np.count_nonzero(valid) < 2:
                break
            jacobian = (x + deltas[valid]) / predicted[valid, None]
            residual = ranges[valid] - predicted[valid]
            step, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
            x += step
            if float(np.linalg.norm(step)) < 1e-6:
                break
        return x

    @staticmethod
    def _range_jacobian(origin_position, deltas):
        vectors = np.asarray(origin_position) + deltas
        distances = np.linalg.norm(vectors, axis=1)
        valid = distances > 1e-6
        return vectors[valid] / distances[valid, None]

    @staticmethod
    def _residual_rms(origin_position, deltas, ranges):
        residual = np.linalg.norm(np.asarray(origin_position) + deltas, axis=1) - ranges
        return float(math.sqrt(np.mean(residual ** 2)))

    @staticmethod
    def _confidence_radius(covariance):
        return float(math.sqrt(5.991 * max(0.0, np.linalg.eigvalsh(covariance)[-1])))

    def _range_rate(self):
        if len(self.samples) < 2:
            return math.nan
        samples = list(self.samples)
        end = samples[-1].time_s
        recent = [sample for sample in samples if end - sample.time_s <= 0.75]
        if len(recent) < 2:
            recent = samples[-2:]
        times = np.asarray([sample.time_s - recent[0].time_s for sample in recent])
        ranges = np.asarray([sample.range_m for sample in recent])
        centered = times - np.mean(times)
        denom = float(centered @ centered)
        return 0.0 if denom <= 1e-12 else float(centered @ (ranges - np.mean(ranges)) / denom)
