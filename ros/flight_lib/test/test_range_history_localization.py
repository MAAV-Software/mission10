import math

import numpy as np

from flight_lib import (
    RangeHistoryRelativeEstimator,
    STATUS_AMBIGUOUS,
    STATUS_TRACKING,
    STATUS_UNOBSERVABLE,
)


def _feed(estimator, rate_hz, duration_s, trajectory, noise_std=0.0, seed=1):
    rng = np.random.default_rng(seed)
    result = None
    for step in range(round(rate_hz * duration_s) + 1):
        time_s = step / rate_hz
        position, velocity = trajectory(time_s)
        measured = np.linalg.norm(position) + rng.normal(0.0, noise_std)
        result = estimator.add_sample(time_s, measured, [0.0, 0.0], velocity)
    return result, position


def test_no_relative_motion_is_explicitly_unobservable():
    estimator = RangeHistoryRelativeEstimator()
    result, _ = _feed(
        estimator, 40.0, 3.0,
        lambda _t: (np.array([3.0, 1.0]), np.zeros(2)),
    )
    assert result.status == STATUS_UNOBSERVABLE
    assert result.position is None


def test_straight_motion_retains_both_mirror_bearings():
    estimator = RangeHistoryRelativeEstimator()

    def trajectory(t):
        return np.array([3.0 + 0.5 * t, 1.5]), np.array([0.5, 0.0])

    result, truth = _feed(estimator, 40.0, 4.0, trajectory)
    assert result.status == STATUS_AMBIGUOUS
    assert result.alternate_position is not None
    candidates = [result.position, result.alternate_position]
    assert min(np.linalg.norm(candidate - truth) for candidate in candidates) < 0.03
    assert candidates[0][1] * candidates[1][1] < 0.0


def test_curved_motion_resolves_bearing_without_absolute_position():
    estimator = RangeHistoryRelativeEstimator()

    def trajectory(t):
        angle = 0.8 * t
        return (
            np.array([4.0 + 1.2 * math.cos(angle), 2.0 + 1.2 * math.sin(angle)]),
            np.array([-0.96 * math.sin(angle), 0.96 * math.cos(angle)]),
        )

    result, truth = _feed(estimator, 40.0, 5.0, trajectory, noise_std=0.02)
    assert result.status == STATUS_TRACKING
    assert np.linalg.norm(result.position - truth) < 0.25
    assert result.confidence_radius_95_m <= 0.5
    assert result.alternate_position is None


def test_estimator_is_event_driven_across_rate_sweep():
    def trajectory(t):
        angle = 0.9 * t
        return (
            np.array([3.5 + math.cos(angle), 1.5 + math.sin(angle)]),
            np.array([-0.9 * math.sin(angle), 0.9 * math.cos(angle)]),
        )

    for rate in (2.0, 5.0, 10.0, 40.0, 100.0):
        estimator = RangeHistoryRelativeEstimator(min_samples=4)
        result, truth = _feed(estimator, rate, 5.0, trajectory, noise_std=0.01)
        assert np.linalg.norm(result.position - truth) < 0.35, rate
        # Low-rate histories can solve the geometry without honestly claiming
        # the plan's 0.5 m 95% confidence requirement. High-rate proximity data
        # crosses that threshold without any rate-specific estimator branch.
        if rate >= 40.0:
            assert result.status == STATUS_TRACKING, rate
        else:
            assert result.status == STATUS_AMBIGUOUS, rate


def test_range_rate_sign_tracks_approach_and_departure():
    approaching = RangeHistoryRelativeEstimator()
    result, _ = _feed(
        approaching, 40.0, 1.0,
        lambda t: (np.array([4.0 - t, 0.0]), np.array([-1.0, 0.0])),
    )
    assert result.range_rate_mps < -0.9

    departing = RangeHistoryRelativeEstimator()
    result, _ = _feed(
        departing, 40.0, 1.0,
        lambda t: (np.array([4.0 + t, 0.0]), np.array([1.0, 0.0])),
    )
    assert result.range_rate_mps > 0.9
