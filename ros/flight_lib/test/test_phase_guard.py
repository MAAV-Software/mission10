import numpy as np

from flight_lib import phase_guard_rate


def test_clear_peer_leaves_rate_unchanged():
    solution = phase_guard_rate(
        [4.6, 0.0], [4.6, 0.0], 0.4,
        [[3.0, 4.6]], [[-1.84, 0.0]],
    )
    assert solution.feasible
    assert solution.rate == 0.4


def test_closing_peer_clamps_own_rate_without_stopping_or_reversing():
    solution = phase_guard_rate(
        [0.0, 0.0], [0.0, 4.6], 0.4,
        [[2.9, 0.0]], [[-2.0, 0.0]],
        protected_distance_m=3.0,
        min_rate_rad_s=0.15,
        max_rate_rad_s=0.65,
    )
    assert solution.feasible
    assert 0.4 < solution.rate <= 0.65
    assert solution.min_slack >= -1e-9


def test_impossible_barrier_is_reported_explicitly():
    solution = phase_guard_rate(
        [0.0, 0.0], [0.0, 0.0], 0.4,
        [[0.0, 0.0]], [[0.0, 0.0]],
        min_rate_rad_s=0.2,
        max_rate_rad_s=0.6,
    )
    assert not solution.feasible
    assert solution.min_slack < 0.0


def test_shape_errors_are_rejected():
    with np.testing.assert_raises(ValueError):
        phase_guard_rate([0.0], [0.0, 1.0], 0.4, [], [])
