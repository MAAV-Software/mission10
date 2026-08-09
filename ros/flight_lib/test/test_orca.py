import numpy as np

from flight_lib import OrcaPeer, orca_effective_radius, orca_solution, orca_velocity


def test_head_on_pair_selects_reciprocal_opposite_sides():
    a = orca_velocity(
        [1.0, 0.0], [1.0, 0.0],
        [OrcaPeer(np.array([4.0, 0.0]), np.array([-1.0, 0.0]), 2.0)],
    )
    b = orca_velocity(
        [-1.0, 0.0], [-1.0, 0.0],
        [OrcaPeer(np.array([-4.0, 0.0]), np.array([1.0, 0.0]), 2.0)],
    )
    assert a[1] < 0.0
    assert b[1] > 0.0
    assert np.allclose(a, -b)


def test_distant_nonconflicting_peer_leaves_velocity_unchanged():
    preferred = np.array([1.0, 0.0])
    safe = orca_velocity(
        preferred, preferred,
        [OrcaPeer(np.array([0.0, 10.0]), np.zeros(2), 2.0)],
    )
    assert np.allclose(safe, preferred)


def test_multiple_peer_constraints_are_satisfied_together():
    own = np.array([1.0, 0.0])
    peers = [
        OrcaPeer(np.array([4.0, 0.5]), np.array([-1.0, 0.0]), 2.0),
        OrcaPeer(np.array([4.0, -0.5]), np.array([-1.0, 0.0]), 2.0),
    ]
    safe = orca_velocity(own, own, peers)
    assert np.linalg.norm(safe) <= 2.0 + 1e-8
    assert safe[0] < own[0]


def test_effective_radius_expands_only_while_closing():
    holding = orca_effective_radius(1.25, 0.5, 0.0, 0.6, 6.0)
    closing = orca_effective_radius(1.25, 0.5, -2.0, 0.6, 6.0)
    separating = orca_effective_radius(1.25, 0.5, 2.0, 0.6, 6.0)

    assert holding == 1.75
    assert closing == 2.95
    assert separating == holding


def test_stationary_three_meter_formation_has_boundary_slack():
    boundary = orca_effective_radius(1.25, 0.5, 0.0, 0.6, 6.0)
    preferred = np.zeros(2)
    safe = orca_velocity(
        preferred,
        preferred,
        [OrcaPeer(np.array([3.0, 0.0]), np.zeros(2), boundary)],
        time_horizon_s=5.0,
    )
    assert np.allclose(safe, preferred)


def test_uncertainty_inflation_can_be_expressed_as_larger_peer_radius():
    preferred = np.array([1.0, 0.0])
    exact = orca_velocity(
        preferred, preferred,
        [OrcaPeer(np.array([5.0, 0.0]), np.array([-1.0, 0.0]), 1.0)],
    )
    uncertain = orca_velocity(
        preferred, preferred,
        [OrcaPeer(np.array([5.0, 0.0]), np.array([-1.0, 0.0]), 2.0)],
    )
    assert np.linalg.norm(uncertain - preferred) > np.linalg.norm(exact - preferred)


def test_solution_reports_infeasible_speed_disc_without_inventing_a_fallback():
    peers = [
        OrcaPeer(np.array([0.1, 0.0]), np.zeros(2), 2.0),
        OrcaPeer(np.array([-0.1, 0.0]), np.zeros(2), 2.0),
    ]
    solution = orca_solution(np.zeros(2), np.zeros(2), peers, max_speed_mps=0.1)
    assert not solution.feasible
    assert solution.max_violation > 0.0
    assert np.linalg.norm(solution.velocity) <= 0.1 + 1e-9


def test_peer_order_permutation_is_symmetric_for_feasible_crossing():
    own = np.array([1.0, 0.0])
    peers = [
        OrcaPeer(np.array([4.0, 1.0]), np.array([-0.5, 0.0]), 1.5),
        OrcaPeer(np.array([4.0, -1.0]), np.array([-0.5, 0.0]), 1.5),
    ]
    forward = orca_solution(own, own, peers)
    reverse = orca_solution(own, own, list(reversed(peers)))
    assert forward.feasible and reverse.feasible
    assert np.allclose(forward.velocity, reverse.velocity)
