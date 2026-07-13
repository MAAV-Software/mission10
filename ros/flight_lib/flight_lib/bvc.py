"""Buffered Voronoi Cell deconfliction.

Plain BVC (Zhou, Wang, Bandyopadhyay, Schwager, RA-L 2017): decentralized, position-only,
provably collision-free — each drone stays on its own side of every pairwise perpendicular
bisector, retracted inward by a safety radius. Given the drone's desired goal, the safe
setpoint is the point of its cell nearest the goal.

B-UAVC (Zhu, Brito, Alonso-Mora, Auton. Robots 2022): the uncertainty-aware variant. The
relative position to each peer is an estimate with covariance (here: UWB range + differential
GNSS + payload velocity). Each bisector's buffer grows by the relative-position uncertainty
projected onto that bisector's normal, scaled by the chance-constraint quantile, so collision
probability stays below delta instead of assuming perfect knowledge. Plain BVC is the
zero-covariance special case.

All in a shared horizontal frame (ENU x, y); altitude handled separately.
"""
from __future__ import annotations

import statistics

import numpy as np


def _project_to_halfspaces(goal, a_mat, b_vec, iters):
    """Nearest point to ``goal`` in ``{x : a_mat x <= b_vec}`` via Dykstra's projection."""
    goal = np.asarray(goal, float)
    if a_mat.shape[0] == 0 or np.all(a_mat @ goal <= b_vec + 1e-9):
        return goal
    x = goal.copy()
    corrections = np.zeros_like(a_mat)
    for _ in range(iters):
        for k in range(a_mat.shape[0]):
            n = a_mat[k]
            y_in = x + corrections[k]
            slack = float(n @ y_in) - b_vec[k]
            y = y_in - slack * n if slack > 0.0 else y_in  # project onto half-space (n unit)
            corrections[k] = y_in - y
            x = y
    return x


def normal_uncertainty_buffer(normal, cov, delta):
    """Chance-constraint buffer along ``normal`` for relative-position covariance ``cov``.

    ``sqrt(n^T cov n) * Phi^{-1}(1 - delta)``: the std of the relative position along the
    bisector normal times the one-sided Gaussian quantile, so the half-space holds with
    probability >= 1 - delta. Uncertainty perpendicular to the normal does not inflate it.
    """
    n = np.asarray(normal, float)
    var = float(n @ np.asarray(cov, float) @ n)
    if var <= 0.0:
        return 0.0
    return float(np.sqrt(var) * statistics.NormalDist().inv_cdf(1.0 - delta))


def buffered_voronoi_halfspaces(own_xy, peer_xys, safety_radius):
    """Return ``(A, b)`` for the BVC ``{x : A x <= b}`` of ``own_xy`` against peers.

    Each row is the buffered perpendicular bisector toward one peer, retracted inward by
    ``safety_radius`` (so two reciprocal cells stay >= 2*safety_radius apart). ``A`` rows are
    unit normals from own toward the peer.
    """
    return _voronoi_halfspaces(own_xy, [(p, None) for p in peer_xys], safety_radius, 0.0)


def buffered_uncertainty_halfspaces(own_xy, peers, safety_radius, delta):
    """Return ``(A, b)`` for the B-UAVC of ``own_xy``. ``peers`` is a list of
    ``(peer_xy, cov2x2)``; ``cov2x2`` is the relative-position covariance (None -> exact)."""
    return _voronoi_halfspaces(own_xy, peers, safety_radius, delta)


def _voronoi_halfspaces(own_xy, peers, safety_radius, delta):
    own = np.asarray(own_xy, float)
    rows_a, rows_b = [], []
    for peer_xy, cov in peers:
        d = np.asarray(peer_xy, float) - own
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            continue
        n = d / dist
        buffer = safety_radius
        if cov is not None and delta > 0.0:
            buffer += normal_uncertainty_buffer(n, cov, delta)
        rows_a.append(n)
        rows_b.append(float(n @ own) + dist / 2.0 - buffer)
    if not rows_a:
        return np.zeros((0, 2)), np.zeros((0,))
    return np.asarray(rows_a), np.asarray(rows_b)


def buffered_voronoi_clip(own_xy, goal_xy, peer_xys, safety_radius, *, iters=30):
    """Point of own's BVC nearest ``goal_xy`` (2D ENU). Transparent when the goal is already
    feasible; otherwise the nearest feasible point (Dykstra). Provably keeps reciprocal pairs
    >= 2*safety_radius apart when all clip and start collision-free."""
    a_mat, b_vec = buffered_voronoi_halfspaces(own_xy, peer_xys, safety_radius)
    return _project_to_halfspaces(goal_xy, a_mat, b_vec, iters)


def buffered_uncertainty_voronoi_clip(own_xy, goal_xy, peers, safety_radius, delta, *, iters=30):
    """B-UAVC clip: like ``buffered_voronoi_clip`` but each bisector buffer is inflated by the
    relative-position uncertainty (per-peer ``(peer_xy, cov2x2)``) at chance level ``delta``."""
    a_mat, b_vec = buffered_uncertainty_halfspaces(own_xy, peers, safety_radius, delta)
    return _project_to_halfspaces(goal_xy, a_mat, b_vec, iters)
