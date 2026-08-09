"""Two-dimensional ORCA planning.

The ORCA construction follows van den Berg et al.'s reciprocal velocity
obstacles.  Peers are expressed relative to the observer, so no absolute origin
is required.  This module is pure NumPy and deliberately has no ROS or PX4 side
effects.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OrcaPeer:
    position: np.ndarray
    velocity: np.ndarray
    combined_radius: float


@dataclass(frozen=True)
class OrcaSolution:
    velocity: np.ndarray
    feasible: bool
    active_constraints: int
    max_violation: float


def _det(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _orca_halfplane(own_velocity, peer, time_horizon, time_step):
    p = np.asarray(peer.position, float)[:2]
    relative_velocity = np.asarray(own_velocity, float)[:2] - np.asarray(peer.velocity, float)[:2]
    radius = float(peer.combined_radius)
    dist_sq = float(p @ p)
    radius_sq = radius * radius

    if dist_sq > radius_sq:
        inv_horizon = 1.0 / max(1e-3, float(time_horizon))
        w = relative_velocity - inv_horizon * p
        w_sq = float(w @ w)
        dot = float(w @ p)
        if dot < 0.0 and dot * dot > radius_sq * w_sq:
            w_len = math.sqrt(max(w_sq, 1e-12))
            unit_w = w / w_len
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (radius * inv_horizon - w_len) * unit_w
        else:
            leg = math.sqrt(max(0.0, dist_sq - radius_sq))
            if _det(p, w) > 0.0:
                direction = np.array([
                    p[0] * leg - p[1] * radius,
                    p[0] * radius + p[1] * leg,
                ]) / dist_sq
            else:
                direction = -np.array([
                    p[0] * leg + p[1] * radius,
                    -p[0] * radius + p[1] * leg,
                ]) / dist_sq
            u = float(relative_velocity @ direction) * direction - relative_velocity
    else:
        inv_step = 1.0 / max(1e-3, float(time_step))
        w = relative_velocity - inv_step * p
        w_len = float(np.linalg.norm(w))
        if w_len <= 1e-9:
            unit_w = -p / max(math.sqrt(dist_sq), 1e-9)
        else:
            unit_w = w / w_len
        direction = np.array([unit_w[1], -unit_w[0]])
        u = (radius * inv_step - w_len) * unit_w

    point = np.asarray(own_velocity, float)[:2] + 0.5 * u
    # ORCA's permitted side is det(direction, point - velocity) <= 0,
    # equivalent to normal dot velocity >= normal dot point.
    normal = np.array([-direction[1], direction[0]])
    return normal, float(normal @ point)


def _clip_speed(velocity, max_speed):
    velocity = np.asarray(velocity, float)[:2].copy()
    speed = float(np.linalg.norm(velocity))
    return velocity if speed <= max_speed else velocity * (max_speed / speed)


def _line_candidate(preferred, constraint, preceding, max_speed):
    """Closest point on one constraint boundary that satisfies earlier lines."""
    normal, bound = constraint
    normal = np.asarray(normal, float)
    point = normal * float(bound)
    radius_sq = max_speed * max_speed - float(point @ point)
    if radius_sq < -1e-10:
        return None
    extent = math.sqrt(max(0.0, radius_sq))
    direction = np.array([-normal[1], normal[0]])
    lower, upper = -extent, extent
    for other_normal, other_bound in preceding:
        coefficient = float(np.asarray(other_normal, float) @ direction)
        required = float(other_bound) - float(np.asarray(other_normal, float) @ point)
        if abs(coefficient) <= 1e-12:
            if required > 1e-9:
                return None
            continue
        limit = required / coefficient
        if coefficient > 0.0:
            lower = max(lower, limit)
        else:
            upper = min(upper, limit)
        if lower > upper + 1e-10:
            return None
    target = float(direction @ (np.asarray(preferred, float)[:2] - point))
    return point + min(upper, max(lower, target)) * direction


def _linear_program(preferred, constraints, max_speed):
    """Canonical incremental 2-D linear program used by ORCA.

    Constraint order is part of the contract.  A failure returns the solution
    for the largest feasible prefix and reports the violated suffix; callers do
    not get an invented stop, turn, or braking behavior.
    """
    result = _clip_speed(preferred, max_speed)
    solved = 0
    for index, constraint in enumerate(constraints):
        normal, bound = constraint
        if float(np.asarray(normal, float) @ result) >= float(bound) - 1e-9:
            solved += 1
            continue
        candidate = _line_candidate(preferred, constraint, constraints[:index], max_speed)
        if candidate is None:
            violations = [
                max(0.0, float(b) - float(np.asarray(n, float) @ result))
                for n, b in constraints
            ]
            return OrcaSolution(
                velocity=result,
                feasible=False,
                active_constraints=solved,
                max_violation=max(violations, default=0.0),
            )
        result = candidate
        solved += 1
    violations = [
        max(0.0, float(b) - float(np.asarray(n, float) @ result))
        for n, b in constraints
    ]
    active = sum(
        abs(float(np.asarray(n, float) @ result) - float(b)) <= 1e-7
        for n, b in constraints
    )
    return OrcaSolution(
        velocity=result,
        feasible=True,
        active_constraints=active,
        max_violation=max(violations, default=0.0),
    )


def orca_solution(preferred_velocity, own_velocity, peers, *, time_horizon_s=3.0,
                  time_step_s=0.05, max_speed_mps=2.0):
    """Return the ORCA velocity together with explicit solver diagnostics."""
    constraints = [
        _orca_halfplane(own_velocity, peer, time_horizon_s, time_step_s)
        for peer in peers
    ]
    return _linear_program(preferred_velocity, constraints, float(max_speed_mps))


def orca_velocity(preferred_velocity, own_velocity, peers, *, time_horizon_s=3.0,
                  time_step_s=0.05, max_speed_mps=2.0):
    """Return the closest reciprocal collision-avoiding horizontal velocity."""
    return orca_solution(
        preferred_velocity,
        own_velocity,
        peers,
        time_horizon_s=time_horizon_s,
        time_step_s=time_step_s,
        max_speed_mps=max_speed_mps,
    ).velocity


def orca_effective_radius(protected_radius_m, uncertainty_radius_m,
                          range_rate_mps, response_time_s,
                          max_closing_speed_mps):
    """Protected geometry inflated by uncertainty and closing response distance."""
    closing = (
        max(0.0, min(float(max_closing_speed_mps), -float(range_rate_mps)))
        if math.isfinite(range_rate_mps) else 0.0
    )
    return (
        float(protected_radius_m)
        + float(uncertainty_radius_m)
        + closing * float(response_time_s)
    )
