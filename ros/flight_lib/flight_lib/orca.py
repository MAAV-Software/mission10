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


def _project_halfplanes(preferred, constraints, max_speed, iterations=80):
    """Nearest point in half-planes plus a speed disc via Dykstra projection."""
    x = np.asarray(preferred, float)[:2].copy()
    sets = [(np.asarray(normal, float), float(bound)) for normal, bound in constraints]
    corrections = [np.zeros(2) for _ in range(len(sets) + 1)]
    for _ in range(iterations):
        previous = x.copy()
        for index, (normal, bound) in enumerate(sets):
            candidate = x + corrections[index]
            shortfall = bound - float(normal @ candidate)
            projected = candidate + max(0.0, shortfall) * normal
            corrections[index] = candidate - projected
            x = projected
        candidate = x + corrections[-1]
        speed = float(np.linalg.norm(candidate))
        projected = candidate if speed <= max_speed else candidate * (max_speed / speed)
        corrections[-1] = candidate - projected
        x = projected
        if float(np.linalg.norm(x - previous)) < 1e-8:
            break
    return x


def orca_velocity(preferred_velocity, own_velocity, peers, *, time_horizon_s=3.0,
                  time_step_s=0.05, max_speed_mps=2.0):
    """Return the closest reciprocal collision-avoiding horizontal velocity."""
    constraints = [
        _orca_halfplane(own_velocity, peer, time_horizon_s, time_step_s)
        for peer in peers
    ]
    return _project_halfplanes(preferred_velocity, constraints, float(max_speed_mps))


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
