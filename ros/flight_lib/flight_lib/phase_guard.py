"""Own-rate control barrier for phased circular flight."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhaseGuardSolution:
    rate: float
    feasible: bool
    active_constraints: int
    min_slack: float


def phase_guard_rate(own_position, own_radial, desired_rate, peer_positions,
                     peer_velocities, *, protected_distance_m=2.8, gamma=1.0,
                     min_rate_rad_s=0.0, max_rate_rad_s=1.0):
    """Clamp one drone's angular rate against its pairwise barriers.

    Peer velocity is measured, not chosen by this controller. Each drone thus
    solves only for the rate it can apply; it never assumes that another drone
    will apply a component of the same optimizer result.
    """
    own_position = np.asarray(own_position, float)
    own_radial = np.asarray(own_radial, float)
    peer_positions = np.asarray(peer_positions, float)
    peer_velocities = np.asarray(peer_velocities, float)
    if own_position.shape != (2,) or own_radial.shape != (2,):
        raise ValueError("own_position and own_radial must have shape (2,)")
    if peer_positions.ndim != 2 or peer_positions.shape[1:] != (2,):
        raise ValueError("peer_positions must have shape (n, 2)")
    if peer_velocities.shape != peer_positions.shape:
        raise ValueError("peer_velocities must match peer_positions")

    rows = []
    bounds = []
    tangent_per_radian = np.array([-own_radial[1], own_radial[0]])
    for peer_position, peer_velocity in zip(peer_positions, peer_velocities):
        separation = own_position - peer_position
        h = float(separation @ separation) - float(protected_distance_m) ** 2
        rows.append(2.0 * float(separation @ tangent_per_radian))
        bounds.append(
            -float(gamma) * h + 2.0 * float(separation @ peer_velocity)
        )

    lower = float(min_rate_rad_s)
    upper = float(max_rate_rad_s)
    feasible = lower <= upper
    for row, bound in zip(rows, bounds):
        if abs(row) <= 1e-9:
            feasible = feasible and bound <= 0.0
        elif row > 0.0:
            lower = max(lower, bound / row)
        else:
            upper = min(upper, bound / row)
    feasible = feasible and lower <= upper

    if feasible:
        rate = float(np.clip(desired_rate, lower, upper))
    else:
        rate = float(np.clip(desired_rate, min_rate_rad_s, max_rate_rad_s))

    slacks = [
        row * rate - bound for row, bound in zip(rows, bounds)
    ] + [rate - min_rate_rad_s, max_rate_rad_s - rate]
    min_slack = min(slacks)
    return PhaseGuardSolution(
        rate=rate,
        feasible=feasible,
        active_constraints=sum(abs(slack) <= 1e-7 for slack in slacks),
        min_slack=float(min_slack),
    )
