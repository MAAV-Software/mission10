"""Pure phase-loop and in-plane reflex primitives for pairwise deconfliction."""
from __future__ import annotations

import math

import numpy as np

from flight_lib.orbit import wrap_angle


def closest_point_of_approach(relative_position, relative_velocity):
    """Return ``(t_cpa, d_cpa)`` for relative ENU position and velocity."""
    p = np.asarray(relative_position, dtype=float)
    v = np.asarray(relative_velocity, dtype=float)
    speed_sq = float(np.dot(v, v))
    if speed_sq <= 1e-9:
        return 0.0, float(np.linalg.norm(p))
    t_cpa = max(0.0, -float(np.dot(p, v)) / speed_sq)
    return t_cpa, float(np.linalg.norm(p + t_cpa * v))


def follower_phase_rate(phase, peer_phase, peer_rate, desired_offset, desired_offset_rate,
                        nominal_rate, kp=0.8, min_scale=0.7, max_scale=1.3):
    """Asymmetric L1 controller for the follower's phase rate and phase error."""
    error = wrap_angle(phase - peer_phase - desired_offset)
    rate = peer_rate + desired_offset_rate - kp * error
    return float(np.clip(rate, min_scale * nominal_rate, max_scale * nominal_rate)), error


def phase_deconflict_rate(own_phase, peer_phase, nominal_rate, t_cpa, d_cpa, vehicle_id,
                          peer_id, *, clearance_m=2.5, horizon_s=3.5, max_scale=1.5,
                          min_scale=0.5):
    """Reciprocal along-track deconfliction: bias own phase rate to widen the angular gap
    when a close approach is predicted. Ahead-in-phase speeds up, behind slows down. Both
    drones compute the same split independently. Returns ``(rate, severity)``."""
    if t_cpa > horizon_s or d_cpa >= clearance_m:
        return float(nominal_rate), 0.0
    severity = (clearance_m - d_cpa) / clearance_m * (1.0 - t_cpa / horizon_s)
    severity = float(np.clip(severity, 0.0, 1.0))
    gap = wrap_angle(own_phase - peer_phase)
    ahead = gap > 0.0 if abs(gap) > 1e-6 else vehicle_id < peer_id
    if ahead:
        rate = nominal_rate * (1.0 + (max_scale - 1.0) * severity)
    else:
        rate = nominal_rate * (1.0 - (1.0 - min_scale) * severity)
    rate = float(np.clip(rate, min_scale * nominal_rate, max_scale * nominal_rate))
    return rate, severity


def reflex_velocity(nominal_velocity, own_position, peer_position, range_m, *,
                    d_reflex=2.0, slow_scale=0.6, right_veer_mps=0.5, repel_gain=0.75,
                    max_repel_mps=0.75):
    """L2 velocity: slowdown, own-right veer, and range-scaled raw-vector repel."""
    nominal = np.asarray(nominal_velocity, dtype=float)
    own = np.asarray(own_position, dtype=float)
    peer = np.asarray(peer_position, dtype=float)
    output = nominal * slow_scale
    horizontal_speed = float(np.linalg.norm(nominal[:2]))
    if horizontal_speed > 1e-6:
        right = np.array([nominal[1], -nominal[0], 0.0]) / horizontal_speed
        output += right * right_veer_mps
    away = own - peer
    away[2] = 0.0
    norm = float(np.linalg.norm(away))
    if norm > 1e-6:
        output += away / norm * min(max_repel_mps, repel_gain * max(0.0, d_reflex - range_m))
    return output
