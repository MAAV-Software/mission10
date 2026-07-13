"""Ring-orbit geometry for evenly-phased multi-drone rings.

Pure math: numpy in, setpoints out. No ROS, no PX4 — the offboard node is
responsible for mapping these into a PX4 TrajectorySetpoint (NED).

Frame convention
----------------
All positions are in a right-handed, z-up world frame (ENU-compatible:
x east, y north, z up; altitude is positive-up). Yaw is measured CCW about
+z from the +x axis, in radians, wrapped to (-pi, pi].

PX4 is NED (x north, y east, z down; yaw CW from north). The consuming node
converts, e.g.::

    ned_pos = (enu_y, enu_x, -enu_z)
    ned_yaw = wrap_angle(pi / 2 - enu_yaw)

(The uXRCE-DDS bridge does NOT auto-convert frames the way MAVROS did, so the
node must do this explicitly.)

Shared ring
-----------
`n` drones share one circle of radius `R` about `center`, spaced by a constant
phase 2*pi*i/n, all rotating at angular rate `omega` (rad/s). Because the
spacing is uniform, the closest pair is always adjacent, so the minimum
inter-drone separation is the adjacent chord::

    d_min = 2 * R * sin(pi / n)

which inverts to size a ring for a required clearance::

    R = d_min / (2 * sin(pi / n))
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wrap_angle",
    "min_separation",
    "radius_for_separation",
    "orbit_setpoint",
    "ring_positions",
]

YAW_MODES = ("tangent", "inward", "outward", "fixed")


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def min_separation(n, radius):
    """Minimum inter-drone separation on an evenly-phased ring (adjacent chord)."""
    if n < 2:
        raise ValueError("need at least 2 drones for a separation")
    return 2.0 * radius * np.sin(np.pi / n)


def radius_for_separation(n, d_min):
    """Smallest orbit radius giving adjacent separation >= d_min."""
    if n < 2:
        raise ValueError("need at least 2 drones for a separation")
    return d_min / (2.0 * np.sin(np.pi / n))


def _yaw_for(theta, yaw_mode, fixed_yaw):
    if yaw_mode == "tangent":  # face the direction of travel (CCW)
        return wrap_angle(theta + np.pi / 2.0)
    if yaw_mode == "inward":  # face the orbit center
        return wrap_angle(theta + np.pi)
    if yaw_mode == "outward":  # face radially out
        return wrap_angle(theta)
    if yaw_mode == "fixed":
        return wrap_angle(fixed_yaw)
    raise ValueError(f"unknown yaw_mode {yaw_mode!r}; expected one of {YAW_MODES}")


def orbit_setpoint(t, index, n, radius, omega, *,
                   center=(0.0, 0.0), altitude=0.0,
                   yaw_mode="inward", fixed_yaw=0.0, phase0=0.0):
    """Position and yaw for drone `index` of `n` at time `t`.

    Returns ``(position, yaw)`` where ``position`` is an (x, y, z) numpy array
    in the z-up world frame and ``yaw`` is a scalar in (-pi, pi].
    """
    if not 0 <= index < n:
        raise ValueError(f"index {index} out of range for n={n}")
    phase = phase0 + 2.0 * np.pi * index / n
    theta = omega * t + phase
    cx, cy = center
    pos = np.array([
        cx + radius * np.cos(theta),
        cy + radius * np.sin(theta),
        altitude,
    ])
    return pos, float(_yaw_for(theta, yaw_mode, fixed_yaw))


def ring_positions(t, n, radius, omega, *, center=(0.0, 0.0), altitude=0.0, phase0=0.0):
    """All `n` drone positions at time `t` as an (n, 3) array (for viz / checks)."""
    i = np.arange(n)
    theta = omega * t + phase0 + 2.0 * np.pi * i / n
    cx, cy = center
    return np.column_stack([
        cx + radius * np.cos(theta),
        cy + radius * np.sin(theta),
        np.full(n, float(altitude)),
    ])
