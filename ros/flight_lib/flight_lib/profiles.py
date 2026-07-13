"""Vertical hover / land profiles over a fixed anchor point.

Same z-up world frame as :mod:`flight_lib.orbit`. The land profile ramps
altitude down at a constant rate to the anchor; intended to sit on top of the
AprilTag EV anchor so the descent stays sub-meter accurate.
"""

from __future__ import annotations

import numpy as np

__all__ = ["hover_setpoint", "land_setpoint", "land_duration"]


def hover_setpoint(center, altitude, yaw=0.0):
    """Static hold at (center_x, center_y, altitude)."""
    cx, cy = center
    return np.array([cx, cy, float(altitude)]), float(yaw)


def land_duration(start_alt, descent_rate, ground=0.0):
    """Time (s) to descend from `start_alt` to `ground` at `descent_rate`."""
    if descent_rate <= 0:
        raise ValueError("descent_rate must be positive")
    return max(0.0, (start_alt - ground) / descent_rate)


def land_setpoint(t, center, start_alt, descent_rate, *, ground=0.0, yaw=0.0):
    """Position+yaw of a constant-rate descent over `center`, clamped at ground."""
    if descent_rate <= 0:
        raise ValueError("descent_rate must be positive")
    cx, cy = center
    z = max(float(ground), start_alt - descent_rate * t)
    return np.array([cx, cy, z]), float(yaw)
