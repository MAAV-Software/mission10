"""NED geometry: quaternion helpers + serpentine survey lanes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (w, x, y, z)


def quat_from_yaw(yaw: float) -> Quat:
    h = yaw / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def yaw_of(q: Quat) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_conj(q: Quat) -> Quat:
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    """Rotate v by q (for body->NED q, takes body vectors to NED)."""
    w, x, y, z = q
    # t = 2 * (q_vec x v); v' = v + w*t + q_vec x t
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + y * tz - z * ty,
        v[1] + w * ty + z * tx - x * tz,
        v[2] + w * tz + x * ty - y * tx,
    )


@dataclass(frozen=True)
class Lane:
    index: int
    start: Tuple[float, float]  # (north, east)
    heading: float  # 0 = north; odd lanes run back at pi
    length: float

    def point_at(self, s: float) -> Tuple[float, float]:
        return (
            self.start[0] + s * math.cos(self.heading),
            self.start[1] + s * math.sin(self.heading),
        )


def serpentine(
    origin: Tuple[float, float],
    length: float,
    n_lanes: int,
    lane_spacing: float,
) -> List[Lane]:
    """Boustrophedon lanes: run north-south along `length`, offset east by
    `lane_spacing` per lane, direction alternating so lane ends adjoin.
    `origin` is the south-west corner of the lane pattern."""
    if n_lanes < 1:
        raise ValueError(f"n_lanes must be >= 1, got {n_lanes}")
    if length <= 0.0 or lane_spacing <= 0.0:
        raise ValueError(f"length/lane_spacing must be positive, got {length}/{lane_spacing}")
    lanes: List[Lane] = []
    for i in range(n_lanes):
        east = origin[1] + i * lane_spacing
        if i % 2 == 0:
            lanes.append(Lane(index=i, start=(origin[0], east), heading=0.0, length=length))
        else:
            lanes.append(
                Lane(index=i, start=(origin[0] + length, east), heading=math.pi, length=length)
            )
    return lanes
