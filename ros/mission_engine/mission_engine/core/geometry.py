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
    heading_rad: float = 0.0,
) -> List[Lane]:
    """Boustrophedon lanes: run along `heading_rad` for `length`, offset 90°
    right of it by `lane_spacing` per lane, direction alternating so lane ends
    adjoin. `origin` is the corner the first lane starts from. The default
    heading runs the lanes north and steps them east."""
    if n_lanes < 1:
        raise ValueError(f"n_lanes must be >= 1, got {n_lanes}")
    if length <= 0.0 or lane_spacing <= 0.0:
        raise ValueError(f"length/lane_spacing must be positive, got {length}/{lane_spacing}")
    fwd = (math.cos(heading_rad), math.sin(heading_rad))
    right = (-fwd[1], fwd[0])  # 90° clockwise in NED; east when heading is north
    lanes: List[Lane] = []
    for i in range(n_lanes):
        base = (
            origin[0] + i * lane_spacing * right[0],
            origin[1] + i * lane_spacing * right[1],
        )
        if i % 2 == 0:
            lanes.append(Lane(index=i, start=base, heading=heading_rad, length=length))
        else:
            lanes.append(
                Lane(
                    index=i,
                    start=(base[0] + length * fwd[0], base[1] + length * fwd[1]),
                    heading=heading_rad + math.pi,
                    length=length,
                )
            )
    return lanes
