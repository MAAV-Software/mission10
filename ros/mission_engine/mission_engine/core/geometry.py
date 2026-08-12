"""NED geometry: quaternion helpers + serpentine survey lanes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (w, x, y, z)
Point2 = Tuple[float, float]


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


def point_in_polygon(point: Point2, polygon: Sequence[Point2]) -> bool:
    """True for points inside or on the edge of a simple N/E polygon."""
    if len(polygon) < 3:
        return False
    n, e = point
    inside = False
    for i, (n1, e1) in enumerate(polygon):
        n2, e2 = polygon[(i + 1) % len(polygon)]
        dn, de = n2 - n1, e2 - e1
        cross = (n - n1) * de - (e - e1) * dn
        if abs(cross) <= 1e-8 and min(n1, n2) - 1e-8 <= n <= max(n1, n2) + 1e-8 \
                and min(e1, e2) - 1e-8 <= e <= max(e1, e2) + 1e-8:
            return True
        if (e1 > e) != (e2 > e):
            crossing_n = n1 + (n2 - n1) * (e - e1) / (e2 - e1)
            if n < crossing_n:
                inside = not inside
    return inside


def distance_to_polygon(point: Point2, polygon: Sequence[Point2]) -> float:
    """Shortest horizontal distance to a polygon boundary."""
    n, e = point
    best = math.inf
    for i, (n1, e1) in enumerate(polygon):
        n2, e2 = polygon[(i + 1) % len(polygon)]
        dn, de = n2 - n1, e2 - e1
        denom = dn * dn + de * de
        u = 0.0 if denom == 0.0 else max(0.0, min(1.0, ((n - n1) * dn + (e - e1) * de) / denom))
        best = min(best, math.hypot(n - (n1 + u * dn), e - (e1 + u * de)))
    return best


def polygon_serpentine(polygon: Sequence[Point2], spacing: float) -> List[Lane]:
    """Clip parallel survey lanes to a convex polygon.

    The first polygon edge selects the lane direction. Corners may be clockwise
    or counter-clockwise, but must follow the perimeter.
    """
    if len(polygon) < 3 or spacing <= 0.0:
        raise ValueError("polygon needs at least three corners and positive spacing")
    n0, e0 = polygon[0]
    dn, de = polygon[1][0] - n0, polygon[1][1] - e0
    edge_len = math.hypot(dn, de)
    if edge_len <= 1e-6:
        raise ValueError("the first two polygon corners must be distinct")
    fwd = (dn / edge_len, de / edge_len)
    right = (-fwd[1], fwd[0])
    along = [(n * fwd[0] + e * fwd[1]) for n, e in polygon]
    across = [(n * right[0] + e * right[1]) for n, e in polygon]
    lo, hi = min(across), max(across)
    count = max(2, int(math.ceil((hi - lo) / spacing)) + 1)
    offsets = [lo + (hi - lo) * i / (count - 1) for i in range(count)]
    lanes: List[Lane] = []
    for offset in offsets:
        hits: List[float] = []
        for i, p1 in enumerate(polygon):
            p2 = polygon[(i + 1) % len(polygon)]
            a1 = p1[0] * right[0] + p1[1] * right[1]
            a2 = p2[0] * right[0] + p2[1] * right[1]
            if abs(a2 - a1) <= 1e-9:
                if abs(offset - a1) <= 1e-8:
                    hits.extend([
                        p1[0] * fwd[0] + p1[1] * fwd[1],
                        p2[0] * fwd[0] + p2[1] * fwd[1],
                    ])
                continue
            u = (offset - a1) / (a2 - a1)
            if -1e-9 <= u <= 1.0 + 1e-9:
                n = p1[0] + u * (p2[0] - p1[0])
                e = p1[1] + u * (p2[1] - p1[1])
                hits.append(n * fwd[0] + e * fwd[1])
        if len(hits) < 2:
            continue
        start_s, end_s = min(hits), max(hits)
        if end_s - start_s <= 1e-6:
            continue
        a, b = ((start_s, end_s) if len(lanes) % 2 == 0 else (end_s, start_s))
        start = (a * fwd[0] + offset * right[0], a * fwd[1] + offset * right[1])
        heading = math.atan2((b - a) * fwd[1], (b - a) * fwd[0])
        lanes.append(Lane(len(lanes), start, heading, abs(b - a)))
    if not lanes:
        raise ValueError("polygon produced no survey lanes")
    return lanes
