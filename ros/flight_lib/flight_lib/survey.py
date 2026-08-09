"""Survey path geometry — serpentine lanes, cross-hatch, and a cube circuit.

Pure math, no ROS. A survey is a schedule of straight segments; the mission
node walks the schedule at a fixed tick and applies a smoothstep inside each
segment, so position and velocity are continuous everywhere and the vehicle
comes to rest at every vertex. Segment durations are chosen so the smoothstep
peak speed (1.5 * length / duration) equals the requested speed.

All points are launch-relative ENU (east, north, up) — the same frame the
phased-orbits choreography uses; the mission node adds its post-climb anchor.
"""
from __future__ import annotations

import math
from typing import List, NamedTuple, Sequence, Tuple

Point = Tuple[float, float, float]


class Segment(NamedTuple):
    p0: Point
    p1: Point
    duration_s: float
    label: str


def _seg(p0: Point, p1: Point, speed: float, label: str) -> Segment:
    length = math.dist(p0, p1)
    # smoothstep peak speed is 1.5 * length / duration
    duration = 1.5 * length / speed if length > 1e-9 else 0.0
    return Segment(p0, p1, duration, label)


def hover(p: Point, duration_s: float, label: str = "hover") -> Segment:
    return Segment(p, p, float(duration_s), label)


def serpentine_waypoints(
    origin_e: float,
    origin_n: float,
    along_deg: float,
    length_m: float,
    width_m: float,
    spacing_m: float,
    altitude_m: float,
    cross: bool = False,
) -> List[Point]:
    """Lane-end vertices of a serpentine over the rectangle spanned by
    ``length_m`` along ``along_deg`` (ENU heading, 0 = east) and ``width_m``
    to its left. ``cross=True`` flies the same rectangle with the roles of
    the two axes swapped (the cross-hatch pass)."""
    if spacing_m <= 0.0:
        raise ValueError("lane spacing must be positive")
    a = math.radians(along_deg)
    u = (math.cos(a), math.sin(a))          # along-lane
    v = (-math.sin(a), math.cos(a))         # across-lane (left)
    if cross:
        u, v = v, u
        length_m, width_m = width_m, length_m

    n_lanes = max(2, int(math.floor(width_m / spacing_m)) + 1)
    step = width_m / (n_lanes - 1)
    pts: List[Point] = []
    for i in range(n_lanes):
        base_e = origin_e + v[0] * step * i
        base_n = origin_n + v[1] * step * i
        near = (base_e, base_n, altitude_m)
        far = (base_e + u[0] * length_m, base_n + u[1] * length_m, altitude_m)
        pts.extend([near, far] if i % 2 == 0 else [far, near])
    return pts


def cube_waypoints(
    center_e: float,
    center_n: float,
    side_m: float,
    base_altitude_m: float,
) -> List[Point]:
    """Edge circuit of a cube sitting ON the survey altitude (z spans
    [base, base+side], never below it): bottom loop, rise, top loop, descend.
    Every leg is a single-axis translation — the point of the exercise."""
    h = side_m / 2.0
    lo, hi = base_altitude_m, base_altitude_m + side_m
    corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
    bottom = [(center_e + e, center_n + n, lo) for e, n in corners]
    top = [(center_e + e, center_n + n, hi) for e, n in corners]
    return bottom + [bottom[0], top[0]] + top[1:] + [top[0], bottom[0]]


def polyline_schedule(
    pts: Sequence[Point], speed: float, label: str
) -> List[Segment]:
    return [
        _seg(pts[i], pts[i + 1], speed, label)
        for i in range(len(pts) - 1)
        if math.dist(pts[i], pts[i + 1]) > 1e-9
    ]


def schedule_duration(schedule: Sequence[Segment]) -> float:
    return sum(s.duration_s for s in schedule)


def schedule_setpoint(
    schedule: Sequence[Segment], t: float
) -> Tuple[Point, Tuple[float, float], bool]:
    """(position, horizontal direction of travel, done) at time ``t``.

    Position smoothsteps inside the active segment. The direction is the
    active segment's horizontal unit vector, held through hovers and zero-
    length tails (the caller's yaw slew wants a stable target, not a jump
    to some default heading mid-pause); it is (0, 0) only before the first
    moving segment.
    """
    direction = (0.0, 0.0)
    if not schedule:
        return (0.0, 0.0, 0.0), direction, True
    t = max(0.0, t)
    for seg in schedule:
        de, dn = seg.p1[0] - seg.p0[0], seg.p1[1] - seg.p0[1]
        horiz = math.hypot(de, dn)
        if horiz > 1e-9:
            new_dir = (de / horiz, dn / horiz)
        else:
            new_dir = direction
        if t <= seg.duration_s:
            u = t / seg.duration_s if seg.duration_s > 0.0 else 1.0
            s = u * u * (3.0 - 2.0 * u)
            pos = (
                seg.p0[0] + (seg.p1[0] - seg.p0[0]) * s,
                seg.p0[1] + (seg.p1[1] - seg.p0[1]) * s,
                seg.p0[2] + (seg.p1[2] - seg.p0[2]) * s,
            )
            return pos, (new_dir if horiz > 1e-9 else direction), False
        t -= seg.duration_s
        direction = new_dir
    return schedule[-1].p1, direction, True


def rotate_schedule(
    schedule: Sequence[Segment], theta_rad: float
) -> List[Segment]:
    """Rotate a schedule about the origin in the horizontal plane. Used to
    align the whole survey with the operator's pre-arm heading, so 'lane
    axis 0' means 'straight out the nose' instead of 'geographic east'."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)

    def rot(p: Point) -> Point:
        return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])

    return [Segment(rot(g.p0), rot(g.p1), g.duration_s, g.label) for g in schedule]


def build_survey_schedule(
    field_e0_m: float,
    field_n0_m: float,
    lane_axis_deg: float,
    field_length_m: float,
    field_width_m: float,
    lane_spacing_m: float,
    altitude_m: float,
    speed_mps: float,
    revisit_gap_s: float,
    crosshatch: bool,
    cube_side_m: float,
    cube_speed_mps: float,
) -> List[Segment]:
    """The full mission: serpentine, gap hover at the field entry, optional
    cross-hatch serpentine, optional cube circuit over the launch anchor,
    then back to the anchor. Starts and ends at (0, 0, altitude)."""
    home: Point = (0.0, 0.0, altitude_m)
    sched: List[Segment] = []

    pass1 = serpentine_waypoints(
        field_e0_m, field_n0_m, lane_axis_deg,
        field_length_m, field_width_m, lane_spacing_m, altitude_m)
    sched += polyline_schedule([home] + pass1, speed_mps, "serpentine")

    if crosshatch:
        entry: Point = (field_e0_m, field_n0_m, altitude_m)
        sched += polyline_schedule([sched[-1].p1, entry], speed_mps, "regap")
        sched.append(hover(entry, revisit_gap_s, "regap"))
        pass2 = serpentine_waypoints(
            field_e0_m, field_n0_m, lane_axis_deg,
            field_length_m, field_width_m, lane_spacing_m, altitude_m,
            cross=True)
        sched += polyline_schedule([entry] + pass2, speed_mps, "crosshatch")

    if cube_side_m > 0.0:
        cube = cube_waypoints(0.0, 0.0, cube_side_m, altitude_m)
        sched += polyline_schedule([sched[-1].p1] + cube, cube_speed_mps, "cube")

    sched += polyline_schedule([sched[-1].p1, home], speed_mps, "home")
    return sched
