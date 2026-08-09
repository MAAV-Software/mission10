import math

import pytest

from flight_lib.survey import (
    build_survey_schedule,
    cube_waypoints,
    hover,
    polyline_schedule,
    schedule_duration,
    schedule_setpoint,
    serpentine_waypoints,
)


def kwargs(**over):
    base = dict(
        field_e0_m=2.0, field_n0_m=3.0, lane_axis_deg=0.0,
        field_length_m=10.0, field_width_m=6.0, lane_spacing_m=2.0,
        altitude_m=4.0, speed_mps=1.5, revisit_gap_s=8.0,
        crosshatch=True, cube_side_m=2.0, cube_speed_mps=1.0,
    )
    base.update(over)
    return base


def test_serpentine_covers_rectangle():
    pts = serpentine_waypoints(0.0, 0.0, 0.0, 10.0, 6.0, 2.0, 4.0)
    es = [p[0] for p in pts]
    ns = [p[1] for p in pts]
    assert min(es) == pytest.approx(0.0) and max(es) == pytest.approx(10.0)
    assert min(ns) == pytest.approx(0.0) and max(ns) == pytest.approx(6.0)
    assert len(pts) == 8  # 4 lanes at 2 m over 6 m width
    assert all(p[2] == pytest.approx(4.0) for p in pts)
    # alternating lane direction: consecutive lane legs reverse east-heading
    assert pts[1][0] > pts[0][0] and pts[3][0] < pts[2][0]


def test_serpentine_axis_rotates():
    pts = serpentine_waypoints(0.0, 0.0, 90.0, 10.0, 6.0, 2.0, 4.0)
    # lanes now run north, stepping west (left of north)
    assert pts[1][1] - pts[0][1] == pytest.approx(10.0)
    assert pts[2][0] - pts[0][0] == pytest.approx(-2.0)


def test_crosshatch_covers_same_rectangle():
    a = serpentine_waypoints(0.0, 0.0, 0.0, 10.0, 6.0, 2.0, 4.0)
    b = serpentine_waypoints(0.0, 0.0, 0.0, 10.0, 6.0, 2.0, 4.0, cross=True)
    for pts in (a, b):
        assert min(p[0] for p in pts) == pytest.approx(0.0)
        assert max(p[0] for p in pts) == pytest.approx(10.0)
        assert min(p[1] for p in pts) == pytest.approx(0.0)
        assert max(p[1] for p in pts) == pytest.approx(6.0)
    # b's lanes run along north (the cross axis)
    assert b[1][1] - b[0][1] == pytest.approx(6.0)


def test_cube_edges_are_single_axis():
    pts = cube_waypoints(1.0, -1.0, 2.0, 4.0)
    assert min(p[2] for p in pts) == pytest.approx(4.0)  # never below base
    assert max(p[2] for p in pts) == pytest.approx(6.0)
    moved_axes = set()
    for p, q in zip(pts, pts[1:]):
        deltas = [abs(q[i] - p[i]) for i in range(3)]
        axes = [i for i, d in enumerate(deltas) if d > 1e-9]
        assert len(axes) == 1  # every leg is a pure single-axis translation
        moved_axes.update(axes)
    assert moved_axes == {0, 1, 2}


def test_schedule_positions_continuous_and_speed_capped():
    sched = build_survey_schedule(**kwargs())
    total = schedule_duration(sched)
    dt = 0.05
    prev, _, _ = schedule_setpoint(sched, 0.0)
    vmax = 0.0
    for k in range(1, int(total / dt) + 2):
        pos, _, _ = schedule_setpoint(sched, k * dt)
        vmax = max(vmax, math.dist(prev, pos) / dt)
        prev = pos
    assert vmax <= 1.5 + 0.01  # smoothstep peak equals the requested speed


def test_schedule_starts_and_ends_home():
    sched = build_survey_schedule(**kwargs())
    start, _, done0 = schedule_setpoint(sched, 0.0)
    end, _, done1 = schedule_setpoint(sched, schedule_duration(sched) + 1.0)
    assert start == pytest.approx((0.0, 0.0, 4.0))
    assert end == pytest.approx((0.0, 0.0, 4.0))
    assert not done0 and done1


def test_gap_hover_present_between_passes():
    sched = build_survey_schedule(**kwargs())
    gaps = [s for s in sched if s.label == "regap" and s.p0 == s.p1]
    assert len(gaps) == 1 and gaps[0].duration_s == pytest.approx(8.0)
    # hover sits at the field entry corner
    assert gaps[0].p0 == pytest.approx((2.0, 3.0, 4.0))


def test_optional_parts_removable():
    sched = build_survey_schedule(**kwargs(crosshatch=False, cube_side_m=0.0))
    labels = {s.label for s in sched}
    assert "crosshatch" not in labels and "cube" not in labels and "regap" not in labels
    end, _, _ = schedule_setpoint(sched, schedule_duration(sched) + 1.0)
    assert end == pytest.approx((0.0, 0.0, 4.0))


def test_rotate_schedule_turns_field_with_nose():
    from flight_lib.survey import rotate_schedule

    sched = build_survey_schedule(**kwargs())
    rot = rotate_schedule(sched, math.pi / 2.0)  # nose pointing north
    # the first serpentine vertex (field entry, pre-rotation east-north) maps
    # east->north, north->west; altitude and timing untouched
    p = sched[0].p1
    q = rot[0].p1
    assert q[0] == pytest.approx(-p[1])
    assert q[1] == pytest.approx(p[0])
    assert q[2] == pytest.approx(p[2])
    assert rot[0].duration_s == pytest.approx(sched[0].duration_s)
    # home stays home
    end, _, _ = schedule_setpoint(rot, schedule_duration(rot) + 1.0)
    assert end == pytest.approx((0.0, 0.0, 4.0))


def test_direction_held_through_hover():
    sched = [
        polyline_schedule([(0, 0, 4), (5, 0, 4)], 1.0, "a")[0],
        hover((5, 0, 4), 3.0),
        polyline_schedule([(5, 0, 4), (5, 5, 4)], 1.0, "b")[0],
    ]
    t_hover = sched[0].duration_s + 1.0
    _, direction, _ = schedule_setpoint(sched, t_hover)
    assert direction == pytest.approx((1.0, 0.0))  # held from the previous leg
