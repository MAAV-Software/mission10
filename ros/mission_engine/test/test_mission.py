"""Closed-loop mission test: engine + kinematic fake + scripted minefield
through the real ingest/back-projection wire shapes."""

import math
import unittest
from dataclasses import replace

from mission_engine.core.config import CameraModel
from mission_engine.core.dumpproto import build_payload, decode_frame, encode_frame
from mission_engine.core.geometry import inset_convex_polygon, point_in_polygon, polygon_serpentine, quat_from_yaw
from mission_engine.core.ingest import PoseHistory, PoseSnapshot, make_observation
from mission_engine.core.minelog import DIPPED, MineLog
from mission_engine.core.mission import (
    ABORT,
    DONE,
    DUMP,
    EGRESS,
    LANE,
    MissionConfig,
    MissionEngine,
)

from .fake import KinematicPoint, ScriptedMinefield

CAM = CameraModel()
HOME_LL = (42.2944, -83.7105)

# lanes 4 m apart so adjacent-lane footprints overlap at 6 m AGL
CFG = MissionConfig(
    lanes_origin=(0.0, 0.0),
    lane_length=16.0,
    n_lanes=3,
    lane_spacing=4.0,
    survey_alt_m=6.0,
    lane_speed=2.0,
    egress_ne=(0.0, 0.0),
    max_dips=2,
)

MINES = [(6.0, 2.0), (12.0, 6.0)]  # between lanes -> seen on two passes


def run_mission(
    cfg=CFG, mines=MINES, sim_s=600.0, dt=0.05, abort_at=None, guard_abort_at=None
):
    log = MineLog(confirm_obs=5, confirm_passes=2)
    eng = MissionEngine(cfg, log)
    body = KinematicPoint(pos=(0.0, 0.0, 0.0))
    field = ScriptedMinefield(CAM, mines)
    hist = PoseHistory()
    eng.start()
    t, last_det = 0.0, 0.0
    while t < sim_s and eng.phase not in (DONE,):
        yaw = eng._last_yaw
        q = quat_from_yaw(yaw)
        snap = PoseSnapshot(
            t=t,
            pos=body.pos,
            q=q,
            ll=(
                HOME_LL[0] + body.pos[0] / 111_320.0,
                HOME_LL[1] + body.pos[1] / (111_320.0 * 0.739),
            ),
        )
        hist.append(snap)
        if abort_at is not None and t >= abort_at:
            eng.operator_abort()
            abort_at = None
        if guard_abort_at is not None and t >= guard_abort_at:
            eng.request_abort("tag anchor disagrees with flight layer by 5.40 m")
            guard_abort_at = None
        if t - last_det >= 0.1:  # 10 Hz detector
            last_det = t
            eng.note_detector_alive(t)
            if -body.pos[2] > 1.0:
                for center, conf in field.detect(body.pos, q):
                    o = make_observation(CAM, hist.nearest(t), t, center, conf, "mine")
                    if o is not None:
                        log.ingest(o)
        sp = eng.tick(t, body.pos)
        if eng.phase == DUMP and not eng.dump_acked:
            payload = build_payload(
                log, "drone2", "test", eng.t_takeoff or 0.0, t,
                coverage=eng.coverage_report(), stats=eng.stats(),
            )
            decoded, _ = decode_frame(encode_frame(payload))
            eng.notify_dump_result(decoded["schema"] == "minefield-dump/1")
        body.step(dt, sp)
        t += dt
    return eng, log, t


class TestClosedLoopMission(unittest.TestCase):
    def setUp(self):
        self.eng, self.log, self.t_end = run_mission()

    def test_mission_completes(self):
        self.assertEqual(self.eng.phase, DONE)
        self.assertLess(self.t_end, 300.0)

    def test_all_mines_found_within_gate(self):
        for mine in MINES:
            d = min(
                ((c.centroid[0] - mine[0]) ** 2 + (c.centroid[1] - mine[1]) ** 2) ** 0.5
                for c in self.log.clusters
            )
            self.assertLess(d, 0.5, f"mine {mine} not localized (best {d:.2f} m)")

    def test_no_spurious_clusters(self):
        self.assertEqual(len(self.log.clusters), len(MINES))

    def test_dips_ran_and_marked(self):
        self.assertGreaterEqual(self.eng.dips_done, 1)
        self.assertTrue(any(c.status == DIPPED for c in self.log.clusters))

    def test_coverage_no_gaps(self):
        cov = self.eng.coverage_report()
        self.assertEqual(cov["gaps"], [], "lane coverage has gaps")
        self.assertGreaterEqual(len(cov["lanes"]), CFG.n_lanes)

    def test_multi_pass_observation(self):
        self.assertTrue(any(c.n_passes >= 2 for c in self.log.clusters))


class TestAbort(unittest.TestCase):
    def test_operator_abort_lands_without_dump(self):
        eng, log, _ = run_mission(abort_at=20.0)
        self.assertEqual(eng.phase, DONE)
        self.assertEqual(eng.abort_reason, "operator")
        self.assertFalse(eng.dump_acked)

    def test_guard_abort_lands_without_dump(self):
        """The rim's guards (tag anchor, link health) reach the same exit."""
        eng, _, _ = run_mission(abort_at=None, guard_abort_at=20.0)
        self.assertEqual(eng.phase, DONE)
        self.assertIn("anchor", eng.abort_reason)
        self.assertFalse(eng.dump_acked)


class TestEnvelope(unittest.TestCase):
    """The envelope guards the deployed survey did not have on 2026-07-24
    (wall-impact REPORT.md, contributing conditions 1 and 3)."""

    def test_a_field_outside_the_fence_never_flies(self):
        cfg = replace(CFG, fence_radius_m=6.0)  # lanes reach 16 m
        eng, _, _ = run_mission(cfg=cfg)
        self.assertEqual(eng.phase, DONE)
        self.assertIn("fence", eng.abort_reason)

    def test_the_fence_catches_the_command_before_the_estimate(self):
        """The command leaves the envelope first, so the abort names it."""
        cfg = replace(CFG, fence_radius_m=6.0)
        eng, _, _ = run_mission(cfg=cfg)
        self.assertIn("command", eng.abort_reason)

    def test_a_survey_that_cannot_finish_comes_home(self):
        cfg = replace(CFG, mission_timeout_s=30.0)
        eng, _, t_end = run_mission(cfg=cfg)
        self.assertEqual(eng.phase, DONE)
        self.assertIn("timeout", eng.abort_reason)
        self.assertLess(t_end, 90.0)

    def test_an_envelope_wide_enough_does_not_fire(self):
        cfg = replace(CFG, fence_radius_m=40.0, mission_timeout_s=600.0)
        eng, _, _ = run_mission(cfg=cfg)
        self.assertEqual(eng.phase, DONE)
        self.assertIsNone(eng.abort_reason)

    def test_polygon_catches_command_before_estimate(self):
        polygon = ((-2.0, -2.0), (18.0, 3.0), (16.0, 11.0), (-4.0, 6.0))
        cfg = replace(CFG, fence_polygon_ne=polygon, fence_radius_m=0.0)
        eng = MissionEngine(cfg)
        self.assertGreater(len(eng.lanes), 1)
        for lane in eng.lanes:
            self.assertTrue(point_in_polygon(lane.start, polygon))
            self.assertTrue(point_in_polygon(lane.point_at(lane.length), polygon))

    def test_polygon_rejects_dip_target_outside_field(self):
        polygon = ((-1.0, -1.0), (20.0, 4.0), (18.0, 14.0), (-3.0, 9.0))
        cfg = replace(CFG, fence_polygon_ne=polygon, fence_radius_m=0.0)
        eng = MissionEngine(cfg)
        eng.phase = LANE
        eng._estimate_fence_violation(0.0, 0.0)
        eng.tick(3.0, (100.0, 100.0, -6.0))
        self.assertEqual(eng.phase, ABORT)
        self.assertIn("field polygon", eng.abort_reason)

    def test_polygon_allows_approach_from_launch_line_then_arms(self):
        polygon = ((10.0, -5.0), (30.0, -5.0), (30.0, 5.0), (10.0, 5.0))
        cfg = replace(CFG, fence_polygon_ne=polygon, max_dips=0)
        eng = MissionEngine(cfg)
        eng.phase = LANE
        self.assertIsNone(eng._estimate_fence_violation(0.0, 0.0))
        self.assertFalse(eng._polygon_fence_engaged)
        self.assertIsNone(eng._estimate_fence_violation(15.0, 0.0))
        self.assertTrue(eng._polygon_fence_engaged)
        self.assertIn("field polygon", eng._estimate_fence_violation(0.0, 0.0))

    def test_polygon_releases_for_egress_home(self):
        polygon = ((10.0, -5.0), (30.0, -5.0), (30.0, 5.0), (10.0, 5.0))
        eng = MissionEngine(replace(CFG, fence_polygon_ne=polygon))
        eng.phase = LANE
        eng._estimate_fence_violation(15.0, 0.0)
        eng.phase = EGRESS
        self.assertIsNone(eng._estimate_fence_violation(0.0, 0.0))


class TestPolygonGeometry(unittest.TestCase):
    def test_inset_is_a_constant_margin_for_both_windings(self):
        for polygon in (
            ((0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)),
            ((0.0, 6.0), (10.0, 6.0), (10.0, 0.0), (0.0, 0.0)),
        ):
            inset = inset_convex_polygon(polygon, 1.0)
            self.assertEqual(
                {(round(n, 6), round(e, 6)) for n, e in inset},
                {(1.0, 1.0), (9.0, 1.0), (9.0, 5.0), (1.0, 5.0)},
            )
            original_heading = math.atan2(
                polygon[1][1] - polygon[0][1], polygon[1][0] - polygon[0][0]
            )
            inset_heading = math.atan2(
                inset[1][1] - inset[0][1], inset[1][0] - inset[0][0]
            )
            self.assertAlmostEqual(
                math.sin(original_heading), math.sin(inset_heading), places=6
            )
            self.assertAlmostEqual(
                math.cos(original_heading), math.cos(inset_heading), places=6
            )

    def test_margin_drives_lanes_and_fence(self):
        polygon = ((0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0))
        cfg = replace(CFG, fence_polygon_ne=polygon, fence_margin_m=2.0)
        eng = MissionEngine(cfg)
        self.assertIsNotNone(eng._fence_violation(1.0, 5.0))
        self.assertIsNone(eng._fence_violation(2.0, 5.0))
        for lane in eng.lanes:
            self.assertTrue(point_in_polygon(lane.start, eng.fence_polygon))
            self.assertTrue(point_in_polygon(lane.point_at(lane.length), eng.fence_polygon))

    def test_non_cardinal_quadrilateral_produces_clipped_lanes(self):
        polygon = ((0.0, 0.0), (12.0, 5.0), (9.0, 13.0), (-3.0, 8.0))
        lanes = polygon_serpentine(polygon, 2.0)
        self.assertGreaterEqual(len(lanes), 4)
        headings = {round(math.degrees(lane.heading) % 180.0, 6) for lane in lanes}
        self.assertEqual(len(headings), 1)
        self.assertNotIn(0.0, headings)
        for lane in lanes:
            self.assertTrue(point_in_polygon(lane.start, polygon))
            self.assertTrue(point_in_polygon(lane.point_at(lane.length), polygon))

    def test_polygon_fence_takes_precedence_over_legacy_radius(self):
        polygon = ((-2.0, -2.0), (20.0, -2.0), (20.0, 2.0), (-2.0, 2.0))
        cfg = replace(CFG, fence_polygon_ne=polygon, fence_radius_m=1.0)
        eng = MissionEngine(cfg)
        eng._home_ne = (0.0, 0.0)
        self.assertIsNone(eng._fence_violation(10.0, 0.0))


class TestRotatedLanes(unittest.TestCase):
    """M-Air runs the survey on cardinal south, not on the NED axes."""

    def test_a_rotated_field_is_covered_the_same(self):
        cfg = replace(CFG, lane_heading_deg=180.0, fence_radius_m=0.0)
        eng, log, _ = run_mission(cfg=cfg, mines=[(-6.0, -2.0), (-12.0, -6.0)])
        self.assertEqual(eng.phase, DONE)
        self.assertEqual(eng.coverage_report()["gaps"], [])
        self.assertEqual(len(log.clusters), 2)

    def test_rotation_preserves_lane_spacing(self):
        for heading in (0.0, 37.0, 180.0, -95.0):
            lanes = MissionEngine(replace(CFG, lane_heading_deg=heading)).lanes
            fwd = (math.cos(math.radians(heading)), math.sin(math.radians(heading)))
            perp = (-fwd[1], fwd[0])
            for i in range(1, len(lanes)):
                d = (lanes[i].start[0] - lanes[i - 1].start[0]) * perp[0] + (
                    lanes[i].start[1] - lanes[i - 1].start[1]
                ) * perp[1]
                self.assertAlmostEqual(d, CFG.lane_spacing, places=6, msg=f"{heading}°")


if __name__ == "__main__":
    unittest.main()
