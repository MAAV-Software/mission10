import math
import unittest

from mission_engine.core.config import CameraModel
from mission_engine.core.geometry import quat_from_yaw
from mission_engine.core.ingest import PoseSnapshot
from mission_engine.core.visibility import (
    EDGE,
    OBSERVED,
    UNSEEN,
    VisibilityLedger,
)

NADIR = CameraModel(tilt_deg=0.0)
Q0 = quat_from_yaw(0.0)


def snap(t=0.0, pos=(0.0, 0.0, -4.0), q=Q0, agl=None):
    return PoseSnapshot(t=t, pos=pos, q=q, agl=agl)


class TestFootprint(unittest.TestCase):
    def test_nadir_extent_matches_fov(self):
        """Level nadir at 4 m AGL: half-extent = agl * tan(half-fov)."""
        led = VisibilityLedger(NADIR, cell_m=0.25)
        led.note_pose(snap())
        half_e = 4.0 * math.tan(math.radians(NADIR.hfov_deg / 2.0))
        vfov = NADIR.hfov_deg * NADIR.height_px / NADIR.width_px
        half_n = 4.0 * math.tan(math.radians(vfov / 2.0))
        self.assertEqual(led.query((0.0, 0.0)), OBSERVED)
        self.assertEqual(led.query((0.0, half_e - 0.3)), OBSERVED)
        self.assertEqual(led.query((0.0, half_e + 0.3)), UNSEEN)
        self.assertEqual(led.query((half_n - 0.3, 0.0)), OBSERVED)
        self.assertEqual(led.query((half_n + 0.3, 0.0)), UNSEEN)

    def test_yaw_rotates_footprint(self):
        """Tilted camera looks ahead; ahead follows body yaw."""
        cam = CameraModel(tilt_deg=30.0)
        ahead = 4.0 * math.tan(math.radians(30.0))
        north = VisibilityLedger(cam)
        east = VisibilityLedger(cam)
        north.note_pose(snap(q=quat_from_yaw(0.0)))
        east.note_pose(snap(q=quat_from_yaw(math.pi / 2.0)))
        self.assertEqual(north.query((ahead, 0.0)), OBSERVED)
        self.assertEqual(north.query((0.0, ahead)), UNSEEN)
        self.assertEqual(east.query((0.0, ahead)), OBSERVED)
        self.assertEqual(east.query((ahead, 0.0)), UNSEEN)

    def test_depression_floor_shrinks_footprint(self):
        loose = VisibilityLedger(CameraModel(tilt_deg=30.0), min_depression_deg=10.0)
        tight = VisibilityLedger(CameraModel(tilt_deg=30.0), min_depression_deg=45.0)
        loose.note_pose(snap())
        tight.note_pose(snap())
        self.assertGreater(len(loose.cells), len(tight.cells))

    def test_max_range_shrinks_footprint(self):
        near = VisibilityLedger(NADIR, max_range_m=5.0)
        far = VisibilityLedger(NADIR, max_range_m=50.0)
        high = snap(pos=(0.0, 0.0, -12.0))
        near.note_pose(high)
        far.note_pose(high)
        self.assertGreater(len(far.cells), len(near.cells))

    def test_agl_overrides_pos_z(self):
        """dToF AGL sets the ground plane, not -pos.z."""
        low = VisibilityLedger(NADIR)
        high = VisibilityLedger(NADIR)
        low.note_pose(snap(agl=2.0))
        high.note_pose(snap(agl=8.0))
        self.assertGreater(len(high.cells), len(low.cells))

    def test_on_ground_records_frame_but_no_cells(self):
        led = VisibilityLedger(NADIR)
        led.note_pose(snap(pos=(0.0, 0.0, 0.0)))
        self.assertEqual(led.n_frames, 1)
        self.assertEqual(len(led.cells), 0)


class TestVisits(unittest.TestCase):
    def test_gap_starts_new_visit(self):
        led = VisibilityLedger(NADIR, revisit_gap_s=5.0)
        led.note_pose(snap(t=0.0))
        led.note_pose(snap(t=4.0))
        led.note_pose(snap(t=20.0))
        visits = led.visits((0.0, 0.0))
        self.assertEqual(len(visits), 2)
        self.assertEqual(visits[0].n_frames, 2)
        self.assertEqual((visits[0].t0, visits[0].t1), (0.0, 4.0))
        self.assertEqual((visits[1].t0, visits[1].t1), (20.0, 20.0))

    def test_lane_revisit_is_per_cell_not_global(self):
        """A continuous stream that moves away and returns: the revisited
        cell gets two visits even though the pose stream never gapped."""
        led = VisibilityLedger(NADIR, revisit_gap_s=5.0)
        for k in range(31):  # 0..30 s, 1 m/s east, then back
            e = float(k) if k <= 15 else float(30 - k)
            led.note_pose(snap(t=float(k), pos=(0.0, e, -4.0)))
        self.assertEqual(len(led.visits((0.0, 0.0))), 2)
        self.assertEqual(len(led.visits((0.0, 15.0))), 1)

    def test_non_monotonic_rejected(self):
        led = VisibilityLedger(NADIR)
        led.note_pose(snap(t=1.0))
        with self.assertRaises(ValueError):
            led.note_pose(snap(t=0.5))

    def test_coverage_counts_visits(self):
        led = VisibilityLedger(NADIR, revisit_gap_s=5.0)
        led.note_pose(snap(t=0.0))
        led.note_pose(snap(t=1.0))
        led.note_pose(snap(t=20.0))
        self.assertEqual(led.coverage_cells()[led.cell_of((0.0, 0.0))], 2)
        self.assertEqual(led.coverage_cells(min_frames=2)[led.cell_of((0.0, 0.0))], 1)


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.led = VisibilityLedger(NADIR, cell_m=0.5)
        self.led.note_pose(snap())
        self.half_e = 4.0 * math.tan(math.radians(NADIR.hfov_deg / 2.0))

    def test_margin_degrades_boundary_to_edge(self):
        """Just inside the footprint: certain without margin, EDGE with a
        margin that crosses the boundary — the transform-uncertainty
        semantics the assignment costs rely on."""
        p = (0.0, self.half_e - 0.3)
        self.assertEqual(self.led.query(p), OBSERVED)
        self.assertEqual(self.led.query(p, margin_m=1.0), EDGE)

    def test_margin_far_outside_stays_unseen(self):
        self.assertEqual(self.led.query((30.0, 30.0), margin_m=1.0), UNSEEN)

    def test_window_selects_visits(self):
        """A tracklet's window only sees visits that overlap it."""
        self.led.note_pose(snap(t=20.0))
        self.assertEqual(self.led.query((0.0, 0.0), window=(19.0, 21.0)), OBSERVED)
        self.assertEqual(self.led.query((0.0, 0.0), window=(8.0, 12.0)), UNSEEN)

    def test_min_frames_gates_observation(self):
        self.assertEqual(self.led.query((0.0, 0.0), min_frames=2), UNSEEN)
        self.led.note_pose(snap(t=1.0))
        self.assertEqual(self.led.query((0.0, 0.0), min_frames=2), OBSERVED)


class TestSerialization(unittest.TestCase):
    def test_round_trippable_dump(self):
        led = VisibilityLedger(NADIR)
        led.note_pose(snap())
        d = led.to_dict()
        self.assertEqual(d["n_frames"], 1)
        self.assertEqual(d["cells"], sorted(d["cells"]))
        i, j, visits = d["cells"][0]
        self.assertEqual(visits, [[0.0, 0.0, 1]])


if __name__ == "__main__":
    unittest.main()
