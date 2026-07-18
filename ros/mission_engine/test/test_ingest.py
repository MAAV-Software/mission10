import math
import unittest

from mission_engine.core.backproject import project_raw
from mission_engine.core.config import CameraModel
from mission_engine.core.geometry import quat_from_yaw
from mission_engine.core.ingest import (
    PoseHistory,
    PoseSnapshot,
    make_observation,
    mine_ll,
)

CAM = CameraModel()


class TestPoseHistory(unittest.TestCase):
    def test_nearest_picks_closest_stamp(self):
        h = PoseHistory()
        for i in range(5):
            h.append(PoseSnapshot(t=float(i), pos=(float(i), 0.0, -6.0), q=quat_from_yaw(0.0)))
        self.assertEqual(h.nearest(2.4).t, 2.0)
        self.assertEqual(h.nearest(2.6).t, 3.0)
        self.assertEqual(h.nearest(-1.0).t, 0.0)
        self.assertEqual(h.nearest(99.0).t, 4.0)

    def test_horizon_trims(self):
        h = PoseHistory(horizon_s=2.0)
        for i in range(10):
            h.append(PoseSnapshot(t=float(i), pos=(0.0, 0.0, -6.0), q=quat_from_yaw(0.0)))
        self.assertEqual(h.nearest(0.0).t, 7.0)

    def test_rejects_time_reversal(self):
        h = PoseHistory()
        h.append(PoseSnapshot(t=1.0, pos=(0.0, 0.0, -6.0), q=quat_from_yaw(0.0)))
        with self.assertRaises(ValueError):
            h.append(PoseSnapshot(t=0.5, pos=(0.0, 0.0, -6.0), q=quat_from_yaw(0.0)))


class TestMakeObservation(unittest.TestCase):
    def test_roundtrip_recovers_mine(self):
        mine = (4.0, 1.0)
        snap = PoseSnapshot(
            t=10.0, pos=(3.0, 1.0, -6.0), q=quat_from_yaw(0.4), ll=(42.2944, -83.7105)
        )
        uv = project_raw(CAM, snap.pos, snap.q, (mine[0], mine[1], 0.0))
        o = make_observation(CAM, snap, 10.0, uv, conf=0.9, class_id="mine")
        self.assertAlmostEqual(o.ground_local[0], mine[0], places=9)
        self.assertAlmostEqual(o.ground_local[1], mine[1], places=9)
        # global fix: drone ll shifted by the local offset
        back = mine_ll(snap.ll, mine[0] - snap.pos[0], mine[1] - snap.pos[1])
        self.assertEqual(o.ll, back)

    def test_mine_ll_scale(self):
        lat, lon = mine_ll((42.0, -83.0), 111.32, 0.0)
        self.assertAlmostEqual(lat, 42.001, places=9)
        self.assertAlmostEqual(lon, -83.0, places=12)

    def test_on_ground_returns_none(self):
        snap = PoseSnapshot(t=0.0, pos=(0.0, 0.0, 0.0), q=quat_from_yaw(0.0))
        self.assertIsNone(
            make_observation(CAM, snap, 0.0, (CAM.cx, CAM.cy), conf=0.9, class_id="mine")
        )


if __name__ == "__main__":
    unittest.main()
