import math
import unittest

from mission_engine.core.backproject import (
    AboveHorizon,
    BehindCamera,
    ground_point,
    project_raw,
)
from mission_engine.core.config import CameraModel
from mission_engine.core.geometry import quat_from_yaw

NADIR = CameraModel(tilt_deg=0.0)
TILTED = CameraModel(tilt_deg=10.0)  # tilt math must hold regardless of mount
Q0 = quat_from_yaw(0.0)


class TestProjectRaw(unittest.TestCase):
    def test_nadir_point_below_hits_center(self):
        u, v = project_raw(NADIR, (0.0, 0.0, -6.0), Q0, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(u, NADIR.cx, places=9)
        self.assertAlmostEqual(v, NADIR.cy, places=9)

    def test_nadir_east_offset_moves_right(self):
        u, v = project_raw(NADIR, (0.0, 0.0, -6.0), Q0, (0.0, 1.0, 0.0))
        self.assertGreater(u, NADIR.cx)
        self.assertAlmostEqual(v, NADIR.cy, places=9)

    def test_nadir_north_offset_moves_image_up(self):
        u, v = project_raw(NADIR, (0.0, 0.0, -6.0), Q0, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(u, NADIR.cx, places=9)
        self.assertLess(v, NADIR.cy)

    def test_tilt_lead_hits_center(self):
        # the optical axis meets the ground alt*tan(tilt) ahead
        lead = 6.0 * math.tan(math.radians(TILTED.tilt_deg))
        u, v = project_raw(TILTED, (0.0, 0.0, -6.0), Q0, (lead, 0.0, 0.0))
        self.assertAlmostEqual(u, TILTED.cx, places=9)
        self.assertAlmostEqual(v, TILTED.cy, places=9)

    def test_yaw_carries_the_lead(self):
        yaw = 2.1
        lead = 6.0 * math.tan(math.radians(TILTED.tilt_deg))
        target = (lead * math.cos(yaw), lead * math.sin(yaw), 0.0)
        u, v = project_raw(TILTED, (0.0, 0.0, -6.0), quat_from_yaw(yaw), target)
        self.assertAlmostEqual(u, TILTED.cx, places=9)
        self.assertAlmostEqual(v, TILTED.cy, places=9)

    def test_known_pixel_scale(self):
        # 1 m east at 6 m below a nadir camera: u - cx = f / 6
        u, _ = project_raw(NADIR, (0.0, 0.0, -6.0), Q0, (0.0, 1.0, 0.0))
        self.assertAlmostEqual(u - NADIR.cx, NADIR.focal_px / 6.0, places=9)

    def test_behind_camera_raises(self):
        with self.assertRaises(BehindCamera):
            project_raw(NADIR, (0.0, 0.0, -6.0), Q0, (0.0, 0.0, -12.0))


class TestGroundPoint(unittest.TestCase):
    def test_roundtrip_with_project_raw(self):
        pos = (3.0, -2.0, -7.5)
        q = quat_from_yaw(0.8)
        for target in ((4.0, -1.0, 0.0), (2.0, -4.0, 0.0), (6.5, 0.5, 0.0)):
            uv = project_raw(TILTED, pos, q, target)
            n, e = ground_point(TILTED, pos, q, uv)
            self.assertAlmostEqual(n, target[0], places=9)
            self.assertAlmostEqual(e, target[1], places=9)

    def test_matches_legacy_nadir_formula(self):
        # legacy IARC_mission_10 ingest: a level nadir camera turns a
        # normalized image offset into ground metres via
        #   offset = (norm - 0.5) * 2 * alt * tan(half_fov)
        # with image-u -> +east and image-v -> -north. Exact in this case.
        alt = 6.0
        u_norm, v_norm = 0.62, 0.31
        east_legacy = (u_norm - 0.5) * 2.0 * alt * math.tan(math.radians(NADIR.hfov_deg / 2.0))
        vfov = 2.0 * math.atan(
            (NADIR.height_px / NADIR.width_px) * math.tan(math.radians(NADIR.hfov_deg / 2.0))
        )
        north_legacy = -(v_norm - 0.5) * 2.0 * alt * math.tan(vfov / 2.0)
        n, e = ground_point(
            NADIR,
            (0.0, 0.0, -alt),
            Q0,
            (u_norm * NADIR.width_px, v_norm * NADIR.height_px),
        )
        self.assertAlmostEqual(e, east_legacy, places=9)
        self.assertAlmostEqual(n, north_legacy, places=9)

    def test_ground_z_offset(self):
        # same pixel, ground plane 1 m higher -> intersection pulls closer
        uv = (NADIR.cx + 200.0, NADIR.cy)
        _, e0 = ground_point(NADIR, (0.0, 0.0, -6.0), Q0, uv, ground_z=0.0)
        _, e1 = ground_point(NADIR, (0.0, 0.0, -6.0), Q0, uv, ground_z=-1.0)
        self.assertAlmostEqual(e1, e0 * 5.0 / 6.0, places=9)

    def test_above_horizon_raises(self):
        # a heavily tilted view's top image row looks over the horizon
        steep = CameraModel(tilt_deg=80.0)
        with self.assertRaises(AboveHorizon):
            ground_point(steep, (0.0, 0.0, -6.0), Q0, (steep.cx, 0.0))


if __name__ == "__main__":
    unittest.main()
