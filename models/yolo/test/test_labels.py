import math
import unittest

from mission_engine.core.config import CameraModel
from mission_engine.core.geometry import quat_from_yaw

from datagen.labels import mine_corners, yolo_box
from datagen.scatter import MinePose

CAM = CameraModel()
DIMS = (0.12, 0.061, 0.020)
POS = (0.0, 0.0, -6.0)
Q0 = quat_from_yaw(0.0)
LEAD = 6.0 * math.tan(math.radians(CAM.tilt_deg))  # image-center ground point


def box(mine, **kw):
    kw.setdefault("min_visible_frac", 0.25)
    kw.setdefault("min_box_px", 4.0)
    return yolo_box(CAM, POS, Q0, mine, DIMS, **kw)


class TestMineCorners(unittest.TestCase):
    def test_eight_corners_on_and_above_ground(self):
        corners = mine_corners(MinePose(3.0, 4.0, 0.7), DIMS)
        self.assertEqual(len(corners), 8)
        self.assertEqual({round(c[2], 6) for c in corners}, {0.0, -0.02})


class TestYoloBox(unittest.TestCase):
    def test_centered_mine(self):
        b = box(MinePose(LEAD, 0.0, 0.0))
        self.assertIsNotNone(b)
        self.assertAlmostEqual(b.cx, 0.5, delta=0.005)
        self.assertAlmostEqual(b.cy, 0.5, delta=0.005)
        self.assertAlmostEqual(b.visible_frac, 1.0, places=9)
        # 0.061 m east extent at ~6.09 m slant: ~13.6 px of 1640
        self.assertAlmostEqual(b.w, 0.0083, delta=0.0015)
        # 0.12 m north extent: ~26 px of 1232 (+ height parallax)
        self.assertAlmostEqual(b.h, 0.022, delta=0.004)

    def test_yaw_rotates_box(self):
        b0 = box(MinePose(LEAD, 0.0, 0.0))
        b90 = box(MinePose(LEAD, 0.0, math.pi / 2.0))
        self.assertGreater(b90.w, b0.w)  # long axis now spans east/u
        self.assertLess(b90.h, b0.h)

    def test_edge_clipping(self):
        # mine centered on the right image edge: half the box clips away
        b = box(MinePose(LEAD, 3.6747, 0.0))
        self.assertIsNotNone(b)
        self.assertAlmostEqual(b.cx + b.w / 2.0, 1.0, places=6)
        self.assertGreater(b.visible_frac, 0.2)
        self.assertLess(b.visible_frac, 0.9)

    def test_far_outside_is_none(self):
        self.assertIsNone(box(MinePose(LEAD, 10.0, 0.0)))

    def test_min_box_px_filters(self):
        self.assertIsNone(box(MinePose(LEAD, 0.0, 0.0), min_box_px=50.0))

    def test_line_format(self):
        parts = box(MinePose(LEAD, 0.0, 0.0)).line().split()
        self.assertEqual(len(parts), 5)
        self.assertEqual(parts[0], "0")
        for tok in parts[1:]:
            val = float(tok)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)


if __name__ == "__main__":
    unittest.main()
