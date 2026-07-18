import math
import unittest

from mission_engine.core.geometry import (
    quat_conj,
    quat_from_yaw,
    quat_rotate,
    serpentine,
    yaw_of,
)


class TestQuat(unittest.TestCase):
    def test_yaw_roundtrip(self):
        for yaw in (-3.0, -math.pi / 2.0, 0.0, 0.7, math.pi / 2.0, 3.0):
            self.assertAlmostEqual(yaw_of(quat_from_yaw(yaw)), yaw, places=12)

    def test_rotate_yaw90_takes_forward_to_east(self):
        q = quat_from_yaw(math.pi / 2.0)
        v = quat_rotate(q, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(v[0], 0.0, places=12)
        self.assertAlmostEqual(v[1], 1.0, places=12)
        self.assertAlmostEqual(v[2], 0.0, places=12)

    def test_rotate_preserves_down(self):
        q = quat_from_yaw(1.1)
        self.assertEqual(quat_rotate(q, (0.0, 0.0, 1.0))[2], 1.0)

    def test_conj_inverts(self):
        q = quat_from_yaw(0.9)
        v = (0.3, -1.2, 4.0)
        back = quat_rotate(quat_conj(q), quat_rotate(q, v))
        for a, b in zip(back, v):
            self.assertAlmostEqual(a, b, places=12)


class TestSerpentine(unittest.TestCase):
    def test_lanes_alternate_and_adjoin(self):
        lanes = serpentine((0.0, 0.0), 25.0, 3, 6.0)
        self.assertEqual([ln.index for ln in lanes], [0, 1, 2])
        self.assertEqual([ln.heading for ln in lanes], [0.0, math.pi, 0.0])
        # lane 0 ends at north=25; lane 1 starts beside it and runs back
        self.assertEqual(lanes[0].point_at(25.0), (25.0, 0.0))
        self.assertEqual(lanes[1].start, (25.0, 6.0))
        n, e = lanes[1].point_at(25.0)
        self.assertAlmostEqual(n, 0.0, places=9)
        self.assertAlmostEqual(e, 6.0, places=9)

    def test_lane_spacing(self):
        lanes = serpentine((2.0, 1.5), 10.0, 4, 3.0)
        self.assertEqual([ln.start[1] for ln in lanes], [1.5, 4.5, 7.5, 10.5])

    def test_rejects_bad_args(self):
        with self.assertRaises(ValueError):
            serpentine((0.0, 0.0), 25.0, 0, 6.0)
        with self.assertRaises(ValueError):
            serpentine((0.0, 0.0), -1.0, 3, 6.0)


if __name__ == "__main__":
    unittest.main()
