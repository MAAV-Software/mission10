import math
import random
import unittest

from mission_engine.core.geometry import yaw_of

from datagen.config import GenConfig
from datagen.flightpath import stations

CFG = GenConfig()


class TestFlightpath(unittest.TestCase):
    def setUp(self):
        self.sts = stations(CFG, alt=6.0, rng=random.Random("fp"))

    def test_station_count(self):
        # 25 m lanes, 1 m interval -> 26 stations per lane, 3 lanes
        self.assertEqual(len(self.sts), 3 * 26)

    def test_lanes_centered_across_field(self):
        # field east width 15, span 12 -> lanes at 1.5, 7.5, 13.5
        easts = sorted({round(s.pos[1], 6) for s in self.sts})
        self.assertEqual(easts, [1.5, 7.5, 13.5])

    def test_altitude_applied(self):
        for s in self.sts:
            self.assertAlmostEqual(s.pos[2], -6.0)

    def test_headings_alternate_with_jitter(self):
        jit = math.radians(CFG.yaw_jitter_deg) + 1e-9
        for s in self.sts:
            yaw = yaw_of(s.q)
            if s.lane % 2 == 0:
                self.assertLessEqual(abs(yaw), jit)
            else:
                self.assertLessEqual(abs(abs(yaw) - math.pi), jit)


if __name__ == "__main__":
    unittest.main()
