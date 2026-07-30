import math
import random
import unittest

from mission_engine.core.geometry import yaw_of

from datagen.config import GenConfig
from datagen.flightpath import stations

CFG = GenConfig()


class TestFlightpath(unittest.TestCase):
    def setUp(self):
        self.sts = stations(
            CFG, alt_rng=random.Random("alt"), rng=random.Random("fp")
        )

    def test_station_count(self):
        # Both lane endpoints are included.
        per_lane = int(
            (CFG.north_extent[1] - CFG.north_extent[0])
            // CFG.station_interval_m
        ) + 1
        self.assertEqual(len(self.sts), CFG.n_lanes * per_lane)

    def test_lanes_centered_across_field(self):
        # field east width 15, span 12 -> lanes at 1.5, 7.5, 13.5
        easts = sorted({round(s.pos[1], 6) for s in self.sts})
        self.assertEqual(easts, [1.5, 7.5, 13.5])

    def test_altitude_varies_per_station_within_envelope(self):
        alts = [-s.pos[2] for s in self.sts]
        for a in alts:
            self.assertGreaterEqual(a, CFG.alt_range_m[0])
            self.assertLessEqual(a, CFG.alt_range_m[1])
        # The production path's uniform draws should span most of the envelope.
        self.assertLess(min(alts), 2.0)
        self.assertGreater(max(alts), 7.0)

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
