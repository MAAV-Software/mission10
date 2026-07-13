import random
import unittest
from dataclasses import replace

from datagen.config import GenConfig
from datagen.scatter import ScatterFailed, scatter

CFG = GenConfig()


class TestScatter(unittest.TestCase):
    def test_deterministic(self):
        a = scatter(CFG, random.Random("s"))
        b = scatter(CFG, random.Random("s"))
        self.assertEqual(a, b)

    def test_separation_and_bounds(self):
        mines = scatter(CFG, random.Random("bounds"))
        self.assertGreaterEqual(len(mines), CFG.mines_min)
        self.assertLessEqual(len(mines), CFG.mines_max)
        m = CFG.edge_margin_m
        for p in mines:
            self.assertGreaterEqual(p.north, CFG.north_extent[0] + m)
            self.assertLessEqual(p.north, CFG.north_extent[1] - m)
            self.assertGreaterEqual(p.east, CFG.east_extent[0] + m)
            self.assertLessEqual(p.east, CFG.east_extent[1] - m)
        for i, a in enumerate(mines):
            for b in mines[i + 1 :]:
                d2 = (a.north - b.north) ** 2 + (a.east - b.east) ** 2
                self.assertGreaterEqual(d2, CFG.min_separation_m**2)

    def test_impossible_packing_fails_fast(self):
        cfg = replace(CFG, mines_min=12, mines_max=12, min_separation_m=50.0)
        with self.assertRaises(ScatterFailed):
            scatter(cfg, random.Random("fail"))


if __name__ == "__main__":
    unittest.main()
