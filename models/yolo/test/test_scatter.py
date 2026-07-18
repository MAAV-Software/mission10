import random
import unittest
from dataclasses import replace

from datagen.config import GenConfig
from datagen.scatter import MinePose, ScatterFailed, scatter

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

    def test_tag_visible_truth_table_is_derived(self):
        cases = (
            ("both", False, True),
            ("both", True, True),
            ("one", False, False),
            ("one", True, True),
            ("none", False, False),
            ("none", True, False),
        )
        for layout, tag_up, expected in cases:
            with self.subTest(layout=layout, tag_up=tag_up):
                pose = MinePose(1.0, 2.0, 3.0, tag_layout=layout, tag_up=tag_up)
                self.assertEqual(pose.tag_visible, expected)

    def test_forced_layout_and_flip_knobs(self):
        cases = (
            ("both", 0.0, False, True),
            ("one", 0.0, False, False),
            ("one", 1.0, True, True),
            ("none", 1.0, True, False),
        )
        for layout, tag_up_prob, expected_up, visible in cases:
            with self.subTest(layout=layout, tag_up_prob=tag_up_prob):
                weights = {
                    f"p_tag_{name}": 1.0 if name == layout else 0.0
                    for name in ("both", "one", "none")
                }
                cfg = replace(CFG, tag_up_prob=tag_up_prob, **weights)
                mines = scatter(
                    cfg,
                    random.Random("geometry"),
                    random.Random("tags"),
                )
                self.assertTrue(all(m.tag_layout == layout for m in mines))
                self.assertTrue(all(m.tag_up == expected_up for m in mines))
                self.assertTrue(all(m.tag_visible == visible for m in mines))


if __name__ == "__main__":
    unittest.main()
