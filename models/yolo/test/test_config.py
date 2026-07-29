import unittest
from dataclasses import replace

from datagen.config import GenConfig


CFG = GenConfig()


class TestRandomizationConfig(unittest.TestCase):
    def test_defaults_keep_untagged_rare_and_unknown_layout_sweepable(self):
        self.assertEqual(CFG.p_tag_both, CFG.p_tag_one)
        self.assertLessEqual(CFG.p_tag_none, 0.01)
        self.assertEqual(CFG.tag_up_prob, 0.5)
        self.assertEqual(
            set(CFG.surface_materials),
            {"grass", "dirt", "gravel", "pavement", "concrete"},
        )
        self.assertEqual(CFG.mine_color_names, ("lime", "green", "muddy_olive"))
        self.assertEqual(CFG.mine_color_weights, (0.10, 0.45, 0.45))

    def test_silent_failure_guards(self):
        for changes in (
            {"tag_up_prob": -0.01},
            {"tag_up_prob": 1.01},
            {"mixed_surface_prob": -0.01},
            {"mixed_surface_prob": 1.01},
            {"p_tag_one": -1.0},
            {"p_tag_both": 0.0, "p_tag_one": 0.0, "p_tag_none": 0.0},
            {"surface_materials": ("grass",), "mixed_surface_prob": 0.1},
            {"grass_blade_m": (0.0, 0.1)},
            {"grass_blade_m": (0.1, 0.04)},
            {"mine_color_names": ()},
            {"mine_color_names": ("green",)},
            {"mine_color_weights": (-1.0, 1.0, 1.0)},
            {"mine_color_weights": (0.0, 0.0, 0.0)},
            {
                "mine_color_palette_srgb": (
                    (0.1, 0.2, 1.1),
                    (0.1, 0.2, 0.3),
                    (0.1, 0.2, 0.3),
                )
            },
            {"mine_color_hue_jitter_deg": -0.1},
            {"mine_color_saturation_scale": (1.0, 0.5)},
            {"mine_color_value_scale": (0.0, 1.0)},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(CFG, **changes)


if __name__ == "__main__":
    unittest.main()
