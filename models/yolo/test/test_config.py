import unittest
import json
from dataclasses import asdict
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
        self.assertEqual(
            CFG.mine_color_names,
            (
                "official_sage_gray",
                "legacy_pale_green",
                "team_lime",
                "green",
                "muddy_olive",
            ),
        )
        self.assertEqual(
            CFG.mine_color_palette_srgb,
            (
                (0x8A / 255, 0xA0 / 255, 0x98 / 255),
                (0xC8 / 255, 0xCC / 255, 0xB5 / 255),
                (0x44 / 255, 0xBE / 255, 0x66 / 255),
                (0x4F / 255, 0x7D / 255, 0x36 / 255),
                (0x55 / 255, 0x57 / 255, 0x37 / 255),
            ),
        )
        self.assertEqual(CFG.mine_color_weights, (0.30, 0.10, 0.10, 0.25, 0.25))
        self.assertEqual(CFG.mine_color_hue_jitter_deg, 6.0)
        self.assertEqual(CFG.mine_color_saturation_scale, (0.50, 1.20))
        self.assertEqual(CFG.mine_color_value_scale, (0.80, 1.20))
        self.assertEqual(CFG.grass_dense_prob, 0.10)
        self.assertEqual(CFG.eevee_render_samples, 8)
        self.assertEqual(CFG.png_compression, 15)
        self.assertEqual(CFG.n_scenes, 300)
        self.assertEqual((CFG.mines_min, CFG.mines_max), (4, 20))
        self.assertEqual(CFG.station_interval_m, 2.0)
        self.assertEqual(CFG.negative_frame_keep, 0.05)
        self.assertEqual(CFG.alt_range_m, (1.0, 7.0))

    def test_manifest_config_roundtrip(self):
        raw = json.loads(json.dumps(asdict(CFG)))
        self.assertEqual(GenConfig.from_dict(raw), CFG)

    def test_silent_failure_guards(self):
        for changes in (
            {"tag_up_prob": -0.01},
            {"tag_up_prob": 1.01},
            {"mixed_surface_prob": -0.01},
            {"mixed_surface_prob": 1.01},
            {"grass_dense_prob": -0.01},
            {"grass_dense_prob": 1.01},
            {"negative_frame_keep": -0.01},
            {"negative_frame_keep": 1.01},
            {"p_tag_one": -1.0},
            {"p_tag_both": 0.0, "p_tag_one": 0.0, "p_tag_none": 0.0},
            {"surface_materials": ("grass",), "mixed_surface_prob": 0.1},
            {"grass_sparse_blade_m": (0.0, 0.1)},
            {"grass_sparse_blade_m": (0.1, 0.04)},
            {"grass_dense_blade_m": (0.0, 0.1)},
            {"grass_sparse_density": (0.0, 600.0)},
            {"grass_dense_density": (6000.0, 3000.0)},
            {"render_samples": 0},
            {"eevee_render_samples": 0},
            {"png_compression": -1},
            {"png_compression": 101},
            {"mine_color_names": ()},
            {"mine_color_names": ("green",)},
            {
                "mine_color_names": (
                    "duplicate",
                    "duplicate",
                    "third",
                    "fourth",
                    "fifth",
                )
            },
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
