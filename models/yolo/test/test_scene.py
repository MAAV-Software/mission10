import colorsys
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from datagen.config import GenConfig
from datagen.dump import write_scene
from datagen.generate import _srgb_to_linear
from datagen.manifest import SCHEMA, scene_manifest
from datagen.scene import build_scene, scene_labels

CFG = GenConfig()


class TestScene(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(build_scene(CFG, 0), build_scene(CFG, 0))

    def test_scenes_differ(self):
        self.assertNotEqual(build_scene(CFG, 0).mines, build_scene(CFG, 1).mines)

    def test_surface_draw_is_deterministic(self):
        self.assertEqual(build_scene(CFG, 4).surface, build_scene(CFG, 4).surface)

    def test_mine_colors_are_one_jittered_batch_per_scene(self):
        scene = build_scene(CFG, 4)
        families = {appearance.color_family for appearance in scene.mine_appearances}
        colors = {appearance.color_srgb for appearance in scene.mine_appearances}
        self.assertEqual(len(families), 1)
        self.assertGreater(len(colors), 1)
        palette_index = CFG.mine_color_names.index(next(iter(families)))
        base_h, base_s, base_v = colorsys.rgb_to_hsv(
            *CFG.mine_color_palette_srgb[palette_index]
        )
        for appearance in scene.mine_appearances:
            self.assertTrue(
                all(0.0 <= channel <= 1.0 for channel in appearance.color_srgb)
            )
            h, s, v = colorsys.rgb_to_hsv(*appearance.color_srgb)
            hue_delta = min(abs(h - base_h), 1.0 - abs(h - base_h)) * 360.0
            self.assertLessEqual(hue_delta, CFG.mine_color_hue_jitter_deg)
            self.assertGreaterEqual(
                s / base_s, CFG.mine_color_saturation_scale[0]
            )
            self.assertLessEqual(
                s / base_s, CFG.mine_color_saturation_scale[1]
            )
            self.assertGreaterEqual(v / base_v, CFG.mine_color_value_scale[0])
            self.assertLessEqual(v / base_v, CFG.mine_color_value_scale[1])

    def test_forced_mine_color_has_exact_anchor_without_jitter(self):
        cfg = replace(
            CFG,
            mine_color_names=("test_green",),
            mine_color_palette_srgb=((0.2, 0.4, 0.3),),
            mine_color_weights=(1.0,),
            mine_color_hue_jitter_deg=0.0,
            mine_color_saturation_scale=(1.0, 1.0),
            mine_color_value_scale=(1.0, 1.0),
        )
        for appearance in build_scene(cfg, 0).mine_appearances:
            self.assertEqual(appearance.color_family, "test_green")
            for actual, expected in zip(
                appearance.color_srgb, (0.2, 0.4, 0.3), strict=True
            ):
                self.assertAlmostEqual(actual, expected)

    def test_srgb_conversion_uses_scene_linear_values(self):
        self.assertEqual(_srgb_to_linear((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
        self.assertEqual(_srgb_to_linear((1.0, 1.0, 1.0)), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(
            _srgb_to_linear((0.5, 0.5, 0.5))[0], 0.214041, places=6
        )

    def test_forced_single_and_mixed_surfaces(self):
        single = replace(
            CFG, surface_materials=("concrete",), mixed_surface_prob=0.0
        )
        self.assertEqual(build_scene(single, 0).surface.primary, "concrete")
        self.assertIsNone(build_scene(single, 0).surface.secondary)

        mixed = replace(
            CFG,
            surface_materials=("grass", "pavement"),
            mixed_surface_prob=1.0,
            mixed_strip_width_m=(3.0, 3.0),
        )
        surface = build_scene(mixed, 0).surface
        self.assertNotEqual(surface.primary, surface.secondary)
        self.assertIn(surface.strip_axis, ("north", "east"))
        self.assertEqual(surface.strip_width_m, 3.0)

    def test_new_knobs_do_not_perturb_geometry_or_labels(self):
        untagged_grass = replace(
            CFG,
            surface_materials=("grass",),
            mixed_surface_prob=0.0,
            p_tag_both=0.0, p_tag_one=0.0, p_tag_none=1.0,
            tag_up_prob=0.0,
        )
        tagged_mixed = replace(
            CFG,
            surface_materials=("dirt", "pavement"),
            mixed_surface_prob=1.0,
            p_tag_both=1.0, p_tag_one=0.0, p_tag_none=0.0,
            tag_up_prob=1.0,
            mine_color_names=("forced",),
            mine_color_palette_srgb=((0.1, 0.2, 0.3),),
            mine_color_weights=(1.0,),
        )
        a = build_scene(untagged_grass, 2)
        b = build_scene(tagged_mixed, 2)
        self.assertEqual(
            [(m.north, m.east, m.yaw) for m in a.mines],
            [(m.north, m.east, m.yaw) for m in b.mines],
        )
        self.assertEqual(a.stations, b.stations)
        self.assertNotEqual(a.mine_appearances, b.mine_appearances)
        self.assertEqual(
            scene_labels(untagged_grass, a), scene_labels(tagged_mixed, b)
        )

    def test_color_config_only_changes_appearance(self):
        recolored = replace(
            CFG,
            mine_color_names=("forced",),
            mine_color_palette_srgb=((0.1, 0.2, 0.3),),
            mine_color_weights=(1.0,),
        )
        original_scene = build_scene(CFG, 2)
        recolored_scene = build_scene(recolored, 2)
        self.assertEqual(original_scene.mines, recolored_scene.mines)
        self.assertEqual(original_scene.stations, recolored_scene.stations)
        self.assertEqual(original_scene.surface, recolored_scene.surface)
        self.assertEqual(original_scene.tilt, recolored_scene.tilt)
        self.assertNotEqual(
            original_scene.mine_appearances, recolored_scene.mine_appearances
        )
        self.assertEqual(
            scene_labels(CFG, original_scene),
            scene_labels(recolored, recolored_scene),
        )

    def test_index_validated(self):
        with self.assertRaises(ValueError):
            build_scene(CFG, CFG.n_scenes)

    def test_labels_cover_all_stations_and_stay_normalized(self):
        total = 0
        for i in range(3):
            scene = build_scene(CFG, i)
            labels = scene_labels(CFG, scene)
            self.assertEqual(len(labels), len(scene.stations))
            for boxes in labels.values():
                for b in boxes:
                    total += 1
                    self.assertGreaterEqual(b.cx - b.w / 2.0, -1e-9)
                    self.assertLessEqual(b.cx + b.w / 2.0, 1.0 + 1e-9)
                    self.assertGreaterEqual(b.cy - b.h / 2.0, -1e-9)
                    self.assertLessEqual(b.cy + b.h / 2.0, 1.0 + 1e-9)
        self.assertGreater(total, 0)


class TestManifestAndDump(unittest.TestCase):
    def test_manifest_json_roundtrip(self):
        scene = build_scene(CFG, 0)
        labels = scene_labels(CFG, scene)
        man = json.loads(json.dumps(scene_manifest(CFG, scene, labels)))
        self.assertEqual(man["schema"], SCHEMA)
        self.assertEqual(len(man["stations"]), len(scene.stations))
        self.assertEqual(len(man["mines"]), len(scene.mines))
        self.assertEqual(
            man["surface"],
            {
                "primary": scene.surface.primary,
                "secondary": scene.surface.secondary,
                "strip_axis": scene.surface.strip_axis,
                "strip_center_m": scene.surface.strip_center_m,
                "strip_width_m": scene.surface.strip_width_m,
            },
        )
        visible = sum(m.tag_visible for m in scene.mines)
        self.assertEqual(man["tag_visible_fraction"], visible / len(scene.mines))
        for mine, appearance, record in zip(
            scene.mines, scene.mine_appearances, man["mines"], strict=True
        ):
            self.assertEqual(record["tag_layout"], mine.tag_layout)
            self.assertEqual(record["tag_up"], mine.tag_up)
            self.assertEqual(record["tag_visible"], mine.tag_visible)
            self.assertEqual(
                record["appearance"]["color_family"], appearance.color_family
            )
            self.assertEqual(
                record["appearance"]["color_srgb"], list(appearance.color_srgb)
            )

    def test_write_scene_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            man = write_scene(CFG, 0, out)
            label_files = sorted((out / "labels").glob("*.txt"))
            self.assertEqual(len(label_files), len(man["stations"]))
            manifest_files = list(out.glob("*.manifest.json"))
            self.assertEqual(len(manifest_files), 1)
            # every non-empty line parses as a 5-token YOLO row, class 0
            for f in label_files:
                for line in f.read_text().splitlines():
                    parts = line.split()
                    self.assertEqual(len(parts), 5)
                    self.assertEqual(parts[0], "0")

    def test_write_scene_is_byte_deterministic(self):
        with (
            tempfile.TemporaryDirectory() as tmp_a,
            tempfile.TemporaryDirectory() as tmp_b,
        ):
            out_a = Path(tmp_a)
            out_b = Path(tmp_b)
            write_scene(CFG, 3, out_a)
            write_scene(CFG, 3, out_b)
            files_a = sorted(path.relative_to(out_a) for path in out_a.rglob("*"))
            files_b = sorted(path.relative_to(out_b) for path in out_b.rglob("*"))
            self.assertEqual(files_a, files_b)
            for relative in files_a:
                if (out_a / relative).is_file():
                    self.assertEqual(
                        (out_a / relative).read_bytes(),
                        (out_b / relative).read_bytes(),
                    )


if __name__ == "__main__":
    unittest.main()
