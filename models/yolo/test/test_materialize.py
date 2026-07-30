import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from datagen.config import GenConfig
from datagen.dump import write_scene
from datagen.labels import raw_extents, yolo_box
from datagen.manifest import OCCLUSION_SCHEMA
from datagen.materialize import TileParams, materialize_scene, tile_scene
from datagen.scene import (
    build_scene,
    image_stem,
    scene_labels,
    selected_station_indices,
)

CFG = GenConfig()
SCENE = build_scene(CFG, 0)
CAM = replace(CFG.camera, tilt_deg=SCENE.tilt)
W, H = CFG.camera.width_px, CFG.camera.height_px
SELECTED = selected_station_indices(CFG, SCENE, scene_labels(CFG, SCENE))


def _boxed_mines(k):
    """(mine index, box) pairs at station k, label-line order."""
    st = SCENE.stations[k]
    out = []
    for i, mine in enumerate(SCENE.mines):
        box = yolo_box(
            CAM,
            st.pos,
            st.q,
            mine,
            CFG.mine_dims_m,
            min_visible_frac=CFG.min_visible_frac,
            min_box_px=CFG.min_box_px,
        )
        if box is not None:
            out.append((i, box))
    return out


def _sidecar(overrides):
    vis = {
        image_stem(CFG, SCENE, k): {str(i): 1.0 for i in range(len(SCENE.mines))}
        for k in SELECTED
    }
    for (k, i), frac in overrides.items():
        vis[image_stem(CFG, SCENE, k)][str(i)] = frac
    return {
        "schema": OCCLUSION_SCHEMA,
        "seed": CFG.seed,
        "scene": 0,
        "station_indices": SELECTED,
        "visible_frac": vis,
    }


class TestMaterialize(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        write_scene(CFG, 0, self.out)
        # a target whose analytic box is fully in frame, so the product
        # rule reduces to the occlusion fraction alone
        self.target_k, self.target_i = next(
            (k, i)
            for k in SELECTED
            for i, box in _boxed_mines(k)
            if box.visible_frac == 1.0
        )

    def _labels(self, sub, stem):
        return (self.out / sub / f"{stem}.txt").read_text()

    def test_all_visible_reproduces_analytic_labels(self):
        kept, dropped = materialize_scene(self.out, _sidecar({}), 0.15)
        self.assertEqual(dropped, 0)
        self.assertGreater(kept, 0)
        for k in SELECTED:
            stem = image_stem(CFG, SCENE, k)
            self.assertEqual(
                self._labels("labels", stem), self._labels("labels_filtered", stem)
            )

    def test_buried_mine_dropped_others_untouched(self):
        occ = _sidecar({(self.target_k, self.target_i): 0.02})
        kept, dropped = materialize_scene(self.out, occ, 0.15)
        self.assertEqual(dropped, 1)
        for k in SELECTED:
            stem = image_stem(CFG, SCENE, k)
            analytic = self._labels("labels", stem).splitlines()
            filtered = self._labels("labels_filtered", stem).splitlines()
            if k == self.target_k:
                drop_at = [i for i, _ in _boxed_mines(k)].index(self.target_i)
                self.assertEqual(filtered, analytic[:drop_at] + analytic[drop_at + 1 :])
            else:
                self.assertEqual(filtered, analytic)

    def test_frac_at_threshold_kept(self):
        occ = _sidecar({(self.target_k, self.target_i): 0.15})
        _, dropped = materialize_scene(self.out, occ, 0.15)
        self.assertEqual(dropped, 0)

    def test_product_rule_combines_edge_and_occlusion(self):
        # an edge-clipped box (visible_frac < 1) under partial occlusion can
        # fall below the threshold even though each factor alone passes
        clipped = next(
            (
                (k, i, box)
                for k in SELECTED
                for i, box in _boxed_mines(k)
                if box.visible_frac < 0.6
            ),
            None,
        )
        if clipped is None:
            self.skipTest("scene 0 has no strongly edge-clipped box")
        k, i, box = clipped
        occ_f = 0.3  # passes alone; product with the clip does not
        self.assertGreater(occ_f, 0.15)
        self.assertLess(occ_f * box.visible_frac, 0.15)
        _, dropped = materialize_scene(self.out, _sidecar({(k, i): occ_f}), 0.15)
        self.assertEqual(dropped, 1)

    def test_manifest_config_is_authoritative(self):
        cfg = replace(
            CFG,
            seed="banked",
            n_scenes=2,
            mines_min=7,
            mines_max=7,
            station_interval_m=2.5,
        )
        scene = build_scene(cfg, 1)
        labels = scene_labels(cfg, scene)
        selected = selected_station_indices(cfg, scene, labels)
        out = Path(tempfile.mkdtemp())
        write_scene(cfg, 1, out)
        occ = {
            "schema": OCCLUSION_SCHEMA,
            "seed": cfg.seed,
            "scene": 1,
            "station_indices": selected,
            "visible_frac": {
                image_stem(cfg, scene, k): {
                    str(i): 1.0 for i in range(len(scene.mines))
                }
                for k in selected
            },
        }
        kept, dropped = materialize_scene(out, occ, 0.15)
        self.assertEqual(dropped, 0)
        self.assertEqual(
            kept,
            sum(len(labels[image_stem(cfg, scene, k)]) for k in selected),
        )

    def test_sidecar_station_selection_must_match_manifest(self):
        occ = _sidecar({})
        occ["station_indices"] = occ["station_indices"][:-1]
        with self.assertRaisesRegex(ValueError, "selections differ"):
            materialize_scene(self.out, occ, 0.15)


class TestTileScene(unittest.TestCase):
    TP = TileParams(images=False, empty_keep=1.0, fullframe_frac=1.0)

    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        write_scene(CFG, 0, self.out)
        self.n_tiles, self.n_poisoned, self.n_boxes = tile_scene(
            self.out, _sidecar({}), 0.15, self.TP
        )
        self.index = json.loads((self.out / "train" / "tiles.json").read_text())[
            f"{CFG.seed}_s0000"
        ]

    def _station_of(self, stem):
        return next(
            k
            for k in range(len(SCENE.stations))
            if image_stem(CFG, SCENE, k) == stem
        )

    def test_interior_mine_roundtrips_to_native_pixels(self):
        # a tile label whose mine sits fully inside the tile must denormalize
        # to the same native-pixel rectangle as the full-frame label
        checked = 0
        for entry in self.index["tiles"]:
            if entry.get("full"):
                continue
            lines = (
                (self.out / "train" / "labels" / f"{entry['tile']}.txt")
                .read_text()
                .splitlines()
            )
            k = self._station_of(entry["src"])
            full_px = [
                (
                    (box.cx - box.w / 2) * W,
                    (box.cy - box.h / 2) * H,
                    (box.cx + box.w / 2) * W,
                    (box.cy + box.h / 2) * H,
                )
                for _, box in _boxed_mines(k)
                if box.visible_frac == 1.0
            ]
            t = self.TP.tile
            for line in lines:
                _, cx, cy, w, h = (float(v) for v in line.split())
                px = (
                    entry["x0"] + (cx - w / 2) * t,
                    entry["y0"] + (cy - h / 2) * t,
                    entry["x0"] + (cx + w / 2) * t,
                    entry["y0"] + (cy + h / 2) * t,
                )
                # 1 px margin: label quantization can nudge an edge-clipped
                # box a hair inside the tile bounds
                interior = (
                    entry["x0"] + 1.0 < px[0]
                    and entry["y0"] + 1.0 < px[1]
                    and px[2] < entry["x0"] + t - 1.0
                    and px[3] < entry["y0"] + t - 1.0
                )
                if not interior:
                    continue
                self.assertTrue(
                    any(
                        all(abs(a - b) < 1e-3 for a, b in zip(px, fp))
                        for fp in full_px
                    ),
                    f"tile box {px} matches no full-frame box",
                )
                checked += 1
        self.assertGreater(checked, 0)

    def test_no_emitted_tile_is_poisoned(self):
        for entry in self.index["tiles"]:
            if entry.get("full"):
                continue
            k = self._station_of(entry["src"])
            st = SCENE.stations[k]
            n_lines = len(
                (self.out / "train" / "labels" / f"{entry['tile']}.txt")
                .read_text()
                .splitlines()
            )
            t = self.TP.tile
            x0, y0 = entry["x0"], entry["y0"]
            learnable = 0
            for mine in SCENE.mines:
                ext = raw_extents(CAM, st.pos, st.q, mine, CFG.mine_dims_m)
                if ext is None:
                    continue
                u0, u1, v0, v1 = ext
                inter = max(0.0, min(u1, x0 + t) - max(u0, x0)) * max(
                    0.0, min(v1, y0 + t) - max(v0, y0)
                )
                raw = (u1 - u0) * (v1 - v0)
                if raw > 0.0 and inter / raw >= 0.15:
                    learnable += 1
            self.assertGreaterEqual(
                n_lines,
                learnable,
                f"{entry['tile']} has learnable mines without labels",
            )

    def test_fullframe_slice_scales_min_box_px(self):
        floor = CFG.min_box_px * W / 640.0 - 1e-6
        for entry in self.index["tiles"]:
            if not entry.get("full"):
                continue
            for line in (
                (self.out / "train" / "labels" / f"{entry['tile']}.txt")
                .read_text()
                .splitlines()
            ):
                _, _, _, w, h = (float(v) for v in line.split())
                self.assertGreaterEqual(w * W, floor)
                self.assertGreaterEqual(h * H, floor)

    def test_deterministic(self):
        out2 = Path(tempfile.mkdtemp())
        write_scene(CFG, 0, out2)
        tile_scene(out2, _sidecar({}), 0.15, self.TP)
        index2 = json.loads((out2 / "train" / "tiles.json").read_text())
        self.assertEqual(self.index, index2[f"{CFG.seed}_s0000"])

    def test_tile_parameter_validation(self):
        for changes in (
            {"tile": 0},
            {"overlap": self.TP.tile},
            {"empty_keep": -0.01},
            {"empty_keep": 1.01},
            {"fullframe_frac": -0.01},
            {"fullframe_frac": 1.01},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(self.TP, **changes)


if __name__ == "__main__":
    unittest.main()
