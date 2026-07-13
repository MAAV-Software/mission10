import json
import tempfile
import unittest
from pathlib import Path

from datagen.config import GenConfig
from datagen.dump import write_scene
from datagen.manifest import SCHEMA, scene_manifest
from datagen.scene import build_scene, scene_labels

CFG = GenConfig()


class TestScene(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(build_scene(CFG, 0), build_scene(CFG, 0))

    def test_scenes_differ(self):
        self.assertNotEqual(build_scene(CFG, 0).mines, build_scene(CFG, 1).mines)

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


if __name__ == "__main__":
    unittest.main()
