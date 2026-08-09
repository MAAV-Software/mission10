import json
import os
import tempfile
import unittest
from pathlib import Path

from train.prepare import LOCK_SCHEMA, SPLIT_SCHEMA, prepare


class TestPrepareDataset(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.raw = self.root / "raw"
        images = self.raw / "train" / "images"
        labels = self.raw / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        index = {}
        for scene in range(3):
            scene_key = f"m10_s{scene:04d}"
            tile = f"{scene_key}_k0000_x0000_y0000"
            (images / f"{tile}.png").write_bytes(b"fake-png" + bytes([scene]))
            label_text = "" if scene == 2 else "0 0.5 0.5 0.2 0.1\n"
            (labels / f"{tile}.txt").write_text(label_text)
            (self.raw / f"{scene_key}.manifest.json").write_text(
                json.dumps({"seed": "m10", "scene": scene})
            )
            index[scene_key] = {
                "tile_px": 640,
                "overlap_px": 192,
                "tiles": [
                    {
                        "tile": tile,
                        "src": f"{scene_key}_k0000",
                        "x0": 0,
                        "y0": 0,
                    }
                ],
            }
        (self.raw / "train" / "tiles.json").write_text(json.dumps(index))
        self.split = self.root / "split.json"
        self.split.write_text(
            json.dumps(
                {
                    "schema": SPLIT_SCHEMA,
                    "seed": "m10",
                    "scenes": {"train": [0], "val": [1], "test": [2]},
                }
            )
        )

    def test_prepares_hardlinked_scene_safe_dataset(self):
        out = self.root / "prepared"
        lock = prepare(self.raw, out, self.split)
        self.assertEqual(lock["schema"], LOCK_SCHEMA)
        self.assertEqual(lock["counts"]["train"], {"tiles": 1, "boxes": 1, "empty": 0})
        self.assertEqual(lock["counts"]["val"], {"tiles": 1, "boxes": 1, "empty": 0})
        self.assertEqual(lock["counts"]["test"], {"tiles": 1, "boxes": 0, "empty": 1})
        source = self.raw / "train" / "images" / "m10_s0000_k0000_x0000_y0000.png"
        linked = out / "images" / "train" / source.name
        self.assertEqual(os.stat(source).st_ino, os.stat(linked).st_ino)
        self.assertIn("0: mine", (out / "dataset.yaml").read_text())

    def test_dataset_hash_is_independent_of_output_path(self):
        first = prepare(self.raw, self.root / "prepared-a", self.split)
        second = prepare(self.raw, self.root / "prepared-b", self.split)
        self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])

    def test_rejects_scene_leakage(self):
        self.split.write_text(
            json.dumps(
                {
                    "schema": SPLIT_SCHEMA,
                    "seed": "m10",
                    "scenes": {"train": [0], "val": [0], "test": [2]},
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "both train and val"):
            prepare(self.raw, self.root / "prepared", self.split)

    def test_rejects_missing_pair(self):
        missing = self.raw / "train" / "images" / "m10_s0001_k0000_x0000_y0000.png"
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "missing image/label pair"):
            prepare(self.raw, self.root / "prepared", self.split)

    def test_rejects_malformed_label(self):
        label = self.raw / "train" / "labels" / "m10_s0000_k0000_x0000_y0000.txt"
        label.write_text("1 0.5 0.5 0.2 0.1\n")
        with self.assertRaisesRegex(ValueError, "expected class 0"):
            prepare(self.raw, self.root / "prepared", self.split)

    def test_rejects_nonempty_output(self):
        out = self.root / "prepared"
        out.mkdir()
        (out / "keep").write_text("user data")
        with self.assertRaisesRegex(ValueError, "not empty"):
            prepare(self.raw, out, self.split)


if __name__ == "__main__":
    unittest.main()
