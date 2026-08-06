import json
import os
import tempfile
import unittest
from pathlib import Path

from export.calibrate import CALIBRATION_SCHEMA, build_calibration
from train.prepare import SPLIT_SCHEMA, prepare


class TestCalibrationDataset(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.raw = self.root / "raw"
        images = self.raw / "train" / "images"
        labels = self.raw / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        index = {}
        for scene in range(4):
            scene_key = f"m10_s{scene:04d}"
            tiles = []
            stations = []
            for station_index, altitude in enumerate((2.0, 4.0, 6.0)):
                source = f"{scene_key}_k{station_index:04d}"
                tile = f"{source}_x0000_y0000"
                (images / f"{tile}.png").write_bytes(
                    b"fake-png" + bytes([scene, station_index])
                )
                if station_index == 0:
                    label_text = ""
                elif station_index == 1:
                    label_text = "0 0.5 0.5 0.02 0.02\n"
                else:
                    label_text = "0 0.5 0.5 0.08 0.08\n"
                (labels / f"{tile}.txt").write_text(label_text)
                tiles.append({"tile": tile, "src": source, "x0": 0, "y0": 0})
                stations.append(
                    {
                        "station_index": station_index,
                        "stem": source,
                        "pos": [0.0, 0.0, -altitude],
                    }
                )
            manifest = {
                "seed": "m10",
                "scene": scene,
                "surface": {
                    "primary": "grass" if scene % 2 else "dirt",
                    "secondary": None,
                },
                "grass": {"profile": "sparse"} if scene % 2 else None,
                "tag_visible_fraction": scene / 3,
                "mines": [
                    {
                        "appearance": {
                            "color_family": "green" if scene % 2 else "muddy_olive"
                        }
                    }
                ],
                "stations": stations,
            }
            (self.raw / f"{scene_key}.manifest.json").write_text(json.dumps(manifest))
            index[scene_key] = {"tile_px": 640, "overlap_px": 192, "tiles": tiles}
        (self.raw / "train" / "tiles.json").write_text(json.dumps(index))
        split = self.root / "split.json"
        split.write_text(
            json.dumps(
                {
                    "schema": SPLIT_SCHEMA,
                    "seed": "m10",
                    "scenes": {"train": [0, 1], "val": [2], "test": [3]},
                }
            )
        )
        self.prepared = self.root / "prepared"
        prepare(self.raw, self.prepared, split)

    def test_builds_train_only_covered_hardlink_set(self):
        out = self.root / "calibration"
        lock = build_calibration(self.prepared, self.raw, out, count=6)
        self.assertEqual(lock["schema"], CALIBRATION_SCHEMA)
        self.assertEqual(lock["count"], 6)
        self.assertEqual({entry["scene"] for entry in lock["images"]}, {"m10_s0000", "m10_s0001"})
        selected_features = lock["selected_feature_counts"]
        for feature in lock["candidate_feature_counts"]:
            self.assertGreater(selected_features[feature], 0)
        first = lock["images"][0]["tile"]
        first_file = lock["images"][0]["file"]
        source = self.prepared / "images" / "train" / f"{first}.png"
        linked = out / "images" / first_file
        self.assertEqual(os.stat(source).st_ino, os.stat(linked).st_ino)

    def test_selection_is_reproducible(self):
        first = build_calibration(
            self.prepared, self.raw, self.root / "calibration-a", count=5
        )
        second = build_calibration(
            self.prepared, self.raw, self.root / "calibration-b", count=5
        )
        self.assertEqual(first["calibration_sha256"], second["calibration_sha256"])
        self.assertEqual(first["images"], second["images"])

    def test_rejects_more_images_than_train_split(self):
        with self.assertRaisesRegex(ValueError, "only 6 candidates"):
            build_calibration(
                self.prepared, self.raw, self.root / "calibration", count=7
            )

    def test_rejects_nonempty_output(self):
        out = self.root / "calibration"
        out.mkdir()
        (out / "keep").write_text("user data")
        with self.assertRaisesRegex(ValueError, "not empty"):
            build_calibration(self.prepared, self.raw, out, count=5)


if __name__ == "__main__":
    unittest.main()
