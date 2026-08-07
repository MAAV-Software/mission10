import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from train.compose import COMPOSITION_PRESETS, compose


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared(root, name, train_count, full_dataset=False):
    prepared = root / name
    splits = ("train", "val", "test") if full_dataset else ("train",)
    entries = {split: [] for split in splits}
    for split in entries:
        count = train_count if split == "train" else 1
        image_dir = prepared / "images" / split
        label_dir = prepared / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(count):
            tile = f"{name}_{split}_{index:03d}"
            image = image_dir / f"{tile}.png"
            label = label_dir / f"{tile}.txt"
            image.write_bytes(f"image:{tile}".encode())
            label.write_text("0 0.5 0.5 0.25 0.25\n")
            entries[split].append(
                {
                    "scene": f"{name}_scene",
                    "tile": tile,
                    "boxes": 1,
                    "image_sha256": _sha256(image),
                    "label_sha256": _sha256(label),
                }
            )
    lock = {
        "schema": (
            "mission10-yolo-dataset/1"
            if full_dataset
            else "mission10-yolo-training-component/1"
        ),
        "dataset_sha256": hashlib.sha256(name.encode()).hexdigest(),
        "entries": entries,
    }
    lock_name = "split.lock.json" if full_dataset else "component.lock.json"
    (prepared / lock_name).write_text(json.dumps(lock) + "\n")
    (prepared / "dataset.yaml").write_text("names:\n  0: mine\n")
    return prepared


class TestComposition(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.components = {
            "production": _prepared(
                self.root, "production", 7, full_dataset=True
            ),
            "appearance": _prepared(
                self.root, "appearance", 2, full_dataset=True
            ),
            "hardneg": _prepared(self.root, "hardneg", 1),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_preset_fractions_are_approved(self):
        self.assertEqual(
            COMPOSITION_PRESETS["real_positive"],
            {
                "production": "0.65",
                "appearance": "0.15",
                "hardneg": "0.10",
                "real_positive": "0.10",
            },
        )

    def test_exact_epoch_fractions_repeat_train_and_not_validation(self):
        out = self.root / "composed"
        lock = compose(out, "combined", self.components)
        self.assertEqual(
            lock["counts"]["train"],
            {"production": 14, "appearance": 3, "hardneg": 3},
        )
        self.assertEqual(
            lock["counts"]["val"],
            {"production": 1},
        )
        self.assertEqual(lock["counts"]["test"], lock["counts"]["val"])
        self.assertEqual(
            lock["components"]["appearance"]["source_counts"],
            {"train": 2, "val": 1, "test": 1},
        )
        self.assertEqual(
            lock["components"]["hardneg"]["source_lock"],
            "component.lock.json",
        )
        self.assertEqual(len(list((out / "images" / "train").iterdir())), 20)
        self.assertEqual(len(list((out / "images" / "val").iterdir())), 1)
        names = [path.name for path in (out / "images" / "train").iterdir()]
        self.assertEqual(len(names), len(set(names)))
        source = (
            self.components["hardneg"]
            / "images"
            / "train"
            / "hardneg_train_000.png"
        )
        repeated = sorted((out / "images" / "train").glob("hardneg__*.png"))
        self.assertEqual(len(repeated), 3)
        self.assertTrue(
            all(
                os.stat(path).st_ino == os.stat(source).st_ino
                for path in repeated
            )
        )

    def test_lock_is_deterministic(self):
        first = compose(self.root / "first", "combined", self.components)
        second = compose(self.root / "second", "combined", self.components)
        self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])
        self.assertEqual(first, second)

    def test_dataset_identity_excludes_path_dependent_provenance(self):
        first = compose(self.root / "identity-first", "combined", self.components)
        lock_path = self.components["appearance"] / "split.lock.json"
        source_lock = json.loads(lock_path.read_text())
        source_lock["source"] = "/a/different/deployment/path"
        lock_path.write_text(json.dumps(source_lock) + "\n")
        second = compose(self.root / "identity-second", "combined", self.components)
        self.assertNotEqual(
            first["components"]["appearance"]["source_lock_sha256"],
            second["components"]["appearance"]["source_lock_sha256"],
        )
        self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])

    def test_train_only_component_requires_component_lock_filename(self):
        component_lock = self.components["hardneg"] / "component.lock.json"
        component_lock.rename(self.components["hardneg"] / "split.lock.json")
        with self.assertRaisesRegex(
            ValueError, "hardneg: missing component.lock.json"
        ):
            compose(self.root / "wrong-lock-name", "combined", self.components)

    def test_rejects_tampered_component_before_creating_output(self):
        label = (
            self.components["appearance"]
            / "labels"
            / "train"
            / "appearance_train_000.txt"
        )
        label.write_text("0 0.5 0.5 0.5 0.5\n")
        out = self.root / "bad"
        with self.assertRaisesRegex(ValueError, "label hash mismatch"):
            compose(
                out,
                "appearance",
                {
                    "production": self.components["production"],
                    "appearance": self.components["appearance"],
                },
            )
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
