import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from train.compose import COMPOSITION_PRESETS, compose


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared(root, name, train_count, full_dataset=False, source_hash=None):
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
            entry = {
                    "scene": f"{name}_scene",
                    "tile": tile,
                    "boxes": 1,
                    "image_sha256": _sha256(image),
                    "label_sha256": _sha256(label),
                }
            if source_hash is not None:
                entry["provenance"] = {"source_sha256": source_hash}
            entries[split].append(entry)
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
                "production": "0.75",
                "hardneg": "0.15",
                "real_positive": "0.10",
            },
        )
        self.assertEqual(
            COMPOSITION_PRESETS["real_positive_appearance"],
            {
                "production": "0.65",
                "appearance": "0.10",
                "hardneg": "0.15",
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

    def test_real_positive_rejects_held_out_photo_provenance(self):
        held_out = "a" * 64
        assignments = [
            {
                "source": "held-out.png",
                "source_sha256": held_out,
                "fold": 0,
                "stratum": "negative",
                "objects": 0,
            }
        ]
        fold = {
            "schema": "mission10-yolo-real-folds/1",
            "labels_sha256": "b" * 64,
            "assignments_sha256": hashlib.sha256(
                json.dumps(
                    assignments, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "role": "training_candidate",
            "grouping": "exact_source_photo",
            "seed": "test",
            "fold_count": 2,
            "assignments": assignments,
            "counts": {
                "0": {"photos": 1, "clear": 0, "partial": 0, "negative": 1},
                "1": {"photos": 0, "clear": 0, "partial": 0, "negative": 0},
            },
        }
        fold_path = self.root / "folds.json"
        fold_path.write_text(json.dumps(fold))
        components = {
            "production": self.components["production"],
            "hardneg": _prepared(
                self.root, "leaking_hardneg", 1, source_hash=held_out
            ),
            "real_positive": _prepared(
                self.root, "real_positive", 1, source_hash="c" * 64
            ),
        }

        with self.assertRaisesRegex(ValueError, "held-out source"):
            compose(
                self.root / "leaking-composition",
                "real_positive",
                components,
                fold_lock_path=fold_path,
                held_out_fold=0,
            )

    def test_real_positive_appearance_requires_fold_lock(self):
        components = {
            **self.components,
            "real_positive": _prepared(self.root, "real_positive", 1),
        }
        with self.assertRaisesRegex(ValueError, "requires a held-out fold lock"):
            compose(
                self.root / "appearance-replay-without-fold",
                "real_positive_appearance",
                components,
            )


if __name__ == "__main__":
    unittest.main()
