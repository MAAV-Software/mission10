import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from audit.folds import (
    DEFAULT_SEED,
    build_fold_document,
    select_records,
    source_hashes,
    validate_fold_document,
)
from audit.labels import LABEL_SCHEMA, certify_labels, freeze_roles, sha256


def _labels(root: Path) -> tuple[Path, dict]:
    images = []
    strata = ["clear"] * 5 + ["partial"] * 7 + ["negative"] * 8
    for index, stratum in enumerate(strata):
        source = root / f"source-{index:02d}.png"
        source.write_bytes(f"source {index}".encode())
        objects = []
        if stratum != "negative":
            objects.append({"xyxy": [1, 1, 3, 3], "visibility": stratum})
        images.append(
            {
                "source": source.name,
                "source_sha256": sha256(source),
                "width": 4,
                "height": 4,
                "capture_group": "phone",
                "role": "training_candidate",
                "review_state": "complete",
                "objects": objects,
                "ignore_regions": [],
            }
        )
    document = certify_labels(
        freeze_roles({"schema": LABEL_SCHEMA, "images": images}, "owner"),
        "reviewer",
    )
    path = root / "labels.json"
    path.write_text(json.dumps(document) + "\n")
    return path, document


class TestRealFolds(unittest.TestCase):
    def test_build_is_deterministic_balanced_and_exact_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, labels = _labels(Path(temporary))
            first = build_fold_document(labels, sha256(path), folds=5)
            second = build_fold_document(labels, sha256(path), folds=5)

            self.assertEqual(first, second)
            validate_fold_document(first, labels=labels, labels_sha256=sha256(path))
            self.assertEqual(len(first["assignments"]), 20)
            self.assertEqual(
                {item["source_sha256"] for item in first["assignments"]},
                {record["source_sha256"] for record in labels["images"]},
            )
            clear_counts = [first["counts"][str(i)]["clear"] for i in range(5)]
            partial_counts = [first["counts"][str(i)]["partial"] for i in range(5)]
            self.assertLessEqual(max(clear_counts) - min(clear_counts), 1)
            self.assertLessEqual(max(partial_counts) - min(partial_counts), 1)

    def test_selection_has_disjoint_complete_sides(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, labels = _labels(Path(temporary))
            folds = build_fold_document(labels, sha256(path), folds=5)
            held_out = source_hashes(folds, 0)
            training = source_hashes(folds, 0, held_out=False)

            self.assertFalse(held_out & training)
            self.assertEqual(len(held_out | training), len(labels["images"]))
            selected = select_records(labels["images"], folds, 0)
            self.assertEqual({record["source_sha256"] for record in selected}, held_out)

    def test_tampering_and_changed_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, labels = _labels(Path(temporary))
            folds = build_fold_document(labels, sha256(path), folds=5)
            changed = deepcopy(folds)
            changed["assignments"][0]["fold"] = 4
            with self.assertRaisesRegex(ValueError, "changed after locking"):
                validate_fold_document(changed)
            with self.assertRaisesRegex(ValueError, "labels hash changed"):
                validate_fold_document(folds, labels_sha256="0" * 64)

    def test_seed_changes_assignments_without_changing_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, labels = _labels(Path(temporary))
            first = build_fold_document(labels, sha256(path), seed=DEFAULT_SEED)
            second = build_fold_document(labels, sha256(path), seed="other-seed")

            self.assertNotEqual(first["assignments"], second["assignments"])
            self.assertEqual(first["counts"], second["counts"])


if __name__ == "__main__":
    unittest.main()
