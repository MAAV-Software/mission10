import json
import tempfile
import unittest
from pathlib import Path

from audit.hard_negatives import COMPONENT_SCHEMA, materialize, propose
from audit.labels import (
    LABEL_SCHEMA,
    certify_labels,
    freeze_roles,
    sha256,
    write_labels,
)


class TestHardNegativeMaterializer(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow unavailable")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "source.png"
        # A gradient ensures deterministic clean crops are not byte duplicates.
        image = Image.new("RGB", (2000, 640))
        for x in range(image.width):
            for y in range(0, image.height, 32):
                image.putpixel((x, y), (x % 251, y % 251, (x + y) % 251))
        image.save(self.image_path)
        document = {
            "schema": LABEL_SCHEMA,
            "images": [
                {
                    "source": "source.png",
                    "source_sha256": sha256(self.image_path),
                    "width": 2000,
                    "height": 640,
                    "capture_group": "phone-candidates",
                    "role": "training_candidate",
                    "review_state": "complete",
                    "objects": [
                        {"xyxy": [10, 10, 50, 50], "visibility": "clear"}
                    ],
                    "ignore_regions": [],
                }
            ],
        }
        self.labels = self.root / "labels.json"
        frozen = freeze_roles(document, "split owner", now="freeze")
        write_labels(
            self.labels, certify_labels(frozen, "human reviewer", now="certify")
        )
        self.baseline = self.root / "audit.json"
        self.baseline.write_text(
            json.dumps(
                {
                    "schema": "mission10-yolo-irl-audit/1",
                    "tile_px": 640,
                    "overlap_px": 192,
                    "candidate_floor": 0.001,
                    "images": [
                        {
                            "source_sha256": sha256(self.image_path),
                            "width": 2000,
                            "height": 640,
                            "candidates": [
                                {
                                    "x0": 10,
                                    "y0": 10,
                                    "x1": 50,
                                    "y1": 50,
                                    "confidence": 0.9,
                                    "tile_x": 0,
                                    "tile_y": 0,
                                },
                                {
                                    "x0": 1500,
                                    "y0": 20,
                                    "x1": 1550,
                                    "y1": 70,
                                    "confidence": 0.2,
                                    "tile_x": 1360,
                                    "tile_y": 0,
                                },
                            ],
                        }
                    ],
                }
            )
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_proposal_excludes_mines_and_requires_human_decisions(self):
        review = propose(self.labels, self.baseline)

        rectangles = [entry["tile_xyxy"] for entry in review["entries"]]
        self.assertNotIn([0, 0, 640, 640], rectangles)
        self.assertIn([1360, 0, 2000, 640], rectangles)
        self.assertTrue(all(entry["confirmation"] == "pending" for entry in review["entries"]))

        review_path = self.root / "review.json"
        review_path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            materialize(review_path, self.labels, self.baseline, self.root / "out")

    def test_confirmed_output_is_train_only_component(self):
        review = propose(self.labels, self.baseline)
        for entry in review["entries"]:
            entry["confirmation"] = (
                "confirmed" if entry["kind"] == "baseline_candidate" else "rejected"
            )
        review_path = self.root / "review.json"
        review_path.write_text(json.dumps(review))

        lock = materialize(review_path, self.labels, self.baseline, self.root / "out")

        self.assertEqual(lock["schema"], COMPONENT_SCHEMA)
        self.assertEqual(lock["scope"], "train_only")
        self.assertEqual(len(lock["entries"]["train"]), 1)
        entry = lock["entries"]["train"][0]
        self.assertEqual(entry["boxes"], 0)
        image_path = self.root / "out" / "images" / "train" / f"{entry['tile']}.png"
        label_path = self.root / "out" / "labels" / "train" / f"{entry['tile']}.txt"
        self.assertTrue(image_path.is_file())
        self.assertTrue(label_path.is_file())
        self.assertEqual(
            lock["selection_basis_counts"],
            {
                "human_confirmed": 1,
                "certified_annotation_absence": 0,
                "human_rejected": len(review["entries"]) - 1,
            },
        )

    def test_certification_backed_output_includes_pending_but_not_rejected(self):
        review = propose(self.labels, self.baseline)
        baseline_entry = next(
            entry for entry in review["entries"]
            if entry["kind"] == "baseline_candidate"
        )
        clean_entries = [
            entry for entry in review["entries"]
            if entry["kind"] == "deterministic_clean"
        ]
        baseline_entry["confirmation"] = "rejected"
        review_path = self.root / "review.json"
        review_path.write_text(json.dumps(review))

        lock = materialize(
            review_path,
            self.labels,
            self.baseline,
            self.root / "out",
            certification_backed=True,
        )

        self.assertEqual(lock["component"], "certification-backed-real-hard-negatives")
        self.assertEqual(lock["counts"]["train"]["tiles"], len(clean_entries))
        self.assertEqual(
            {
                entry["provenance"]["review_entry_id"]
                for entry in lock["entries"]["train"]
            },
            {entry["id"] for entry in clean_entries},
        )
        self.assertEqual(
            lock["entries"]["train"][0]["provenance"]["selection_basis"],
            "certified_annotation_absence",
        )
        self.assertEqual(
            lock["selection_basis_counts"],
            {
                "human_confirmed": 0,
                "certified_annotation_absence": len(clean_entries),
                "human_rejected": 1,
            },
        )

    def test_review_can_change_confirmation_but_not_provenance(self):
        review = propose(self.labels, self.baseline)
        for entry in review["entries"]:
            entry["confirmation"] = "rejected"
        review["entries"][0]["capture_group"] = "tampered"
        review_path = self.root / "review.json"
        review_path.write_text(json.dumps(review))

        with self.assertRaisesRegex(ValueError, "provenance was modified"):
            materialize(review_path, self.labels, self.baseline, self.root / "out")


if __name__ == "__main__":
    unittest.main()
