import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from audit.hard_negative_review import (
    ReviewSession,
    SourceImageCache,
    confirmation_counts,
    display_indices,
    qa_display_indices,
    resume_position,
    serve,
    update_confirmation,
)
from audit.hard_negatives import propose
from audit.labels import (
    LABEL_SCHEMA,
    certify_labels,
    freeze_roles,
    sha256,
    write_labels,
)


class TestHardNegativeReview(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow unavailable")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.png"
        image = Image.new("RGB", (12, 4))
        for x in range(image.width):
            for y in range(image.height):
                image.putpixel((x, y), (x * 10, y * 20, x + y))
        image.save(self.source)

        labels = {
            "schema": LABEL_SCHEMA,
            "images": [
                {
                    "source": "source.png",
                    "source_sha256": sha256(self.source),
                    "width": 12,
                    "height": 4,
                    "capture_group": "phone-training",
                    "role": "training_candidate",
                    "review_state": "complete",
                    "objects": [{"xyxy": [0, 0, 2, 2], "visibility": "clear"}],
                    "ignore_regions": [],
                }
            ],
        }
        self.labels = self.root / "labels.json"
        labels = freeze_roles(labels, "split owner", now="freeze")
        labels = certify_labels(labels, "reviewer", now="certify")
        write_labels(self.labels, labels)

        self.baseline = self.root / "baseline.json"
        self.baseline.write_text(
            json.dumps(
                {
                    "schema": "mission10-yolo-irl-audit/1",
                    "tile_px": 4,
                    "overlap_px": 0,
                    "candidate_floor": 0.001,
                    "images": [
                        {
                            "source_sha256": sha256(self.source),
                            "width": 12,
                            "height": 4,
                            "candidates": [
                                {
                                    "x0": 8,
                                    "y0": 0,
                                    "x1": 10,
                                    "y1": 2,
                                    "confidence": 0.42,
                                    "tile_x": 8,
                                    "tile_y": 0,
                                }
                            ],
                        }
                    ],
                }
            )
        )
        self.review = self.root / "review.json"
        self.document = propose(self.labels, self.baseline)
        self.review.write_text(json.dumps(self.document, indent=2) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_update_authorizes_only_a_confirmation_decision(self):
        original = deepcopy(self.document)
        entry_id = self.document["entries"][0]["id"]

        updated = update_confirmation(
            self.document, {"id": entry_id, "confirmation": "confirmed"}
        )

        self.assertEqual(self.document, original)
        changed = [
            (before, after)
            for before, after in zip(original["entries"], updated["entries"])
            if before != after
        ]
        self.assertEqual(len(changed), 1)
        before, after = changed[0]
        self.assertEqual(
            {key: value for key, value in before.items() if key != "confirmation"},
            {key: value for key, value in after.items() if key != "confirmation"},
        )
        with self.assertRaisesRegex(ValueError, "only id and confirmation"):
            update_confirmation(
                self.document,
                {"id": entry_id, "confirmation": "rejected", "source": "other"},
            )
        with self.assertRaisesRegex(ValueError, "confirmed or rejected"):
            update_confirmation(
                self.document, {"id": entry_id, "confirmation": "pending"}
            )

    def test_resume_uses_source_grouped_display_order_and_counts(self):
        labels = json.loads(self.labels.read_text())
        order = display_indices(self.document, labels)
        kinds = [self.document["entries"][index]["kind"] for index in order]
        self.assertEqual(kinds[0], "baseline_candidate")

        reviewed = deepcopy(self.document)
        for entry in reviewed["entries"]:
            entry["confirmation"] = (
                "confirmed" if entry["kind"] == "baseline_candidate" else "pending"
            )
        position = resume_position(reviewed, display_indices(reviewed, labels))
        self.assertEqual(kinds[position], "deterministic_clean")
        self.assertEqual(
            confirmation_counts(reviewed),
            {"total": 2, "pending": 1, "confirmed": 1, "rejected": 0},
        )

    def test_qa_sample_is_bounded_and_preserves_existing_decisions(self):
        review = deepcopy(self.document)
        decided_id = review["entries"][0]["id"]
        review["entries"][0]["confirmation"] = "confirmed"
        labels = json.loads(self.labels.read_text())

        order = qa_display_indices(review, labels, 1)

        self.assertEqual(len(order), 1)
        self.assertEqual(review["entries"][order[0]]["id"], decided_id)

    def test_session_atomically_preserves_provenance(self):
        session = ReviewSession(self.review, self.labels, self.baseline)
        entry_id = session.state()["resume_id"]
        before = json.loads(self.review.read_text())

        state = session.update({"id": entry_id, "confirmation": "rejected"})

        after = json.loads(self.review.read_text())
        before_by_id = {entry["id"]: entry for entry in before["entries"]}
        after_by_id = {entry["id"]: entry for entry in after["entries"]}
        self.assertEqual(after_by_id[entry_id]["confirmation"], "rejected")
        for key in after:
            if key != "entries":
                self.assertEqual(after[key], before[key])
        for candidate_id in after_by_id:
            immutable_before = dict(before_by_id[candidate_id])
            immutable_after = dict(after_by_id[candidate_id])
            immutable_before.pop("confirmation")
            immutable_after.pop("confirmation")
            self.assertEqual(immutable_after, immutable_before)
        self.assertEqual(state["counts"]["rejected"], 1)
        self.assertEqual(list(self.root.glob(".review.json.*.tmp")), [])

    def test_crop_is_lossless_and_exact_deployment_size(self):
        from PIL import Image

        session = ReviewSession(self.review, self.labels, self.baseline)
        baseline_entry = next(
            entry for entry in session.state()["entries"]
            if entry["kind"] == "baseline_candidate"
        )
        content = session.crop(baseline_entry["id"], SourceImageCache())
        with Image.open(io.BytesIO(content)) as crop:
            self.assertEqual(crop.size, (4, 4))
            self.assertEqual(crop.getpixel((0, 0)), (80, 0, 8))

    def test_non_loopback_bind_is_rejected_before_loading_files(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            serve(
                Path("missing-review"),
                Path("missing-labels"),
                Path("missing-baseline"),
                host="0.0.0.0",
            )


if __name__ == "__main__":
    unittest.main()
