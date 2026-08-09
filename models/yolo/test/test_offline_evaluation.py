import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from audit.labels import LABEL_SCHEMA, certify_labels, freeze_roles, write_labels
from audit.offline_evaluation import OFFLINE_EVALUATION_SCHEMA, run


TRAIN_HASH = "a" * 64
DEV_HASH = "b" * 64


def _labels():
    document = {
        "schema": LABEL_SCHEMA,
        "images": [
            {
                "source": "training.png",
                "source_sha256": TRAIN_HASH,
                "width": 1500,
                "height": 640,
                "capture_group": "phone-training",
                "role": "training_candidate",
                "review_state": "complete",
                "objects": [
                    {"xyxy": [10, 10, 50, 50], "visibility": "clear"},
                    {"xyxy": [500, 10, 540, 50], "visibility": "partial"},
                ],
                "ignore_regions": [],
            },
            {
                "source": "development.png",
                "source_sha256": DEV_HASH,
                "width": 640,
                "height": 640,
                "capture_group": "phone-development",
                "role": "development_eval",
                "review_state": "complete",
                "objects": [],
                "ignore_regions": [],
            },
        ],
    }
    return certify_labels(
        freeze_roles(document, "split owner", now="freeze"),
        "human reviewer",
        now="certify",
    )


def _candidate(x0, y0, x1, y1, confidence, tile_x, tile_y=0):
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "confidence": confidence,
        "tile_x": tile_x,
        "tile_y": tile_y,
    }


def _audit():
    return {
        "schema": "mission10-yolo-irl-audit/1",
        "weights": "/frozen/best.pt",
        "weights_sha256": "c" * 64,
        "threshold": 0.37,
        "candidate_floor": 0.001,
        "tile_px": 640,
        "overlap_px": 192,
        "merge_overlap": 0.5,
        "images": [
            {
                "source": "/capture/training.png",
                "source_sha256": TRAIN_HASH,
                "width": 1500,
                "height": 640,
                # mission_engine tile_grid gives 0, 430, 860, not a fixed stride.
                "tiles": 3,
                "candidates": [
                    _candidate(10, 10, 50, 50, 0.9, 0),
                    _candidate(500, 10, 540, 50, 0.2, 430),
                    _candidate(1000, 100, 1040, 140, 0.8, 860),
                ],
            },
            {
                "source": "/capture/development.png",
                "source_sha256": DEV_HASH,
                "width": 640,
                "height": 640,
                "tiles": 1,
                "candidates": [],
            },
        ],
    }


class TestOfflineEvaluation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.labels_path = self.root / "labels.json"
        self.audit_path = self.root / "audit.json"
        write_labels(self.labels_path, _labels())
        self.audit_path.write_text(json.dumps(_audit()))

    def tearDown(self):
        self.temporary.cleanup()

    def test_training_candidate_is_diagnostic_with_exact_metrics(self):
        report = run(
            self.audit_path,
            self.labels_path,
            "training_candidate",
            self.root / "report.json",
        )

        self.assertEqual(report["schema"], OFFLINE_EVALUATION_SCHEMA)
        self.assertEqual(report["report_classification"], "diagnostic_only")
        self.assertFalse(report["promotion_use_permitted"])
        self.assertIn("not an evaluation role", report["non_promotion_reason"])
        self.assertEqual(report["weights_sha256"], "c" * 64)
        self.assertEqual(len(report["audit_sha256"]), 64)
        self.assertEqual(len(report["labels_sha256"]), 64)

        metrics = report["metrics"]
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["recall_by_visibility"]["clear"]["recall"], 1.0)
        self.assertEqual(metrics["recall_by_visibility"]["partial"]["recall"], 0.0)
        self.assertEqual(metrics["empty_real_tiles"], 1)
        self.assertEqual(metrics["empty_real_tiles_with_false_positive"], 1)
        self.assertEqual(metrics["empty_real_tile_false_positive_rate"], 1.0)
        self.assertEqual(
            report["extra_unselected_audit_records"],
            [
                {
                    "source": "/capture/development.png",
                    "source_sha256": DEV_HASH,
                    "label_role": "development_eval",
                }
            ],
        )

    def test_configurable_threshold_and_merge_override_are_recorded(self):
        report = run(
            self.audit_path,
            self.labels_path,
            "training_candidate",
            self.root / "report.json",
            threshold=0.1,
            merge_overlap=0.7,
        )

        self.assertEqual(report["threshold"], 0.1)
        self.assertEqual(report["merge_overlap"], 0.7)
        self.assertEqual((report["metrics"]["tp"], report["metrics"]["fn"]), (2, 0))

    def test_evaluation_role_is_not_marked_diagnostic(self):
        report = run(
            self.audit_path,
            self.labels_path,
            "development_eval",
            self.root / "report.json",
        )

        self.assertEqual(report["report_classification"], "promotion_evaluation")
        self.assertTrue(report["promotion_use_permitted"])
        self.assertIsNone(report["non_promotion_reason"])
        self.assertIn("clear", report["metrics"]["recall_by_visibility"])
        self.assertIn("partial", report["metrics"]["recall_by_visibility"])

    def test_rejects_missing_selected_source_and_existing_output(self):
        audit = _audit()
        audit["images"] = audit["images"][1:]
        self.audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "missing labeled sources"):
            run(
                self.audit_path,
                self.labels_path,
                "training_candidate",
                self.root / "report.json",
            )

        output = self.root / "exists.json"
        output.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            run(
                self.audit_path,
                self.labels_path,
                "training_candidate",
                output,
            )
        self.assertEqual(output.read_text(), "keep me")

    def test_rejects_duplicate_and_invalid_origin_but_allows_unlabeled_extra(self):
        duplicate = _audit()
        duplicate["images"].append(deepcopy(duplicate["images"][0]))
        self.audit_path.write_text(json.dumps(duplicate))
        with self.assertRaisesRegex(ValueError, "duplicate source hash"):
            run(
                self.audit_path,
                self.labels_path,
                "training_candidate",
                self.root / "duplicate.json",
            )

        unmatched = _audit()
        unmatched["images"][1]["source_sha256"] = "d" * 64
        self.audit_path.write_text(json.dumps(unmatched))
        report = run(
            self.audit_path,
            self.labels_path,
            "training_candidate",
            self.root / "unmatched.json",
        )
        self.assertEqual(
            report["extra_unselected_audit_records"][0]["source_sha256"],
            "d" * 64,
        )
        self.assertNotIn("label_role", report["extra_unselected_audit_records"][0])

        invalid_origin = _audit()
        invalid_origin["images"][0]["candidates"][2]["tile_x"] = 896
        self.audit_path.write_text(json.dumps(invalid_origin))
        with self.assertRaisesRegex(ValueError, "invalid deployment tile origin"):
            run(
                self.audit_path,
                self.labels_path,
                "training_candidate",
                self.root / "origin.json",
            )

    def test_rejects_threshold_below_retained_candidate_floor(self):
        with self.assertRaisesRegex(ValueError, "below audit candidate floor"):
            run(
                self.audit_path,
                self.labels_path,
                "training_candidate",
                self.root / "report.json",
                threshold=0.0001,
            )


if __name__ == "__main__":
    unittest.main()
