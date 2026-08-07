import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_color_scale_matrix import (
    BACKGROUND_PLATES,
    INPUT_SCHEMA,
    POSITIVE_CASES,
    Box,
    _declared_image_sha256,
    _load_manifest,
    acceptance_summary,
    iou,
    score_case,
    sha256,
)


def _manifest():
    positives = []
    for index in range(POSITIVE_CASES):
        positives.append(
            {
                "case_id": f"positive-{index}",
                "image": f"images/positive-{index}.png",
                "ground_truth": [{"xyxy_px": [10, 20, 40, 50]}],
            }
        )
    plates = []
    for index in range(BACKGROUND_PLATES):
        plates.append(
            {
                "case_id": f"plate-{index}",
                "image": f"images/plate-{index}.png",
                "ground_truth": [],
            }
        )
    return {
        "schema": INPUT_SCHEMA,
        "positive_cases": positives,
        "background_plates": plates,
    }


class TestColorScaleMetrics(unittest.TestCase):
    def test_iou_and_confidence_ordered_one_to_one_matching(self):
        truth = [Box(0, 0, 10, 10)]
        predictions = [
            Box(0, 0, 10, 10, 0.7),
            Box(1, 1, 9, 9, 0.9),
        ]
        self.assertEqual(iou(truth[0], predictions[0]), 1.0)
        result = score_case(truth, predictions)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["matches"][0]["prediction_index"], 1)
        self.assertAlmostEqual(result["matches"][0]["iou"], 0.64)

    def test_iou_threshold_is_inclusive(self):
        truth = [Box(0, 0, 10, 10)]
        prediction = [Box(0, 0, 5, 10, 0.5)]
        result = score_case(truth, prediction, iou_threshold=0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))

    def test_acceptance_requires_15_matches_and_clean_plates(self):
        positives = [{"tp": 1, "fp": 1, "fn": 0}] * POSITIVE_CASES
        plates = [{"tp": 0, "fp": 0, "fn": 0}] * BACKGROUND_PLATES
        result = acceptance_summary(positives, plates)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["all_positive_cases_matched"])
        self.assertTrue(result["background_plates_clean"])
        self.assertEqual(result["positive_false_positives"], POSITIVE_CASES)

        dirty = [*plates]
        dirty[0] = {"tp": 0, "fp": 1, "fn": 0}
        result = acceptance_summary(positives, dirty)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["background_plate_false_positives"], 1)

    def test_acceptance_rejects_one_missed_positive(self):
        positives = [{"tp": 1, "fp": 0, "fn": 0}] * POSITIVE_CASES
        positives[-1] = {"tp": 0, "fp": 0, "fn": 1}
        plates = [{"tp": 0, "fp": 0, "fn": 0}] * BACKGROUND_PLATES
        result = acceptance_summary(positives, plates)
        self.assertEqual(result["positive_cases_matched"], 14)
        self.assertFalse(result["all_positive_cases_matched"])
        self.assertFalse(result["accepted"])


class TestColorScaleInputs(unittest.TestCase):
    def test_manifest_contract_and_hash_reporting_are_dependency_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest()))
            loaded = _load_manifest(manifest_path)
            self.assertEqual(len(loaded["positive_cases"]), POSITIVE_CASES)
            self.assertEqual(len(loaded["background_plates"]), BACKGROUND_PLATES)

            artifact = root / "artifact.bin"
            artifact.write_bytes(b"mine diagnostic")
            self.assertEqual(
                sha256(artifact),
                "e24a1ac4b3968726b72c05752a448ae4ca083d66d93c76b09ab79aa15675288d",
            )

    def test_manifest_rejects_bad_counts_and_plate_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            document = _manifest()
            document["positive_cases"].pop()
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "exactly 15"):
                _load_manifest(path)

            document = _manifest()
            document["background_plates"][0]["ground_truth"] = [
                {"xyxy_px": [0, 0, 1, 1]}
            ]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "empty ground truth"):
                _load_manifest(path)

    def test_declared_hash_compatibility(self):
        manifest = {"hashes": {"images/a.png": {"sha256": "a" * 64}}}
        record = {"image": "images/a.png"}
        self.assertEqual(_declared_image_sha256(manifest, record), "a" * 64)
        record["image_sha256"] = "b" * 64
        self.assertEqual(_declared_image_sha256(manifest, record), "b" * 64)


if __name__ == "__main__":
    unittest.main()
