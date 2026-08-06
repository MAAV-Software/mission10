import unittest

from train.evaluate import (
    Box,
    ImageRecord,
    choose_threshold,
    evaluate_records,
    iou,
    match_image,
    threshold_sweep,
)


class TestOperationalEvaluation(unittest.TestCase):
    def test_iou_and_one_to_one_matching(self):
        ground_truth = [Box(0, 0, 10, 10)]
        predictions = [
            Box(0, 0, 10, 10, 0.9),
            Box(1, 1, 9, 9, 0.8),
        ]
        self.assertEqual(iou(ground_truth[0], predictions[0]), 1.0)
        matched, false_positives = match_image(ground_truth, predictions, 0.5)
        self.assertEqual(matched, {0})
        self.assertEqual(false_positives, 1)

    def test_sweep_selects_recall_first_threshold_with_precision_floor(self):
        records = [
            ImageRecord(
                "positive",
                (Box(0, 0, 10, 10),),
                (
                    Box(0, 0, 10, 10, 0.8),
                    Box(20, 20, 30, 30, 0.2),
                ),
            ),
            ImageRecord("empty", (), (Box(20, 20, 30, 30, 0.2),)),
        ]
        chosen = choose_threshold(threshold_sweep(records), 0.9)
        self.assertTrue(chosen["precision_floor_met"])
        self.assertAlmostEqual(chosen["threshold"], 0.21)
        self.assertEqual(chosen["precision"], 1.0)
        self.assertEqual(chosen["recall"], 1.0)

    def test_precision_floor_fallback_is_explicit(self):
        sweep = [
            {"threshold": 0.1, "precision": 0.4, "recall": 1.0, "fbeta": 0.7},
            {"threshold": 0.2, "precision": 0.5, "recall": 0.8, "fbeta": 0.72},
        ]
        chosen = choose_threshold(sweep, 0.9)
        self.assertFalse(chosen["precision_floor_met"])
        self.assertEqual(chosen["threshold"], 0.2)

    def test_empty_tile_rate_and_recall_groups(self):
        records = [
            ImageRecord(
                "positive",
                (Box(0, 0, 10, 10), Box(20, 20, 30, 30)),
                (Box(0, 0, 10, 10, 0.9),),
                (("surface", "grass"),),
                ("small", "large"),
            ),
            ImageRecord("empty-hit", (), (Box(0, 0, 10, 10, 0.8),)),
            ImageRecord("empty-clean", (), ()),
        ]
        result = evaluate_records(records, 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["empty_tile_false_positive_rate"], 0.5)
        self.assertEqual(result["recall_by"]["surface"]["grass"]["recall"], 0.5)
        self.assertEqual(result["recall_by"]["box_size"]["small"]["recall"], 1.0)
        self.assertEqual(result["recall_by"]["box_size"]["large"]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
