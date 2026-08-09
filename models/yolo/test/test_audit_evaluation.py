import unittest

from audit.evaluation import (
    Box,
    aggregate_metrics,
    evaluate_image,
    find_empty_tiles,
    scale_probe,
)


class TestRealEvaluation(unittest.TestCase):
    def test_tile_fragment_is_separate_from_false_positive(self):
        truth = [Box(100, 100, 200, 200)]
        candidates = [
            Box(100, 100, 200, 200, 0.9, 0, 0),
            Box(160, 100, 200, 200, 0.8, 448, 0),
            Box(300, 300, 350, 350, 0.7, 448, 0),
        ]

        result = evaluate_image(truth, [], candidates, threshold=0.37)

        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["tile_fragments"], 1)
        aggregate = aggregate_metrics([result], tiles=2)
        self.assertEqual(aggregate["false_positives_per_tile"], 0.5)

    def test_ignore_region_suppresses_candidate_from_scoring(self):
        candidate = Box(10, 10, 20, 20, 0.5)
        result = evaluate_image([], [Box(0, 0, 30, 30)], [candidate], threshold=0.37)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["ignored_candidates"], 1)

    def test_thresholds_preserve_candidate_floor(self):
        candidate = Box(0, 0, 10, 10, 0.05)
        low = evaluate_image([Box(0, 0, 10, 10)], [], [candidate], threshold=0.001)
        operating = evaluate_image(
            [Box(0, 0, 10, 10)], [], [candidate], threshold=0.37
        )
        self.assertEqual((low["tp"], operating["fn"]), (1, 1))

    def test_scale_probe_has_exact_requested_object_side(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow unavailable")
        image = Image.new("RGB", (1000, 800))
        annotation = {"xyxy": [400, 300, 500, 350], "visibility": "clear"}

        probe, truth = scale_probe(image, annotation, 60)

        self.assertEqual(probe.size, (640, 640))
        self.assertAlmostEqual(max(truth.x1 - truth.x0, truth.y1 - truth.y0), 60)

    def test_clear_recall_and_empty_real_tile_gate_are_exact(self):
        truth = [Box(10, 10, 20, 20), Box(650, 10, 670, 30)]
        origins = [(0, 0), (640, 0), (1280, 0)]
        empty = find_empty_tiles(origins, 640, 1920, 640, truth)
        candidates = [
            Box(10, 10, 20, 20, 0.9, 0, 0),
            Box(1400, 100, 1450, 150, 0.8, 1280, 0),
        ]
        result = evaluate_image(
            truth,
            [],
            candidates,
            threshold=0.37,
            ground_truth_visibility=["clear", "partial"],
            empty_tile_origins=empty,
        )
        aggregate = aggregate_metrics([result], tiles=3)

        self.assertEqual(aggregate["recall_by_visibility"]["clear"]["recall"], 1.0)
        self.assertEqual(aggregate["recall_by_visibility"]["partial"]["recall"], 0.0)
        self.assertEqual(aggregate["empty_real_tiles"], 1)
        self.assertEqual(aggregate["empty_real_tiles_with_false_positive"], 1)
        self.assertEqual(aggregate["empty_real_tile_false_positive_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
