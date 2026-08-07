import unittest

from tools.audit_irl import CANDIDATE_FLOOR, Detection, merge_detections, predict_tiled


def _detection(x0, y0, x1, y1, confidence):
    return Detection(x0, y0, x1, y1, confidence, 0, 0)


class TestIrlAudit(unittest.TestCase):
    def test_clipped_tile_duplicate_is_suppressed(self):
        complete = _detection(100, 100, 200, 200, 0.9)
        clipped = _detection(160, 100, 200, 200, 0.8)

        self.assertEqual(merge_detections([clipped, complete]), [complete])

    def test_disjoint_detections_are_preserved(self):
        first = _detection(100, 100, 200, 200, 0.8)
        second = _detection(300, 300, 400, 400, 0.7)

        self.assertEqual(merge_detections([second, first]), [first, second])

    def test_tiled_inference_preserves_candidates_below_operating_threshold(self):
        class Values:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class Model:
            def __init__(self):
                self.calls = []

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                boxes = type(
                    "Boxes",
                    (),
                    {
                        "xyxy": Values([[10, 20, 30, 40]]),
                        "conf": Values([0.005]),
                        "cls": Values([0]),
                    },
                )()
                return [type("Result", (), {"boxes": boxes})()]

        class Image:
            width = height = 640

            def crop(self, box):
                return box

        model = Model()
        origins, candidates = predict_tiled(
            model, Image(), tile=640, overlap=192, batch=16
        )

        self.assertEqual(origins, [(0, 0)])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].confidence, 0.005)
        self.assertEqual(model.calls[0]["conf"], CANDIDATE_FLOOR)


if __name__ == "__main__":
    unittest.main()
