import json
import tempfile
import unittest
from pathlib import Path

from audit.labels import LABEL_SCHEMA, certify_labels, freeze_roles, sha256
from tools.probe_certified_scale import (
    TILE_PX,
    candidates_from_result,
    probe_transform,
    run,
    select_certified_role,
)


class _Tensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class _Boxes:
    def __init__(self, xyxy, confidence, classes):
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(confidence)
        self.cls = _Tensor(classes)


class _Result:
    def __init__(self, xyxy, confidence, classes):
        self.boxes = _Boxes(xyxy, confidence, classes)


class _Model:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        results = []
        for target_px in (30, 60, 120):
            results.append(
                _Result(
                    [
                        [
                            320 - target_px / 2,
                            320 - target_px / 4,
                            320 + target_px / 2,
                            320 + target_px / 4,
                        ],
                        [0, 0, 20, 20],
                    ],
                    [0.8, 0.9],
                    [0, 1],
                )
            )
        return results


def _certified_document(source: Path) -> dict:
    document = {
        "schema": LABEL_SCHEMA,
        "images": [
            {
                "source": source.name,
                "source_sha256": sha256(source),
                "width": 100,
                "height": 80,
                "capture_group": "candidate-phone",
                "role": "training_candidate",
                "review_state": "complete",
                "objects": [
                    {"xyxy": [30, 30, 70, 50], "visibility": "partial"},
                    {"xyxy": [4, 4, 12, 12], "visibility": "not_visible"},
                ],
                "ignore_regions": [
                    {"xyxy": [80, 60, 90, 70], "reason": "ambiguous debris"}
                ],
            }
        ],
    }
    return certify_labels(freeze_roles(document, "split owner"), "reviewer")


class TestCertifiedScaleProbe(unittest.TestCase):
    def test_transform_is_exact_and_centered(self):
        transform = probe_transform({"xyxy": [30, 30, 70, 50]}, 60)

        self.assertEqual(transform["output_width"], TILE_PX)
        self.assertEqual(transform["output_height"], TILE_PX)
        self.assertEqual(transform["source_to_probe_scale"], 1.5)
        crop = transform["source_crop_xyxy"]
        self.assertAlmostEqual((crop[0] + crop[2]) / 2, 50)
        self.assertAlmostEqual((crop[1] + crop[3]) / 2, 40)

    def test_role_is_explicit_and_training_candidate_is_supported(self):
        with self.assertRaisesRegex(ValueError, "explicitly requested"):
            select_certified_role({}, None)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "photo.png"
            source.write_bytes(b"content only needed for label validation")
            document = _certified_document(source)
            records = select_certified_role(document, "training_candidate")

        self.assertEqual(len(records), 1)

    def test_candidate_adapter_keeps_only_class_zero_at_floor(self):
        result = _Result(
            [[-1, 2, 650, 700], [1, 2, 3, 4], [5, 6, 7, 8]],
            [0.001, 0.9, 0.0009],
            [0, 1, 0],
        )

        candidates = candidates_from_result(result)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            (candidates[0].x0, candidates[0].y0, candidates[0].x1, candidates[0].y1),
            (0.0, 2.0, 640.0, 640.0),
        )

    def test_run_batches_all_probes_and_marks_non_promotion(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "photo.png"
            Image.new("RGB", (100, 80)).save(source)
            labels = root / "labels.json"
            labels.write_text(json.dumps(_certified_document(source)))
            weights = root / "best.pt"
            weights.write_bytes(b"stub weights")
            out = root / "report.json"
            model = _Model()

            report = run(
                weights,
                labels,
                "training_candidate",
                out,
                batch=8,
                model=model,
            )

            self.assertEqual(len(model.calls), 1)
            self.assertEqual(len(model.calls[0]["source"]), 3)
            self.assertEqual(model.calls[0]["imgsz"], 640)
            self.assertEqual(report["summary"]["overall"]["probes"], 3)
            self.assertEqual(
                report["summary"]["by_visibility"]["partial"]["probes"], 3
            )
            self.assertEqual(
                report["summary"]["overall"]["thresholds"]["0.37"]["tp"], 3
            )
            self.assertFalse(report["automatic_training_promotion"])
            self.assertEqual(
                report["training_candidate_policy"],
                "diagnostic_only_no_promotion",
            )
            self.assertEqual(len(report["probes"][0]["candidates"]), 1)
            self.assertEqual(report["probes"][0]["object_index"], 0)
            self.assertIn("probe_transform", report["probes"][0])
            self.assertTrue(out.is_file())

            with self.assertRaisesRegex(ValueError, "overwrite"):
                run(
                    weights,
                    labels,
                    "training_candidate",
                    out,
                    model=model,
                )


if __name__ == "__main__":
    unittest.main()
