import json
import tempfile
import unittest
from pathlib import Path

from audit.labels import LABEL_SCHEMA, certify_labels, freeze_roles, sha256
from audit.mask_review import update_confirmation
from audit.masks import propose_masks


class _Masks:
    def __init__(self, data):
        self.data = [data]


class _Result:
    def __init__(self, data):
        self.masks = _Masks(data)


class _Model:
    def predict(self, **kwargs):
        import numpy as np

        mask = np.zeros((64, 64), dtype=float)
        mask[12:29, 11:30] = 1.0
        return [_Result(mask)]


class TestMasks(unittest.TestCase):
    def test_proposal_uses_clear_only_and_review_changes_only_decision(self):
        try:
            from PIL import Image
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("Pillow/numpy unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "photo.png"
            Image.new("RGB", (64, 64), "green").save(source)
            document = {
                "schema": LABEL_SCHEMA,
                "images": [
                    {
                        "source": source.name,
                        "source_sha256": sha256(source),
                        "width": 64,
                        "height": 64,
                        "capture_group": "phone",
                        "role": "training_candidate",
                        "review_state": "complete",
                        "objects": [
                            {"xyxy": [10, 10, 30, 30], "visibility": "clear"},
                            {"xyxy": [35, 35, 50, 50], "visibility": "partial"},
                        ],
                        "ignore_regions": [],
                    }
                ],
            }
            document = certify_labels(
                freeze_roles(document, "owner", now="freeze"),
                "reviewer",
                now="certify",
            )
            labels = root / "labels.json"
            labels.write_text(json.dumps(document))
            weights = root / "sam.pt"
            weights.write_bytes(b"stub")
            review = propose_masks(labels, weights, root / "review", model=_Model())

            self.assertEqual(len(review["entries"]), 1)
            self.assertEqual(review["entries"][0]["object_index"], 0)
            self.assertEqual(
                review["policy"]["partial_use"], "whole_context_positive_only"
            )
            self.assertTrue(
                (root / "review" / review["entries"][0]["cutout"]).is_file()
            )
            before = dict(review["entries"][0])
            updated = update_confirmation(
                review,
                {"id": before["id"], "confirmation": "confirmed"},
            )
            after = dict(updated["entries"][0])
            self.assertEqual(after.pop("confirmation"), "confirmed")
            before.pop("confirmation")
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
