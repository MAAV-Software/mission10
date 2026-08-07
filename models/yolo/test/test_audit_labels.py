import unittest

from audit.labels import LABEL_SCHEMA, certify_labels, freeze_roles, validate_labels
from audit.evaluation import select_role


def _document(state="complete", role="development_eval"):
    return {
        "schema": LABEL_SCHEMA,
        "images": [
            {
                "source": "photo.jpg",
                "source_sha256": "a" * 64,
                "width": 100,
                "height": 80,
                "capture_group": "phone-audited",
                "role": role,
                "review_state": state,
                "objects": [
                    {"xyxy": [10, 10, 30, 35], "visibility": "partial"}
                ],
                "ignore_regions": [
                    {"xyxy": [60, 5, 90, 30], "reason": "ambiguous debris"}
                ],
            }
        ],
    }


class TestRealLabelSchema(unittest.TestCase):
    def test_role_freeze_detects_capture_group_reassignment(self):
        document = freeze_roles(_document(), "split owner", now="2026-08-06T00:00:00Z")
        document["images"][0]["role"] = "training_candidate"

        with self.assertRaisesRegex(ValueError, "roles changed"):
            validate_labels(document)

    def test_role_freeze_is_per_source_not_only_per_group(self):
        document = _document()
        second = {
            **document["images"][0],
            "source": "second.jpg",
            "source_sha256": "b" * 64,
            "capture_group": "phone-candidates",
            "role": "training_candidate",
        }
        document["images"].append(second)
        frozen = freeze_roles(document, "split owner")
        frozen["images"][0]["capture_group"] = "phone-candidates"
        frozen["images"][0]["role"] = "training_candidate"
        frozen["images"][1]["capture_group"] = "phone-audited"
        frozen["images"][1]["role"] = "development_eval"

        with self.assertRaisesRegex(ValueError, "roles changed"):
            validate_labels(frozen)

    def test_certification_is_explicit_and_content_addressed(self):
        document = freeze_roles(_document(), "split owner", now="freeze")
        certified = certify_labels(document, "human reviewer", now="certify")

        self.assertEqual(certified["images"][0]["review_state"], "certified")
        self.assertEqual(
            select_role(certified, "development_eval")[0]["source"], "photo.jpg"
        )
        certified["images"][0]["objects"][0]["xyxy"][0] = 11
        with self.assertRaisesRegex(ValueError, "changed after human certification"):
            validate_labels(certified)

    def test_incomplete_images_cannot_be_certified(self):
        document = freeze_roles(_document(state="unreviewed"), "split owner")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            certify_labels(document, "reviewer")

    def test_evaluation_rejects_merely_complete_document(self):
        document = freeze_roles(_document(), "split owner")
        with self.assertRaisesRegex(ValueError, "final human certification"):
            select_role(document, "development_eval")

    def test_full_object_box_and_visibility_are_required(self):
        document = _document()
        del document["images"][0]["objects"][0]["visibility"]
        with self.assertRaisesRegex(ValueError, "visibility"):
            validate_labels(document)


if __name__ == "__main__":
    unittest.main()
