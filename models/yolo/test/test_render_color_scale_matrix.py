import tempfile
import unittest
from pathlib import Path

from datagen.config import GenConfig
from tools.render_color_scale_matrix import (
    BACKGROUND_PLATES,
    PLATE_REFERENCE_WIDTH_PX,
    POSITIVE_BACKGROUND,
    SCHEMA,
    TARGET_WIDTHS_PX,
    _arguments,
    _prepare_output,
    altitude_for_projected_width,
    centered_ground_truth,
    diagnostic_cases,
    diagnostic_plates,
    validate_template_dimensions,
)


class TestColorScaleMatrix(unittest.TestCase):
    def test_schema_and_exact_acceptance_corpus_are_stable(self):
        self.assertEqual(SCHEMA, "mine-color-scale-diagnostic/1")
        cfg = GenConfig()
        cases = diagnostic_cases(cfg)
        plates = diagnostic_plates(cfg)
        self.assertEqual(
            len(cases), len(cfg.mine_color_names) * len(TARGET_WIDTHS_PX)
        )
        self.assertEqual(len(cases), 15)
        self.assertEqual(len(plates), 3)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))
        self.assertEqual(len({plate.case_id for plate in plates}), len(plates))
        self.assertEqual(
            {case.color_family for case in cases}, set(cfg.mine_color_names)
        )
        self.assertEqual(
            {case.target_width_px for case in cases}, set(TARGET_WIDTHS_PX)
        )
        self.assertEqual({case.background for case in cases}, {POSITIVE_BACKGROUND})
        self.assertEqual({plate.background for plate in plates}, set(BACKGROUND_PLATES))
        self.assertEqual(
            {plate.altitude_m for plate in plates},
            {
                altitude_for_projected_width(
                    PLATE_REFERENCE_WIDTH_PX,
                    cfg.camera.focal_px,
                    cfg.mine_dims_m[0],
                )
            },
        )

    def test_altitudes_exactly_invert_pinhole_projection(self):
        cfg = GenConfig()
        for width in TARGET_WIDTHS_PX:
            altitude = altitude_for_projected_width(
                width, cfg.camera.focal_px, cfg.mine_dims_m[0]
            )
            projected = cfg.camera.focal_px * cfg.mine_dims_m[0] / altitude
            self.assertAlmostEqual(projected, width)
        altitudes = [
            altitude_for_projected_width(
                width, cfg.camera.focal_px, cfg.mine_dims_m[0]
            )
            for width in TARGET_WIDTHS_PX
        ]
        self.assertGreater(altitudes[0], altitudes[1])
        self.assertGreater(altitudes[1], altitudes[2])

    def test_centered_ground_truth_matches_physical_aspect_ratio(self):
        cfg = GenConfig()
        truth = centered_ground_truth(cfg, 60)
        x0, y0, x1, y1 = truth["xyxy_px"]
        self.assertAlmostEqual(x1 - x0, 60.0)
        self.assertAlmostEqual(
            y1 - y0, 60.0 * cfg.mine_dims_m[1] / cfg.mine_dims_m[0]
        )
        self.assertEqual(truth["yolo_xywhn"][:2], [0.5, 0.5])
        self.assertTrue(truth["yolo_line"].startswith("0 0.500000000 0.500000000 "))

    def test_cases_use_exact_palette_anchors_without_jitter(self):
        cfg = GenConfig()
        anchors = dict(zip(cfg.mine_color_names, cfg.mine_color_palette_srgb))
        for case in diagnostic_cases(cfg):
            self.assertEqual(case.color_srgb, anchors[case.color_family])

    def test_template_dimensions_are_checked_and_returned(self):
        expected = (0.12, 0.061, 0.02)
        self.assertEqual(
            validate_template_dimensions((0.1201, 0.0609, 0.0201), expected),
            (0.1201, 0.0609, 0.0201),
        )
        with self.assertRaisesRegex(RuntimeError, "differ from configured"):
            validate_template_dimensions((0.13, 0.061, 0.02), expected)

    def test_output_must_be_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            images, labels = _prepare_output(out)
            self.assertTrue(images.is_dir())
            self.assertTrue(labels.is_dir())
            with self.assertRaisesRegex(ValueError, "not empty"):
                _prepare_output(out)

    def test_script_backend_option_does_not_collide_with_blender(self):
        args = _arguments(
            ["--out", "/tmp/matrix", "--cycles-backend", "optix"]
        )
        self.assertEqual(args.cycles_backend, "optix")

    def test_invalid_pure_inputs_are_rejected(self):
        cfg = GenConfig()
        with self.assertRaises(ValueError):
            altitude_for_projected_width(0, 1000.0, cfg.mine_dims_m[0])
        with self.assertRaises(ValueError):
            diagnostic_plates(backgrounds=("not-a-material",))
        with self.assertRaises(ValueError):
            diagnostic_cases(mine_dims_m=(0.12, -0.01, 0.02))


if __name__ == "__main__":
    unittest.main()
