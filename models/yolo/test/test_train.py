import unittest
from pathlib import Path

from train.run import training_args


class TestTrainingConfig(unittest.TestCase):
    def test_pilot_settings_are_explicit_and_hailo_sized(self):
        args = training_args(
            Path("/dataset/dataset.yaml"),
            Path("/runs"),
            "pilot",
        )
        self.assertEqual(args["imgsz"], 640)
        self.assertEqual(args["batch"], 16)
        self.assertEqual(args["epochs"], 50)
        self.assertEqual(args["optimizer"], "AdamW")
        self.assertEqual(args["seed"], 10)
        self.assertTrue(args["deterministic"])
        self.assertEqual(args["flipud"], 0.5)
        self.assertEqual(args["fliplr"], 0.5)
        self.assertEqual(args["mixup"], 0.0)
        self.assertEqual(args["copy_paste"], 0.0)
        self.assertEqual(args["close_mosaic"], 10)

    def test_one_epoch_preflight_disables_patience_and_warmup(self):
        args = training_args(Path("data.yaml"), Path("runs"), "preflight", 1, 8)
        self.assertEqual(args["epochs"], 1)
        self.assertEqual(args["batch"], 8)
        self.assertEqual(args["patience"], 0)
        self.assertEqual(args["warmup_epochs"], 0.0)
        self.assertEqual(args["close_mosaic"], 0)


if __name__ == "__main__":
    unittest.main()
