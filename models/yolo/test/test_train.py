import json
import tempfile
import unittest
from pathlib import Path

from train.run import (
    resume_inputs,
    sha256,
    source_commit,
    training_args,
    training_config_sha256,
)


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
        self.assertEqual(args["cache"], "ram")
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

    def test_cache_can_be_disabled_for_memory_limited_pods(self):
        args = training_args(
            Path("data.yaml"), Path("runs"), "streamed", 50, 16, False
        )
        self.assertFalse(args["cache"])

    def test_fine_tune_presets_are_locked(self):
        args = training_args(
            Path("data.yaml"), Path("runs"), "combined", preset="combined"
        )
        self.assertEqual(args["epochs"], 20)
        self.assertEqual(args["patience"], 8)
        self.assertEqual(args["optimizer"], "AdamW")
        self.assertEqual(args["lr0"], 0.0001)
        self.assertTrue(args["cos_lr"])
        self.assertEqual(args["batch"], 16)
        self.assertEqual(args["imgsz"], 640)
        self.assertEqual(args["device"], 0)
        self.assertEqual(args["workers"], 8)
        self.assertFalse(args["cache"])
        self.assertEqual(args["seed"], 10)
        self.assertTrue(args["deterministic"])
        self.assertEqual(args["mosaic"], 1.0)
        self.assertEqual(args["close_mosaic"], 10)

    def test_fine_tune_presets_reject_behavior_overrides(self):
        with self.assertRaisesRegex(ValueError, "does not permit overrides"):
            training_args(
                Path("data.yaml"),
                Path("runs"),
                "combined",
                epochs=21,
                preset="combined",
            )

    def test_training_config_hash_ignores_run_locations(self):
        first = training_args(Path("a.yaml"), Path("first"), "one")
        second = training_args(Path("b.yaml"), Path("second"), "two")
        self.assertEqual(
            training_config_sha256(first), training_config_sha256(second)
        )
        second["lr0"] = 0.2
        self.assertNotEqual(
            training_config_sha256(first), training_config_sha256(second)
        )

    def test_archive_source_marker_supplies_commit(self):
        root = Path(tempfile.mkdtemp())
        commit = "a" * 40
        (root / ".source-commit").write_text(commit + "\n")
        self.assertEqual(source_commit(root), commit)

    def _interrupted_run(self) -> Path:
        root = Path(tempfile.mkdtemp())
        data_dir = root / "dataset"
        data_dir.mkdir()
        data = data_dir / "dataset.yaml"
        data.write_text("names:\n  0: mine\n")
        dataset_lock = data_dir / "split.lock.json"
        dataset_identity = "d" * 64
        dataset_lock.write_text(
            json.dumps({"dataset_sha256": dataset_identity}) + "\n"
        )
        source = root / "source.pt"
        source.write_bytes(b"source")
        run_dir = root / "runs" / "hardneg"
        weights = run_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "last.pt").write_bytes(b"last")
        args = training_args(
            data,
            root / "runs",
            "hardneg",
            preset="hardneg",
        )
        lock = {
            "schema": "mission10-yolo-run/1",
            "status": "started",
            "dataset_lock_sha256": sha256(dataset_lock),
            "dataset_sha256": dataset_identity,
            "source_weights": str(source),
            "source_weights_sha256": sha256(source),
            "training_config_sha256": training_config_sha256(args),
            "training_args": args,
        }
        (run_dir / "run.lock.json").write_text(json.dumps(lock) + "\n")
        (run_dir / "results.csv").write_text("epoch,time\n1,100\n")
        return run_dir

    def test_resume_inputs_preserve_locked_dataset_and_epoch(self):
        state = resume_inputs(self._interrupted_run())
        self.assertEqual(state["completed_epochs"], 1)
        self.assertEqual(state["target_epochs"], 20)
        self.assertEqual(state["last"].name, "last.pt")

    def test_resume_inputs_reject_changed_dataset_lock(self):
        run_dir = self._interrupted_run()
        lock = json.loads((run_dir / "run.lock.json").read_text())
        data = Path(lock["training_args"]["data"])
        (data.parent / "split.lock.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "dataset lock changed"):
            resume_inputs(run_dir)


if __name__ == "__main__":
    unittest.main()
