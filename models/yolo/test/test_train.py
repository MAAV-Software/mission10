import json
import tempfile
import unittest
from pathlib import Path

from train.run import (
    _finish_run,
    planned_stop_callback,
    resume_inputs,
    sha256,
    source_commit,
    training_args,
    training_config_sha256,
    validate_planned_stop,
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
        self.assertEqual(args["save_period"], 5)

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
        self.assertEqual(args["save_period"], 1)

        replay = training_args(
            Path("data.yaml"),
            Path("runs"),
            "appearance-replay",
            preset="real_positive_appearance",
        )
        self.assertEqual(replay, {**args, "name": "appearance-replay"})

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

    def test_planned_stop_preserves_schedule_and_changes_identity(self):
        args = training_args(
            Path("data.yaml"), Path("runs"), "realpositive", preset="real_positive"
        )
        self.assertEqual(validate_planned_stop(9, args, "real_positive"), 9)
        self.assertEqual(args["epochs"], 20)
        self.assertEqual(args["close_mosaic"], 10)
        self.assertNotEqual(
            training_config_sha256(args), training_config_sha256(args, 9)
        )

    def test_planned_stop_rejects_invalid_or_pilot_requests(self):
        fine_tune = training_args(
            Path("data.yaml"), Path("runs"), "hardneg", preset="hardneg"
        )
        with self.assertRaisesRegex(ValueError, "between 1"):
            validate_planned_stop(20, fine_tune, "hardneg")
        pilot = training_args(Path("data.yaml"), Path("runs"), "pilot")
        with self.assertRaisesRegex(ValueError, "fine-tune"):
            validate_planned_stop(9, pilot, "pilot")

    def test_planned_stop_callback_uses_completed_epoch_count(self):
        trainer = type("Trainer", (), {"epoch": 7, "stop": False})()
        callback = planned_stop_callback(9)
        callback(trainer)
        self.assertFalse(trainer.stop)
        trainer.epoch = 8
        callback(trainer)
        self.assertTrue(trainer.stop)

    def test_planned_stop_evaluates_and_locks_last_checkpoint(self):
        root = Path(tempfile.mkdtemp())
        run_dir = root / "run"
        weights = run_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"best")
        (weights / "last.pt").write_bytes(b"last")
        (run_dir / "results.csv").write_text("epoch,time\n1,1\n2,2\n")

        class Metrics:
            results_dict = {"metrics/mAP50-95(B)": 0.9}

        loaded = []

        class Model:
            def val(self, **kwargs):
                return Metrics()

        def fake_yolo(path):
            loaded.append(path)
            return Model()

        lock = {"planned_stop_after_epochs": 2}
        args = {"batch": 16, "project": str(root), "name": "run"}
        _finish_run(run_dir, lock, root / "dataset.yaml", args, fake_yolo)
        self.assertEqual(loaded, [str((weights / "last.pt").resolve())])
        self.assertEqual(lock["completed_epochs"], 2)
        self.assertEqual(lock["evaluation_weights"]["role"], "frozen_planned_stop")
        self.assertEqual(
            lock["evaluation_weights"]["sha256"], sha256(weights / "last.pt")
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
            "training_preset": "hardneg",
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
        self.assertEqual(state["scheduled_epochs"], 20)
        self.assertFalse(state["training_complete"])
        self.assertEqual(state["last"].name, "last.pt")

    def test_resume_can_finalize_a_completed_planned_stop(self):
        run_dir = self._interrupted_run()
        lock_path = run_dir / "run.lock.json"
        lock = json.loads(lock_path.read_text())
        lock["planned_stop_after_epochs"] = 9
        lock["training_config_sha256"] = training_config_sha256(
            lock["training_args"], 9
        )
        lock_path.write_text(json.dumps(lock) + "\n")
        rows = "".join(f"{epoch},{epoch * 100}\n" for epoch in range(1, 10))
        (run_dir / "results.csv").write_text("epoch,time\n" + rows)
        state = resume_inputs(run_dir)
        self.assertEqual(state["target_epochs"], 9)
        self.assertEqual(state["scheduled_epochs"], 20)
        self.assertTrue(state["training_complete"])

    def test_resume_inputs_reject_changed_dataset_lock(self):
        run_dir = self._interrupted_run()
        lock = json.loads((run_dir / "run.lock.json").read_text())
        data = Path(lock["training_args"]["data"])
        (data.parent / "split.lock.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "dataset lock changed"):
            resume_inputs(run_dir)


if __name__ == "__main__":
    unittest.main()
