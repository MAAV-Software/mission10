"""Run the pinned YOLO11m pilot and record a reproducibility lock.

Ultralytics is imported only inside ``run`` so local config tests do not need
the GPU environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


RUN_SCHEMA = "mission10-yolo-run/1"
COMPOSITION_SCHEMA = "mission10-yolo-composition/1"
FINE_TUNE_PRESETS = (
    "control",
    "appearance",
    "hardneg",
    "combined",
    "real_positive",
    "real_positive_appearance",
)
TRAINING_PRESETS = ("pilot", *FINE_TUNE_PRESETS)
_CONFIG_RUNTIME_KEYS = {"data", "project", "name", "exist_ok"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_args(
    data: Path,
    project: Path,
    name: str,
    epochs: int | None = None,
    batch: int | None = None,
    cache: str | bool | None = None,
    preset: str = "pilot",
) -> dict:
    """Return one allow-listed, fully explicit Ultralytics configuration."""
    if preset not in TRAINING_PRESETS:
        raise ValueError(f"unknown training preset: {preset!r}")
    if preset == "pilot":
        epochs = 50 if epochs is None else epochs
        batch = 16 if batch is None else batch
        cache = "ram" if cache is None else cache
        patience = 15 if epochs > 1 else 0
        lr0 = 0.001
        save_period = 5 if epochs > 1 else -1
    else:
        required = {"epochs": 20, "batch": 16, "cache": False}
        supplied = {"epochs": epochs, "batch": batch, "cache": cache}
        conflicts = {
            key: value
            for key, value in supplied.items()
            if value is not None and value != required[key]
        }
        if conflicts:
            raise ValueError(f"{preset} preset does not permit overrides: {conflicts}")
        epochs = 20
        batch = 16
        cache = False
        patience = 8
        lr0 = 0.0001
        save_period = 1
    return {
        "data": str(data.resolve()),
        "project": str(project.resolve()),
        "name": name,
        "exist_ok": True,
        "epochs": epochs,
        "patience": patience,
        "imgsz": 640,
        "batch": batch,
        "device": 0,
        "workers": 8,
        "cache": cache,
        "amp": True,
        "optimizer": "AdamW",
        "lr0": lr0,
        "lrf": 0.01,
        "cos_lr": True,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0 if epochs > 1 else 0.0,
        "seed": 10,
        "deterministic": True,
        "save": True,
        "save_period": save_period,
        "plots": True,
        "val": True,
        "hsv_h": 0.01,
        "hsv_s": 0.25,
        "hsv_v": 0.25,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.25,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": min(10, max(0, epochs - 1)),
    }


def training_config_sha256(
    args: dict, planned_stop_after_epochs: int | None = None
) -> str:
    """Hash behavior-changing settings independently of run locations."""
    config = {
        key: value for key, value in args.items() if key not in _CONFIG_RUNTIME_KEYS
    }
    if planned_stop_after_epochs is not None:
        config["planned_stop_after_epochs"] = planned_stop_after_epochs
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_planned_stop(
    planned_stop_after_epochs: int | None, args: dict, preset: str
) -> int | None:
    """Validate an intentional stop without changing the scheduled horizon."""
    if planned_stop_after_epochs is None:
        return None
    if preset not in FINE_TUNE_PRESETS:
        raise ValueError("planned stop is only supported for fine-tune presets")
    if (
        isinstance(planned_stop_after_epochs, bool)
        or planned_stop_after_epochs < 1
        or planned_stop_after_epochs >= args["epochs"]
    ):
        raise ValueError(
            "planned stop must be between 1 and one less than the scheduled epochs"
        )
    return planned_stop_after_epochs


def planned_stop_callback(planned_stop_after_epochs: int):
    """Stop after an exact completed epoch without shortening the LR schedule."""

    def stop_after_epoch(trainer) -> None:
        if trainer.epoch + 1 >= planned_stop_after_epochs:
            trainer.stop = True

    return stop_after_epoch


def _command_output(argv: list[str]) -> str:
    return subprocess.run(
        argv,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def source_commit(repo: Path) -> str:
    """Read the revision from git or a deployment archive marker."""
    try:
        commit = _command_output(["git", "-C", str(repo), "rev-parse", "HEAD"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        marker = repo / ".source-commit"
        if not marker.is_file():
            raise ValueError(f"no git checkout or source marker under {repo}")
        commit = marker.read_text().strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError(f"invalid source commit {commit!r}")
    return commit


def _json_metrics(metrics) -> dict:
    return {
        key: float(value)
        for key, value in metrics.results_dict.items()
    }


def _runtime_environment(torch, ultralytics) -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "packages": sorted(
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        ),
    }


def resume_inputs(run_dir: Path) -> dict:
    """Validate one interrupted run without importing the GPU framework."""
    run_dir = Path(run_dir).resolve()
    lock_path = run_dir / "run.lock.json"
    if not lock_path.is_file():
        raise ValueError(f"missing run lock: {lock_path}")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != RUN_SCHEMA:
        raise ValueError(f"invalid run-lock schema: {lock.get('schema')!r}")
    if lock.get("status") not in {"started", "resuming"}:
        raise ValueError(f"run is not interrupted: status={lock.get('status')!r}")
    args = lock.get("training_args")
    if not isinstance(args, dict):
        raise ValueError("run lock has no training arguments")
    planned_stop_after_epochs = lock.get("planned_stop_after_epochs")
    validate_planned_stop(
        planned_stop_after_epochs, args, lock.get("training_preset", "pilot")
    )
    if training_config_sha256(
        args, planned_stop_after_epochs
    ) != lock.get("training_config_sha256"):
        raise ValueError("training arguments changed after the run started")

    data = Path(args.get("data", ""))
    if not data.is_file():
        raise ValueError(f"missing resume dataset YAML: {data}")
    dataset_lock = data.parent / "split.lock.json"
    if not dataset_lock.is_file():
        raise ValueError(f"missing resume dataset lock: {dataset_lock}")
    if sha256(dataset_lock) != lock.get("dataset_lock_sha256"):
        raise ValueError("resume dataset lock changed after the run started")
    dataset = json.loads(dataset_lock.read_text())
    if dataset.get("dataset_sha256") != lock.get("dataset_sha256"):
        raise ValueError("resume dataset identity changed after the run started")

    source_weights = Path(lock.get("source_weights", ""))
    if not source_weights.is_file():
        raise ValueError(f"missing original source weights: {source_weights}")
    if sha256(source_weights) != lock.get("source_weights_sha256"):
        raise ValueError("original source weights changed after the run started")

    last = run_dir / "weights" / "last.pt"
    if not last.is_file():
        raise ValueError(f"missing resume checkpoint: {last}")
    results = run_dir / "results.csv"
    if not results.is_file():
        raise ValueError(f"missing completed-epoch history: {results}")
    with results.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("cannot resume before one epoch has completed")
    try:
        completed_epochs = int(rows[-1]["epoch"])
        scheduled_epochs = int(args["epochs"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid epoch history in results.csv") from error
    target_epochs = planned_stop_after_epochs or scheduled_epochs
    if completed_epochs != len(rows):
        raise ValueError(
            f"epoch history is not contiguous: last={completed_epochs}, rows={len(rows)}"
        )
    if completed_epochs > target_epochs:
        raise ValueError(
            f"training exceeded its target: {completed_epochs}/{target_epochs} epochs"
        )
    if completed_epochs == target_epochs and planned_stop_after_epochs is None:
        raise ValueError(
            f"training already reached {completed_epochs}/{target_epochs} epochs"
        )
    return {
        "run_dir": run_dir,
        "lock_path": lock_path,
        "lock": lock,
        "args": args,
        "data": data,
        "last": last,
        "completed_epochs": completed_epochs,
        "target_epochs": target_epochs,
        "scheduled_epochs": scheduled_epochs,
        "training_complete": completed_epochs == target_epochs,
    }


def _record_weights(run_dir: Path, lock: dict) -> tuple[Path, Path]:
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError("training finished without best.pt and last.pt")
    lock["best_weights_sha256"] = sha256(best)
    lock["last_weights_sha256"] = sha256(last)
    return best, last


def _completed_epochs(run_dir: Path) -> int:
    results = run_dir / "results.csv"
    with results.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("training finished without a completed epoch")
    completed = int(rows[-1]["epoch"])
    if completed != len(rows):
        raise RuntimeError(
            f"epoch history is not contiguous: last={completed}, rows={len(rows)}"
        )
    return completed


def _finish_run(run_dir: Path, lock: dict, data: Path, args: dict, YOLO):
    best, last = _record_weights(run_dir, lock)
    completed = _completed_epochs(run_dir)
    planned_stop = lock.get("planned_stop_after_epochs")
    if planned_stop is not None and completed != planned_stop:
        raise RuntimeError(
            f"planned stop completed {completed} epochs instead of {planned_stop}"
        )
    evaluation_weights = last if planned_stop is not None else best
    lock["completed_epochs"] = completed
    lock["evaluation_weights"] = {
        "role": (
            "frozen_planned_stop" if planned_stop is not None else "validation_best"
        ),
        "path": str(evaluation_weights.resolve()),
        "sha256": sha256(evaluation_weights),
    }
    trained = YOLO(str(evaluation_weights))
    test_metrics = trained.val(
        data=str(data.resolve()),
        split="test",
        imgsz=640,
        batch=args["batch"],
        device=0,
        workers=8,
        plots=True,
        project=str(Path(args["project"]).resolve()),
        name=f"{args['name']}-test",
        exist_ok=False,
    )
    lock["test_metrics"] = _json_metrics(test_metrics)
    (run_dir / "test_metrics.json").write_text(
        json.dumps(lock["test_metrics"], indent=2) + "\n"
    )
    return trained


def resume_run(run_dir: Path) -> Path:
    """Resume an interrupted checkpoint with its saved optimizer and schedule."""
    state = resume_inputs(run_dir)
    import torch
    import ultralytics
    from ultralytics import YOLO

    lock = state["lock"]
    expected = lock.get("environment", {})
    if ultralytics.__version__ != expected.get("ultralytics"):
        raise ValueError(
            "resume Ultralytics version differs: "
            f"{ultralytics.__version__} != {expected.get('ultralytics')}"
        )
    if torch.__version__ != expected.get("torch"):
        raise ValueError(
            f"resume Torch version differs: {torch.__version__} != {expected.get('torch')}"
        )
    if not torch.cuda.is_available():
        raise ValueError("resume requires a CUDA GPU")

    repo = Path(__file__).resolve().parents[3]
    attempts = lock.setdefault("resume_attempts", [])
    attempts.append(
        {
            "started_unix": time.time(),
            "source_commit": source_commit(repo),
            "checkpoint": str(state["last"]),
            "checkpoint_sha256": sha256(state["last"]),
            "completed_epochs_before": state["completed_epochs"],
            "environment": _runtime_environment(torch, ultralytics),
        }
    )
    lock["status"] = "resuming"
    state["lock_path"].write_text(json.dumps(lock, indent=2) + "\n")

    if not state["training_complete"]:
        model = YOLO(str(state["last"]))
        planned_stop = lock.get("planned_stop_after_epochs")
        if planned_stop is not None:
            model.add_callback(
                "on_train_epoch_end", planned_stop_callback(planned_stop)
            )
        model.train(resume=True)
    _finish_run(state["run_dir"], lock, state["data"], state["args"], YOLO)
    attempts[-1]["completed_unix"] = time.time()
    lock["status"] = "complete"
    lock["completed_unix"] = time.time()
    state["lock_path"].write_text(json.dumps(lock, indent=2) + "\n")
    return state["run_dir"]


def run(
    data: Path,
    model_path: Path,
    project: Path,
    name: str,
    epochs: int | None,
    batch: int | None,
    cache: str | bool | None,
    qualitative: Path | None,
    preset: str = "pilot",
    planned_stop_after_epochs: int | None = None,
) -> Path:
    if not data.is_file():
        raise ValueError(f"missing dataset YAML: {data}")
    if not model_path.is_file():
        raise ValueError(f"missing source weights: {model_path}")
    dataset_lock = data.parent / "split.lock.json"
    if not dataset_lock.is_file():
        raise ValueError(f"missing dataset lock: {dataset_lock}")
    dataset = json.loads(dataset_lock.read_text())
    if "dataset_sha256" not in dataset:
        raise ValueError(f"dataset lock has no dataset_sha256: {dataset_lock}")
    if preset in FINE_TUNE_PRESETS and (
        dataset.get("schema") != COMPOSITION_SCHEMA
        or dataset.get("preset") != preset
    ):
        raise ValueError(
            f"{preset} training requires a matching locked composition dataset"
        )
    args = training_args(data, project, name, epochs, batch, cache, preset)
    planned_stop_after_epochs = validate_planned_stop(
        planned_stop_after_epochs, args, preset
    )

    run_dir = project.resolve() / name
    if run_dir.exists():
        raise ValueError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    import torch
    import ultralytics
    from ultralytics import YOLO

    repo = Path(__file__).resolve().parents[3]
    runner_path = Path(__file__).resolve()
    lock = {
        "schema": RUN_SCHEMA,
        "status": "started",
        "started_unix": time.time(),
        "git_commit": source_commit(repo),
        "runner": str(runner_path),
        "runner_sha256": sha256(runner_path),
        "dataset_lock_sha256": sha256(dataset_lock),
        "dataset_sha256": dataset["dataset_sha256"],
        "source_weights": str(model_path.resolve()),
        "source_weights_sha256": sha256(model_path),
        "training_preset": preset,
        "planned_stop_after_epochs": planned_stop_after_epochs,
        "training_config_sha256": training_config_sha256(
            args, planned_stop_after_epochs
        ),
        "training_args": args,
        "environment": _runtime_environment(torch, ultralytics),
    }
    lock_path = run_dir / "run.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    model = YOLO(str(model_path))
    if planned_stop_after_epochs is not None:
        model.add_callback(
            "on_train_epoch_end",
            planned_stop_callback(planned_stop_after_epochs),
        )
    model.train(**args)
    if args["epochs"] > 1:
        trained = _finish_run(run_dir, lock, data, args, YOLO)
        if qualitative is not None:
            if not qualitative.exists():
                raise ValueError(f"missing qualitative source: {qualitative}")
            trained.predict(
                source=str(qualitative.resolve()),
                imgsz=640,
                conf=0.10,
                device=0,
                save=True,
                save_txt=True,
                save_conf=True,
                project=str(project.resolve()),
                name=f"{name}-scene49",
                exist_ok=False,
            )
    else:
        _record_weights(run_dir, lock)
        lock["completed_epochs"] = _completed_epochs(run_dir)

    lock["status"] = "complete"
    lock["completed_unix"] = time.time()
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return run_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume an interrupted run directory from its last.pt checkpoint",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--preset", choices=TRAINING_PRESETS, default="pilot")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument(
        "--cache",
        choices=("ram", "disk", "none"),
        default=None,
        help="image cache policy; 'none' streams from the prepared dataset",
    )
    parser.add_argument("--qualitative", type=Path)
    parser.add_argument(
        "--stop-after-epochs",
        type=int,
        help=(
            "intentionally stop after this many completed epochs while preserving "
            "the preset's scheduled horizon"
        ),
    )
    args = parser.parse_args(argv)
    if args.resume is not None:
        conflicts = [
            name
            for name in (
                "data",
                "model",
                "project",
                "name",
                "epochs",
                "batch",
                "cache",
                "qualitative",
                "stop_after_epochs",
            )
            if getattr(args, name) is not None
        ]
        if conflicts or args.preset != "pilot":
            parser.error("--resume does not accept training overrides")
        try:
            run_dir = resume_run(args.resume)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(f"completed {run_dir}")
        return
    missing = [
        name for name in ("data", "model", "project", "name")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(f"missing required arguments: {', '.join('--' + name for name in missing)}")
    if (args.epochs is not None and args.epochs < 1) or (
        args.batch is not None and args.batch < 1
    ):
        parser.error("epochs and batch must be positive")
    run_dir = run(
        args.data,
        args.model,
        args.project,
        args.name,
        args.epochs,
        args.batch,
        False if args.cache == "none" else args.cache,
        args.qualitative,
        args.preset,
        args.stop_after_epochs,
    )
    print(f"completed {run_dir}")


if __name__ == "__main__":
    main()
