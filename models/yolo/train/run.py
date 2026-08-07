"""Run the pinned YOLO11m pilot and record a reproducibility lock.

Ultralytics is imported only inside ``run`` so local config tests do not need
the GPU environment.
"""

from __future__ import annotations

import argparse
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
        "save_period": 5 if epochs > 1 else -1,
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


def training_config_sha256(args: dict) -> str:
    """Hash behavior-changing settings independently of run locations."""
    config = {
        key: value for key, value in args.items() if key not in _CONFIG_RUNTIME_KEYS
    }
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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

    run_dir = project.resolve() / name
    if run_dir.exists():
        raise ValueError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    import torch
    import ultralytics
    from ultralytics import YOLO

    repo = Path(__file__).resolve().parents[3]
    lock = {
        "schema": RUN_SCHEMA,
        "status": "started",
        "started_unix": time.time(),
        "git_commit": source_commit(repo),
        "dataset_lock_sha256": sha256(dataset_lock),
        "dataset_sha256": dataset["dataset_sha256"],
        "source_weights": str(model_path.resolve()),
        "source_weights_sha256": sha256(model_path),
        "training_preset": preset,
        "training_config_sha256": training_config_sha256(args),
        "training_args": args,
        "environment": {
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
        },
    }
    lock_path = run_dir / "run.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    model = YOLO(str(model_path))
    model.train(**args)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError("training finished without best.pt and last.pt")

    lock["best_weights_sha256"] = sha256(best)
    lock["last_weights_sha256"] = sha256(last)
    if args["epochs"] > 1:
        trained = YOLO(str(best))
        test_metrics = trained.val(
            data=str(data.resolve()),
            split="test",
            imgsz=640,
            batch=args["batch"],
            device=0,
            workers=8,
            plots=True,
            project=str(project.resolve()),
            name=f"{name}-test",
            exist_ok=False,
        )
        lock["test_metrics"] = _json_metrics(test_metrics)
        (run_dir / "test_metrics.json").write_text(
            json.dumps(lock["test_metrics"], indent=2) + "\n"
        )
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

    lock["status"] = "complete"
    lock["completed_unix"] = time.time()
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return run_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
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
    args = parser.parse_args(argv)
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
    )
    print(f"completed {run_dir}")


if __name__ == "__main__":
    main()
