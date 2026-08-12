"""Run the hosted Qwen detector on a CM2 review set and checkpoint proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from PIL import Image

DEFAULT_PROMPT = "green circular prop landmine"
_thread_local = threading.local()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"missing {name}")
    return value


def client() -> InferenceHTTPClient:
    cached = getattr(_thread_local, "client", None)
    if cached is None:
        cached = InferenceHTTPClient(
            api_url=require_environment("ROBOFLOW_API_URL"),
            api_key=require_environment("ROBOFLOW_API_KEY"),
        )
        _thread_local.client = cached
    return cached


def workflow_outputs(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        outputs = result
    elif isinstance(result, dict):
        outputs = result.get("outputs")
    else:
        raise TypeError(f"unexpected Workflow result type: {type(result).__name__}")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Workflow response has no outputs")
    if not all(isinstance(output, dict) for output in outputs):
        raise TypeError("Workflow outputs are not objects")
    return outputs


def prediction_document(result: Any) -> dict[str, Any]:
    predictions = workflow_outputs(result)[0].get("predictions")
    if not isinstance(predictions, dict):
        raise TypeError("Workflow output has no predictions document")
    return predictions


def normalized_predictions(result: Any) -> list[dict[str, Any]]:
    normalized = []
    for prediction in prediction_document(result).get("predictions", []):
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
        normalized.append(
            {
                "xyxy": [
                    x - width / 2,
                    y - height / 2,
                    x + width / 2,
                    y + height / 2,
                ],
                "confidence": float(prediction["confidence"]),
                "class": prediction.get("class"),
            }
        )
    return normalized


def remove_visualizations(result: Any) -> bool:
    removed = False
    for output in workflow_outputs(result):
        removed = output.pop("label_visualization", None) is not None or removed
    return removed


def run_sample(
    sample: dict[str, Any],
    image_path: Path,
    response_path: Path,
    prompt: str,
) -> dict[str, Any]:
    started = time.monotonic()
    result = client().run_workflow(
        workspace_name=require_environment("ROBOFLOW_WORKSPACE"),
        workflow_id=require_environment("ROBOFLOW_WORKFLOW_ID"),
        images={"image": str(image_path)},
        parameters={
            "classes": [prompt],
            "model_api_key": require_environment("OPENROUTER_API_KEY"),
        },
        excluded_fields=["label_visualization"],
        use_cache=True,
    )
    elapsed = time.monotonic() - started
    remove_visualizations(result)
    record = {
        "sample_id": f"{sample['bag']}_f{sample['frame']:04d}",
        "elapsed_seconds": round(elapsed, 3),
        "result": result,
    }
    temporary = response_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(response_path)
    return record


def load_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    prediction_document(record["result"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be positive")
    load_dotenv(args.env, override=False)
    for name in (
        "OPENROUTER_API_KEY",
        "ROBOFLOW_API_KEY",
        "ROBOFLOW_WORKSPACE",
        "ROBOFLOW_WORKFLOW_ID",
        "ROBOFLOW_API_URL",
    ):
        require_environment(name)
    if Path("/etc/ssl/certs/ca-certificates.crt").exists():
        os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")

    selection_path = args.review / "selection.json"
    selection = json.loads(selection_path.read_text())
    if args.limit is not None:
        selection = selection[: args.limit]
    response_directory = args.output / "responses"
    response_directory.mkdir(parents=True, exist_ok=True)

    samples = []
    pending = []
    for sample in selection:
        sample_id = f"{sample['bag']}_f{sample['frame']:04d}"
        image_path = args.review / "raw" / f"{sample_id}.png"
        response_path = response_directory / f"{sample_id}.json"
        with Image.open(image_path) as image:
            width, height = image.size
        entry = {
            "id": sample_id,
            "bag": sample["bag"],
            "frame": sample["frame"],
            "image": str(image_path.relative_to(args.review)),
            "image_sha256": sha256(image_path),
            "image_size": [width, height],
            "response": str(response_path.relative_to(args.output)),
        }
        samples.append(entry)
        if response_path.exists():
            record = load_record(response_path)
            if remove_visualizations(record["result"]):
                response_path.write_text(json.dumps(record, indent=2) + "\n")
        else:
            pending.append((sample, image_path, response_path))

    print(f"Qwen proposals: {len(samples)} samples, {len(pending)} pending")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_sample, sample, image_path, response_path, args.prompt): (
                sample,
                response_path,
            )
            for sample, image_path, response_path in pending
        }
        completed = len(samples) - len(pending)
        for future in as_completed(futures):
            sample, _ = futures[future]
            record = future.result()
            completed += 1
            count = len(normalized_predictions(record["result"]))
            print(
                f"[{completed}/{len(samples)}] {sample['bag']} f{sample['frame']:04d}: "
                f"{count} proposal(s), {record['elapsed_seconds']:.1f}s",
                flush=True,
            )

    rows = []
    for sample in samples:
        response = load_record(args.output / sample["response"])
        predictions = normalized_predictions(response["result"])
        rows.append(
            {
                **sample,
                "elapsed_seconds": response["elapsed_seconds"],
                "predictions": predictions,
                "detections_at_0.60": sum(
                    prediction["confidence"] >= 0.60 for prediction in predictions
                ),
            }
        )
    manifest = {
        "prompt": args.prompt,
        "selection_sha256": sha256(selection_path),
        "sample_count": len(samples),
        "completed_count": len(rows),
        "rows": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
