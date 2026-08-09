"""Extract comparable summary metrics from Hailo HTML profiler reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping


STATS_MARKER = '"stats": {"profiler_parameters"'
METRICS = (
    "fps",
    "latency",
    "number_of_contexts",
    "input_bw",
    "output_bw",
    "ops_per_second",
    "measured_mac_util",
)


def load_stats(path: Path) -> dict:
    """Decode the embedded ``stats`` object without executing report JavaScript."""
    text = path.read_text(errors="replace")
    marker_at = text.find(STATS_MARKER)
    if marker_at < 0:
        raise ValueError(f"cannot find embedded profiler stats in {path}")
    object_at = text.find("{", marker_at)
    stats, _ = json.JSONDecoder().raw_decode(text, object_at)
    if not isinstance(stats, dict):
        raise ValueError(f"invalid profiler stats in {path}")
    return stats


def summarize(path: Path) -> dict:
    stats = load_stats(path)
    parameters = stats.get("profiler_parameters", {})
    model = stats.get("model_details", {})
    performance = stats.get("performance_details", [])
    if not isinstance(performance, list) or not performance:
        raise ValueError(f"profiler report has no performance data: {path}")
    batch1 = next(
        (record for record in performance if record.get("batch_size") == 1),
        performance[0],
    )
    return {
        "report": str(path),
        "compiler_version": parameters.get("compiler_version"),
        "profiling_mode": parameters.get("profiling_mode"),
        "is_quantized": parameters.get("is_quantized"),
        "model_name": model.get("model_name"),
        "hw_arch": model.get("hw_arch"),
        "weights": model.get("weights"),
        "total_ops_per_frame": model.get("total_ops_per_frame"),
        "input_shapes": model.get("input_shapes"),
        "output_shapes": model.get("output_shapes"),
        "post_processing": model.get("post_processing"),
        "batch1": {metric: batch1.get(metric) for metric in METRICS},
    }


def _delta(candidate: object, baseline: object) -> dict:
    if not isinstance(candidate, (int, float)) or not isinstance(
        baseline, (int, float)
    ):
        return {"baseline": baseline, "candidate": candidate}
    absolute = candidate - baseline
    percent = None if baseline == 0 else absolute / baseline * 100.0
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute": absolute,
        "percent": percent,
    }


def compare(baseline: Mapping[str, object], candidate: Mapping[str, object]) -> dict:
    baseline_batch = baseline["batch1"]
    candidate_batch = candidate["batch1"]
    if not isinstance(baseline_batch, Mapping) or not isinstance(
        candidate_batch, Mapping
    ):
        raise ValueError("summaries have invalid batch-one data")
    return {
        "model": {
            "weights": _delta(candidate.get("weights"), baseline.get("weights")),
            "total_ops_per_frame": _delta(
                candidate.get("total_ops_per_frame"),
                baseline.get("total_ops_per_frame"),
            ),
            "output_shapes": {
                "baseline": baseline.get("output_shapes"),
                "candidate": candidate.get("output_shapes"),
            },
        },
        "batch1": {
            metric: _delta(candidate_batch.get(metric), baseline_batch.get(metric))
            for metric in METRICS
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    candidate = summarize(args.candidate)
    result: Dict[str, object] = {"candidate": candidate}
    if args.baseline is not None:
        baseline = summarize(args.baseline)
        result = {
            "baseline": baseline,
            "candidate": candidate,
            "comparison": compare(baseline, candidate),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
