#!/usr/bin/env python3
"""One CLI for the July 24 SVO and CM2 localization replay experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import write_json
from flow import run_flow, synthetic_self_check
from prepare import prepare
from report import generate
from svo import run_matrix, score_matrix
from tags import run_tag_anchors


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DEFAULT_BAG = (
    WORKSPACE
    / "reference/flight_bags/20260725_000141_07-24-2005-survey"
)
DEFAULT_WORK = Path("/tmp/maav_localization_20260724_2005")
DEFAULT_ANALYSIS = DEFAULT_BAG / "analysis/localization_experiments"
DEFAULT_SVO = WORKSPACE / "reference/rpg_svo_pro_with_digital_twins"
DEFAULT_GENERATED = (
    WORKSPACE
    / "reference/rpg-consult/vio-research/revision/svo_generated"
)
DEFAULT_OV_CALIBRATION = (
    DEFAULT_GENERATED / "calibrations/A_static_allan.yaml"
)
DEFAULT_CM2_CALIBRATION = (
    WORKSPACE
    / "reference/cyclops_offline/calibration/"
    "20260724_drone4_intrinsics/cm2_intrinsics_rs.yaml"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    result.add_argument("--work", type=Path, default=DEFAULT_WORK)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--max-frames", type=int)
    prepare_command.add_argument("--no-hash", action="store_true")
    svo_command = commands.add_parser("svo")
    svo_command.add_argument("--timeout", type=int, default=900)
    flow_command = commands.add_parser("flow")
    flow_command.add_argument("--max-pairs", type=int)
    flow_command.add_argument("--start-s", type=float)
    commands.add_parser("tags")
    commands.add_parser("report")
    all_command = commands.add_parser("all")
    all_command.add_argument("--timeout", type=int, default=900)
    all_command.add_argument("--max-pairs", type=int)
    return result


def ensure_prepared(args) -> dict:
    return prepare(
        args.bag,
        args.work / "prepared",
        DEFAULT_OV_CALIBRATION,
        hash_sources=not getattr(args, "no_hash", False),
        max_frames=getattr(args, "max_frames", None),
    )


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        print(json.dumps(ensure_prepared(args), indent=2))
        return 0
    self_check = synthetic_self_check()
    write_json(args.work / "flow_self_check.json", self_check)
    if not self_check["passed"]:
        raise SystemExit("flow sign self-check failed")
    manifest = None
    if args.command in {"svo", "report", "all"}:
        manifest = ensure_prepared(args)
    if args.command in {"svo", "all"}:
        assert manifest is not None
        run_matrix(
            Path(manifest["svo_dataset"]),
            args.work / "svo",
            DEFAULT_SVO,
            DEFAULT_GENERATED,
            timeout_seconds=args.timeout,
        )
        score_matrix(
            args.work / "svo/runs",
            HERE / "segments.json",
            args.work / "svo",
        )
    if args.command in {"flow", "all"}:
        run_flow(
            args.bag,
            args.work / "flow",
            DEFAULT_CM2_CALIBRATION,
            max_pairs=getattr(args, "max_pairs", None),
            start_s=getattr(args, "start_s", None),
        )
    if args.command in {"tags", "all"}:
        decision = run_tag_anchors(
            args.bag,
            args.work / "flow/flow_selected.csv",
            args.work / "tags",
        )
        print(json.dumps(decision, indent=2))
    if args.command in {"report", "all"}:
        assert manifest is not None
        report = generate(
            args.work,
            DEFAULT_ANALYSIS,
            manifest,
            self_check,
        )
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
