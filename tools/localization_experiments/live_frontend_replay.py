"""Replay the exact live CM2 frontend against an MCAP before deployment."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SENSING = HERE.parents[1] / "ros" / "sensing"
sys.path.insert(0, str(SENSING))

from sensing.cm2_flow import Cm2FlowFrontend, ImuHistory, STATUS_VALID  # noqa: E402
from sensing.cm2_svo_flow import Cm2SvoFlowFrontend  # noqa: E402
from common import iter_messages, stamp_ns  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument(
        "--backend", choices=("klt", "svo"), default="klt"
    )
    parser.add_argument(
        "--svo-build",
        type=Path,
        default=(
            HERE.parents[2]
            / "reference/rl_vo/svo-lib/build/svo_env"
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=SENSING / "sensing" / "config" / "cm2_intrinsics_rs.yaml",
    )
    args = parser.parse_args()

    imu = ImuHistory(horizon_s=300.0)
    for topic, _, message in iter_messages(args.bag, ["/imu"]):
        angular = message.angular_velocity
        imu.note(
            stamp_ns(message.header),
            (angular.x, angular.y, angular.z),
        )
    if args.backend == "svo":
        frontend = Cm2SvoFlowFrontend(
            args.calibration,
            imu,
            args.svo_build,
            SENSING / "sensing" / "config" / "svo_flow_params.yaml",
            SENSING / "sensing" / "config" / "svo_flow_cm2_820.yaml",
        )
    else:
        frontend = Cm2FlowFrontend(args.calibration, imu)
    results = []
    latency_ms = []
    for topic, _, image in iter_messages(
        args.bag, ["/camera_down/image_raw"]
    ):
        started = time.perf_counter()
        results.append(frontend.process(image, stamp_ns(image.header)))
        latency_ms.append((time.perf_counter() - started) * 1e3)
        if len(results) >= args.max_pairs + 1:
            break
    valid = [result for result in results if result.status == STATUS_VALID]
    if not valid:
        raise RuntimeError("live frontend produced no valid pairs")
    tracked = [result for result in results[1:] if result.tracked >= 8]
    # PX4 computes (-raw + gyro) / dt. This must be the negative of the
    # frontend's compensated translational pixel-flow convention.
    contract_error = [
        np.linalg.norm(
            -result.pixel_flow_raw
            + result.delta_angle[:2]
            + result.pixel_flow_compensated
        )
        for result in valid
    ]
    print(
        f"backend={args.backend} frames={len(results)} valid={len(valid)} "
        f"tracker_availability={len(tracked) / max(1, len(results) - 1):.3f} "
        f"quality_availability={len(valid) / max(1, len(results) - 1):.3f} "
        f"quality_median={np.median([r.quality for r in valid]):.1f} "
        f"latency_median_ms={np.median(latency_ms):.1f} "
        f"latency_p95_ms={np.percentile(latency_ms, 95):.1f} "
        f"contract_error_max={max(contract_error):.3e} "
        f"statuses={dict(Counter(r.status for r in results))}"
    )
    residuals = [
        result.residual_p95_rad
        for result in results
        if np.isfinite(result.residual_p95_rad)
    ]
    print(
        "residual_p95_rad percentiles="
        + ",".join(
            f"{percentile}:{np.percentile(residuals, percentile):.5f}"
            for percentile in (50, 90, 95, 97.5, 99)
        )
    )


if __name__ == "__main__":
    main()
