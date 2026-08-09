"""Generate compact plots and an evidence-limited experiment report."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row, key):
    try:
        value = float(row[key])
        return value if np.isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def _svo_plot(svo_dir: Path, output: Path) -> None:
    rows = _csv(svo_dir / "svo_summary.csv")
    if not rows:
        return
    names = [row["run_id"].split("_", 1)[0] for row in rows]
    speed = [_float(row, "max_speed_m_s") or 0 for row in rows]
    accel = [_float(row, "max_accel_bias_m_s2") or 0 for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(names, speed)
    axes[0].axhline(3.0, color="tab:red", linestyle="--", label="sanity gate")
    axes[0].set_ylabel("maximum speed (m/s)")
    axes[0].legend()
    axes[1].bar(names, accel)
    axes[1].axhline(0.5, color="tab:red", linestyle="--", label="sanity gate")
    axes[1].set_ylabel("maximum accel-bias norm (m/s²)")
    axes[1].legend()
    figure.suptitle("Focused SVO replay — internal physical sanity")
    figure.tight_layout()
    figure.savefig(output / "svo_physical_sanity.png", dpi=170)
    plt.close(figure)


def _flow_plots(flow_dir: Path, output: Path) -> None:
    hypotheses = _csv(flow_dir / "flow_hypothesis_summary.csv")
    if hypotheses:
        labels = [
            f"yaw{row['yaw_quadrants']} {row['rs_bands']}b "
            f"{row['readout_sign']}"
            for row in hypotheses
        ]
        values = [
            1000.0 * (_float(row, "residual_p95_rad") or 0)
            for row in hypotheses
        ]
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.barh(labels[::-1], values[::-1])
        axis.set_xlabel("p95 centered residual (mrad)")
        axis.set_title("CM2 gyro/rolling-shutter hypotheses")
        figure.tight_layout()
        figure.savefig(output / "cm2_hypotheses.png", dpi=170)
        plt.close(figure)
    selected = _csv(flow_dir / "flow_selected.csv")
    if selected:
        flow_n = [
            _float(row, "path_north_m") or 0.0 for row in selected
        ]
        flow_e = [
            _float(row, "path_east_m") or 0.0 for row in selected
        ]
        local_n = [_float(row, "local_x_m") for row in selected]
        local_e = [_float(row, "local_y_m") for row in selected]
        figure, axis = plt.subplots(figsize=(6.5, 6.5))
        axis.plot(flow_e, flow_n, label="CM2+dToF diagnostic")
        valid = [
            index
            for index, (north, east) in enumerate(zip(local_n, local_e))
            if north is not None and east is not None
        ]
        if valid:
            origin_n = local_n[valid[0]]
            origin_e = local_e[valid[0]]
            axis.plot(
                [local_e[index] - origin_e for index in valid],
                [local_n[index] - origin_n for index in valid],
                label="PX4 contextual comparison",
                alpha=0.8,
            )
        axis.axis("equal")
        axis.set_xlabel("east displacement (m)")
        axis.set_ylabel("north displacement (m)")
        axis.set_title("Integrated horizontal path (not independent truth)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "cm2_integrated_path.png", dpi=170)
        plt.close(figure)


def _tag_plots(flow_dir: Path, tag_dir: Path, output: Path) -> None:
    selected = _csv(flow_dir / "flow_selected.csv")
    bins = _csv(tag_dir / "tag_anchor_bins.csv")
    decision_path = tag_dir / "tag_anchor_decision.json"
    if not selected or not bins or not decision_path.exists():
        return
    decision = json.loads(decision_path.read_text())
    translation = decision["initial_translation_anchor"]["north_east_m"]
    flow_n = np.asarray(
        [_float(row, "path_north_m") or 0.0 for row in selected]
    ) + translation[0]
    flow_e = np.asarray(
        [_float(row, "path_east_m") or 0.0 for row in selected]
    ) + translation[1]
    tag_n = np.asarray([_float(row, "tag_north_m") for row in bins])
    tag_e = np.asarray([_float(row, "tag_east_m") for row in bins])
    times = np.asarray([_float(row, "timestamp_s") for row in bins])
    errors = np.asarray([_float(row, "anchor_error_m") for row in bins])
    times -= times[0]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(
        flow_e,
        flow_n,
        color="0.65",
        linewidth=1.2,
        label="CM2 flow (initial translation only)",
    )
    points = axes[0].scatter(
        tag_e,
        tag_n,
        c=times,
        cmap="viridis",
        s=18,
        label="tag-derived vehicle fixes",
        zorder=3,
    )
    tag6 = decision["anchor_definition"]["tag6_north_east_m"]
    tag7 = decision["anchor_definition"]["tag7_north_east_m"]
    axes[0].scatter(
        [tag6[1], tag7[1]],
        [tag6[0], tag7[0]],
        marker="x",
        s=70,
        color="tab:red",
        label="fixed tag map",
        zorder=4,
    )
    axes[0].axis("equal")
    axes[0].set_xlabel("east (m)")
    axes[0].set_ylabel("north (m)")
    axes[0].set_title("Path with tag encounters")
    axes[0].legend(fontsize=8)
    figure.colorbar(points, ax=axes[0], label="seconds after first tag fix")

    axes[1].scatter(times, errors, s=14)
    axes[1].axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("seconds after first tag fix")
    axes[1].set_ylabel("translation-only flow error (m)")
    axes[1].set_title("Anchor error reveals blind-gap drift")
    figure.tight_layout()
    figure.savefig(output / "cm2_tag_anchors.png", dpi=170)
    plt.close(figure)


def generate(
    work: Path, analysis: Path, manifest: dict, self_check: dict
) -> Path:
    analysis.mkdir(parents=True, exist_ok=True)
    svo_dir = work / "svo"
    flow_dir = work / "flow"
    tag_dir = work / "tags"
    svo = (
        json.loads((svo_dir / "svo_decision.json").read_text())
        if (svo_dir / "svo_decision.json").exists()
        else None
    )
    flow = (
        json.loads((flow_dir / "flow_decision.json").read_text())
        if (flow_dir / "flow_decision.json").exists()
        else None
    )
    tag_anchors = (
        json.loads((tag_dir / "tag_anchor_decision.json").read_text())
        if (tag_dir / "tag_anchor_decision.json").exists()
        else None
    )
    _svo_plot(svo_dir, analysis)
    _flow_plots(flow_dir, analysis)
    _tag_plots(flow_dir, tag_dir, analysis)
    for source in (
        svo_dir / "svo_summary.csv",
        svo_dir / "svo_segments.csv",
        svo_dir / "svo_decision.json",
        flow_dir / "flow_hypothesis_summary.csv",
        flow_dir / "flow_decision.json",
        flow_dir / "flow_selected.csv",
        tag_dir / "tag_anchor_decision.json",
        tag_dir / "tag_anchor_fixes.csv",
        tag_dir / "tag_anchor_bins.csv",
        tag_dir / "tag_anchor_epochs.csv",
        tag_dir / "tag_anchor_local_steps.csv",
        tag_dir / "tag_anchor_legs.csv",
    ):
        if source.exists():
            shutil.copy2(source, analysis / source.name)

    lines = [
        "# July 24 localization replay experiments",
        "",
        "Status: offline evidence only. Neither estimator is qualified for flight.",
        "",
        "## Input lineage",
        "",
        f"- Source: `{manifest['bag']}`",
        f"- Forward frames: {manifest['camera_frames']}",
        f"- Forward nominal/effective rate: "
        f"{manifest['camera_rate_hz']:.2f}/{manifest['camera_effective_rate_hz']:.2f} Hz",
        f"- IMU samples: {manifest['imu_samples']}",
        f"- IMU median/effective rate: "
        f"{manifest['imu_rate_hz']:.2f}/{manifest['imu_effective_rate_hz']:.2f} Hz",
        f"- IMU covers camera: {manifest['imu_covers_camera']}",
        f"- `/imu` timestamps also present in the independent raw "
        f"SensorCombined subscription: "
        f"{manifest['clock_mapping']['exact_timestamp_matches']}/"
        f"{manifest['clock_mapping']['derived_imu_samples']}",
        f"- Deterministic sign self-check: {self_check['passed']}",
        "",
    ]
    if svo:
        lines.extend(
            [
                "## SVO",
                "",
                f"- Decision: {svo['interpretation']}",
                "- The replay includes preflight and takeoff; the earlier "
                "20:30 replay did not.",
                "- Every trial initialized about 3.4 s after takeoff. Seeing "
                "takeoff was therefore insufficient to make SVO stable.",
                "- A/B/C/G already use the accepted July 19 final-installed "
                "native-flip OV9281–IMU `T_B_C`. The generated calibration "
                "files' `provisional` label is stale metadata.",
                "",
                "| Run | Init after takeoff | Max speed | Max accel bias | Resets | Sane |",
                "| --- | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        for row in svo["runs"]:
            lines.append(
                f"| {row['run_id']} | "
                f"{row['first_metric_init_after_takeoff_s']:.2f} s | "
                f"{row['max_speed_m_s']:.2f} | "
                f"{row['max_accel_bias_m_s2']:.3f} | "
                f"{row['reinitialization_count']} | "
                f"{row['physically_sane']} |"
            )
        lines.append("")
    if flow:
        best = flow["selected_hypothesis"]
        context = flow["px4_contextual_velocity_comparison"]
        tags = flow["tag_sighting_context"]
        lines.extend(
            [
                "## CM2 flow",
                "",
                f"- Accepted pairs: {flow['accepted_pairs']} "
                f"({100*flow['accepted_pair_availability']:.1f}% of transitions).",
                f"- Recorded rate: {flow['recorded_rate_hz']:.2f} Hz.",
                f"- Lowest-residual hypothesis: yaw quadrant "
                f"{best['yaw_quadrants']}, {best['rs_bands']} bands, "
                f"readout sign {best['readout_sign']}.",
                f"- Axis mapping uniquely supported at the predeclared 5% "
                f"margin: {flow['axis_mapping_is_unique']} "
                f"({100*flow['axis_selection_margin_p95_fraction']:.1f}% margin).",
                f"- Rolling-shutter variant uniquely supported at the same "
                f"margin: {flow['rolling_shutter_variant_is_unique']} "
                f"({100*flow['timing_selection_margin_p95_fraction']:.1f}% margin).",
                f"- Diagnostic endpoint: north "
                f"{flow['diagnostic_endpoint_north_m']:.2f} m, east "
                f"{flow['diagnostic_endpoint_east_m']:.2f} m.",
                f"- PX4 contextual endpoint: north "
                f"{flow['px4_contextual_endpoint']['north_m']:.2f} m, east "
                f"{flow['px4_contextual_endpoint']['east_m']:.2f} m.",
                f"- Against PX4 velocity only as a contextual comparator: "
                f"{100*context['direction_agreement_fraction']:.1f}% direction "
                f"agreement, {context['median_speed_ratio_to_px4']:.2f}× median "
                f"speed, N/E correlations {context['north_correlation']:.2f}/"
                f"{context['east_correlation']:.2f}.",
                f"- All {tags['matched_flow_pairs']} tag-sighting frames had "
                f"matched flow; median quality was {tags['quality_median']:.0f}/255.",
                "",
                "The integrated path uses dToF and PX4 heading and is a "
                "diagnostic proxy, not ground truth.",
                "",
            ]
        )
    if tag_anchors:
        local = tag_anchors["within_encounter_motion"]
        encounter = tag_anchors["encounter_anchor_error"]
        disagreement = tag_anchors["two_tag_fix_disagreement"]
        lines.extend(
            [
                "## Tag-anchor comparison",
                "",
                "- Tags 6 and 7 define a fixed local map: their midpoint is "
                "the known center-square origin, and their baseline is learned "
                "from the first five seconds of the centered encounter.",
                "- This comparison uses PX4 attitude and downward dToF, but no "
                "EKF2 horizontal position and no GNSS position or velocity.",
                f"- Map baseline: "
                f"{tag_anchors['anchor_definition']['tag_separation_m']:.3f} m "
                f"from "
                f"{tag_anchors['anchor_definition']['map_calibration_pair_samples']} "
                "simultaneous two-tag frames.",
                f"- Two-tag vehicle-fix disagreement: median "
                f"{disagreement['median_m']:.3f} m, p95 "
                f"{disagreement['p95_m']:.3f} m.",
                f"- During tag-visible motion, flow displacement was "
                f"{local['median_flow_m_per_tag_m']:.3f}× tag-derived "
                f"displacement; the bootstrap interval for the median was "
                f"{local['bootstrap_p05_median_flow_m_per_tag_m']:.3f}–"
                f"{local['bootstrap_p95_median_flow_m_per_tag_m']:.3f}× "
                f"over {local['eligible_half_second_steps']} half-second steps.",
                f"- Direction error during those steps: median "
                f"{local['median_direction_error_deg']:.1f}°, p95 "
                f"{local['p95_direction_error_deg']:.1f}°.",
                f"- After the initial translation alignment, later tag fixes "
                f"saw an accumulated flow-path offset: epoch 2 had "
                f"{encounter['second_epoch_median_m']:.2f} m median error, "
                f"and epoch medians from epoch 3 onward ranged "
                f"{encounter['epochs_3_onward_median_range_m'][0]:.2f}–"
                f"{encounter['epochs_3_onward_median_range_m'][1]:.2f} m.",
                "",
                "The tag evidence supports the local flow scale. It does not "
                "support treating the raw integrated flow path as a stable "
                "position estimate across tag-blind legs.",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusions",
            "",
            "- Takeoff excitation allowed SVO to initialize during the climb, "
            "but did not prevent later divergence and resets.",
            "- SVO is strongly noise-sensitive: powered settings avoid the "
            "catastrophic bias growth of the static setting, but none pass.",
            "- Relaxing the SVO solver budget does not cure the failure.",
            "- CM2 angular flow is available and directionally consistent "
            "enough to justify a dedicated 41 Hz qualification capture.",
            "- Tag-derived anchors indicate the flow magnitude itself is "
            "approximately correct; the earlier 1.19× PX4 comparison reflects "
            "a disagreement with GNSS-fused EKF2, not evidence of a 19% flow "
            "scale error.",
            "- Repeated tag encounters expose accumulated open-loop position "
            "offsets during blind legs. PX4 EKF2 flow fusion should own "
            "inertial propagation instead of integrating this diagnostic path.",
            "- Yaw quadrant 0 is supported; this flight does not distinguish "
            "global-shutter approximation from 8/16-band positive-readout "
            "correction strongly enough to choose among them.",
            "",
            "## Boundaries",
            "",
            "- The CM2 capture ran near 30 Hz, so this does not qualify 41 Hz operation.",
            "- CM2-to-Pixhawk rotation is inferred from discrete hypotheses.",
            "- Tag anchors share PX4 attitude and dToF with the flow pipeline. "
            "They independently constrain horizontal image displacement, not "
            "attitude or height errors.",
            "- The tag baseline is inferred from the initial centered "
            "encounter; the tag locations are not surveyed.",
            "- CM2 and dToF origin offsets are neglected in this replay; they "
            "are small but must be measured for a flight implementation.",
            "- OV9281-to-IMU extrinsics are the accepted July 19 installed-camera "
            "solve. A new solve is needed only if the camera/IMU geometry changes.",
            "- Exposure and gain were measured during capture but not serialized.",
            "- No live SensorOpticalFlow publisher or PX4 configuration was changed.",
            "",
            "Before a publisher is implemented, capture one installed-camera "
            "41 Hz dataset with deliberate props-off body-axis rotations and "
            "taped translations. That closes the mount/sign and rate gaps "
            "without requiring RTK, mocap, or additional sensors.",
            "",
        ]
    )
    report = analysis / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
