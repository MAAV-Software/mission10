"""Use the center-square tags as independent horizontal anchors for CM2 flow."""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import median, percentile, write_csv, write_json


# Native 1640x1232 CM2 calibration used by the tag detector output.
FX, FY, CX, CY = 1298.69385194, 1299.56328818, 827.70242273, 617.60425847
K1, K2, P1, P2 = 0.15046410, -0.23368367, 0.00000042, -0.00209991

# image-right -> body-right, image-down -> body-back, optical axis -> body-down
R_BODY_CAMERA = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
TAG_IDS = (6, 7)
MAP_CALIBRATION_SECONDS = 5.0


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class _Nearest:
    def __init__(self, rows: list[dict]):
        self.rows = sorted(rows, key=lambda row: float(row["t"]))
        self.timestamps = [float(row["t"]) for row in self.rows]

    def get(self, timestamp: float, max_age: float = 0.1) -> dict | None:
        index = bisect_left(self.timestamps, timestamp)
        candidates = [
            chosen
            for chosen in (index - 1, index)
            if 0 <= chosen < len(self.rows)
        ]
        if not candidates:
            return None
        chosen = min(
            candidates, key=lambda value: abs(self.timestamps[value] - timestamp)
        )
        if abs(self.timestamps[chosen] - timestamp) > max_age:
            return None
        return self.rows[chosen]


def _undistort(u: float, v: float) -> tuple[float, float]:
    """Invert the calibrated radial-tangential model."""
    xd, yd = (u - CX) / FX, (v - CY) / FY
    x, y = xd, yd
    for _ in range(8):
        r2 = x * x + y * y
        radial = 1.0 + K1 * r2 + K2 * r2 * r2
        dx = 2.0 * P1 * x * y + P2 * (r2 + 2.0 * x * x)
        dy = P1 * (r2 + 2.0 * y * y) + 2.0 * P2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def _rotation_ned_body(q: list[float]) -> np.ndarray:
    """PX4 body-FRD to local-NED rotation for a w,x,y,z quaternion."""
    w, x, y, z = (float(value) for value in q)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _relative_tag_ne(
    u: float, v: float, q: list[float], slant_range_m: float
) -> np.ndarray | None:
    """Vehicle-to-tag NE vector from a ground-plane ray intersection."""
    x, y = _undistort(u, v)
    ray_camera = np.array([x, y, 1.0], dtype=np.float64)
    rotation = _rotation_ned_body(q)
    ray_ned = rotation @ R_BODY_CAMERA @ ray_camera
    # The dToF points along body +Z. Its projection on NED +Z is the vertical
    # camera-to-ground distance used by the ray/ground intersection.
    agl = float(slant_range_m) * float(rotation[2, 2])
    if not 0.2 <= agl <= 10.0 or ray_ned[2] <= 0.1:
        return None
    return ray_ned[:2] * (agl / ray_ned[2])


def _interp_flow(flow: list[dict], timestamp: float) -> np.ndarray | None:
    t = np.asarray([float(row["timestamp_sample_s"]) for row in flow])
    if timestamp < t[0] or timestamp > t[-1]:
        return None
    north = np.asarray([float(row["path_north_m"]) for row in flow])
    east = np.asarray([float(row["path_east_m"]) for row in flow])
    return np.array(
        [np.interp(timestamp, t, north), np.interp(timestamp, t, east)]
    )


def _bootstrap_epoch_median(
    rows: list[dict], samples: int = 2000
) -> tuple[float, float]:
    generator = np.random.default_rng(20260727)
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["flow_to_tag_ratio"] is not None:
            grouped[int(row["epoch"])].append(row["flow_to_tag_ratio"])
    epochs = sorted(grouped)
    estimates = []
    for _ in range(samples):
        sampled = generator.integers(0, len(epochs), len(epochs))
        values = [
            value
            for index in sampled
            for value in grouped[epochs[index]]
        ]
        estimates.append(float(np.median(values)))
    return (
        float(np.percentile(estimates, 5)),
        float(np.percentile(estimates, 95)),
    )


def _time_bins(fixes: list[dict], seconds: float = 0.5) -> list[dict]:
    """Downsample correlated detector frames before fitting scale."""
    bins: dict[int, list[dict]] = defaultdict(list)
    origin = fixes[0]["timestamp_s"]
    for row in fixes:
        bins[int((row["timestamp_s"] - origin) // seconds)].append(row)
    out = []
    for rows in bins.values():
        out.append(
            {
                "timestamp_s": median(row["timestamp_s"] for row in rows),
                "tag_north_m": median(row["tag_north_m"] for row in rows),
                "tag_east_m": median(row["tag_east_m"] for row in rows),
                "flow_north_m": median(row["flow_north_m"] for row in rows),
                "flow_east_m": median(row["flow_east_m"] for row in rows),
                "frames": len(rows),
                "tags": ",".join(
                    str(tag)
                    for tag in sorted(
                        {
                            tag
                            for row in rows
                            for tag in str(row["tags"]).split(",")
                        }
                    )
                ),
                "two_tag_disagreement_m": median(
                    row["two_tag_disagreement_m"] for row in rows
                ),
            }
        )
    return out


def _epochs(bins: list[dict], gap_seconds: float = 2.0) -> list[list[dict]]:
    result: list[list[dict]] = []
    for row in bins:
        if (
            not result
            or row["timestamp_s"] - result[-1][-1]["timestamp_s"] > gap_seconds
        ):
            result.append([])
        result[-1].append(row)
    return result


def run_tag_anchors(
    bag: Path, flow_csv: Path, output: Path
) -> dict:
    """Build tag fixes and compare them with the integrated CM2 flow path."""
    analysis = bag / "analysis"
    telemetry = json.loads(
        (
            analysis
            / "20260725_000141_07-24-2005-survey_telemetry.json"
        ).read_text()
    )
    attitudes = _Nearest(telemetry["attitude"])
    ranges = _Nearest(
        [
            row
            for row in telemetry["range"]
            if int(row.get("orientation", 25)) == 25
        ]
    )
    flow = _rows(flow_csv)
    raw_detections = _rows(analysis / "flight1_cm2_tags.csv")

    frame_vectors: dict[int, dict[int, dict]] = defaultdict(dict)
    for detection in raw_detections:
        tag = int(detection["tag_id"])
        if tag not in TAG_IDS:
            continue
        timestamp = int(detection["timestamp_ns"]) / 1e9
        attitude = attitudes.get(timestamp)
        distance = ranges.get(timestamp)
        if attitude is None or distance is None:
            continue
        relative = _relative_tag_ne(
            float(detection["cx"]),
            float(detection["cy"]),
            attitude["q"],
            float(distance["distance"]),
        )
        if relative is None:
            continue
        frame_vectors[int(detection["frame"])][tag] = {
            "timestamp_s": timestamp,
            "relative": relative,
        }

    pair_rows = []
    pair_distances = []
    for detections in frame_vectors.values():
        if all(tag in detections for tag in TAG_IDS):
            delta = detections[7]["relative"] - detections[6]["relative"]
            distance = float(np.linalg.norm(delta))
            if distance > 0.1:
                pair_rows.append(
                    (detections[6]["timestamp_s"], delta)
                )
                pair_distances.append(distance)
    if not pair_rows:
        raise RuntimeError("no simultaneous tag 6/7 observations")
    # During the first encounter the aircraft is over the known center of the
    # concrete square. The tag-to-tag vector cancels vehicle translation, so
    # the early dToF-backed observations establish both map direction and
    # metric scale without flow or EKF horizontal position. Freeze that map
    # before evaluating later encounters.
    first_pair_time = min(row[0] for row in pair_rows)
    calibration_vectors = np.asarray(
        [
            vector
            for timestamp, vector in pair_rows
            if timestamp <= first_pair_time + MAP_CALIBRATION_SECONDS
        ]
    )
    baseline = np.median(calibration_vectors, axis=0)
    tag_separation_m = float(np.linalg.norm(baseline))
    tag_positions = {
        6: -0.5 * baseline,
        7: 0.5 * baseline,
    }

    fixes = []
    for frame, detections in sorted(frame_vectors.items()):
        positions = []
        for tag, detection in detections.items():
            positions.append(tag_positions[tag] - detection["relative"])
        timestamp = median(
            detection["timestamp_s"] for detection in detections.values()
        )
        flow_position = _interp_flow(flow, timestamp)
        if flow_position is None:
            continue
        position = np.median(np.asarray(positions), axis=0)
        disagreement = (
            float(np.linalg.norm(positions[0] - positions[1]))
            if len(positions) == 2
            else None
        )
        fixes.append(
            {
                "frame": frame,
                "timestamp_s": timestamp,
                "tags": ",".join(str(tag) for tag in sorted(detections)),
                "tag_north_m": float(position[0]),
                "tag_east_m": float(position[1]),
                "flow_north_m": float(flow_position[0]),
                "flow_east_m": float(flow_position[1]),
                "two_tag_disagreement_m": disagreement,
            }
        )
    bins = _time_bins(fixes)

    # Establish only the arbitrary translation offset during the same five
    # seconds used to define the tag map. Do not rotate or rescale the flow
    # path before measuring its later error.
    initial = [
        row
        for row in bins
        if row["timestamp_s"] <= bins[0]["timestamp_s"] + MAP_CALIBRATION_SECONDS
    ]
    initial_translation = np.median(
        np.asarray(
            [
                [
                    row["tag_north_m"] - row["flow_north_m"],
                    row["tag_east_m"] - row["flow_east_m"],
                ]
                for row in initial
            ]
        ),
        axis=0,
    )
    for row in bins:
        row["anchored_flow_north_m"] = (
            row["flow_north_m"] + initial_translation[0]
        )
        row["anchored_flow_east_m"] = (
            row["flow_east_m"] + initial_translation[1]
        )
        row["anchor_error_m"] = math.hypot(
            row["tag_north_m"] - row["anchored_flow_north_m"],
            row["tag_east_m"] - row["anchored_flow_east_m"],
        )

    epoch_rows = []
    grouped = _epochs(bins)
    for index, rows in enumerate(grouped, 1):
        epoch_rows.append(
            {
                "epoch": index,
                "start_s": rows[0]["timestamp_s"],
                "end_s": rows[-1]["timestamp_s"],
                "duration_s": rows[-1]["timestamp_s"] - rows[0]["timestamp_s"],
                "bins": len(rows),
                "tag_north_m": median(row["tag_north_m"] for row in rows),
                "tag_east_m": median(row["tag_east_m"] for row in rows),
                "flow_north_m": median(row["flow_north_m"] for row in rows),
                "flow_east_m": median(row["flow_east_m"] for row in rows),
                "anchor_error_median_m": median(
                    row["anchor_error_m"] for row in rows
                ),
                "anchor_error_p95_m": percentile(
                    [row["anchor_error_m"] for row in rows], 95
                ),
            }
        )

    local_steps = []
    for group_index, group in enumerate(grouped, 1):
        for first, second in zip(group, group[1:]):
            flow_delta = np.array(
                [
                    second["flow_north_m"] - first["flow_north_m"],
                    second["flow_east_m"] - first["flow_east_m"],
                ]
            )
            tag_delta = np.array(
                [
                    second["tag_north_m"] - first["tag_north_m"],
                    second["tag_east_m"] - first["tag_east_m"],
                ]
            )
            flow_distance = float(np.linalg.norm(flow_delta))
            tag_distance = float(np.linalg.norm(tag_delta))
            cosine = (
                float(
                    np.dot(flow_delta, tag_delta)
                    / (flow_distance * tag_distance)
                )
                if flow_distance > 0.03 and tag_distance > 0.03
                else None
            )
            eligible = tag_distance >= 0.15
            local_steps.append(
                {
                    "epoch": group_index,
                    "from_s": first["timestamp_s"],
                    "to_s": second["timestamp_s"],
                    "flow_displacement_m": flow_distance,
                    "tag_displacement_m": tag_distance,
                    "flow_to_tag_ratio": (
                        flow_distance / tag_distance if eligible else None
                    ),
                    "direction_error_deg": (
                        math.degrees(
                            math.acos(np.clip(cosine, -1.0, 1.0))
                        )
                        if cosine is not None
                        else None
                    ),
                    "used_for_ratio": eligible,
                }
            )

    legs = []
    for group_index, (first_group, second_group) in enumerate(
        zip(grouped, grouped[1:]), 1
    ):
        first, second = first_group[-1], second_group[0]
        flow_delta = np.array(
            [
                second["flow_north_m"] - first["flow_north_m"],
                second["flow_east_m"] - first["flow_east_m"],
            ]
        )
        tag_delta = np.array(
            [
                second["tag_north_m"] - first["tag_north_m"],
                second["tag_east_m"] - first["tag_east_m"],
            ]
        )
        flow_distance = float(np.linalg.norm(flow_delta))
        tag_distance = float(np.linalg.norm(tag_delta))
        cosine = (
            float(np.dot(flow_delta, tag_delta) / (flow_distance * tag_distance))
            if flow_distance > 0.1 and tag_distance > 0.1
            else None
        )
        legs.append(
            {
                "from_epoch": group_index,
                "to_epoch": group_index + 1,
                "gap_s": second["timestamp_s"] - first["timestamp_s"],
                "flow_displacement_m": flow_distance,
                "tag_displacement_m": tag_distance,
                "flow_to_tag_ratio": (
                    flow_distance / tag_distance if tag_distance >= 0.75 else None
                ),
                "direction_error_deg": (
                    math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
                    if cosine is not None
                    else None
                ),
                "used_for_ratio": tag_distance >= 0.75,
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "tag_anchor_fixes.csv", fixes)
    write_csv(output / "tag_anchor_bins.csv", bins)
    write_csv(output / "tag_anchor_epochs.csv", epoch_rows)
    write_csv(output / "tag_anchor_local_steps.csv", local_steps)
    write_csv(output / "tag_anchor_legs.csv", legs)
    local_ratios = [
        row["flow_to_tag_ratio"]
        for row in local_steps
        if row["flow_to_tag_ratio"] is not None
    ]
    local_ratio_p05, local_ratio_p95 = _bootstrap_epoch_median(local_steps)
    decision = {
        "anchor_definition": {
            "origin": "midpoint of tag36h11 6 and 7",
            "geometry": (
                "opposing corners of center concrete square; baseline learned "
                "from the first centered encounter"
            ),
            "map_calibration_seconds": MAP_CALIBRATION_SECONDS,
            "map_calibration_pair_samples": len(calibration_vectors),
            "tag_separation_m": tag_separation_m,
            "tag6_north_east_m": tag_positions[6].tolist(),
            "tag7_north_east_m": tag_positions[7].tolist(),
            "horizontal_ekf_position_used": False,
            "attitude_source": "PX4 vehicle attitude",
            "height_source": "downward dToF",
        },
        "observed_two_tag_separation": {
            "samples": len(pair_distances),
            "median_m": median(pair_distances),
            "p05_m": percentile(pair_distances, 5),
            "p95_m": percentile(pair_distances, 95),
        },
        "fixes": len(fixes),
        "half_second_bins": len(bins),
        "encounter_epochs": len(epoch_rows),
        "two_tag_fix_disagreement": {
            "median_m": median(
                row["two_tag_disagreement_m"] for row in fixes
            ),
            "p95_m": percentile(
                [row["two_tag_disagreement_m"] for row in fixes], 95
            ),
        },
        "initial_translation_anchor": {
            "calibration_bins": len(initial),
            "north_east_m": initial_translation.tolist(),
            "later_error_median_m": median(
                row["anchor_error_m"] for row in bins[len(initial) :]
            ),
            "later_error_p95_m": percentile(
                [row["anchor_error_m"] for row in bins[len(initial) :]], 95
            ),
        },
        "within_encounter_motion": {
            "eligible_half_second_steps": len(local_ratios),
            "median_flow_m_per_tag_m": median(local_ratios),
            "bootstrap_p05_median_flow_m_per_tag_m": local_ratio_p05,
            "bootstrap_p95_median_flow_m_per_tag_m": local_ratio_p95,
            "median_direction_error_deg": median(
                row["direction_error_deg"]
                for row in local_steps
                if row["used_for_ratio"]
            ),
            "p95_direction_error_deg": percentile(
                [
                    row["direction_error_deg"]
                    for row in local_steps
                    if row["used_for_ratio"]
                ],
                95,
            ),
        },
        "encounter_anchor_error": {
            "initial_epoch_median_m": epoch_rows[0][
                "anchor_error_median_m"
            ],
            "second_epoch_median_m": epoch_rows[1][
                "anchor_error_median_m"
            ],
            "epochs_3_onward_median_range_m": [
                min(
                    row["anchor_error_median_m"]
                    for row in epoch_rows[2:]
                ),
                max(
                    row["anchor_error_median_m"]
                    for row in epoch_rows[2:]
                ),
            ],
        },
        "inter_encounter_ratios": {
            "eligible_legs": sum(row["used_for_ratio"] for row in legs),
            "median_flow_m_per_tag_m": median(
                row["flow_to_tag_ratio"] for row in legs
            ),
            "ratios": [
                row["flow_to_tag_ratio"]
                for row in legs
                if row["flow_to_tag_ratio"] is not None
            ],
        },
    }
    write_json(output / "tag_anchor_decision.json", decision)
    return decision
