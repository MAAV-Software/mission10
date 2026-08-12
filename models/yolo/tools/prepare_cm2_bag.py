"""Validate a CM2 MCAP bag and export synchronized analysis artifacts."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

CAMERA_TOPIC = "/camera_down/image_raw"
TOPICS = {
    CAMERA_TOPIC,
    "/fmu/out/distance_sensor",
    "/fmu/out/estimator_aid_src_optical_flow",
    "/fmu/out/estimator_status_flags",
    "/fmu/out/vehicle_attitude",
    "/fmu/out/vehicle_command_ack_v1",
    "/fmu/out/vehicle_local_position_v1",
    "/fmu/out/vehicle_status_v4",
}
SYNC_TOPICS = TOPICS - {CAMERA_TOPIC, "/fmu/out/vehicle_command_ack_v1"}


def split_paths(bag: Path) -> list[Path]:
    def key(path: Path) -> tuple[str, int]:
        prefix, suffix = path.stem.rsplit("_", 1)
        return prefix, int(suffix)

    return sorted(bag.glob("*.mcap"), key=key)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_ns(message: Any) -> int:
    return int(message.timestamp) * 1000


def camera_stamp_ns(message: Any) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class Series:
    timestamps: list[int]
    rows: list[dict[str, Any]]

    def __init__(self) -> None:
        self.timestamps = []
        self.rows = []

    def add(self, stamp: int, row: dict[str, Any]) -> None:
        if self.timestamps and stamp < self.timestamps[-1]:
            raise RuntimeError("telemetry timestamps are not monotonic")
        self.timestamps.append(stamp)
        self.rows.append(row)

    def nearest(self, stamp: int) -> tuple[dict[str, Any] | None, float | None]:
        if not self.timestamps:
            return None, None
        index = bisect.bisect_left(self.timestamps, stamp)
        candidates = []
        if index < len(self.timestamps):
            candidates.append(index)
        if index:
            candidates.append(index - 1)
        best = min(candidates, key=lambda item: abs(self.timestamps[item] - stamp))
        age_ms = (self.timestamps[best] - stamp) / 1_000_000
        return self.rows[best], age_ms


def fields(topic: str, message: Any) -> dict[str, Any]:
    if topic == "/fmu/out/distance_sensor":
        names = ("current_distance", "min_distance", "max_distance", "variance")
    elif topic == "/fmu/out/vehicle_local_position_v1":
        names = (
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "heading",
            "dist_bottom",
            "xy_valid",
            "v_xy_valid",
            "z_valid",
            "v_z_valid",
            "xy_reset_counter",
            "vxy_reset_counter",
            "z_reset_counter",
            "vz_reset_counter",
            "heading_reset_counter",
        )
    elif topic == "/fmu/out/vehicle_attitude":
        return {f"q{index}": finite(value) for index, value in enumerate(message.q)}
    elif topic == "/fmu/out/vehicle_status_v4":
        names = (
            "arming_state",
            "nav_state",
            "nav_state_user_intention",
            "failsafe",
            "pre_flight_checks_pass",
        )
    elif topic == "/fmu/out/estimator_status_flags":
        names = (
            "cs_in_air",
            "cs_opt_flow",
            "cs_gnss_pos",
            "cs_gnss_vel",
            "cs_rng_hgt",
            "cs_inertial_dead_reckoning",
            "cs_vehicle_at_rest",
            "cs_rng_fault",
            "fs_bad_optflow_x",
            "fs_bad_optflow_y",
        )
    elif topic == "/fmu/out/estimator_aid_src_optical_flow":
        row = {
            "fused": bool(message.fused),
            "innovation_rejected": bool(message.innovation_rejected),
        }
        row.update(
            {f"test_ratio_{index}": finite(value) for index, value in enumerate(message.test_ratio)}
        )
        return row
    else:
        raise ValueError(f"unsupported telemetry topic: {topic}")
    return {
        name: finite(getattr(message, name))
        for name in names
        if hasattr(message, name)
    }


def encoder_command(args: argparse.Namespace, width: int, height: int, part: Path) -> list[str]:
    base = [
        args.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "yuyv422",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{args.fps:g}",
        "-i",
        "pipe:0",
        "-an",
    ]
    if args.codec == "libsvtav1":
        codec = [
            "-c:v",
            "libsvtav1",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-g",
            str(round(args.fps * 10)),
            "-pix_fmt",
            "yuv420p10le",
        ]
    else:
        codec = [
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
        ]
    return [*base, *codec, str(part)]


def write_frame_csv(path: Path, frames: list[dict[str, Any]], series: dict[str, Series]) -> None:
    prefixes = {
        "/fmu/out/distance_sensor": "range",
        "/fmu/out/vehicle_local_position_v1": "local",
        "/fmu/out/vehicle_attitude": "attitude",
        "/fmu/out/vehicle_status_v4": "status",
        "/fmu/out/estimator_status_flags": "flags",
        "/fmu/out/estimator_aid_src_optical_flow": "flow_aid",
    }
    joined = []
    for frame in frames:
        row = dict(frame)
        for topic, prefix in prefixes.items():
            sample, age_ms = series[topic].nearest(frame["camera_timestamp_ns"])
            row[f"{prefix}_age_ms"] = finite(age_ms)
            if sample:
                row.update({f"{prefix}_{key}": value for key, value in sample.items()})
        joined.append(row)
    columns = sorted({key for row in joined for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(joined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--codec", choices=("libsvtav1", "libx264"), default="libsvtav1")
    parser.add_argument("--crf", type=int, default=12)
    parser.add_argument("--preset", default="6")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output}")
    paths = split_paths(args.bag)
    if not paths:
        raise RuntimeError(f"no MCAP files found: {args.bag}")
    args.output.mkdir(parents=True, exist_ok=True)

    series = {topic: Series() for topic in SYNC_TOPICS}
    frames: list[dict[str, Any]] = []
    command_acks: list[dict[str, Any]] = []
    topic_counts = {topic: 0 for topic in TOPICS}
    first_camera_stamp = None
    previous_camera_stamp = None
    encoder = None
    video = args.output / "cm2_native_crf12.mkv"
    part = args.output / "cm2_native_crf12.part.mkv"

    try:
        for source in paths:
            with source.open("rb") as stream:
                reader = make_reader(stream, decoder_factories=[DecoderFactory()])
                for _, channel, record, message in reader.iter_decoded_messages(topics=TOPICS):
                    topic = channel.topic
                    topic_counts[topic] += 1
                    if topic == CAMERA_TOPIC:
                        stamp = camera_stamp_ns(message)
                        if previous_camera_stamp is not None and stamp <= previous_camera_stamp:
                            raise RuntimeError("camera timestamps are not strictly increasing")
                        if first_camera_stamp is None:
                            first_camera_stamp = stamp
                        actual = (
                            int(message.width),
                            int(message.height),
                            str(message.encoding).lower(),
                            int(message.step),
                            len(message.data),
                        )
                        expected = (
                            int(message.width),
                            int(message.height),
                            "yuyv",
                            int(message.width) * 2,
                            int(message.width) * int(message.height) * 2,
                        )
                        if actual != expected:
                            raise RuntimeError(f"unexpected CM2 frame format: {actual}")
                        if encoder is None and not args.skip_video:
                            command = encoder_command(
                                args, int(message.width), int(message.height), part
                            )
                            encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
                            if encoder.stdin is None:
                                raise RuntimeError("ffmpeg stdin was not created")
                        if encoder is not None:
                            assert encoder.stdin is not None
                            encoder.stdin.write(message.data)
                        frames.append(
                            {
                                "frame": len(frames),
                                "output_time_s": len(frames) / args.fps,
                                "camera_timestamp_ns": stamp,
                                "camera_time_s": (stamp - first_camera_stamp) * 1e-9,
                                "log_time_ns": int(record.log_time),
                                "width": int(message.width),
                                "height": int(message.height),
                                "encoding": str(message.encoding).lower(),
                            }
                        )
                        previous_camera_stamp = stamp
                    elif topic == "/fmu/out/vehicle_command_ack_v1":
                        command_acks.append(
                            {
                                "timestamp_ns": timestamp_ns(message),
                                "command": int(message.command),
                                "result": int(message.result),
                                "target_system": int(message.target_system),
                                "target_component": int(message.target_component),
                            }
                        )
                    else:
                        series[topic].add(timestamp_ns(message), fields(topic, message))
    finally:
        if encoder is not None and encoder.stdin is not None:
            encoder.stdin.close()

    if encoder is not None:
        status = encoder.wait()
        if status:
            raise RuntimeError(f"ffmpeg failed with status {status}: {part}")
        subprocess.run(
            [
                args.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-err_detect",
                "explode",
                "-i",
                str(part),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        part.replace(video)

    if not frames:
        raise RuntimeError("bag has no CM2 frames")
    write_frame_csv(args.output / "frames.csv", frames, series)
    (args.output / "command_acks.json").write_text(
        json.dumps(command_acks, indent=2) + "\n"
    )
    gaps = [
        (second["camera_timestamp_ns"] - first["camera_timestamp_ns"]) * 1e-9
        for first, second in pairwise(frames)
    ]
    manifest = {
        "schema": "mission10-cm2-bag-preparation/1",
        "bag": str(args.bag.resolve()),
        "sources": [
            {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
        ],
        "topic_counts": topic_counts,
        "frames": len(frames),
        "format": {
            "width": frames[0]["width"],
            "height": frames[0]["height"],
            "encoding": frames[0]["encoding"],
        },
        "camera": {
            "first_timestamp_ns": frames[0]["camera_timestamp_ns"],
            "last_timestamp_ns": frames[-1]["camera_timestamp_ns"],
            "duration_s": frames[-1]["camera_time_s"],
            "median_period_s": statistics.median(gaps) if gaps else None,
            "maximum_period_s": max(gaps) if gaps else None,
        },
        "video": None
        if args.skip_video
        else {
            "path": str(video.resolve()),
            "codec": args.codec,
            "crf": args.crf,
            "preset": args.preset,
            "fps": args.fps,
            "sha256": sha256(video),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
