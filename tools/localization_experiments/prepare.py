"""Prepare the synchronized July 24 replay for SVO without touching source data."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

from common import (
    image_to_gray,
    iter_messages,
    sha256,
    split_mcaps,
    stamp_ns,
    write_json,
    write_pgm,
)


TOPICS = [
    "/camera/image_raw",
    "/imu",
    "/fmu/out/sensor_combined",
]


def prepare(
    bag: Path,
    output: Path,
    calibration: Path,
    *,
    max_frames: int | None = None,
    hash_sources: bool = True,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    euroc = output / "euroc"
    camera_dir = euroc / "cam0" / "data"
    imu_dir = euroc / "imu0"
    dataset = output / "svo_dataset"
    dataset_images = dataset / "data" / "img"
    for path in (camera_dir, imu_dir, dataset_images):
        path.mkdir(parents=True, exist_ok=True)

    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if "imu_effective_rate_hz" not in manifest:
            with (euroc / "cam0" / "data.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                camera_times = [
                    int(row[0])
                    for row in csv.reader(stream)
                    if row and not row[0].startswith("#")
                ]
            with (imu_dir / "data.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                imu_times = [
                    int(row[0])
                    for row in csv.reader(stream)
                    if row and not row[0].startswith("#")
                ]
            manifest.update(
                {
                    "camera_effective_rate_hz": (
                        (len(camera_times) - 1)
                        * 1e9
                        / (camera_times[-1] - camera_times[0])
                    ),
                    "imu_effective_rate_hz": (
                        (len(imu_times) - 1)
                        * 1e9
                        / (imu_times[-1] - imu_times[0])
                    ),
                }
            )
            write_json(output / "manifest.json", manifest)
        return manifest

    imu_stamp: list[int] = []
    imu_gyro: list[tuple[float, float, float]] = []
    imu_accel: list[tuple[float, float, float]] = []
    sensor_combined_hrt: list[int] = []
    camera_rows: list[tuple[int, str]] = []

    for topic, _, message in iter_messages(bag, TOPICS):
        if topic == "/imu":
            imu_stamp.append(stamp_ns(message.header))
            gyro = message.angular_velocity
            accel = message.linear_acceleration
            imu_gyro.append((gyro.x, gyro.y, gyro.z))
            imu_accel.append((accel.x, accel.y, accel.z))
        elif topic == "/fmu/out/sensor_combined":
            sensor_combined_hrt.append(int(message.timestamp) * 1000)
        elif topic == "/camera/image_raw":
            if max_frames is not None and len(camera_rows) >= max_frames:
                continue
            timestamp = stamp_ns(message.header)
            filename = f"{timestamp}.pgm"
            destination = camera_dir / filename
            if not destination.exists():
                write_pgm(destination, image_to_gray(message))
            camera_rows.append((timestamp, filename))

    if not camera_rows or not imu_stamp:
        raise RuntimeError(
            f"incomplete replay: camera={len(camera_rows)}, imu={len(imu_stamp)}"
        )
    camera_rows.sort()
    imu_order = np.argsort(np.asarray(imu_stamp))
    imu_stamp = [imu_stamp[index] for index in imu_order]
    imu_gyro = [imu_gyro[index] for index in imu_order]
    imu_accel = [imu_accel[index] for index in imu_order]

    # Firmware DDS-adjusts PX4 timestamps into ROS realtime before delivery.
    # The recorder then copies SensorCombined.timestamp into /imu.header.stamp.
    # Five raw samples precede/follow the derived stream in this bag, so compare
    # the intersection instead of requiring equal topic counts.
    sensor_set = set(sensor_combined_hrt)
    common = [timestamp for timestamp in imu_stamp if timestamp in sensor_set]
    clock = {
        "px4_timestamp_domain": "DDS-adjusted ROS realtime",
        "sensor_combined_samples": len(sensor_combined_hrt),
        "derived_imu_samples": len(imu_stamp),
        "exact_timestamp_matches": len(common),
        "unmatched_sensor_combined": len(sensor_combined_hrt) - len(common),
        "unmatched_derived_imu": len(imu_stamp) - len(common),
    }

    with (euroc / "cam0" / "data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["#timestamp [ns]", "filename"])
        writer.writerows(camera_rows)

    with (imu_dir / "data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["#timestamp [ns]", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"]
        )
        for timestamp, gyro, accel in zip(
            imu_stamp, imu_gyro, imu_accel, strict=True
        ):
            writer.writerow([timestamp, *gyro, *accel])

    with (dataset / "data" / "images.txt").open(
        "w", encoding="utf-8"
    ) as stream:
        stream.write("# id time_sec image\n")
        for image_id, (timestamp, filename) in enumerate(camera_rows):
            source = camera_dir / filename
            destination = dataset_images / filename
            if not destination.exists():
                os.link(source, destination)
            stream.write(
                f"{image_id} {timestamp / 1e9:.9f} img/{filename}\n"
            )

    first_imu = camera_rows[0][0] - 100_000_000
    last_imu = camera_rows[-1][0] + 100_000_000
    selected_imu = [
        row
        for row in zip(imu_stamp, imu_gyro, imu_accel, strict=True)
        if first_imu <= row[0] <= last_imu
    ]
    with (dataset / "data" / "imu.txt").open(
        "w", encoding="utf-8"
    ) as stream:
        stream.write("# id time_sec wx wy wz ax ay az\n")
        for imu_id, (timestamp, gyro, accel) in enumerate(selected_imu):
            stream.write(
                f"{imu_id} {timestamp / 1e9:.9f} "
                f"{gyro[0]:.9g} {gyro[1]:.9g} {gyro[2]:.9g} "
                f"{accel[0]:.9g} {accel[1]:.9g} {accel[2]:.9g}\n"
            )
    shutil.copy2(calibration, dataset / "calib.yaml")

    source_files = split_mcaps(bag)
    sources = []
    for path in source_files:
        stat = path.stat()
        entry = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if hash_sources:
            entry["sha256"] = sha256(path)
        sources.append(entry)

    camera_dt = np.diff([row[0] for row in camera_rows]) / 1e9
    imu_dt = np.diff(imu_stamp) / 1e9
    manifest = {
        "bag": str(bag.resolve()),
        "sources": sources,
        "camera_frames": len(camera_rows),
        "imu_samples": len(imu_stamp),
        "selected_imu_samples": len(selected_imu),
        "camera_start_s": camera_rows[0][0] / 1e9,
        "camera_end_s": camera_rows[-1][0] / 1e9,
        "camera_rate_hz": float(1.0 / np.median(camera_dt)),
        "imu_rate_hz": float(1.0 / np.median(imu_dt)),
        "camera_effective_rate_hz": float(
            (len(camera_rows) - 1)
            / ((camera_rows[-1][0] - camera_rows[0][0]) / 1e9)
        ),
        "imu_effective_rate_hz": float(
            (len(imu_stamp) - 1)
            / ((imu_stamp[-1] - imu_stamp[0]) / 1e9)
        ),
        "strict_camera_timestamps": bool(np.all(camera_dt > 0)),
        "strict_imu_timestamps": bool(np.all(imu_dt > 0)),
        "imu_covers_camera": bool(
            imu_stamp[0] <= camera_rows[0][0]
            and imu_stamp[-1] >= camera_rows[-1][0]
        ),
        "clock_mapping": clock,
        "calibration": str(calibration.resolve()),
        "svo_dataset": str(dataset.resolve()),
    }
    write_json(output / "manifest.json", manifest)
    return manifest
