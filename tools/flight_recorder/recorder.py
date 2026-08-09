#!/usr/bin/env python3
"""Isolated full-bag recorder for PX4, OV9281, and shared CM2 frames.

The recorder never owns the downward camera and never holds a local sensing
lease. It copies selected frames from shared memory under a generation seqlock.
If this process blocks or dies, mission_sensing continues unchanged.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image
from std_msgs.msg import Int64, String
from px4_msgs.msg import SensorCombined, SensorOpticalFlow, VehicleOdometry, VehicleStatus
from picamera2 import Picamera2

from image_formats import YUYV_ENCODING
import recording
from sensing.camera_tuning import load_ov9281_daylight_tuning
from sensing.px4_topics import live_px4_topic
from sensing.shared_frame_pool import SharedFramePool


def make_cm2_image(metadata, payload, pool) -> Image:
    image = Image()
    image.header.stamp = recording.stamp_ns(metadata.realtime_ns)
    image.header.frame_id = "imx219_nadir"
    image.height = pool.height
    image.width = pool.width
    image.encoding = YUYV_ENCODING
    image.is_bigendian = 0
    image.step = pool.stride
    image.data = payload
    return image


def record_shared_cm2(
    pool: SharedFramePool,
    bag: recording.Bag,
    stop: threading.Event,
    *,
    record_fps: float,
) -> dict[str, int]:
    """Record the selected cadence. A torn frame is dropped, never retried stale."""
    stats = {"seen": 0, "written": 0, "sequence_drops": 0, "torn": 0}
    last_seen = 0
    next_record_ns = None
    period_ns = int(1e9 / record_fps) if record_fps > 0 else 0
    camera_info = recording.make_camera_info(pool.width, pool.height)
    clock_msg = Int64()
    last_clock_ns = 0
    while not stop.is_set():
        pool.set_recorder_heartbeat()
        stop.wait(0.01)
        metadata = pool.latest_after(last_seen)
        if metadata is None:
            continue
        stats["seen"] += 1
        if last_seen and metadata.sequence > last_seen + 1:
            stats["sequence_drops"] += metadata.sequence - last_seen - 1
        last_seen = metadata.sequence
        should_record = (
            period_ns == 0
            or next_record_ns is None
            or metadata.realtime_ns >= next_record_ns
        )
        if not should_record:
            continue
        payload = pool.copy(metadata)
        if payload is None:
            stats["torn"] += 1
            continue
        image = make_cm2_image(metadata, payload, pool)
        bag.write("/camera_down/image_raw", serialize_message(image), metadata.realtime_ns)
        camera_info.header = image.header
        bag.write(
            "/camera_down/camera_info",
            serialize_message(camera_info),
            metadata.realtime_ns,
        )
        stats["written"] += 1
        if metadata.realtime_ns - last_clock_ns >= 1_000_000_000:
            # This is an audit of the sensing process's already-mapped stamp,
            # not a second clock mapping in the recorder.
            clock_msg.data = metadata.realtime_ns - metadata.sensor_boottime_ns
            bag.write(
                "/camera_down/realtime_minus_boottime_ns",
                serialize_message(clock_msg),
                metadata.realtime_ns,
            )
            last_clock_ns = metadata.realtime_ns
        if period_ns:
            if next_record_ns is None:
                next_record_ns = metadata.realtime_ns + period_ns
            else:
                periods = max(
                    1,
                    (metadata.realtime_ns - next_record_ns) // period_ns + 1,
                )
                next_record_ns += periods * period_ns
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-name", default="cm2")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--ov-record-fps", type=float, default=1.0)
    ap.add_argument("--ov-max-exposure-us", type=int, default=1000)
    ap.add_argument("--down-record-fps", type=float, default=10.0)
    ap.add_argument("--split-mb", type=int, default=0)
    ap.add_argument("--storage-config", default="")
    ap.add_argument("--stop-on-disarm", action="store_true")
    ap.add_argument("--sync-timeout", type=float, default=12.0)
    ap.add_argument("--max-sync-rtt-ms", type=float, default=50.0)
    ap.add_argument("--max-clock-step-ms", type=float, default=5.0)
    ap.add_argument("--px4-namespace", default="")
    ap.add_argument("--detect-topic", default="/detections/down")
    args = ap.parse_args()
    if args.fps <= 0 or args.ov_max_exposure_us <= 0 or args.down_record_fps < 0:
        ap.error("camera rates and exposure ceiling are invalid")

    stop = threading.Event()
    errors: list[str] = []
    threads: list[threading.Thread] = []
    pool = None
    camera = None
    node = None
    spin = None
    cm2_stats: dict[str, int] = {}
    ov_stats = recording.CameraStats(args.fps)
    ov_clock = recording.ClockMapper(int(args.max_clock_step_ms * 1e6))
    sync = recording.SyncMonitor(int(args.max_sync_rtt_ms * 1000))
    bag = recording.Bag(
        args.out,
        split_bytes=args.split_mb * 1024 * 1024,
        storage_config=args.storage_config,
    )

    def request_stop(*_args) -> None:
        if not stop.is_set():
            print("recorder: stop requested; sensing is independent", file=sys.stderr, flush=True)
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        for topic, type_name in (
            ("/camera/image_raw", "sensor_msgs/msg/Image"),
            ("/camera/camera_info", "sensor_msgs/msg/CameraInfo"),
            ("/camera_down/image_raw", "sensor_msgs/msg/Image"),
            ("/camera_down/camera_info", "sensor_msgs/msg/CameraInfo"),
            ("/capture/realtime_minus_boottime_ns", "std_msgs/msg/Int64"),
            ("/camera_down/realtime_minus_boottime_ns", "std_msgs/msg/Int64"),
            ("/imu", "sensor_msgs/msg/Imu"),
            ("/fmu/in/vehicle_visual_odometry", "px4_msgs/msg/VehicleOdometry"),
            ("/localization/tag_ev/status", "std_msgs/msg/String"),
        ):
            bag.topic(topic, type_name)
        for name, type_name, _cls in recording.PX4_SUBS:
            bag.topic(name, type_name)
        bag.topic("/fmu/in/sensor_optical_flow", "px4_msgs/msg/SensorOpticalFlow")
        bag.topic("/localization/cm2_flow/debug", "std_msgs/msg/String")
        bag.topic(args.detect_topic, "vision_msgs/msg/Detection2DArray")
        bag.topic("/detections/down/debug", "std_msgs/msg/String")

        rclpy.init()
        node = Node("full_bag_recorder")
        qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        for name, _type_name, cls in recording.PX4_SUBS:
            node.create_subscription(
                cls,
                live_px4_topic(name, args.px4_namespace),
                (lambda topic: lambda msg: recording.record_px4(bag, sync, topic, msg))(name),
                qos,
            )
        node.create_subscription(
            SensorCombined,
            live_px4_topic("/fmu/out/sensor_combined", args.px4_namespace),
            lambda msg: recording.record_imu(bag, sync, msg),
            qos,
        )
        node.create_subscription(
            VehicleOdometry,
            live_px4_topic("/fmu/in/vehicle_visual_odometry", args.px4_namespace),
            lambda msg: bag.write(
                "/fmu/in/vehicle_visual_odometry",
                serialize_message(msg),
                recording.message_time_ns(msg),
            ),
            qos,
        )
        node.create_subscription(
            String,
            "/localization/tag_ev/status",
            lambda msg: bag.write(
                "/localization/tag_ev/status", serialize_message(msg), recording.realtime_ns()
            ),
            qos,
        )
        if args.stop_on_disarm:
            node.create_subscription(
                VehicleStatus,
                live_px4_topic("/fmu/out/vehicle_status_v4", args.px4_namespace),
                recording.StopOnDisarm(stop),
                qos,
            )
        node.create_subscription(
            SensorOpticalFlow,
            live_px4_topic("/fmu/in/sensor_optical_flow", args.px4_namespace),
            lambda msg: bag.write(
                "/fmu/in/sensor_optical_flow",
                serialize_message(msg),
                recording.message_time_ns(msg),
            ),
            qos,
        )
        node.create_subscription(
            String,
            "/localization/cm2_flow/debug",
            lambda msg: bag.write(
                "/localization/cm2_flow/debug", serialize_message(msg), recording.realtime_ns()
            ),
            qos,
        )

        from vision_msgs.msg import Detection2DArray

        node.create_subscription(
            Detection2DArray,
            args.detect_topic,
            lambda msg: bag.write(
                args.detect_topic,
                serialize_message(msg),
                int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec),
            ),
            10,
        )
        node.create_subscription(
            String,
            "/detections/down/debug",
            lambda msg: bag.write(
                "/detections/down/debug", serialize_message(msg), recording.realtime_ns()
            ),
            10,
        )
        spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin.start()

        ok, detail = sync.wait_ready(args.sync_timeout)
        if not ok:
            raise RuntimeError(f"refusing recording: {detail}")
        print(f"recorder: {detail}", file=sys.stderr, flush=True)

        tuning = load_ov9281_daylight_tuning(Picamera2, args.ov_max_exposure_us)
        camera = Picamera2(args.cam, tuning=tuning)
        model = camera.camera_properties.get("Model")
        rotation = camera.camera_properties.get("Rotation")
        if model != "ov9281" or rotation != 180:
            raise RuntimeError(
                f"expected rotated OV9281 at camera {args.cam}, "
                f"got model={model!r} rotation={rotation!r}"
            )
        duration_us = int(1_000_000 / args.fps)
        camera.configure(
            camera.create_video_configuration(
                main={"size": (args.width, args.height), "format": "YUV420"},
                controls={
                    "FrameDurationLimits": (duration_us, duration_us),
                    "AeExposureMode": 1,
                    "ExposureTimeMode": 0,
                    "AnalogueGainMode": 0,
                },
                buffer_count=12,
            )
        )
        pool = SharedFramePool.attach(args.pool_name)
        camera.start()
        pool.set_recorder_heartbeat()

        def worker(name, function) -> None:
            try:
                function()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                print(f"recorder: {name} failed: {exc}", file=sys.stderr, flush=True)
                stop.set()

        threads.append(
            threading.Thread(
                target=worker,
                args=(
                    "OV9281",
                    lambda: recording.record_ov9281(
                        camera,
                        bag,
                        ov_clock,
                        ov_stats,
                        stop,
                        width=args.width,
                        height=args.height,
                        max_exposure_us=args.ov_max_exposure_us,
                        record_fps=args.ov_record_fps,
                    ),
                ),
                name="record-ov9281",
            )
        )

        def cm2_worker() -> None:
            nonlocal cm2_stats
            cm2_stats = record_shared_cm2(
                pool, bag, stop, record_fps=args.down_record_fps
            )

        threads.append(
            threading.Thread(
                target=worker,
                args=("CM2 shared reader", cm2_worker),
                name="record-cm2-shm",
            )
        )
        for thread in threads:
            thread.start()
        print(
            f"recorder: isolated bag -> {args.out}; CM2 shared YUYV @"
            f"{args.down_record_fps:g} Hz; OV9281 @{args.ov_record_fps:g} Hz",
            file=sys.stderr,
            flush=True,
        )
        while not stop.is_set() and any(thread.is_alive() for thread in threads):
            time.sleep(0.05)
    except Exception as exc:
        errors.append(str(exc))
        print(f"recorder: FATAL (sensing remains independent): {exc}", file=sys.stderr, flush=True)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2.0)
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
            camera.close()
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()
        if spin is not None:
            spin.join(timeout=1.0)
        if pool is not None:
            pool.close()
        bag.close()
        print("=== recorder summary ===", file=sys.stderr)
        print(
            f"  CM2 seen={cm2_stats.get('seen', 0)} written={cm2_stats.get('written', 0)} "
            f"sequence_drops={cm2_stats.get('sequence_drops', 0)} "
            f"torn_rejections={cm2_stats.get('torn', 0)}",
            file=sys.stderr,
        )
        print(
            f"  OV9281 frames={ov_stats.frames} rate={ov_stats.hz():.2f} Hz "
            f"large_gaps={ov_stats.large_gaps}",
            file=sys.stderr,
        )
        for error in errors:
            print(f"  ERROR: {error}", file=sys.stderr)
    return 4 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
