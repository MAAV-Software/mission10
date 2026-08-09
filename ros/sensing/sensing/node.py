#!/usr/bin/env python3
"""Mission-critical CM2 owner and bounded local sensing fanout.

This process never opens a bag.  It owns the camera, maps the hardware capture
timestamp once, copies each frame into a bounded shared-memory pool, and lends
immutable views to flow/tag consumers.  The optional recorder is an external
reader and cannot exert backpressure on this process.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
from picamera2 import MappedArray, Picamera2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from px4_msgs.msg import (
    DistanceSensor,
    SensorCombined,
    SensorOpticalFlow,
    TimesyncStatus,
    VehicleStatus,
)

from .camera_tuning import load_imx219_daylight_tuning
from .cm2_flow import ImuHistory, RangeHistory
from .frame_sinks import DetectorSink, FlowSink
from .px4_topics import live_px4_topic
from .shared_frame_pool import DEFAULT_SLOTS, SharedFramePool
from .timebase import ClockMapper, SyncMonitor, boottime_ns


class CameraStats:
    def __init__(self, fps: float) -> None:
        self.nominal_period_ns = int(1e9 / fps)
        self.frames = 0
        self.first_ns = 0
        self.last_ns = 0
        self.max_gap_ns = 0
        self.large_gaps = 0
        self.max_exposure_us = 0
        self.max_gain = 0.0

    def note(self, sensor_ns: int, exposure_us: int, gain: float) -> None:
        if not self.first_ns:
            self.first_ns = sensor_ns
        if self.last_ns:
            gap = sensor_ns - self.last_ns
            self.max_gap_ns = max(self.max_gap_ns, gap)
            self.large_gaps += int(gap > 1.5 * self.nominal_period_ns)
        self.last_ns = sensor_ns
        self.frames += 1
        self.max_exposure_us = max(self.max_exposure_us, exposure_us)
        self.max_gain = max(self.max_gain, gain)

    def hz(self) -> float:
        span = self.last_ns - self.first_ns
        return (self.frames - 1) * 1e9 / span if self.frames > 1 and span > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-name", default="cm2")
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--width", type=int, default=1640)
    ap.add_argument("--height", type=int, default=1232)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    ap.add_argument("--max-exposure-us", type=int, default=1000)
    ap.add_argument("--max-clock-step-ms", type=float, default=5.0)
    ap.add_argument("--sync-timeout", type=float, default=12.0)
    ap.add_argument("--max-sync-rtt-ms", type=float, default=50.0)
    ap.add_argument("--px4-namespace", default="")
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--flow-shadow", action="store_true")
    ap.add_argument("--flow-backend", choices=("klt", "svo"), default="klt")
    ap.add_argument(
        "--flow-calibration",
        default=str(Path(__file__).with_name("config") / "cm2_intrinsics_rs.yaml"),
    )
    ap.add_argument("--svo-build", default="/home/maav/rl_vo_cm2_flow/svo-lib/build/svo_env")
    ap.add_argument(
        "--svo-params",
        default=str(Path(__file__).with_name("config") / "svo_flow_params.yaml"),
    )
    ap.add_argument(
        "--svo-calibration",
        default=str(Path(__file__).with_name("config") / "svo_flow_cm2_820.yaml"),
    )
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--detect-topic", default="/detections/down")
    ap.add_argument("--stop-on-disarm", action="store_true")
    args = ap.parse_args()
    if args.fps <= 0 or args.max_exposure_us <= 0:
        ap.error("camera rate and exposure ceiling must be positive")

    stop = threading.Event()
    errors: list[str] = []
    stats = CameraStats(args.fps)
    pool = None
    camera = None
    flow_sink = None
    detector_sink = None
    node = None
    spin = None
    clock = ClockMapper(int(args.max_clock_step_ms * 1e6))

    def request_stop(*_args) -> None:
        if not stop.is_set():
            print("sensing: stop requested", file=sys.stderr, flush=True)
            stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        rclpy.init()
        node = Node("mission_sensing")
        qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        sync = SyncMonitor(int(args.max_sync_rtt_ms * 1000))
        imu_history = ImuHistory()
        range_history = RangeHistory()
        armed_seen = [False]

        def on_imu(msg) -> None:
            sample_ns = int(msg.timestamp) * 1000
            sync.note_imu(sample_ns)
            imu_history.note(sample_ns, msg.gyro_rad)

        def on_range(msg) -> None:
            distance = float(msg.current_distance)
            if msg.min_distance <= distance <= msg.max_distance:
                range_history.note(
                    int(msg.timestamp) * 1000, distance, int(msg.signal_quality)
                )

        def on_status(msg) -> None:
            if msg.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                armed_seen[0] = True
            elif armed_seen[0]:
                print(
                    "sensing: vehicle disarmed after flight -> stopping",
                    file=sys.stderr,
                    flush=True,
                )
                stop.set()

        node.create_subscription(
            TimesyncStatus,
            live_px4_topic("/fmu/out/timesync_status", args.px4_namespace),
            sync.note_timesync,
            qos,
        )
        node.create_subscription(
            SensorCombined,
            live_px4_topic("/fmu/out/sensor_combined", args.px4_namespace),
            on_imu,
            qos,
        )
        node.create_subscription(
            DistanceSensor,
            live_px4_topic("/fmu/out/distance_sensor", args.px4_namespace),
            on_range,
            qos,
        )
        if args.stop_on_disarm:
            node.create_subscription(
                VehicleStatus,
                live_px4_topic("/fmu/out/vehicle_status_v4", args.px4_namespace),
                on_status,
                qos,
            )
        spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin.start()

        tuning = load_imx219_daylight_tuning(Picamera2, args.max_exposure_us)
        try:
            camera = Picamera2(args.camera, tuning=tuning)
        except IndexError as exc:
            available = [
                info.get("Model", "unknown")
                for info in Picamera2.global_camera_info()
            ]
            raise RuntimeError(
                f"CM2 camera index {args.camera} is unavailable; "
                f"libcamera sees {available}"
            ) from exc
        model = camera.camera_properties.get("Model")
        rotation = camera.camera_properties.get("Rotation")
        if model != "imx219":
            raise RuntimeError(f"expected IMX219 at camera {args.camera}, got {model!r}")

        # Failure to allocate this pool is a sensing startup failure: local
        # fanout, not just recording, depends on it.
        pool = SharedFramePool.create(
            args.pool_name,
            args.width,
            args.height,
            slots=args.slots,
        )
        sync_ok, sync_detail = sync.wait_ready(args.sync_timeout, stop)
        if not sync_ok:
            raise RuntimeError(f"refusing CM2 capture: {sync_detail}")
        print(f"sensing: {sync_detail}", file=sys.stderr, flush=True)

        duration_us = int(1_000_000 / args.fps)
        camera.configure(
            camera.create_video_configuration(
                main={"size": (args.width, args.height), "format": "YUYV"},
                controls={
                    "FrameDurationLimits": (duration_us, duration_us),
                    "AeExposureMode": 1,
                    "ExposureTimeMode": 0,
                    "AnalogueGainMode": 0,
                },
                buffer_count=8,
            )
        )

        if not args.no_flow:
            if args.flow_backend == "svo":
                from .cm2_svo_flow import Cm2SvoFlowFrontend

                frontend = Cm2SvoFlowFrontend(
                    args.flow_calibration,
                    imu_history,
                    args.svo_build,
                    args.svo_params,
                    args.svo_calibration,
                )
            else:
                from .cm2_flow import Cm2FlowFrontend

                frontend = Cm2FlowFrontend(args.flow_calibration, imu_history)
            flow_pub = node.create_publisher(
                SensorOpticalFlow,
                live_px4_topic("/fmu/in/sensor_optical_flow", args.px4_namespace),
                10,
            )
            flow_debug_pub = node.create_publisher(
                String, "/localization/cm2_flow/debug", 10
            )
            flow_sink = FlowSink(
                frontend,
                range_history,
                flow_pub.publish,
                flow_debug_pub.publish,
                depth=1,
                publish_enabled=not args.flow_shadow,
            )

        if args.detect:
            from mission_engine.rim.tag_detector import TagDetector, detect_image_with_debug
            from vision_msgs.msg import Detection2DArray

            detector = TagDetector()
            detection_pub = node.create_publisher(Detection2DArray, args.detect_topic, 10)
            detection_debug_pub = node.create_publisher(
                String, "/detections/down/debug", 10
            )
            detector_sink = DetectorSink(
                detector=lambda image: detect_image_with_debug(detector, image),
                publish=detection_pub.publish,
                depth=1,
                debug_publish=detection_debug_pub.publish,
            )

        camera.start()
        print(
            f"sensing: IMX219 cam{args.camera} {args.width}x{args.height}@{args.fps:g} "
            f"rotation={rotation} pool={args.pool_name} slots={args.slots} "
            f"clock_offset={clock.initial_offset_ns} ns",
            file=sys.stderr,
            flush=True,
        )
        recorder_was_alive = False
        while not stop.is_set():
            request = camera.capture_request()
            try:
                metadata = request.get_metadata()
                sensor_ns = int(metadata.get("SensorTimestamp", boottime_ns()))
                exposure_us = int(metadata.get("ExposureTime", 0))
                gain = float(metadata.get("AnalogueGain", 0.0))
                if exposure_us > args.max_exposure_us:
                    raise RuntimeError(
                        f"exposure ceiling violated: {exposure_us} > {args.max_exposure_us} us"
                    )
                mapped_ns, _clock_offset_ns = clock.map(sensor_ns)
                writable = pool.begin_write()
                if writable is None:
                    continue
                with MappedArray(request, "main") as mapped:
                    source = mapped.array
                    if source.shape[0] != args.height or source.shape[1] < args.width:
                        writable.abort()
                        raise RuntimeError(f"unexpected CM2 buffer shape {source.shape}")
                    destination = np.ndarray(
                        (args.height, args.width, 2),
                        dtype=np.uint8,
                        buffer=writable.buffer,
                    )
                    np.copyto(destination, source[:, : args.width, :])
                published = writable.commit(
                    sensor_boottime_ns=sensor_ns,
                    realtime_ns=mapped_ns,
                    exposure_us=exposure_us,
                    analogue_gain=gain,
                )
                del destination, source, writable
                stats.note(sensor_ns, exposure_us, gain)
            finally:
                request.release()

            for sink in (flow_sink, detector_sink):
                if sink is None:
                    continue
                lease = pool.lease(published)
                if lease is not None:
                    sink.submit(lease.image(), lease.ts_ns, lease.release)

            recorder_alive = pool.recorder_alive()
            if recorder_alive != recorder_was_alive:
                print(
                    "sensing: recorder "
                    + ("attached" if recorder_alive else "absent; sensing continues"),
                    file=sys.stderr,
                    flush=True,
                )
                recorder_was_alive = recorder_alive
    except Exception as exc:
        errors.append(str(exc))
        print(f"sensing: FATAL: {exc}", file=sys.stderr, flush=True)
    finally:
        stop.set()
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        for sink in (flow_sink, detector_sink):
            if sink is not None:
                sink.close()
        if camera is not None:
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
            try:
                pool.close()
            except BufferError as exc:
                errors.append(f"pool close: {exc}")
        print("=== sensing summary ===", file=sys.stderr)
        print(
            f"  frames={stats.frames} rate={stats.hz():.2f} Hz "
            f"large_gaps={stats.large_gaps} max_gap={stats.max_gap_ns / 1e6:.2f} ms",
            file=sys.stderr,
        )
        if pool is not None:
            print(
                f"  pool_full_drops={pool.pool_full_drops}",
                file=sys.stderr,
            )
        for sink in (flow_sink, detector_sink):
            if sink is not None:
                print(f"  {sink.summary()}", file=sys.stderr)
        for error in errors:
            print(f"  ERROR: {error}", file=sys.stderr)
    return 4 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
