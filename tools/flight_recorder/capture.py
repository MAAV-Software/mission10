#!/usr/bin/env python3
"""Unified dual-camera flight capture -> single mcap, no DDS hop for images.

Sources:
  - OV9281 forward GS cam via Picamera2 (YUV420 luma -> mono8), written DIRECTLY
    to mcap (no ROS publish / DDS). libcamera SensorTimestamp is CLOCK_BOOTTIME;
    it is mapped to CLOCK_REALTIME so its header shares the ROS/PX4 DDS clock.
  - IMX219 Camera Module 2 via a second Picamera2 instance. The flight default
    processes 1640x1232 YUYV at 30 Hz into PX4 optical flow and compact
    diagnostics, with a 1 Hz preview and event-triggered raw clips. Continuous
    raw YUYV remains available for calibration captures.
  - IMU via the uXRCE-DDS bridge: /fmu/out/sensor_combined (~194 Hz over TELEM2
    serial, NO USB cable). Mapped to sensor_msgs/Imu; header.stamp =
    SensorCombined.timestamp (DDS-adjusted ROS time, us). Requires firmware >= 8551f635c5 with
    `rate_limit: 250` on sensor_combined (older fw caps this topic at 100 Hz).
  - PX4 pose/timesync/gps over the same uXRCE bridge (typed subscriptions),
    re-serialized into the same bag. timesync_status is the PX4<->CM5 clock
    bridge for offline align.
  - Raw dToF and EKF2 range-height diagnostics over uXRCE-DDS.

Bag timestamps and camera/IMU headers are all CLOCK_REALTIME ns. The original
PX4 messages plus timesync_status and /capture/realtime_minus_boottime_ns keep
the mapping auditable. CM2 flow uses the installed-camera July 24 intrinsics
and rolling-shutter calibration.
SIGINT (e.g. `timeout -s INT`) stops capture and safely finalizes the bag.
"""
import argparse
from collections import deque
import multiprocessing
from pathlib import Path
import signal
import struct
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import Image, CameraInfo, Imu
from std_msgs.msg import Bool, Int64, String
import rosbag2_py
from picamera2 import Picamera2

from camera_tuning import (
    load_imx219_daylight_tuning,
    load_ov9281_daylight_tuning,
)
from image_formats import YUYV_BYTES_PER_PIXEL, YUYV_ENCODING, pack_yuyv_frame
from px4_msgs.msg import (
    DistanceSensor, EstimatorAidSource1d, EstimatorAidSource2d,
    EstimatorAidSource3d, EstimatorGpsStatus, EstimatorStatus,
    EstimatorStatusFlags, FailsafeFlags, SensorCombined, SensorGps,
    TimesyncStatus, VehicleAttitude, VehicleCommandAck,
    VehicleGlobalPosition, VehicleLocalPosition, VehicleOdometry, VehicleStatus,
    SensorOpticalFlow,
)

PX4_SUBS = [
    ("/fmu/out/timesync_status", "px4_msgs/msg/TimesyncStatus", TimesyncStatus),
    ("/fmu/out/vehicle_local_position_v1", "px4_msgs/msg/VehicleLocalPosition", VehicleLocalPosition),
    ("/fmu/out/vehicle_attitude", "px4_msgs/msg/VehicleAttitude", VehicleAttitude),
    ("/fmu/out/vehicle_global_position", "px4_msgs/msg/VehicleGlobalPosition", VehicleGlobalPosition),
    ("/fmu/out/vehicle_gps_position", "px4_msgs/msg/SensorGps", SensorGps),
    # raw IMU alongside the converted /imu: keeps the clipping counters and
    # integration dt fields for initialization analysis.
    ("/fmu/out/sensor_combined", "px4_msgs/msg/SensorCombined", SensorCombined),
    # The raw range distinguishes the dToF measurement from EKF HAGL. The aid
    # source records acceptance, rejection, innovation, and fusion state.
    ("/fmu/out/distance_sensor", "px4_msgs/msg/DistanceSensor", DistanceSensor),
    ("/fmu/out/estimator_status_flags", "px4_msgs/msg/EstimatorStatusFlags", EstimatorStatusFlags),
    ("/fmu/out/estimator_status", "px4_msgs/msg/EstimatorStatus", EstimatorStatus),
    ("/fmu/out/estimator_gps_status", "px4_msgs/msg/EstimatorGpsStatus", EstimatorGpsStatus),
    ("/fmu/out/estimator_aid_src_gnss_pos", "px4_msgs/msg/EstimatorAidSource2d", EstimatorAidSource2d),
    ("/fmu/out/estimator_aid_src_gnss_vel", "px4_msgs/msg/EstimatorAidSource3d", EstimatorAidSource3d),
    ("/fmu/out/estimator_aid_src_rng_hgt", "px4_msgs/msg/EstimatorAidSource1d", EstimatorAidSource1d),
    ("/fmu/out/estimator_aid_src_optical_flow", "px4_msgs/msg/EstimatorAidSource2d", EstimatorAidSource2d),
    ("/fmu/out/estimator_aid_src_ev_pos", "px4_msgs/msg/EstimatorAidSource2d", EstimatorAidSource2d),
    # Preserve command acceptance and the complete arming/failsafe requirement
    # state so pre-arm failures can be diagnosed from the bag.
    ("/fmu/out/vehicle_command_ack_v1", "px4_msgs/msg/VehicleCommandAck", VehicleCommandAck),
    ("/fmu/out/failsafe_flags", "px4_msgs/msg/FailsafeFlags", FailsafeFlags),
    # vehicle_status_v4 is logged for offline arm/nav-state timing AND, when
    # --stop-on-disarm is set, drives the auto-stop (armed->disarmed = mission end).
    ("/fmu/out/vehicle_status_v4", "px4_msgs/msg/VehicleStatus", VehicleStatus),
]

_running = True
_seen_armed = False  # set once the vehicle has armed; gates the disarm auto-stop


def boottime_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def realtime_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_REALTIME)


def stamp_ns(ns: int) -> TimeMsg:
    return TimeMsg(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


class ClockMapper:
    """Map libcamera CLOCK_BOOTTIME stamps into the ROS CLOCK_REALTIME domain."""

    def __init__(self, max_step_ns: int):
        self.max_step_ns = max_step_ns
        self.initial_offset_ns = self._sample_offset()
        self.last_offset_ns = self.initial_offset_ns
        self.max_offset_delta_ns = 0
        self.last_camera_realtime_ns = 0

    @staticmethod
    def _sample_offset() -> int:
        # CLOCK_REALTIME is sampled between two BOOTTIME reads. Keep the sample
        # with the narrowest bracket to minimize scheduler/preemption error.
        best = None
        for _ in range(7):
            b0 = boottime_ns()
            rt = realtime_ns()
            b1 = boottime_ns()
            candidate = (b1 - b0, rt - ((b0 + b1) // 2))
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best[1]

    def camera_realtime_ns(self, sensor_boottime_ns: int):
        offset_ns = self._sample_offset()
        step_ns = offset_ns - self.last_offset_ns
        if abs(step_ns) > self.max_step_ns:
            raise RuntimeError(
                "CLOCK_REALTIME stepped relative to CLOCK_BOOTTIME by "
                f"{step_ns / 1e6:.3f} ms"
            )
        self.last_offset_ns = offset_ns
        self.max_offset_delta_ns = max(
            self.max_offset_delta_ns, abs(offset_ns - self.initial_offset_ns)
        )
        mapped_ns = sensor_boottime_ns + offset_ns
        if mapped_ns <= self.last_camera_realtime_ns:
            raise RuntimeError("non-monotonic mapped camera timestamp")
        self.last_camera_realtime_ns = mapped_ns
        return mapped_ns, offset_ns


class SyncMonitor:
    """Require a live, stable DDS time map and IMU timestamps in ROS time."""

    def __init__(self, max_rtt_us: int, max_offset_spread_us: int = 5000):
        self.max_rtt_us = max_rtt_us
        self.max_offset_spread_us = max_offset_spread_us
        self.cv = threading.Condition()
        self.sync_samples = deque(maxlen=5)
        self.last_sync_monotonic = 0.0
        self.last_imu_monotonic = 0.0
        self.last_imu_ns = 0
        self.source_protocol = 0

    def note_timesync(self, msg):
        with self.cv:
            self.source_protocol = int(msg.source_protocol)
            self.sync_samples.append((int(msg.estimated_offset), int(msg.round_trip_time)))
            self.last_sync_monotonic = time.monotonic()
            self.cv.notify_all()

    def note_imu(self, timestamp_ns: int):
        with self.cv:
            self.last_imu_ns = timestamp_ns
            self.last_imu_monotonic = time.monotonic()
            self.cv.notify_all()

    def _status(self):
        now_mono = time.monotonic()
        if len(self.sync_samples) < 3:
            return False, f"waiting for timesync samples ({len(self.sync_samples)}/3)"
        if self.source_protocol != TimesyncStatus.SOURCE_PROTOCOL_DDS:
            return False, f"timesync source is {self.source_protocol}, expected DDS"
        if now_mono - self.last_sync_monotonic > 2.0:
            return False, "timesync_status is stale"
        offsets = [x[0] for x in self.sync_samples]
        rtts = [x[1] for x in self.sync_samples]
        if any(x == 0 for x in offsets):
            return False, "timesync offset is zero"
        if max(rtts) > self.max_rtt_us:
            return False, f"timesync RTT {max(rtts) / 1000:.1f} ms exceeds limit"
        spread = max(offsets) - min(offsets)
        if spread > self.max_offset_spread_us:
            return False, f"timesync offset spread {spread / 1000:.3f} ms is unstable"
        if not self.last_imu_ns or now_mono - self.last_imu_monotonic > 1.0:
            return False, "sensor_combined is stale"
        imu_skew_ns = abs(realtime_ns() - self.last_imu_ns)
        if imu_skew_ns > 2_000_000_000:
            return False, (
                "IMU timestamp is not in ROS realtime "
                f"(difference {imu_skew_ns / 1e9:.3f} s)"
            )
        return True, (
            f"DDS time sync ready: RTT max={max(rtts) / 1000:.1f} ms, "
            f"offset spread={spread / 1000:.3f} ms, "
            f"IMU age={imu_skew_ns / 1e6:.1f} ms"
        )

    def wait_ready(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        with self.cv:
            while True:
                ready, status = self._status()
                if ready:
                    return True, status
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, status
                self.cv.wait(timeout=min(remaining, 0.25))


class Bag:
    def __init__(self, uri, split_bytes=0, storage_config=""):
        self.w = rosbag2_py.SequentialWriter()
        # max_cache_size>0 enables rosbag2's async writer (CacheConsumer thread):
        # write() buffers into RAM while a background thread drains to storage.
        # max_bagfile_size>0 optionally splits the bag into <uri>_N.mcap chunks;
        # zero writes one file and avoids split-finalization stalls.
        # storage_config (mcap plugin yaml) enables per-chunk zstd: ~1.78x on
        # mono8. See mcap_zstd.yaml.
        so = rosbag2_py.StorageOptions(uri=uri, storage_id="mcap",
                                       max_cache_size=512*1024*1024,
                                       max_bagfile_size=split_bytes)
        if storage_config:
            so.storage_config_uri = storage_config
        self.w.open(so, rosbag2_py.ConverterOptions("", ""))
        self.lock = threading.Lock()
        self._id = 0
        self.counts = {}

    def topic(self, name, type_str):
        self.w.create_topic(rosbag2_py.TopicMetadata(
            id=self._id, name=name, type=type_str, serialization_format="cdr"))
        self._id += 1
        self.counts[name] = 0

    def write(self, name, data, ts_ns):
        with self.lock:
            if self.w is None:  # closed during teardown -> drop late callbacks
                return
            self.w.write(name, data, ts_ns)
            self.counts[name] += 1

    def close(self):
        # Detach the writer under the lock so any in-flight ROS callback that
        # races teardown sees w is None and no-ops, then finalize (flush cache +
        # write metadata) outside the lock.
        with self.lock:
            w, self.w = self.w, None
        print(
            "recorder: finalizing MCAP cache and metadata; keep power connected",
            file=sys.stderr,
            flush=True,
        )
        del w
        print("recorder: MCAP finalized", file=sys.stderr, flush=True)


def message_record_time_ns(msg):
    """Prefer a valid DDS-adjusted sample time; otherwise use callback time."""
    msg_ns = int(getattr(msg, "timestamp", 0)) * 1000
    now_ns = realtime_ns()
    return msg_ns if msg_ns and abs(now_ns - msg_ns) < 60_000_000_000 else now_ns


def on_px4_message(bag, sync_monitor, name, msg):
    if name == "/fmu/out/timesync_status":
        sync_monitor.note_timesync(msg)
    bag.write(name, serialize_message(msg), message_record_time_ns(msg))


def on_sensor_combined(bag, sync_monitor, msg, imu_history=None):
    """SensorCombined (DDS) -> sensor_msgs/Imu in the ROS realtime domain."""
    sample_ns = int(msg.timestamp) * 1000
    sync_monitor.note_imu(sample_ns)
    if imu_history is not None:
        imu_history.note(sample_ns, msg.gyro_rad)
    m = Imu()
    m.header.stamp = stamp_ns(sample_ns)  # DDS already mapped PX4 hrt -> ROS time
    m.header.frame_id = "px4_imu_frd"
    m.angular_velocity.x = float(msg.gyro_rad[0])
    m.angular_velocity.y = float(msg.gyro_rad[1])
    m.angular_velocity.z = float(msg.gyro_rad[2])
    m.linear_acceleration.x = float(msg.accelerometer_m_s2[0])
    m.linear_acceleration.y = float(msg.accelerometer_m_s2[1])
    m.linear_acceleration.z = float(msg.accelerometer_m_s2[2])
    m.orientation_covariance[0] = -1.0  # no orientation provided
    bag.write("/imu", serialize_message(m), sample_ns)


def on_vehicle_status(msg):
    """Auto-stop the recording one flight after it begins: latch _seen_armed when
    the vehicle arms, then stop on the first disarm. The recorder is started on
    the pad (disarmed), so we must NOT stop until we've actually seen an arm --
    otherwise the very first sample (arming_state=DISARMED) would end it instantly.
    Disarm is the ground-truth mission-end signal (autonomous RTL+land disarms)."""
    global _running, _seen_armed
    if msg.arming_state == VehicleStatus.ARMING_STATE_ARMED:
        _seen_armed = True
    elif _seen_armed:
        print("recorder: vehicle disarmed after flight -> stopping", file=sys.stderr, flush=True)
        _running = False


def make_uncalibrated_camera_info(w, h):
    """ROS convention: K[0] == 0 declares an uncalibrated camera."""
    ci = CameraInfo()
    ci.width, ci.height = w, h
    return ci


class CameraStats:
    def __init__(self, nominal_fps):
        self.nominal_period_ns = int(1_000_000_000 / nominal_fps)
        self.frames = 0
        self.first_sensor_ns = 0
        self.last_sensor_ns = 0
        self.max_gap_ns = 0
        self.large_gaps = 0
        self.max_exposure_us = 0
        self.max_analogue_gain = 0.0

    def note(self, sensor_ns, exposure_us=0, analogue_gain=0.0):
        if not self.first_sensor_ns:
            self.first_sensor_ns = sensor_ns
        if self.last_sensor_ns:
            gap_ns = sensor_ns - self.last_sensor_ns
            self.max_gap_ns = max(self.max_gap_ns, gap_ns)
            if gap_ns > 1.5 * self.nominal_period_ns:
                self.large_gaps += 1
        self.last_sensor_ns = sensor_ns
        self.frames += 1
        self.max_exposure_us = max(self.max_exposure_us, int(exposure_us or 0))
        self.max_analogue_gain = max(
            self.max_analogue_gain, float(analogue_gain or 0.0)
        )

    def measured_hz(self):
        span_ns = self.last_sensor_ns - self.first_sensor_ns
        return ((self.frames - 1) * 1e9 / span_ns
                if self.frames > 1 and span_ns > 0 else 0.0)


def capture_camera(
    picam, bag, clock_mapper, stats, *, width, height, frame_id,
    image_topic, info_topic, camera_info, clock_topic, max_exposure_us,
    errors, errors_lock, record_fps=0.0,
):
    """Drain one Picamera2 stream until the session stops."""
    global _running
    last_clock_record_ns = 0
    clock_msg = Int64()
    last_record_ns = 0
    try:
        while _running:
            req = picam.capture_request()
            try:
                md = req.get_metadata()
                sensor_boottime_ns = int(md.get("SensorTimestamp", boottime_ns()))
                exposure_us = int(md.get("ExposureTime", 0))
                analogue_gain = float(md.get("AnalogueGain", 0.0))
                yuv = req.make_array("main")
            finally:
                req.release()

            if exposure_us > max_exposure_us:
                raise RuntimeError(
                    "exposure ceiling violated: "
                    f"{exposure_us} us > {max_exposure_us} us"
                )
            ts_ns, clock_offset_ns = clock_mapper.camera_realtime_ns(
                sensor_boottime_ns
            )
            stats.note(sensor_boottime_ns, exposure_us, analogue_gain)
            luma = np.ascontiguousarray(yuv[:height, :width])
            st = stamp_ns(ts_ns)

            img = Image()
            img.header.stamp = st
            img.header.frame_id = frame_id
            img.height, img.width = height, width
            img.encoding = "mono8"
            img.is_bigendian = 0
            img.step = width
            img.data = luma.tobytes()
            record_period_ns = int(1e9 / record_fps) if record_fps > 0 else 0
            if record_period_ns == 0 or ts_ns - last_record_ns >= record_period_ns:
                bag.write(image_topic, serialize_message(img), ts_ns)
                camera_info.header.stamp = st
                camera_info.header.frame_id = frame_id
                bag.write(info_topic, serialize_message(camera_info), ts_ns)
                last_record_ns = ts_ns

            if clock_topic and ts_ns - last_clock_record_ns >= 1_000_000_000:
                clock_msg.data = clock_offset_ns
                bag.write(clock_topic, serialize_message(clock_msg), ts_ns)
                last_clock_record_ns = ts_ns
    except Exception as exc:
        if _running:
            with errors_lock:
                errors.append(f"{frame_id}: {exc}")
            print(
                f"recorder: CAMERA ERROR ({frame_id}): {exc}",
                file=sys.stderr,
                flush=True,
            )
            _running = False


CM2_FRAME_HEADER = struct.Struct("<QIf")


def capture_imx219_worker(
    camera_index, width, height, fps, max_exposure_us,
    frame_connection, status_connection, configure_event, start_event,
    stop_event,
):
    """Own the CM2 in a process with its sensor-specific capped AGC tuning."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    picam = None
    ready = False
    try:
        tuning = load_imx219_daylight_tuning(Picamera2, max_exposure_us)
        picam = Picamera2(camera_index, tuning=tuning)
        model = picam.camera_properties.get("Model")
        rotation = picam.camera_properties.get("Rotation")
        if model != "imx219":
            raise RuntimeError(
                f"expected IMX219 at camera {camera_index}, got model={model!r}"
            )
        status_connection.send(("initialized", rotation))
        if not configure_event.wait(10) or stop_event.is_set():
            raise RuntimeError("IMX219 configuration was not requested")
        duration_us = int(1_000_000 / fps)
        picam.configure(picam.create_video_configuration(
            main={"size": (width, height), "format": "YUYV"},
            controls={
                "FrameDurationLimits": (duration_us, duration_us),
                "AeExposureMode": 1,
                "ExposureTimeMode": 0,
                "AnalogueGainMode": 0,
            },
            buffer_count=8,
        ))
        status_connection.send(("configured", rotation))
        if not start_event.wait(10) or stop_event.is_set():
            raise RuntimeError("IMX219 start was not requested")
        picam.start()
        status_connection.send(("ready", rotation))
        ready = True
        while not stop_event.is_set():
            req = picam.capture_request()
            try:
                md = req.get_metadata()
                sensor_ns = int(md.get("SensorTimestamp", boottime_ns()))
                exposure_us = int(md.get("ExposureTime", 0))
                analogue_gain = float(md.get("AnalogueGain", 0.0))
                yuyv = req.make_array("main")
            finally:
                req.release()
            packed_yuyv = pack_yuyv_frame(yuyv, width, height)
            frame_connection.send_bytes(
                CM2_FRAME_HEADER.pack(
                    sensor_ns, exposure_us, analogue_gain
                ) + packed_yuyv
            )
    except Exception as exc:
        try:
            status_connection.send(("error", str(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if picam is not None:
            try:
                picam.stop()
            except Exception:
                pass
            picam.close()
        frame_connection.close()
        status_connection.close()
        if not ready:
            stop_event.set()


def capture_camera_pipe(
    connection, status_connection, process, bag, clock_mapper, stats, *,
    width, height, frame_id, image_topic, info_topic, camera_info,
    max_exposure_us, errors, errors_lock, sink=None, record_raw=True,
    preview_topic="/camera_down/image_preview", preview_fps=1.0,
):
    """Record packed CM2 YUYV frames received from its tuning-isolated process."""
    global _running
    expected_size = (
        CM2_FRAME_HEADER.size + width * height * YUYV_BYTES_PER_PIXEL
    )
    last_preview_ns = 0
    try:
        while _running:
            if not connection.poll(0.25):
                if process.is_alive():
                    continue
                detail = "camera process exited"
                if status_connection.poll():
                    status, value = status_connection.recv()
                    if status == "error":
                        detail = value
                raise RuntimeError(detail)
            payload = connection.recv_bytes()
            if len(payload) != expected_size:
                raise RuntimeError(
                    f"expected {expected_size} frame bytes, got {len(payload)}"
                )
            sensor_ns, exposure_us, analogue_gain = CM2_FRAME_HEADER.unpack_from(
                payload
            )
            if exposure_us > max_exposure_us:
                raise RuntimeError(
                    "exposure ceiling violated: "
                    f"{exposure_us} us > {max_exposure_us} us"
                )
            ts_ns, _ = clock_mapper.camera_realtime_ns(sensor_ns)
            stats.note(sensor_ns, exposure_us, analogue_gain)
            st = stamp_ns(ts_ns)

            img = Image()
            img.header.stamp = st
            img.header.frame_id = frame_id
            img.height, img.width = height, width
            img.encoding = YUYV_ENCODING
            img.is_bigendian = 0
            img.step = width * YUYV_BYTES_PER_PIXEL
            img.data = payload[CM2_FRAME_HEADER.size:]
            if record_raw:
                bag.write(image_topic, serialize_message(img), ts_ns)
                camera_info.header.stamp = st
                camera_info.header.frame_id = frame_id
                bag.write(info_topic, serialize_message(camera_info), ts_ns)
            elif (
                preview_fps > 0
                and ts_ns - last_preview_ns >= int(1e9 / preview_fps)
            ):
                preview = Image()
                preview.header = img.header
                preview.height, preview.width = height, width
                preview.encoding = "mono8"
                preview.is_bigendian = 0
                preview.step = width
                yuyv = np.frombuffer(img.data, dtype=np.uint8).reshape(
                    height, width * YUYV_BYTES_PER_PIXEL
                )
                preview.data = np.ascontiguousarray(yuyv[:, 0::2]).tobytes()
                bag.write(
                    preview_topic, serialize_message(preview), ts_ns
                )
                camera_info.header.stamp = st
                camera_info.header.frame_id = frame_id
                bag.write(info_topic, serialize_message(camera_info), ts_ns)
                last_preview_ns = ts_ns

            # The recording is complete at this point. A sink is a tap on the
            # frame we already hold; it cannot block here and cannot fail here.
            if sink is not None:
                sink.submit(img, ts_ns)
    except Exception as exc:
        if _running:
            with errors_lock:
                errors.append(f"{frame_id}: {exc}")
            print(
                f"recorder: CAMERA ERROR ({frame_id}): {exc}",
                file=sys.stderr,
                flush=True,
            )
            _running = False


def main():
    global _running
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=800)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--ov-max-exposure-us", type=int, default=1000,
                    help="hard OV9281 daylight shutter ceiling (default: 1000 us)")
    ap.add_argument("--down-cam", type=int, default=1)
    ap.add_argument("--down-w", type=int, default=1640)
    ap.add_argument("--down-h", type=int, default=1232)
    ap.add_argument("--down-fps", type=float, default=10.0)
    ap.add_argument("--ov-record-fps", type=float, default=0.0,
                    help="record OV9281 at this rate while capturing at --fps (0=all)")
    ap.add_argument("--down-preview-fps", type=float, default=1.0)
    ap.add_argument("--no-down-raw", action="store_true",
                    help="record processed flow plus a low-rate CM2 preview, not continuous raw CM2")
    ap.add_argument("--flow", action="store_true",
                    help="publish CM2 angular flow to PX4 and record compact diagnostics")
    ap.add_argument(
        "--flow-calibration",
        default=str(Path(__file__).with_name("config") / "cm2_intrinsics_rs.yaml"),
    )
    ap.add_argument("--down-max-exposure-us", type=int, default=1000,
                    help="hard CM2 daylight shutter ceiling (default: 1000 us)")
    ap.add_argument("--no-down-camera", action="store_true",
                    help="record only the legacy forward OV9281 stream")
    ap.add_argument("--detect", action="store_true",
                    help="also run the AprilTag detector on nadir frames "
                         "and publish/record /detections/down")
    ap.add_argument("--detect-topic", default="/detections/down")
    ap.add_argument("--detect-queue", type=int, default=2,
                    help="nadir frames the detector may fall behind by")
    ap.add_argument("--split-mb", type=int, default=0,
                    help="split bag into N-MB mcap chunks (0=single file)")
    ap.add_argument("--storage-config", default="",
                    help="mcap storage plugin yaml (e.g. config/mcap_zstd.yaml)")
    ap.add_argument("--stop-on-disarm", action="store_true",
                    help="auto-stop after the vehicle arms then disarms (mission end)")
    ap.add_argument("--sync-timeout", type=float, default=12.0,
                    help="seconds to wait for stable DDS time sync before opening camera")
    ap.add_argument("--max-sync-rtt-ms", type=float, default=50.0,
                    help="reject capture when recent DDS timesync RTT exceeds this")
    ap.add_argument("--max-clock-step-ms", type=float, default=5.0,
                    help="abort if realtime steps this far relative to boottime")
    args = ap.parse_args()

    if args.fps <= 0 or args.down_fps <= 0:
        ap.error("camera frame rates must be positive")
    if args.ov_max_exposure_us <= 0:
        ap.error("--ov-max-exposure-us must be positive")
    if args.down_max_exposure_us <= 0:
        ap.error("--down-max-exposure-us must be positive")
    if not args.no_down_camera and args.cam == args.down_cam:
        ap.error("forward and downward camera indices must differ")
    if args.detect and args.no_down_camera:
        ap.error("--detect needs the downward camera")
    if args.flow and args.no_down_camera:
        ap.error("--flow needs the downward camera")
    if args.detect_queue < 1:
        ap.error("--detect-queue must be at least 1")

    ov_clock_mapper = ClockMapper(max_step_ns=int(args.max_clock_step_ms * 1e6))
    down_clock_mapper = ClockMapper(max_step_ns=int(args.max_clock_step_ms * 1e6))
    sync_monitor = SyncMonitor(max_rtt_us=int(args.max_sync_rtt_ms * 1000))

    bag = Bag(args.out, split_bytes=args.split_mb * 1024 * 1024,
              storage_config=args.storage_config)
    bag.topic("/camera/image_raw", "sensor_msgs/msg/Image")
    bag.topic("/camera/camera_info", "sensor_msgs/msg/CameraInfo")
    if not args.no_down_camera:
        if not args.no_down_raw:
            bag.topic("/camera_down/image_raw", "sensor_msgs/msg/Image")
        else:
            bag.topic("/camera_down/image_preview", "sensor_msgs/msg/Image")
        bag.topic("/camera_down/camera_info", "sensor_msgs/msg/CameraInfo")
    if args.flow:
        bag.topic("/fmu/in/sensor_optical_flow", "px4_msgs/msg/SensorOpticalFlow")
        bag.topic("/localization/cm2_flow/debug", "std_msgs/msg/String")
        bag.topic("/camera_down/image_fault", "sensor_msgs/msg/Image")
    if args.detect:
        bag.topic(args.detect_topic, "vision_msgs/msg/Detection2DArray")
        bag.topic("/detections/down/debug", "std_msgs/msg/String")
    bag.topic(
        "/fmu/in/vehicle_visual_odometry", "px4_msgs/msg/VehicleOdometry"
    )
    bag.topic("/localization/tag_ev/status", "std_msgs/msg/String")
    bag.topic("/imu", "sensor_msgs/msg/Imu")
    bag.topic("/capture/realtime_minus_boottime_ns", "std_msgs/msg/Int64")
    for name, type_str, _ in PX4_SUBS:
        bag.topic(name, type_str)

    # ROS subscriptions for PX4 pose/timesync/gps
    rclpy.init()
    node = Node("flight_recorder")
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    for name, _type_str, cls in PX4_SUBS:
        node.create_subscription(
            cls, name,
            (lambda n: (lambda msg: on_px4_message(bag, sync_monitor, n, msg)))(name),
            qos)
    node.create_subscription(
        VehicleOdometry,
        "/fmu/in/vehicle_visual_odometry",
        lambda msg: bag.write(
            "/fmu/in/vehicle_visual_odometry",
            serialize_message(msg),
            message_record_time_ns(msg),
        ),
        qos,
    )
    node.create_subscription(
        String,
        "/localization/tag_ev/status",
        lambda msg: bag.write(
            "/localization/tag_ev/status",
            serialize_message(msg),
            realtime_ns(),
        ),
        qos,
    )
    # IMU: SensorCombined over the same bridge, converted to sensor_msgs/Imu on /imu.
    imu_history = None
    range_history = None
    if args.flow:
        from cm2_flow import ImuHistory, RangeHistory

        imu_history = ImuHistory()
        range_history = RangeHistory()
    node.create_subscription(
        SensorCombined, "/fmu/out/sensor_combined",
        lambda msg: on_sensor_combined(
            bag, sync_monitor, msg, imu_history
        ), qos)
    if range_history is not None:
        def note_range(msg):
            distance = float(msg.current_distance)
            if msg.min_distance <= distance <= msg.max_distance:
                range_history.note(
                    int(msg.timestamp) * 1000,
                    distance,
                    int(msg.signal_quality),
                )
        node.create_subscription(
            DistanceSensor, "/fmu/out/distance_sensor", note_range, qos
        )
    # Mission-end auto-stop: a dedicated vehicle_status sub (the PX4_SUBS one only
    # logs). Off by default so standalone/bench captures are unaffected.
    if args.stop_on_disarm:
        node.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v4", on_vehicle_status, qos)
        print("recorder: --stop-on-disarm armed (will end one flight after arm)",
              file=sys.stderr, flush=True)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    def stop(*_):
        global _running
        if _running:
            print(
                "recorder: stop requested; finishing current frames",
                file=sys.stderr,
                flush=True,
            )
            _running = False
        else:
            print(
                "recorder: finalization is already in progress",
                file=sys.stderr,
                flush=True,
            )
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    sync_ok, sync_status = sync_monitor.wait_ready(args.sync_timeout)
    if not sync_ok:
        print(f"recorder: REFUSING CAPTURE: {sync_status}", file=sys.stderr, flush=True)
        node.destroy_node()
        rclpy.shutdown()
        bag.close()
        return 3
    print(f"recorder: {sync_status}", file=sys.stderr, flush=True)
    print(
        "recorder: realtime_minus_boottime="
        f"{ov_clock_mapper.initial_offset_ns} ns",
        file=sys.stderr, flush=True,
    )

    cameras = []
    threads = []
    errors = []
    errors_lock = threading.Lock()

    # The nadir detector, when asked for. The recording is the product, so a
    # detector that will not load is reported and the capture proceeds without
    # it; it never costs the flight its bag.
    detector_sink = None
    from frame_sinks import FanoutSink
    if args.detect:
        try:
            from frame_sinks import DetectorSink
            from mission_engine.rim.tag_detector import (
                TagDetector, detect_image_with_debug,
            )
            from vision_msgs.msg import Detection2DArray

            tag_detector = TagDetector()
            publisher = node.create_publisher(Detection2DArray, args.detect_topic, 10)
            debug_publisher = node.create_publisher(
                String, "/detections/down/debug", 10
            )
            detector_sink = DetectorSink(
                detector=lambda img: detect_image_with_debug(tag_detector, img),
                publish=publisher.publish,
                bag=bag,
                topic=args.detect_topic,
                depth=args.detect_queue,
                debug_publish=debug_publisher.publish,
                debug_topic="/detections/down/debug",
            )
            print(f"recorder: nadir detector -> {args.detect_topic} "
                  f"(queue {args.detect_queue})", file=sys.stderr, flush=True)
        except Exception as exc:
            errors.append(f"detector: {exc}")
            print(f"recorder: DETECTOR DISABLED: {exc}", file=sys.stderr, flush=True)
    flow_sink = None
    if args.flow:
        try:
            from cm2_flow import Cm2FlowFrontend
            from frame_sinks import FlowSink

            frontend = Cm2FlowFrontend(args.flow_calibration, imu_history)
            flow_publisher = node.create_publisher(
                SensorOpticalFlow, "/fmu/in/sensor_optical_flow", 10
            )
            flow_debug_publisher = node.create_publisher(
                String, "/localization/cm2_flow/debug", 10
            )
            flow_sink = FlowSink(
                frontend,
                range_history,
                flow_publisher.publish,
                flow_debug_publisher.publish,
                bag,
            )
            node.create_subscription(
                Bool,
                "/capture/trigger_raw_cm2",
                lambda msg: flow_sink.trigger_raw_clip() if msg.data else None,
                10,
            )
            print(
                "recorder: live CM2 flow -> /fmu/in/sensor_optical_flow",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            errors.append(f"flow: {exc}")
            print(f"recorder: FLOW DISABLED: {exc}", file=sys.stderr, flush=True)
    down_process = None
    down_frames = None
    down_status = None
    down_configure = None
    down_start = None
    down_stop = None
    session_t0 = 0.0
    ov_stats = CameraStats(args.fps)
    down_stats = CameraStats(args.down_fps)
    try:
        if not args.no_down_camera:
            context = multiprocessing.get_context("spawn")
            down_frames, child_frames = context.Pipe(duplex=False)
            down_status, child_status = context.Pipe(duplex=False)
            down_configure = context.Event()
            down_start = context.Event()
            down_stop = context.Event()
            down_process = context.Process(
                target=capture_imx219_worker,
                args=(
                    args.down_cam,
                    args.down_w,
                    args.down_h,
                    args.down_fps,
                    args.down_max_exposure_us,
                    child_frames,
                    child_status,
                    down_configure,
                    down_start,
                    down_stop,
                ),
                name="capture-imx219-camera",
            )
            down_process.start()
            child_frames.close()
            child_status.close()
            if not down_status.poll(10):
                raise RuntimeError("IMX219 camera did not initialize within 10 seconds")
            status, value = down_status.recv()
            if status != "initialized":
                raise RuntimeError(f"IMX219 camera failed: {value}")
            down_rotation = value

        # Initialize both libcamera managers before either sensor streams. The
        # CM2 needs a separate manager for its process-local exposure tuning.
        ov_tuning = load_ov9281_daylight_tuning(
            Picamera2, args.ov_max_exposure_us
        )
        ov = Picamera2(args.cam, tuning=ov_tuning)
        cameras.append(ov)
        ov_model = ov.camera_properties.get("Model")
        ov_rotation = ov.camera_properties.get("Rotation")
        if ov_model != "ov9281" or ov_rotation != 180:
            raise RuntimeError(
                "expected OV9281 with device-tree rotation=180 at "
                f"camera {args.cam}, got model={ov_model!r} "
                f"rotation={ov_rotation!r}"
            )
        if down_process:
            down_configure.set()
            if not down_status.poll(10):
                raise RuntimeError(
                    "IMX219 camera did not configure within 10 seconds"
                )
            status, value = down_status.recv()
            if status != "configured":
                raise RuntimeError(f"IMX219 camera failed: {value}")

        # Configure the OV9281 last. With rotation=180 in device tree and an
        # identity output transform, libcamera selects native H+V sensor flips.
        ov_duration_us = int(1_000_000 / args.fps)
        ov.configure(ov.create_video_configuration(
            main={"size": (args.w, args.h), "format": "YUV420"},
            controls={
                "FrameDurationLimits": (ov_duration_us, ov_duration_us),
                "AeExposureMode": 1,
                "ExposureTimeMode": 0,
                "AnalogueGainMode": 0,
            },
            buffer_count=12,
        ))
        if down_process:
            down_start.set()
            if not down_status.poll(10):
                raise RuntimeError("IMX219 camera did not start within 10 seconds")
            status, value = down_status.recv()
            if status != "ready":
                raise RuntimeError(f"IMX219 camera failed: {value}")
        ov.start()
        session_t0 = time.monotonic()
        print(
            f"recorder: OV9281 cam{args.cam} {args.w}x{args.h}@{args.fps} "
            f"rotation={ov_rotation} "
            f"automatic exposure<={args.ov_max_exposure_us}us "
            f"-> {args.out}",
            file=sys.stderr,
            flush=True,
        )
        if down_process:
            print(
                f"recorder: IMX219 cam{args.down_cam} "
                f"{args.down_w}x{args.down_h}@{args.down_fps} "
                f"rotation={down_rotation} "
                f"automatic exposure<={args.down_max_exposure_us}us "
                f"-> {args.out}",
                file=sys.stderr,
                flush=True,
            )

        threads.append(threading.Thread(
            target=capture_camera,
            kwargs={
                "picam": ov,
                "bag": bag,
                "clock_mapper": ov_clock_mapper,
                "stats": ov_stats,
                "width": args.w,
                "height": args.h,
                "frame_id": "ov9281",
                "image_topic": "/camera/image_raw",
                "info_topic": "/camera/camera_info",
                "camera_info": make_uncalibrated_camera_info(args.w, args.h),
                "clock_topic": "/capture/realtime_minus_boottime_ns",
                "max_exposure_us": args.ov_max_exposure_us,
                "errors": errors,
                "errors_lock": errors_lock,
                "record_fps": args.ov_record_fps,
            },
            name="capture-ov9281",
        ))
        if down_process:
            threads.append(threading.Thread(
                target=capture_camera_pipe,
                kwargs={
                    "connection": down_frames,
                    "status_connection": down_status,
                    "process": down_process,
                    "bag": bag,
                    "clock_mapper": down_clock_mapper,
                    "stats": down_stats,
                    "width": args.down_w,
                    "height": args.down_h,
                    "frame_id": "imx219_nadir",
                    "image_topic": "/camera_down/image_raw",
                    "info_topic": "/camera_down/camera_info",
                    "camera_info": make_uncalibrated_camera_info(
                        args.down_w, args.down_h
                    ),
                    "max_exposure_us": args.down_max_exposure_us,
                    "errors": errors,
                    "errors_lock": errors_lock,
                    "sink": FanoutSink(flow_sink, detector_sink),
                    "record_raw": not args.no_down_raw,
                    "preview_fps": args.down_preview_fps,
                },
                name="capture-imx219",
            ))
        for thread in threads:
            thread.start()
        while _running and any(thread.is_alive() for thread in threads):
            time.sleep(0.05)
    except Exception as exc:
        errors.append(f"setup: {exc}")
        print(f"recorder: REFUSING CAPTURE: {exc}", file=sys.stderr, flush=True)
    finally:
        _running = False
        if down_stop is not None:
            down_stop.set()
        if down_configure is not None:
            down_configure.set()
        if down_start is not None:
            down_start.set()
        for thread in threads:
            thread.join(timeout=0.5)
        for camera in reversed(cameras):
            try:
                camera.stop()
            except Exception:
                pass
        for thread in threads:
            thread.join(timeout=2.0)
        if down_frames is not None:
            # Unblock a CM2 worker that is inside send_bytes after recording
            # stops and the receiver thread exits.
            down_frames.close()
        if down_process is not None:
            down_process.join(timeout=3.0)
            if down_process.is_alive():
                errors.append("imx219_nadir: camera process did not stop")
                down_process.terminate()
                down_process.join(timeout=2.0)
        if down_status is not None:
            down_status.close()
        for camera in reversed(cameras):
            try:
                camera.close()
            except Exception:
                pass
        for sink in (flow_sink, detector_sink):
            if sink is not None:
                sink.close()
        try:
            node.destroy_node()  # stop ROS callbacks before finalizing the writer
        except Exception:
            pass
        rclpy.shutdown()
        bag.close()              # race-free finalize (flush cache + metadata)

        dt = time.monotonic() - session_t0 if session_t0 else 0.0
        print("=== capture summary ===", file=sys.stderr)
        for k, v in bag.counts.items():
            hz = v / dt if dt > 0 else 0.0
            print(f"  {k:38s} {v:6d}  ({hz:.1f} Hz)", file=sys.stderr)
        for name, stats, mapper in (
            ("ov9281", ov_stats, ov_clock_mapper),
            ("imx219_nadir", down_stats, down_clock_mapper),
        ):
            if stats.frames:
                print(
                    f"  {name}: sensor_rate={stats.measured_hz():.2f} Hz, "
                    f"large_gaps={stats.large_gaps}, "
                    f"max_gap={stats.max_gap_ns / 1e6:.2f} ms, "
                    f"max_exposure={stats.max_exposure_us} us, "
                    f"max_analogue_gain={stats.max_analogue_gain:.2f}, "
                    "max realtime/boottime offset change="
                    f"{mapper.max_offset_delta_ns / 1e6:.3f} ms",
                    file=sys.stderr,
                )
        if detector_sink is not None:
            print(f"  {detector_sink.summary()}", file=sys.stderr)
            if detector_sink.last_fault:
                print(f"  last detector fault: {detector_sink.last_fault}",
                      file=sys.stderr)
        if flow_sink is not None:
            print(f"  {flow_sink.summary()}", file=sys.stderr)
            if flow_sink.last_fault:
                print(f"  last flow fault: {flow_sink.last_fault}",
                      file=sys.stderr)
        for error in errors:
            print(f"  ERROR: {error}", file=sys.stderr)
    return 4 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
