"""Small, recorder-only building blocks."""
from __future__ import annotations

import sys
import threading

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Int64
from px4_msgs.msg import (
    DistanceSensor,
    EstimatorAidSource1d,
    EstimatorAidSource2d,
    EstimatorAidSource3d,
    EstimatorGpsStatus,
    EstimatorStatus,
    EstimatorStatusFlags,
    FailsafeFlags,
    SensorCombined,
    SensorGps,
    TimesyncStatus,
    VehicleAttitude,
    VehicleCommandAck,
    VehicleGlobalPosition,
    VehicleLocalPosition,
    VehicleStatus,
)
from sensing.timebase import ClockMapper, SyncMonitor, boottime_ns, realtime_ns


PX4_SUBS = [
    ("/fmu/out/timesync_status", "px4_msgs/msg/TimesyncStatus", TimesyncStatus),
    (
        "/fmu/out/vehicle_local_position_v1",
        "px4_msgs/msg/VehicleLocalPosition",
        VehicleLocalPosition,
    ),
    ("/fmu/out/vehicle_attitude", "px4_msgs/msg/VehicleAttitude", VehicleAttitude),
    (
        "/fmu/out/vehicle_global_position",
        "px4_msgs/msg/VehicleGlobalPosition",
        VehicleGlobalPosition,
    ),
    ("/fmu/out/vehicle_gps_position", "px4_msgs/msg/SensorGps", SensorGps),
    ("/fmu/out/sensor_combined", "px4_msgs/msg/SensorCombined", SensorCombined),
    ("/fmu/out/distance_sensor", "px4_msgs/msg/DistanceSensor", DistanceSensor),
    ("/fmu/out/estimator_status_flags", "px4_msgs/msg/EstimatorStatusFlags", EstimatorStatusFlags),
    ("/fmu/out/estimator_status", "px4_msgs/msg/EstimatorStatus", EstimatorStatus),
    ("/fmu/out/estimator_gps_status", "px4_msgs/msg/EstimatorGpsStatus", EstimatorGpsStatus),
    (
        "/fmu/out/estimator_aid_src_gnss_pos",
        "px4_msgs/msg/EstimatorAidSource2d",
        EstimatorAidSource2d,
    ),
    (
        "/fmu/out/estimator_aid_src_gnss_vel",
        "px4_msgs/msg/EstimatorAidSource3d",
        EstimatorAidSource3d,
    ),
    (
        "/fmu/out/estimator_aid_src_rng_hgt",
        "px4_msgs/msg/EstimatorAidSource1d",
        EstimatorAidSource1d,
    ),
    (
        "/fmu/out/estimator_aid_src_optical_flow",
        "px4_msgs/msg/EstimatorAidSource2d",
        EstimatorAidSource2d,
    ),
    (
        "/fmu/out/estimator_aid_src_ev_pos",
        "px4_msgs/msg/EstimatorAidSource2d",
        EstimatorAidSource2d,
    ),
    ("/fmu/out/vehicle_command_ack_v1", "px4_msgs/msg/VehicleCommandAck", VehicleCommandAck),
    ("/fmu/out/failsafe_flags", "px4_msgs/msg/FailsafeFlags", FailsafeFlags),
    ("/fmu/out/vehicle_status_v4", "px4_msgs/msg/VehicleStatus", VehicleStatus),
]


def stamp_ns(value: int) -> TimeMsg:
    return TimeMsg(sec=int(value // 1_000_000_000), nanosec=int(value % 1_000_000_000))


class Bag:
    def __init__(self, uri: str, split_bytes: int = 0, storage_config: str = "") -> None:
        self.writer = rosbag2_py.SequentialWriter()
        options = rosbag2_py.StorageOptions(
            uri=uri,
            storage_id="mcap",
            max_cache_size=512 * 1024 * 1024,
            max_bagfile_size=split_bytes,
        )
        if storage_config:
            options.storage_config_uri = storage_config
        self.writer.open(options, rosbag2_py.ConverterOptions("", ""))
        self.lock = threading.Lock()
        self.next_topic_id = 0
        self.counts: dict[str, int] = {}

    def topic(self, name: str, type_name: str) -> None:
        self.writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=self.next_topic_id,
                name=name,
                type=type_name,
                serialization_format="cdr",
            )
        )
        self.next_topic_id += 1
        self.counts[name] = 0

    def write(self, name: str, data: bytes, timestamp_ns: int) -> None:
        with self.lock:
            if self.writer is None:
                return
            self.writer.write(name, data, timestamp_ns)
            self.counts[name] += 1

    def close(self) -> None:
        with self.lock:
            writer, self.writer = self.writer, None
        print("recorder: finalizing MCAP; keep power connected", file=sys.stderr, flush=True)
        del writer
        print("recorder: MCAP finalized", file=sys.stderr, flush=True)


class CameraStats:
    def __init__(self, fps: float) -> None:
        self.nominal_period_ns = int(1e9 / fps)
        self.frames = 0
        self.first_sensor_ns = 0
        self.last_sensor_ns = 0
        self.max_gap_ns = 0
        self.large_gaps = 0

    def note(self, sensor_ns: int) -> None:
        if not self.first_sensor_ns:
            self.first_sensor_ns = sensor_ns
        if self.last_sensor_ns:
            gap = sensor_ns - self.last_sensor_ns
            self.max_gap_ns = max(self.max_gap_ns, gap)
            self.large_gaps += int(gap > 1.5 * self.nominal_period_ns)
        self.last_sensor_ns = sensor_ns
        self.frames += 1

    def hz(self) -> float:
        span = self.last_sensor_ns - self.first_sensor_ns
        return (self.frames - 1) * 1e9 / span if self.frames > 1 and span > 0 else 0.0


def make_camera_info(width: int, height: int) -> CameraInfo:
    info = CameraInfo()
    info.width = width
    info.height = height
    return info


def record_ov9281(
    camera,
    bag: Bag,
    clock: ClockMapper,
    stats: CameraStats,
    stop: threading.Event,
    *,
    width: int,
    height: int,
    max_exposure_us: int,
    record_fps: float,
) -> None:
    period_ns = int(1e9 / record_fps) if record_fps > 0 else 0
    last_record_ns = 0
    last_clock_ns = 0
    info = make_camera_info(width, height)
    clock_msg = Int64()
    while not stop.is_set():
        request = camera.capture_request()
        try:
            metadata = request.get_metadata()
            sensor_ns = int(metadata.get("SensorTimestamp", boottime_ns()))
            exposure_us = int(metadata.get("ExposureTime", 0))
            if exposure_us > max_exposure_us:
                raise RuntimeError(
                    f"OV9281 exposure ceiling violated: {exposure_us} > {max_exposure_us} us"
                )
            yuv = request.make_array("main")
        finally:
            request.release()
        timestamp_ns, offset_ns = clock.map(sensor_ns)
        stats.note(sensor_ns)
        if period_ns and timestamp_ns - last_record_ns < period_ns:
            continue
        image = Image()
        image.header.stamp = stamp_ns(timestamp_ns)
        image.header.frame_id = "ov9281"
        image.height = height
        image.width = width
        image.encoding = "mono8"
        image.is_bigendian = 0
        image.step = width
        image.data = np.ascontiguousarray(yuv[:height, :width]).tobytes()
        bag.write("/camera/image_raw", serialize_message(image), timestamp_ns)
        info.header = image.header
        bag.write("/camera/camera_info", serialize_message(info), timestamp_ns)
        last_record_ns = timestamp_ns
        if timestamp_ns - last_clock_ns >= 1_000_000_000:
            clock_msg.data = offset_ns
            bag.write(
                "/capture/realtime_minus_boottime_ns",
                serialize_message(clock_msg),
                timestamp_ns,
            )
            last_clock_ns = timestamp_ns


def message_time_ns(msg) -> int:
    timestamp_ns = int(getattr(msg, "timestamp", 0)) * 1000
    now = realtime_ns()
    return timestamp_ns if timestamp_ns and abs(now - timestamp_ns) < 60_000_000_000 else now


def record_px4(bag: Bag, sync: SyncMonitor, name: str, msg) -> None:
    if name == "/fmu/out/timesync_status":
        sync.note_timesync(msg)
    bag.write(name, serialize_message(msg), message_time_ns(msg))


def record_imu(bag: Bag, sync: SyncMonitor, msg) -> None:
    sample_ns = int(msg.timestamp) * 1000
    sync.note_imu(sample_ns)
    imu = Imu()
    imu.header.stamp = stamp_ns(sample_ns)
    imu.header.frame_id = "px4_imu_frd"
    imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (
        float(value) for value in msg.gyro_rad
    )
    imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = (
        float(value) for value in msg.accelerometer_m_s2
    )
    imu.orientation_covariance[0] = -1.0
    bag.write("/imu", serialize_message(imu), sample_ns)


class StopOnDisarm:
    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop
        self.seen_armed = False

    def __call__(self, msg) -> None:
        if msg.arming_state == VehicleStatus.ARMING_STATE_ARMED:
            self.seen_armed = True
        elif self.seen_armed:
            print(
                "recorder: vehicle disarmed after flight -> stopping",
                file=sys.stderr,
                flush=True,
            )
            self.stop.set()
