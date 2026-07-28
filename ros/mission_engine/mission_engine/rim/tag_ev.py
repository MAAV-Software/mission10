"""Pad-prior and AprilTag horizontal-position adapter for PX4 EKF2.

The observation never consumes PX4 local X/Y. It uses only the known pad
centre, image geometry, image-time attitude, and the downward range sensor.
"""
from __future__ import annotations

from collections import deque
import json
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

from px4_msgs.msg import (
    DistanceSensor,
    VehicleAttitude,
    VehicleOdometry,
    VehicleStatus,
)


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _quat_rotate(q, v):
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return np.array(
        [
            v[0] + w * tx + y * tz - z * ty,
            v[1] + w * ty + z * tx - x * tz,
            v[2] + w * tz + x * ty - y * tx,
        ],
        dtype=np.float64,
    )


class TimedHistory:
    def __init__(self, maxlen=500):
        self.rows = deque(maxlen=maxlen)

    def add(self, timestamp_ns, value):
        if not self.rows or timestamp_ns > self.rows[-1][0]:
            self.rows.append((int(timestamp_ns), value))

    def nearest(self, timestamp_ns, max_age_ns):
        if not self.rows:
            return None
        row = min(self.rows, key=lambda item: abs(item[0] - timestamp_ns))
        return row if abs(row[0] - timestamp_ns) <= max_age_ns else None


class TagEvNode(Node):
    def __init__(self):
        super().__init__("tag_ev")
        self.declare_parameter("publish_ev", False)
        self.declare_parameter("detections_topic", "/detections/down")
        self.declare_parameter("tag_ids", [6, 7])
        self.declare_parameter("registration_samples", 10)
        self.declare_parameter("registration_scatter_m", 0.10)
        self.declare_parameter("pair_disagreement_m", 0.25)
        self.declare_parameter("prior_timeout_after_arm_s", 10.0)
        self.declare_parameter("position_noise_floor_m", 0.10)

        self.publish_ev = bool(self.get_parameter("publish_ev").value)
        self.tag_ids = {
            int(value) for value in self.get_parameter("tag_ids").value
        }
        self.registration_samples = int(
            self.get_parameter("registration_samples").value
        )
        self.registration_scatter_m = float(
            self.get_parameter("registration_scatter_m").value
        )
        self.pair_disagreement_m = float(
            self.get_parameter("pair_disagreement_m").value
        )
        self.prior_timeout_after_arm_s = float(
            self.get_parameter("prior_timeout_after_arm_s").value
        )
        self.noise_floor_m = float(
            self.get_parameter("position_noise_floor_m").value
        )

        self.k = np.array(
            [
                [1298.69385194, 0.0, 827.70242273],
                [0.0, 1299.56328818, 617.60425847],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.distortion = np.array(
            [0.15046410, -0.23368367, 0.00000042, -0.00209991],
            dtype=np.float64,
        )
        self.line_delay_s = 9.68869339923e-6
        self.r_b_c = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.attitudes = TimedHistory()
        self.ranges = TimedHistory()
        self.registration = {tag_id: deque(maxlen=self.registration_samples)
                             for tag_id in self.tag_ids}
        self.references = {}
        self.armed_since = None
        self.reset_counter = 0

        sensor_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.ev_pub = self.create_publisher(
            VehicleOdometry, "/fmu/in/vehicle_visual_odometry", 10
        )
        self.status_pub = self.create_publisher(
            String, "/localization/tag_ev/status", 10
        )
        self.create_subscription(
            Detection2DArray,
            str(self.get_parameter("detections_topic").value),
            self._detections,
            10,
        )
        self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self._attitude,
            sensor_qos,
        )
        self.create_subscription(
            DistanceSensor,
            "/fmu/out/distance_sensor",
            self._range,
            sensor_qos,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v4",
            self._status,
            sensor_qos,
        )
        self.create_timer(0.1, self._prior_tick)
        self.get_logger().info(
            "tag EV up in "
            + ("ACTIVE" if self.publish_ev else "SHADOW")
            + " mode; no PX4 local XY is subscribed"
        )

    def _now_us(self):
        return int(self.get_clock().now().nanoseconds // 1000)

    def _attitude(self, msg):
        q = np.asarray(msg.q, dtype=np.float64)
        if np.all(np.isfinite(q)) and np.linalg.norm(q) > 0.5:
            q /= np.linalg.norm(q)
            self.attitudes.add(int(msg.timestamp) * 1000, q)

    def _range(self, msg):
        distance = float(msg.current_distance)
        if (
            math.isfinite(distance)
            and float(msg.min_distance) <= distance <= float(msg.max_distance)
            and int(msg.signal_quality) != 0
        ):
            self.ranges.add(
                int(msg.timestamp) * 1000,
                (distance, int(msg.signal_quality)),
            )

    def _status(self, msg):
        if msg.arming_state == VehicleStatus.ARMING_STATE_ARMED:
            if self.armed_since is None:
                self.armed_since = time.monotonic()
        else:
            self.armed_since = None

    def _prior_allowed(self):
        if self.references.keys() >= self.tag_ids:
            return False
        if self.armed_since is None:
            return True
        return time.monotonic() - self.armed_since <= self.prior_timeout_after_arm_s

    def _prior_tick(self):
        if not self._prior_allowed():
            return
        # The launch prior is only a bootstrap measurement. Once one tag has
        # a pad-frame reference, that tag owns the horizontal observation.
        if self.references:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self.publish_ev:
            self._publish_ev(now_ns, np.zeros(2), self.noise_floor_m ** 2, 100)
        self._publish_status(
            now_ns,
            "pad_prior" if not any(self.registration.values()) else "registering",
            None,
            None,
            self.publish_ev,
        )

    def _camera_to_tag(self, center_px, timestamp_ns):
        # Use the detected tag row's physical rolling-shutter time.
        row_ns = timestamp_ns + int(center_px[1] * self.line_delay_s * 1e9)
        attitude = self.attitudes.nearest(row_ns, 100_000_000)
        range_row = self.ranges.nearest(row_ns, 100_000_000)
        if attitude is None or range_row is None:
            return None
        q = attitude[1]
        distance, quality = range_row[1]
        normalized = cv2.undistortPoints(
            np.asarray(center_px, dtype=np.float64).reshape(1, 1, 2),
            self.k,
            self.distortion,
        )[0, 0]
        ray_camera = np.array([normalized[0], normalized[1], 1.0])
        ray_ned = _quat_rotate(q, self.r_b_c @ ray_camera)
        body_down_ned = _quat_rotate(q, np.array([0.0, 0.0, 1.0]))
        agl = distance * body_down_ned[2]
        if agl <= 0.0 or ray_ned[2] <= 0.1:
            return None
        scale = agl / ray_ned[2]
        offset = ray_ned[:2] * scale
        return row_ns, offset, quality

    @staticmethod
    def _tag_id(detection):
        value = str(detection.id)
        try:
            return int(value.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return None

    def _detections(self, msg):
        image_ns = _stamp_ns(msg.header.stamp)
        observations = []
        used_ns = image_ns
        for detection in msg.detections:
            tag_id = self._tag_id(detection)
            if tag_id not in self.tag_ids:
                continue
            center = (
                float(detection.bbox.center.position.x),
                float(detection.bbox.center.position.y),
            )
            relative = self._camera_to_tag(center, image_ns)
            if relative is None:
                continue
            used_ns, offset, _ = relative
            observations.append((tag_id, offset))

        if not observations:
            return

        # Bootstrap exactly one tag against the known launch-pad origin. A
        # short stable window bounds the handoff jump without incorrectly
        # requiring the aircraft to remain stationary for the whole climb.
        if not self.references and self._prior_allowed():
            tag_id, offset = max(
                observations,
                key=lambda item: (
                    len(self.registration[item[0]]),
                    -item[0],
                ),
            )
            rows = self.registration[tag_id]
            rows.append(offset)
            if len(rows) == self.registration_samples:
                values = np.asarray(rows)
                median = np.median(values, axis=0)
                scatter = float(
                    np.max(np.linalg.norm(values - median, axis=1))
                )
                if scatter <= self.registration_scatter_m:
                    self.references[tag_id] = median
                    self.get_logger().info(
                        f"bootstrapped tag {tag_id} at "
                        f"N={median[0]:+.3f} E={median[1]:+.3f}"
                    )

        candidates = [
            self.references[tag_id] - offset
            for tag_id, offset in observations
            if tag_id in self.references
        ]
        if not candidates:
            return

        # Register every later tag relative to the position supplied by an
        # existing tag. This removes vehicle motion from its registration
        # samples and allows a second anchor to converge during flight.
        provisional_position = np.median(np.asarray(candidates), axis=0)
        for tag_id, offset in observations:
            if tag_id in self.references:
                continue
            rows = self.registration[tag_id]
            rows.append(provisional_position + offset)
            if len(rows) != self.registration_samples:
                continue
            values = np.asarray(rows)
            median = np.median(values, axis=0)
            scatter = float(np.max(np.linalg.norm(values - median, axis=1)))
            if scatter <= self.registration_scatter_m:
                self.references[tag_id] = median
                self.get_logger().info(
                    f"registered tag {tag_id} relative to active anchor at "
                    f"N={median[0]:+.3f} E={median[1]:+.3f}"
                )

        candidates = [
            self.references[tag_id] - offset
            for tag_id, offset in observations
            if tag_id in self.references
        ]

        values = np.asarray(candidates)
        position = np.median(values, axis=0)
        disagreement = (
            float(np.max(np.linalg.norm(values - position, axis=1)))
            if len(values) > 1
            else 0.0
        )
        if disagreement > self.pair_disagreement_m:
            self._publish_status(
                used_ns, "rejected", position, disagreement, False
            )
            return
        scatter_var = (
            float(np.max(np.var(values, axis=0))) if len(values) > 1 else 0.0
        )
        variance = max(self.noise_floor_m ** 2, scatter_var)
        if self.publish_ev:
            self._publish_ev(used_ns, position, variance, 100)
        self._publish_status(
            used_ns, "tag_position", position, disagreement, self.publish_ev
        )

    def _publish_ev(self, sample_ns, position, variance, quality):
        msg = VehicleOdometry()
        msg.timestamp = self._now_us()
        msg.timestamp_sample = int(sample_ns // 1000)
        msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
        # PX4 checks all three position elements before extracting horizontal
        # XY. Z is a finite parser placeholder; EKF2_EV_CTRL=1 does not fuse it.
        msg.position[:] = [float(position[0]), float(position[1]), 0.0]
        msg.q[:] = [math.nan, math.nan, math.nan, math.nan]
        msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_UNKNOWN
        msg.velocity[:] = [math.nan, math.nan, math.nan]
        msg.angular_velocity[:] = [math.nan, math.nan, math.nan]
        msg.position_variance[:] = [variance, variance, 1.0]
        msg.orientation_variance[:] = [math.nan, math.nan, math.nan]
        msg.velocity_variance[:] = [math.nan, math.nan, math.nan]
        msg.reset_counter = self.reset_counter
        msg.quality = int(quality)
        self.ev_pub.publish(msg)

    def _publish_status(
        self, timestamp_ns, state, position, disagreement, ev_published
    ):
        msg = String()
        msg.data = json.dumps(
            {
                "timestamp_sample_ns": int(timestamp_ns),
                "state": state,
                "publish_ev": self.publish_ev,
                "ev_published": bool(ev_published),
                "registered": {
                    str(tag_id): self.references[tag_id].tolist()
                    for tag_id in sorted(self.references)
                },
                "registration_counts": {
                    str(tag_id): len(self.registration[tag_id])
                    for tag_id in sorted(self.registration)
                },
                "camera_position_ned_m": (
                    np.asarray(position).tolist()
                    if position is not None
                    else None
                ),
                "pair_disagreement_m": disagreement,
                "reset_counter": self.reset_counter,
            },
            separators=(",", ":"),
        )
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TagEvNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
