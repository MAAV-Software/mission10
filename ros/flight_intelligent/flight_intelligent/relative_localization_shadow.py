"""UWB range-history relative-localization node.

The node consumes only scalar range and shared-ENU velocity from ``UwbState``.
Absolute/GNSS position fields are intentionally never read. Avoidance lives in
the mission node, where the intended velocity is available.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from flight_interfaces.msg import RelativePeerState, UwbRange, UwbState
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from flight_lib import RangeHistoryRelativeEstimator


def _seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _flat2(matrix):
    return [float(value) for value in np.asarray(matrix, float).reshape(-1)]


class RelativeLocalization(Node):
    def __init__(self):
        super().__init__("relative_localization")
        self.declare_parameter("vehicle_namespace", "px4_0")
        self.declare_parameter("drone_index", 0)
        self.declare_parameter("peer_namespaces", ["px4_1"])
        self.declare_parameter("window_s", 5.0)
        self.declare_parameter("range_noise_std_m", 0.1)
        self.declare_parameter("velocity_noise_std_mps", 0.3)

        self.namespace = str(self.get_parameter("vehicle_namespace").value)
        self.index = int(self.get_parameter("drone_index").value)
        self.peer_namespaces = [
            value for value in self.get_parameter("peer_namespaces").value if value
        ]
        self.peer_ids = {
            int(namespace.rsplit("_", 1)[-1]): namespace
            for namespace in self.peer_namespaces
        }
        self.states: dict[int, UwbState] = {}
        self.estimators = {
            peer_id: RangeHistoryRelativeEstimator(
                window_s=float(self.get_parameter("window_s").value),
                range_noise_std=float(self.get_parameter("range_noise_std_m").value),
                velocity_noise_std=float(
                    self.get_parameter("velocity_noise_std_mps").value),
            )
            for peer_id in self.peer_ids
        }
        self.latest_estimates = {}
        self.last_status = {}

        self.relative_pub = self.create_publisher(
            RelativePeerState, f"/{self.namespace}/uwb/relative_state", 20)
        self.create_subscription(
            UwbRange, f"/{self.namespace}/uwb/range", self._range_cb, 50)
        for namespace in [self.namespace, *self.peer_namespaces]:
            self.create_subscription(
                UwbState, f"/{namespace}/uwb/state", self._state_cb, 50)
        self.get_logger().info(
            f"relative localization up: observer={self.index} "
            f"peers={sorted(self.peer_ids)}")

    def _state_cb(self, msg):
        # Only velocity is consumed. In particular, position_enu_m and
        # gnss_enu_m are not read anywhere in this node.
        self.states[int(msg.vehicle_id)] = msg

    def _range_cb(self, msg):
        peer_id = int(msg.source_id)
        if int(msg.receiver_id) != self.index or peer_id not in self.estimators:
            return
        own = self.states.get(self.index)
        peer = self.states.get(peer_id)
        if own is None or peer is None:
            return
        timestamp = _seconds(msg.stamp)
        if timestamp <= 0.0:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        try:
            estimate = self.estimators[peer_id].add_sample(
                timestamp,
                float(msg.range_m),
                own.velocity_enu_mps,
                peer.velocity_enu_mps,
            )
        except ValueError as exc:
            self.get_logger().warn(f"discarding UWB sample from {peer_id}: {exc}")
            return
        self.latest_estimates[peer_id] = estimate
        self._publish_estimate(msg.stamp, peer_id, estimate)
        if self.last_status.get(peer_id) != estimate.status:
            self.last_status[peer_id] = estimate.status
            labels = {0: "UNOBSERVABLE", 1: "AMBIGUOUS", 2: "TRACKING"}
            self.get_logger().info(
                f"peer {peer_id}: {labels.get(estimate.status, estimate.status)} "
                f"samples={estimate.sample_count} obs={estimate.observability:.3f} "
                f"r95={estimate.confidence_radius_95_m:.2f}m")

    def _publish_estimate(self, stamp, peer_id, estimate):
        msg = RelativePeerState()
        msg.stamp = stamp
        msg.observer_id = self.index
        msg.peer_id = peer_id
        msg.status = int(estimate.status)
        msg.position_enu_m = (
            [float(v) for v in estimate.position] if estimate.position is not None
            else [math.nan, math.nan])
        msg.velocity_enu_mps = [float(v) for v in estimate.relative_velocity]
        msg.covariance_xy = (
            _flat2(estimate.covariance) if estimate.covariance is not None
            else [math.inf] * 4)
        msg.alternate_valid = estimate.alternate_position is not None
        msg.alternate_position_enu_m = (
            [float(v) for v in estimate.alternate_position]
            if estimate.alternate_position is not None else [math.nan, math.nan])
        msg.alternate_covariance_xy = (
            _flat2(estimate.alternate_covariance)
            if estimate.alternate_covariance is not None else [math.inf] * 4)
        msg.range_m = float(estimate.range_m)
        msg.range_rate_mps = float(estimate.range_rate_mps)
        msg.residual_rms_m = float(estimate.residual_rms_m)
        msg.observability = float(estimate.observability)
        msg.confidence_radius_95_m = float(estimate.confidence_radius_95_m)
        msg.sample_count = min(65535, int(estimate.sample_count))
        self.relative_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RelativeLocalization()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
