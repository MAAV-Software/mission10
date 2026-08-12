"""Fused relative localization from shared ENU state and UWB range."""
from __future__ import annotations

import math

import numpy as np
import rclpy
from flight_interfaces.msg import RelativePeerState, UwbRange, UwbState
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from flight_lib import FusedRelativeTracker, STATUS_TRACKING


def _flat2(matrix):
    return [float(value) for value in np.asarray(matrix, float).reshape(-1)]


def _peer_id_map(namespaces, configured_ids):
    namespaces = [value for value in namespaces if value]
    configured_ids = [int(value) for value in configured_ids]
    if configured_ids == [-1]:
        ids = [int(namespace.rsplit("_", 1)[-1]) for namespace in namespaces]
    else:
        if len(configured_ids) != len(namespaces):
            raise ValueError("peer_ids must match peer_namespaces")
        ids = configured_ids
    if any(peer_id < 0 for peer_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("peer_ids must be unique nonnegative integers")
    return dict(zip(ids, namespaces))


class RelativeLocalization(Node):
    def __init__(self):
        super().__init__("relative_localization")
        self.declare_parameter("vehicle_namespace", "px4_0")
        self.declare_parameter("drone_index", 0)
        self.declare_parameter("peer_namespaces", ["px4_1"])
        # Explicit IDs decouple temporary UWB/formation identity from hostname
        # digits. [-1] retains namespace-suffix inference for existing launchers.
        self.declare_parameter("peer_ids", [-1])
        self.declare_parameter("position_noise_std_m", 0.6)
        self.declare_parameter("range_noise_std_m", 0.1)
        self.declare_parameter("velocity_noise_std_mps", 0.3)

        self.namespace = str(self.get_parameter("vehicle_namespace").value)
        self.index = int(self.get_parameter("drone_index").value)
        self.peer_namespaces = [
            value for value in self.get_parameter("peer_namespaces").value if value
        ]
        self.peer_ids = _peer_id_map(
            self.peer_namespaces, self.get_parameter("peer_ids").value)
        self.states: dict[int, UwbState] = {}
        self.estimators = {
            peer_id: FusedRelativeTracker(
                position_noise_std=float(
                    self.get_parameter("position_noise_std_m").value),
                range_noise_std=float(self.get_parameter("range_noise_std_m").value),
                velocity_noise_std=float(
                    self.get_parameter("velocity_noise_std_mps").value),
            )
            for peer_id in self.peer_ids
        }
        self.latest_estimates = {}

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
        vehicle_id = int(msg.vehicle_id)
        self.states[vehicle_id] = msg
        peer_ids = self.peer_ids if vehicle_id == self.index else [vehicle_id]
        for peer_id in peer_ids:
            if peer_id in self.estimators:
                self._update_peer(peer_id, msg.stamp)

    def _range_cb(self, msg):
        peer_id = int(msg.source_id)
        if int(msg.receiver_id) != self.index or peer_id not in self.estimators:
            return
        timestamp = self.get_clock().now().nanoseconds * 1e-9
        self._update_peer(peer_id, msg.stamp, timestamp=timestamp, range_m=float(msg.range_m))

    def _update_peer(self, peer_id, stamp, *, timestamp=None, range_m=math.nan):
        own = self.states.get(self.index)
        peer = self.states.get(peer_id)
        if own is None or peer is None:
            return
        required = UwbState.VALID_POSITION | UwbState.VALID_VELOCITY
        if ((int(own.validity) & required) != required
                or (int(peer.validity) & required) != required):
            return
        if timestamp is None:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        try:
            estimate = self.estimators[peer_id].update(
                timestamp,
                own.position_enu_m,
                peer.position_enu_m,
                own.velocity_enu_mps,
                peer.velocity_enu_mps,
                range_m=range_m,
                own_epoch=int(own.frame_epoch),
                peer_epoch=int(peer.frame_epoch),
            )
        except ValueError as exc:
            self.get_logger().warn(f"discarding UWB sample from {peer_id}: {exc}")
            return
        self.latest_estimates[peer_id] = estimate
        self._publish_estimate(stamp, peer_id, estimate)
        if estimate.reseeded:
            self.get_logger().info(
                f"peer {peer_id}: fused tracker seeded at frame epochs "
                f"{int(own.frame_epoch)}/{int(peer.frame_epoch)}")

    def _publish_estimate(self, stamp, peer_id, estimate):
        msg = RelativePeerState()
        msg.stamp = stamp
        msg.observer_id = self.index
        msg.peer_id = peer_id
        msg.status = STATUS_TRACKING
        msg.position_enu_m = [float(v) for v in estimate.position]
        msg.velocity_enu_mps = [float(v) for v in estimate.relative_velocity]
        msg.covariance_xy = _flat2(estimate.covariance)
        msg.alternate_valid = False
        msg.alternate_position_enu_m = [math.nan, math.nan]
        msg.alternate_covariance_xy = [math.inf] * 4
        msg.range_m = float(estimate.range_m)
        msg.range_rate_mps = float(estimate.range_rate_mps)
        msg.residual_rms_m = float(estimate.residual_m)
        msg.observability = 1.0
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
