"""Truth-only scorer for relative localization.

This process is deliberately separate from the estimator.  It sees Gazebo
odometry solely to measure error and never republishes truth into a control or
localization topic.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from flight_interfaces.msg import RelativePeerState
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


class RelativeTruthMonitor(Node):
    def __init__(self):
        super().__init__("relative_truth_monitor")
        self.declare_parameter("vehicle_namespaces", ["px4_0", "px4_1"])
        self.declare_parameter("spawn_e_m", [0.0, 3.0])
        self.declare_parameter("spawn_n_m", [0.0, 0.0])
        self.namespaces = list(self.get_parameter("vehicle_namespaces").value)
        self.spawn_e = [float(v) for v in self.get_parameter("spawn_e_m").value]
        self.spawn_n = [float(v) for v in self.get_parameter("spawn_n_m").value]
        self.odom = {}
        self.latest = {}
        self.errors = []
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        for index, namespace in enumerate(self.namespaces):
            self.create_subscription(
                Odometry,
                f"/{namespace}/ground_truth/odometry",
                lambda msg, index=index: self.odom.__setitem__(index, msg),
                qos,
            )
            self.create_subscription(
                RelativePeerState,
                f"/{namespace}/uwb/relative_state",
                self._estimate_cb,
                50,
            )
        self.create_timer(1.0, self._report)
        self.get_logger().info(
            "relative truth monitor up (evaluation only; no estimator output)")

    def _world_xy(self, index):
        msg = self.odom.get(index)
        if msg is None:
            return None
        p = msg.pose.pose.position
        return np.array([self.spawn_e[index] + p.x, self.spawn_n[index] + p.y])

    def _estimate_cb(self, msg):
        observer = int(msg.observer_id)
        peer = int(msg.peer_id)
        own = self._world_xy(observer)
        other = self._world_xy(peer)
        if own is None or other is None or not all(math.isfinite(v) for v in msg.position_enu_m):
            return
        truth = other - own
        primary = np.asarray(msg.position_enu_m, float)
        error = float(np.linalg.norm(primary - truth))
        if msg.alternate_valid and all(math.isfinite(v) for v in msg.alternate_position_enu_m):
            alternate = np.asarray(msg.alternate_position_enu_m, float)
            error = min(error, float(np.linalg.norm(alternate - truth)))
        self.latest[(observer, peer)] = (error, int(msg.status), float(msg.confidence_radius_95_m))
        if int(msg.status) == RelativePeerState.STATUS_TRACKING:
            self.errors.append(error)
            if len(self.errors) > 20_000:
                del self.errors[:10_000]

    def _report(self):
        if not self.latest:
            return
        tracking = sum(status == RelativePeerState.STATUS_TRACKING
                       for _, status, _ in self.latest.values())
        current = [error for error, _, _ in self.latest.values()]
        historical = np.asarray(self.errors, float)
        p95 = float(np.percentile(historical, 95)) if historical.size else math.nan
        self.get_logger().info(
            f"relative-localization truth: tracking={tracking}/{len(self.latest)} "
            f"current_max={max(current):.3f}m tracking_p95={p95:.3f}m")


def main(args=None):
    rclpy.init(args=args)
    node = RelativeTruthMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
