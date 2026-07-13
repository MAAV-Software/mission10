"""UWB range sensor: pairwise distance from gz ground truth plus noise/dropout.

Mirrors a real UWB radio (e.g. the reference Eliko/uwb_gazebo models): each
measurement is a scalar range between two nodes, derived from ground-truth
geometry, never carrying peer telemetry (phase/yaw/position ride UwbState).

Truth source is each drone's ground_truth/odometry feed (bridged from gz by
the sim_truth_ev nodes) — the same ground truth that stands in for mocap/EV.
That odometry is launch-relative, so spawn offsets reconstruct true world
positions; the range is independent of any EKF estimate.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from flight_interfaces.msg import UwbRange
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


class UwbRangeSim(Node):
    def __init__(self):
        super().__init__("uwb_range_sim")
        self.declare_parameter("vehicle_namespaces", ["px4_0", "px4_1"])
        self.declare_parameter("spawn_e_m", [0.0, 3.0])
        self.declare_parameter("spawn_n_m", [0.0, 0.0])
        self.declare_parameter("odom_topic", "ground_truth/odometry")
        self.declare_parameter("noise_std_m", 0.1)
        self.declare_parameter("outlier_probability", 0.0)
        self.declare_parameter("outlier_std_m", 0.5)
        self.declare_parameter("dropout_probability", 0.0)
        self.declare_parameter("far_rate_hz", 10.0)
        self.declare_parameter("near_rate_hz", 50.0)
        self.declare_parameter("near_range_m", 3.0)
        self.declare_parameter("seed", 7)

        self.namespaces = list(self.get_parameter("vehicle_namespaces").value)
        self.spawn_e = [float(v) for v in self.get_parameter("spawn_e_m").value]
        self.spawn_n = [float(v) for v in self.get_parameter("spawn_n_m").value]
        if len(self.spawn_e) != len(self.namespaces) or len(self.spawn_n) != len(self.namespaces):
            raise RuntimeError("spawn_e_m / spawn_n_m must match vehicle_namespaces length")
        self.noise_std = float(self.get_parameter("noise_std_m").value)
        self.outlier_probability = float(self.get_parameter("outlier_probability").value)
        self.outlier_std = float(self.get_parameter("outlier_std_m").value)
        self.dropout_probability = float(self.get_parameter("dropout_probability").value)
        self.near_range = float(self.get_parameter("near_range_m").value)
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.odom: dict[int, Odometry] = {}
        self.sequence = 0
        self._far_divisor = max(1, round(float(self.get_parameter("near_rate_hz").value) /
                                         float(self.get_parameter("far_rate_hz").value)))
        self.range_pubs = {
            i: self.create_publisher(UwbRange, f"/{ns}/uwb/range", 20)
            for i, ns in enumerate(self.namespaces)
        }
        # sim_truth_ev publishes ground-truth odometry BEST_EFFORT; a
        # reliable subscriber is silently incompatible and would receive nothing.
        odom_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                              history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        odom_topic = str(self.get_parameter("odom_topic").value)
        for i, ns in enumerate(self.namespaces):
            self.create_subscription(Odometry, f"/{ns}/{odom_topic}",
                                     lambda msg, i=i: self.odom.__setitem__(i, msg), odom_qos)
        near_rate = float(self.get_parameter("near_rate_hz").value)
        self.tick = 0
        self.create_timer(1.0 / near_rate, self._publish)

    def _world_pos(self, i: int):
        """True world position: launch-relative gz odometry plus spawn offset (ENU)."""
        if i not in self.odom:
            return None
        p = self.odom[i].pose.pose.position
        return (self.spawn_e[i] + p.x, self.spawn_n[i] + p.y, p.z)

    def _publish(self):
        self.tick += 1
        pos = {i: self._world_pos(i) for i in range(len(self.namespaces))}
        for receiver in range(len(self.namespaces)):
            for source in range(len(self.namespaces)):
                if receiver == source or pos[source] is None or pos[receiver] is None:
                    continue
                distance = math.dist(pos[source], pos[receiver])
                if distance >= self.near_range and self.tick % self._far_divisor:
                    continue
                if self.rng.random() < self.dropout_probability:
                    continue
                msg = UwbRange()
                msg.stamp = self.get_clock().now().to_msg()
                msg.sequence = self.sequence
                msg.source_id = source
                msg.receiver_id = receiver
                sigma = self.outlier_std if self.rng.random() < self.outlier_probability else self.noise_std
                msg.range_m = max(0.0, distance + float(self.rng.normal(0.0, sigma)))
                self.range_pubs[receiver].publish(msg)
                self.sequence += 1


def main(args=None):
    rclpy.init(args=args)
    node = UwbRangeSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
