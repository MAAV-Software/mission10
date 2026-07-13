"""Split the gz world ground-truth pose feed into per-drone Odometry.

Truth source is `/world/<world>/dynamic_pose/info` (bridged to a TFMessage on
`/uwb/world_poses`). Unlike a per-model OdometryPublisher it works for
runtime-spawned PX4 models (gz-sim#2165), so it is the faithful mocap/QTM stand-in
here. dynamic_pose/info is used in preference to pose/info because it carries only
the moving entities -- the drone models (in world frame) and their links (in
model-relative frame, pinned near the origin) -- with none of the static
ground_plane / world-root / sun poses that sit at (0,0,0) in pose/info and would
collide with a drone parked at the world origin.

ros_gz_bridge drops the gz entity name when it converts Pose_V -> TFMessage, so
transforms can't be keyed by name. Instead each drone is tracked by nearest-match:
seeded at its spawn, every frame we claim the unclaimed transform closest to the
drone's last known world position. Because the drones spawn metres off the origin
and the only other transforms (links) stay pinned near the origin, the nearest
transform to a tracked drone is unambiguously that drone's model pose -- in flight
too, where a per-index scheme breaks (gz omits an unmoving entity for a frame,
shifting every index). A miss (no transform within the gate) leaves the tracker
untouched and publishes nothing that frame, so EV never sees a garbage pose.

Each drone republishes `/{ns}/ground_truth/odometry` with pose only (launch-relative
ENU = world pose minus the spawn offset), matching what the per-model
OdometryPublisher used to emit. gt_to_ev feeds the pose to EKF2 as EV (anchored at the
drone's own spawn), and uwb_range_sim adds the offset back to recover the true world
position for ranging. No velocity is published: the EV airframe (EKF2_EV_CTRL=11) does
not fuse EV velocity -- like a real mocap rig it derives velocity from the IMU.
"""
from __future__ import annotations

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_msgs.msg import TFMessage


class WorldTruthToOdom(Node):
    def __init__(self):
        super().__init__("world_truth_to_odom")
        self.declare_parameter("vehicle_namespaces", ["px4_0", "px4_1"])
        self.declare_parameter("spawn_e_m", [0.0, 3.0])
        self.declare_parameter("spawn_n_m", [0.0, 0.0])
        self.declare_parameter("z_seed_m", 0.18)
        self.declare_parameter("match_gate_m", 1.5)
        self.declare_parameter("origin_clear_m", 0.8)
        self.declare_parameter("world_poses_topic", "/uwb/world_poses")

        self.namespaces = list(self.get_parameter("vehicle_namespaces").value)
        self.spawn_e = [float(v) for v in self.get_parameter("spawn_e_m").value]
        self.spawn_n = [float(v) for v in self.get_parameter("spawn_n_m").value]
        z_seed = float(self.get_parameter("z_seed_m").value)
        self.gate = float(self.get_parameter("match_gate_m").value)
        self.origin_clear = float(self.get_parameter("origin_clear_m").value)
        n = len(self.namespaces)
        if not (len(self.spawn_e) == len(self.spawn_n) == n):
            raise RuntimeError("spawn_e_m / spawn_n_m must match vehicle_namespaces length")

        # tracked world ENU position per drone, seeded at its spawn point
        self.track = [(self.spawn_e[i], self.spawn_n[i], z_seed) for i in range(n)]

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.odom_pubs = [
            self.create_publisher(Odometry, f"/{ns}/ground_truth/odometry", qos)
            for ns in self.namespaces
        ]
        topic = str(self.get_parameter("world_poses_topic").value)
        self.create_subscription(TFMessage, topic, self._poses_cb, qos)
        self.get_logger().info(
            f"world_truth_to_odom up: {topic} -> {[f'/{ns}/ground_truth/odometry' for ns in self.namespaces]}"
        )

    def _poses_cb(self, msg: TFMessage):
        # Model poses are world-frame (off-origin); link poses are model-relative and
        # stay pinned near the raw origin. Keep only the world-frame (off-origin)
        # candidates so a drone can never be matched to a link.
        models = [tf for tf in msg.transforms
                  if math.hypot(tf.transform.translation.x, tf.transform.translation.y) > self.origin_clear]
        claimed = set()
        for i in range(len(self.namespaces)):
            best_j, best_d = None, self.gate
            tx = self.track[i]
            for j, tf in enumerate(models):
                if j in claimed:
                    continue
                t = tf.transform.translation
                d = math.dist((t.x, t.y, t.z), tx)
                if d < best_d:
                    best_j, best_d = j, d
            if best_j is None:
                continue
            claimed.add(best_j)
            self._publish(i, models[best_j])

    def _publish(self, i: int, tf):
        t = tf.transform.translation
        r = tf.transform.rotation
        self.track[i] = (t.x, t.y, t.z)
        rel = (t.x - self.spawn_e[i], t.y - self.spawn_n[i], t.z)

        out = Odometry()
        out.header.stamp = tf.header.stamp
        out.header.frame_id = "odom"
        out.child_frame_id = self.namespaces[i]
        out.pose.pose.position.x = rel[0]
        out.pose.pose.position.y = rel[1]
        out.pose.pose.position.z = rel[2]
        out.pose.pose.orientation = r
        self.odom_pubs[i].publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = WorldTruthToOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
