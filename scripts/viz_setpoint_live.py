#!/usr/bin/env python3
"""Per-drone commanded-setpoint overlay, as visualization_msgs/MarkerArray.

For each drone draws a thin rod from its actual position to where its controller
is commanding it right now (the TrajectorySetpoint it publishes to PX4) -- shows
tracking error / intent per drone. Published on `gz_markers`; the C++
`ros_gz_marker_bridge` relays it into the Gazebo GUI (and RViz2). A drone's rod
is dropped when its setpoint goes stale (controller stopped / landed), so it
clears via the marker lifetime.

  source install/setup.bash
  ros2 run ros_gz_marker_bridge marker_bridge &
  python3 scripts/viz_setpoint_live.py
"""
import math

import rclpy
import yaml
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from visualization_msgs.msg import MarkerArray
from px4_msgs.msg import TrajectorySetpoint, VehicleLocalPosition

from viz_markers import COLORS, rod

FLEET = "/home/muku/Projects/MAAV/mission10/ros/bringup/config/fleet.yaml"
NS = "po_setpoint"
RATE_HZ = 30.0
ROD_DIA = 0.03
STALE_S = 0.5


class SetpointLive(Node):
    def __init__(self):
        super().__init__("viz_setpoint_live")
        with open(FLEET) as f:
            fleet = yaml.safe_load(f)
        self.spawn = {}
        for i, v in enumerate(fleet["vehicles"]):
            e, n, _ = (float(x) for x in v["pose"].split(",")[:3])
            self.spawn[i] = (e, n)
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.pos, self.sp, self.sp_t = {}, {}, {}
        for i in self.spawn:
            self.create_subscription(
                VehicleLocalPosition, f"/px4_{i}/fmu/out/vehicle_local_position",
                lambda m, i=i: self._pos_cb(i, m), qos)
            self.create_subscription(
                TrajectorySetpoint, f"/px4_{i}/fmu/in/trajectory_setpoint",
                lambda m, i=i: self._sp_cb(i, m), qos)
        self.pub = self.create_publisher(MarkerArray, "gz_markers", 10)
        self.n_pub = 0
        self.create_timer(1.0 / RATE_HZ, self._tick)
        self.create_timer(2.0, self._status_log)

    def _pos_cb(self, i, m):
        if math.isfinite(m.x) and math.isfinite(m.y):
            e, n = self.spawn[i]
            self.pos[i] = (e + m.y, n + m.x, -m.z)  # NED(local) -> world ENU

    def _sp_cb(self, i, m):
        p = m.position
        if all(math.isfinite(x) for x in p):
            e, n = self.spawn[i]
            self.sp[i] = (e + p[1], n + p[0], -p[2])  # NED -> world ENU
            self.sp_t[i] = self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        life = Duration(sec=0, nanosec=300_000_000)
        markers = []
        for i in self.spawn:
            if i not in self.pos or i not in self.sp:
                continue
            if now - self.sp_t.get(i, 0.0) > STALE_S:  # controller stopped -> let it expire
                continue
            markers.append(rod(NS, i + 1, self.pos[i], self.sp[i],
                               COLORS[i % 4], dia=ROD_DIA, life=life))
        if markers:
            self.pub.publish(MarkerArray(markers=markers))
            self.n_pub += 1

    def _status_log(self):
        live = [i for i in self.spawn
                if i in self.pos and i in self.sp
                and self.get_clock().now().nanoseconds * 1e-9 - self.sp_t.get(i, 0.0) <= STALE_S]
        self.get_logger().info(f"{self.n_pub / 2.0:.0f} Hz | active rods: {live}")
        self.n_pub = 0


def main():
    rclpy.init()
    node = SetpointLive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
