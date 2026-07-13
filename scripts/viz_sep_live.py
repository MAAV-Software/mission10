#!/usr/bin/env python3
"""Live min-separation overlay, published as visualization_msgs/MarkerArray.

Reconstructs each drone's WORLD position (spawn offset + launch-relative local
NED, same source as /tmp/sep_monitor.py) and publishes a rod between the current
closest pair plus the distance as text, on `gz_markers`. The C++
`ros_gz_marker_bridge` relays it into the Gazebo GUI at the publish rate (no gz
service CLI cost), and the same topic renders in RViz2. Colour ramps red
(<=danger) -> green (>=safe).

  source install/setup.bash
  ros2 run ros_gz_marker_bridge marker_bridge &     # the relay
  python3 scripts/viz_sep_live.py                    # this publisher
"""
import math

import rclpy
import yaml
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from visualization_msgs.msg import MarkerArray
from px4_msgs.msg import VehicleLocalPosition

from viz_markers import rod, text

FLEET = "/home/muku/Projects/MAAV/mission10/ros/bringup/config/fleet.yaml"
NS = "po_live"
RATE_HZ = 30.0
DANGER_M = 1.5
SAFE_M = 3.0


def ramp(d):
    t = max(0.0, min(1.0, (d - DANGER_M) / (SAFE_M - DANGER_M)))
    return (1.0 - t, t, 0.0)  # red -> green


class SepLive(Node):
    def __init__(self):
        super().__init__("viz_sep_live")
        with open(FLEET) as f:
            fleet = yaml.safe_load(f)
        self.spawn = {}
        for i, v in enumerate(fleet["vehicles"]):
            e, n, _ = (float(x) for x in v["pose"].split(",")[:3])
            self.spawn[i] = (e, n)
        sub_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.pos = {}
        for i in self.spawn:
            self.create_subscription(
                VehicleLocalPosition, f"/px4_{i}/fmu/out/vehicle_local_position",
                lambda m, i=i: self._cb(i, m), sub_qos)
        self.pub = self.create_publisher(MarkerArray, "gz_markers", 10)
        self.n_pub = 0
        self.last = None
        self.create_timer(1.0 / RATE_HZ, self._tick)
        self.create_timer(2.0, self._status_log)

    def _cb(self, i, m):
        if math.isfinite(m.x) and math.isfinite(m.y):
            e, n = self.spawn[i]
            self.pos[i] = (e + m.y, n + m.x, -m.z)  # world east, north, alt

    def _tick(self):
        if len(self.pos) < 2:
            return
        ids = sorted(self.pos)
        dmin, worst = math.inf, None
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pa, pb = self.pos[ids[a]], self.pos[ids[b]]
                d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                if d < dmin:
                    dmin, worst = d, (ids[a], ids[b])
        pa, pb = self.pos[worst[0]], self.pos[worst[1]]
        self.last = (dmin, worst)
        c = ramp(dmin)
        life = Duration(sec=0, nanosec=300_000_000)  # auto-clear if publisher stops
        mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2)
        self.pub.publish(MarkerArray(markers=[
            rod(NS, 1, pa, pb, c, dia=0.06, life=life),
            text(NS, 2, mid, c, f"{dmin:.2f} m", life=life),
        ]))
        self.n_pub += 1

    def _status_log(self):
        hz = self.n_pub / 2.0
        self.n_pub = 0
        if self.last is None:
            self.get_logger().info(f"{hz:.0f} Hz | waiting for >=2 drones")
        else:
            d, w = self.last
            self.get_logger().info(f"{hz:.0f} Hz | min_sep={d:.2f}m pair={w}")


def main():
    rclpy.init()
    node = SepLive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
