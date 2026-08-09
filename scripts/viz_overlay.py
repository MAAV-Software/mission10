#!/usr/bin/env python3
"""Static phased-orbits geometry overlay, as visualization_msgs/MarkerArray.

Per-drone orbit circle + D-label, insertion path, exit/peel path, and the
takeoff/hover line, published on `gz_markers`. The C++ `ros_gz_marker_bridge`
relays it into the Gazebo GUI; the same topic renders in RViz2. Reuses
flight_lib so the overlay always matches the flown trajectory for a config.
Publishes a few times then exits (markers persist in the GUI; lifetime 0).

  source install/setup.bash
  ros2 run ros_gz_marker_bridge marker_bridge &
  python3 scripts/viz_overlay.py /tmp/phased_orbits_exitA.yaml
  python3 scripts/viz_overlay.py --clear
"""
import argparse
import math

import numpy as np
import rclpy
import yaml
from rclpy.qos import QoSProfile
from visualization_msgs.msg import Marker, MarkerArray
from flight_lib import (
    peeloff_duration,
    phased_orbit_insertion,
    phased_orbit_peeloff,
)
from viz_markers import COLORS, line, text

FLEET = "/home/muku/Projects/MAAV/mission10/ros/bringup/config/fleet.yaml"
NS = "phased_orbits"
LINE_W = (1.0, 1.0, 1.0)


def build(config):
    with open(config) as f:
        d = yaml.safe_load(f)["/**"]["ros__parameters"]
    with open(FLEET) as f:
        fleet = yaml.safe_load(f)
    spawn = [tuple(float(x) for x in v["pose"].split(",")[:3]) for v in fleet["vehicles"]]
    n = int(d["drone_count"])
    R = float(d["orbit_radius_m"])
    ce, cn = float(d["orbit_center_e_m"]), float(d["orbit_center_n_m"])
    alt = float(d["takeoff_altitude_m"])
    omega = float(d["orbit_speed_mps"]) / R
    phase0 = math.radians(float(d["phase0_deg"]))
    step = math.radians(float(d["phase_step_deg"]))

    # Each mission node generates the same launch-relative trajectory; the
    # fleet's two-dimensional spawn offsets place it in the shared world.
    kw = dict(spacing=0.0, downrange=cn, base=(ce, 0.0), altitude=alt)
    pkw = dict(phase_step=step, phase0=phase0)

    ins_spin1 = math.radians(float(d.get("insertion_spin_deg", 235.0)))

    centers = [(spawn[i][0] + ce, spawn[i][1] + cn) for i in range(n)]
    out, mid = [], 1  # ids start at 1 (gz treats id 0 as auto-assign)
    for i in range(n):
        c = COLORS[i % 4]
        cx, cy = centers[i]
        ring = [(cx + R * math.cos(a), cy + R * math.sin(a), alt)
                for a in np.linspace(0, 2 * math.pi, 73)]
        out.append(line(NS, mid, ring, c)); mid += 1
        out.append(text(NS, mid, (cx, cy, alt), c, "D%d" % i)); mid += 1

        # entry/exit curves disabled — circles only
        # ins = [phased_orbit_insertion(s, i, n, R, spin=ins_spin1, **kw, **pkw)[0]
        #        for s in np.linspace(0, 1, 60)]
        # out.append(line(NS, mid, ins, c)); mid += 1

        # lead = float(d.get("peel_lead_in_s", 0.5))
        # stag = float(d.get("peel_stagger_s", 3.0))
        # pdur = float(d.get("peel_duration_s", 4.0))
        # pspin = math.radians(float(d.get("peel_spin_deg", 90.0)))
        # order = [int(x) for x in str(d.get("peel_order", "")).split(",") if x.strip()] or None
        # T = peeloff_duration(n, lead_in=lead, stagger=stag, peel_duration=pdur)
        # ex = [phased_orbit_peeloff(t, i, n, R, omega, peel_order=order, lead_in=lead,
        #       stagger=stag, peel_duration=pdur, spin=pspin, **kw, **pkw)[0]
        #       for t in np.linspace(0, T, 80)]
        # out.append(line(NS, mid, ex, c)); mid += 1

    ground = [(spawn[i][0], spawn[i][1], 0.05) for i in range(n)]
    out.append(line(NS, mid, ground, LINE_W)); mid += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("viz_overlay")
    pub = node.create_publisher(MarkerArray, "gz_markers", QoSProfile(depth=1))

    if args.clear:
        m = Marker(); m.ns = NS; m.action = Marker.DELETEALL
        arr = MarkerArray(markers=[m])
    else:
        if not args.config:
            ap.error("config required (or --clear)")
        arr = MarkerArray(markers=build(args.config))

    for _ in range(5):
        pub.publish(arr)
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info(
        "cleared" if args.clear else f"published {len(arr.markers)} markers")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
