#!/usr/bin/env python3
"""Passively sample /fmu/out/sensor_combined arrival rate over the uXRCE-DDS
bridge and gate on it.

Does NOT touch FC state -- it only subscribes, so it is safe to run before a
capture. Exits non-zero if the measured rate is below --min. Replaces the old
MAVLink HIGHRES_IMU probe: the IMU now rides the DDS bridge (firmware
>= 8551f635c5 with `rate_limit: 250` on sensor_combined -> ~194 Hz; older fw
caps it at 100 Hz, which is why the gate would catch a stale flash).
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import SensorCombined

ap = argparse.ArgumentParser()
ap.add_argument("--secs", type=float, default=3.0)
ap.add_argument("--min", type=float, default=120.0, help="fail below this Hz")
args = ap.parse_args()

rclpy.init()
node = Node("imu_rate")
qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST)

n = 0


def _cb(_msg):
    global n
    n += 1


node.create_subscription(SensorCombined, "/fmu/out/sensor_combined", _cb, qos)

t0 = time.monotonic()
while time.monotonic() - t0 < args.secs:
    rclpy.spin_once(node, timeout_sec=0.2)
elapsed = time.monotonic() - t0
node.destroy_node()
rclpy.shutdown()

hz = n / elapsed if elapsed else 0.0
print(f"imu_rate: {hz:.1f} Hz ({n} msgs / {elapsed:.1f}s) on /fmu/out/sensor_combined")
if hz < args.min:
    print(f"imu_rate: BELOW MIN {args.min:.0f} Hz -- DDS bridge down, or firmware "
          f"without the sensor_combined rate_limit (100 Hz cap)?", file=sys.stderr)
    sys.exit(1)
