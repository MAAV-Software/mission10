#!/usr/bin/env python3
"""Exercise the phased-orbits gates without using the ROS 2 CLI."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
import time

from flight_interfaces.msg import UwbState
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleStatus
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool
import yaml


def _pose_xy(pose: str) -> tuple[float, float]:
    east, north, *_ = (float(value) for value in pose.split(","))
    return east, north


def _fleet_geometry(path: str, count: int):
    with Path(path).open() as stream:
        fleet = yaml.safe_load(stream)
    vehicles = fleet["vehicles"][:count]
    if len(vehicles) != count:
        raise ValueError(f"fleet has {len(vehicles)} vehicles, need {count}")
    spawns = [_pose_xy(vehicle["pose"]) for vehicle in vehicles]
    slots = [
        _pose_xy(vehicle.get("staging_pose", vehicle["pose"]))
        for vehicle in vehicles
    ]
    return spawns, slots


class GateTest(Node):
    def __init__(self, count: int, spawns: list[tuple[float, float]]):
        super().__init__("phased_orbits_gate_test")
        self.count = count
        self.spawns = spawns
        self.states: dict[int, UwbState] = {}
        self.statuses: dict[int, VehicleStatus] = {}
        self.truth: dict[int, tuple[float, float, float]] = {}
        self.minimum_separation = math.inf
        self.minimum_orbit_separation = math.inf
        self.minimum_pair = None
        self.minimum_orbit_pair = None
        self.hard_breach = False
        self.extent = [math.inf, -math.inf, math.inf, -math.inf]

        gate_qos = QoSProfile(depth=1)
        gate_qos.reliability = ReliabilityPolicy.RELIABLE
        gate_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.gates = {
            name: self.create_publisher(Bool, name, gate_qos)
            for name in ("start_mission", "begin_orbit", "end_mission", "abort_mission")
        }
        for index in range(count):
            status_qos = QoSProfile(depth=10)
            status_qos.reliability = ReliabilityPolicy.BEST_EFFORT
            for suffix in ("vehicle_status", "vehicle_status_v1", "vehicle_status_v4"):
                self.create_subscription(
                    VehicleStatus,
                    f"/px4_{index}/fmu/out/{suffix}",
                    lambda msg, index=index: self._status_cb(index, msg),
                    status_qos,
                )
            self.create_subscription(
                UwbState,
                f"/px4_{index}/uwb/state",
                self._state_cb,
                20,
            )
            truth_qos = QoSProfile(depth=10)
            truth_qos.reliability = ReliabilityPolicy.BEST_EFFORT
            self.create_subscription(
                Odometry,
                f"/px4_{index}/ground_truth/odometry",
                lambda msg, index=index: self._truth_cb(index, msg),
                truth_qos,
            )

    def _state_cb(self, msg: UwbState):
        self.states[int(msg.vehicle_id)] = msg

    def _status_cb(self, index: int, msg: VehicleStatus):
        self.statuses[index] = msg

    def _truth_cb(self, index: int, msg: Odometry):
        point = msg.pose.pose.position
        spawn_east, spawn_north = self.spawns[index]
        self.truth[index] = (
            float(point.x) + spawn_east,
            float(point.y) + spawn_north,
            float(point.z),
        )
        if len(self.truth) != self.count:
            return
        positions = self.truth
        self.extent[0] = min(self.extent[0], *(p[0] for p in positions.values()))
        self.extent[1] = max(self.extent[1], *(p[0] for p in positions.values()))
        self.extent[2] = min(self.extent[2], *(p[1] for p in positions.values()))
        self.extent[3] = max(self.extent[3], *(p[1] for p in positions.values()))
        closest, pair = min(
            (math.dist(positions[a][:2], positions[b][:2]), (a, b))
            for a, b in itertools.combinations(sorted(positions), 2)
        )
        if closest < self.minimum_separation:
            self.minimum_separation = closest
            self.minimum_pair = pair
        if closest < 1.0 and not self.hard_breach:
            self.hard_breach = True
            print(
                f"HARD BREACH {closest:.3f} m "
                f"(px4_{pair[0]}, px4_{pair[1]}); firing /abort_mission",
                flush=True,
            )
            self.gates["abort_mission"].publish(Bool(data=True))
        if len(self.states) == self.count and all(
            msg.validity & UwbState.VALID_PHASE for msg in self.states.values()
        ):
            if closest < self.minimum_orbit_separation:
                self.minimum_orbit_separation = closest
                self.minimum_orbit_pair = pair

    def wait_for(self, predicate, timeout_s: float, label: str):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.hard_breach:
                print(f"FAIL {label}", flush=True)
                return False
            if predicate():
                print(f"PASS {label}", flush=True)
                return True
        print(f"FAIL {label}", flush=True)
        return False

    def wait_for_stable(self, predicate, stable_s: float, timeout_s: float, label: str):
        deadline = time.monotonic() + timeout_s
        stable_since = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.hard_breach:
                print(f"FAIL {label}", flush=True)
                return False
            if predicate():
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= stable_s:
                    print(f"PASS {label}", flush=True)
                    return True
            else:
                stable_since = None
        print(f"FAIL {label}", flush=True)
        return False

    def fire(self, name: str):
        publisher = self.gates[name]
        if not self.wait_for(
            lambda: publisher.get_subscription_count() >= self.count,
            30.0,
            f"{name} has {self.count} subscribers",
        ):
            return False
        message = Bool(data=True)
        publisher.publish(message)
        time.sleep(0.5)
        publisher.publish(message)
        print(f"FIRED /{name}", flush=True)
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--fleet", default="/tmp/maav_sitl_effective_fleet.yaml",
        help="effective fleet YAML written by sitl.sh",
    )
    parser.add_argument("--orbit-seconds", type=float, default=30.0)
    parser.add_argument("--center-east", type=float, default=4.6)
    parser.add_argument("--center-north", type=float, default=0.0)
    parser.add_argument("--center-tolerance", type=float, default=0.35)
    parser.add_argument("--altitude", type=float, default=5.4)
    return parser.parse_args()


def main():
    args = parse_args()
    spawns, slots = _fleet_geometry(args.fleet, args.count)
    rclpy.init()
    node = GateTest(args.count, spawns)
    ok = node.wait_for(
        lambda: len(node.statuses) == args.count and all(
            status.pre_flight_checks_pass for status in node.statuses.values()
        ),
        60.0,
        "fleet preflight ready",
    )
    ok = ok and node.fire("start_mission")

    def at_centers():
        if len(node.truth) != args.count:
            return False
        for index, position in node.truth.items():
            slot_east, slot_north = slots[index]
            expected_north = slot_north + args.center_north
            expected_east = slot_east + args.center_east
            east, north, up = position
            if math.hypot(east - expected_east, north - expected_north) > args.center_tolerance:
                return False
            if up < args.altitude:
                return False
        return True

    ok = ok and node.wait_for_stable(
        at_centers, 1.0, 60.0, "fleet settled at orbit-ready centers")
    ok = ok and node.fire("begin_orbit")
    ok = ok and node.wait_for(
        lambda: len(node.states) == args.count and all(
            state.validity & UwbState.VALID_PHASE for state in node.states.values()
        ),
        20.0,
        "all phase states valid",
    )
    if ok:
        end = time.monotonic() + args.orbit_seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        print(f"PASS observed orbit for {args.orbit_seconds:.0f}s", flush=True)

    home_sent = node.fire("end_mission")
    landed = node.wait_for(
        lambda: len(node.truth) == args.count
        and all(
            position[2] < 0.5
            and math.dist(position[:2], spawns[index]) < 0.75
            for index, position in node.truth.items()
        ),
        60.0,
        "fleet returned near its launch points and landed",
    )
    ok = ok and home_sent and landed

    print(
        f"minimum separation: {node.minimum_separation:.3f} m "
        f"(px4_{node.minimum_pair[0]}, px4_{node.minimum_pair[1]})",
        flush=True,
    )
    if node.minimum_orbit_pair is None:
        print("minimum phase-valid separation: unavailable", flush=True)
    else:
        print(
            f"minimum phase-valid separation: {node.minimum_orbit_separation:.3f} m "
            f"(px4_{node.minimum_orbit_pair[0]}, px4_{node.minimum_orbit_pair[1]})",
            flush=True,
        )
    print(
        "observed EN envelope: "
        f"east [{node.extent[0]:.2f}, {node.extent[1]:.2f}] m, "
        f"north [{node.extent[2]:.2f}, {node.extent[3]:.2f}] m",
        flush=True,
    )
    ok = ok and node.minimum_separation >= 1.0
    ok = ok and node.minimum_orbit_pair is not None
    ok = ok and node.minimum_orbit_separation >= 2.8
    print("GATE TEST PASS" if ok else "GATE TEST FAIL", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
