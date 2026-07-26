#!/usr/bin/env python3
"""Fleet separation, formation quality, and active-ORCA command metrics.

The instrument for the peel-off. `test_peeloff_separation_beats_retrace`
(flight_lib) puts the analytic floor at 2.8 m, but that is the trajectory
generator on its own. This measures what the vehicles actually do, with PX4's
tracking error and active ORCA in the loop.

Two things bite anyone who writes this ad hoc:

  - `/{ns}/ground_truth/odometry` is launch-relative -- world pose minus the
    drone's own spawn offset (sim_truth_ev/world_truth_to_odom.py). The spawn
    offsets go back on here. Skip that and every pair reads short by its spawn
    separation, which fabricates breaches that never happened.
  - The publisher is BEST_EFFORT. A RELIABLE subscription matches nothing and
    this sits silent forever, looking like a fleet that never moved.

Separation is horizontal. The orbit and the peel-off are planar, so the 2.8 m
figure is a horizontal one, and horizontal <= 3D always -- it is both the
comparable number and the conservative one. Altitudes are printed alongside to
show off-plane motion.

The 2.8 m quality floor begins only after the required 3 m line has held for two
seconds.  Before that, acquisition/staging use a 1.0 m hard physical-safety
floor.  AvoidanceDecision samples provide command acceleration/jerk and ORCA
activity without adding another control path.

Usage: python3 scripts/sep_monitor.py [--fleet <path>] [--floor 2.8]
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
import signal
import sys

from flight_interfaces.msg import AvoidanceDecision
import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FLEET = REPO / "ros" / "bringup" / "config" / "fleet.yaml"
REPORT_HZ = 2.0
HARD_FLOOR_M = 1.0
FORMATION_SPACING_M = 3.0
FORMATION_CROSSTRACK_M = 0.25
FORMATION_SPACING_ERROR_M = 0.20
FORMATION_DWELL_S = 2.0
FORMATION_HOLD_S = 10.0
FORMATION_MAX_ORCA_ACTIVE_PCT = 5.0


def percentile(values, pct):
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(pct) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def read_fleet(path: Path):
    """Give [(namespace, spawn_east, spawn_north)] from a fleet config.

    `pose` is the gz spawn string "x,y,z,roll,pitch,yaw" -- x is east, y is
    north, matching the ENU the odometry is expressed in.
    """
    fleet = yaml.safe_load(path.read_text())
    out = []
    for i, vehicle in enumerate(fleet["vehicles"]):
        namespace = vehicle.get("namespace", f"px4_{i}")
        pose = [float(v) for v in str(vehicle.get("pose", "0,0,0,0,0,0")).split(",")]
        out.append((namespace, pose[0], pose[1]))
    return out


class SeparationMonitor(Node):
    def __init__(self, fleet, floor_m: float):
        super().__init__("sep_monitor")
        self.names = [ns for ns, _, _ in fleet]
        self.spawn = {ns: (e, n) for ns, e, n in fleet}
        self.floor_m = floor_m
        self.world: dict[str, tuple[float, float, float]] = {}
        self.worst = math.inf
        self.worst_pair = None
        self.hard_breaches = 0
        self.formation_breaches = 0
        self.formation_worst = math.inf
        self.formation_candidate_ns = 0
        self.formation_ns = 0
        self.evaluation_complete_ns = 0
        self.airborne_ns = 0
        self.command_state = {}
        self.accelerations = []
        self.jerks = []
        self.avoidance_samples = 0
        self.decision_samples = 0
        self.formation_avoidance_samples = 0
        self.formation_decision_samples = 0
        self.start_ns = self.get_clock().now().nanoseconds

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        for ns in self.names:
            self.create_subscription(
                Odometry,
                f"/{ns}/ground_truth/odometry",
                lambda msg, ns=ns: self._odom_cb(ns, msg),
                qos,
            )
            self.create_subscription(
                AvoidanceDecision,
                f"/{ns}/avoidance/active",
                lambda msg, ns=ns: self._decision_cb(ns, msg),
                50,
            )
        self.create_timer(1.0 / REPORT_HZ, self._report)
        print(f"sep_monitor up: {len(self.names)} drones, floor {floor_m:.2f} m", flush=True)

    def _odom_cb(self, ns: str, msg: Odometry):
        spawn_e, spawn_n = self.spawn[ns]
        p = msg.pose.pose.position
        self.world[ns] = (p.x + spawn_e, p.y + spawn_n, p.z)

    def _decision_cb(self, ns: str, msg: AvoidanceDecision):
        if self.evaluation_complete_ns:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        safe = tuple(float(value) for value in msg.safe_velocity_enu_mps)
        nominal = tuple(float(value) for value in msg.nominal_velocity_enu_mps)
        active = math.dist(safe, nominal) > 0.02
        self.decision_samples += 1
        if active:
            self.avoidance_samples += 1
        if self.formation_ns:
            self.formation_decision_samples += 1
            if active:
                self.formation_avoidance_samples += 1

        previous = self.command_state.get(ns)
        if previous is not None:
            previous_t, previous_v, previous_a = previous
            dt = now - previous_t
            if 0.005 <= dt <= 0.2:
                accel_vector = tuple((safe[i] - previous_v[i]) / dt for i in range(2))
                acceleration = math.hypot(*accel_vector)
                self.accelerations.append(acceleration)
                if previous_a is not None:
                    self.jerks.append(math.dist(accel_vector, previous_a) / dt)
                self.command_state[ns] = (now, safe, accel_vector)
                return
        self.command_state[ns] = (now, safe, None)

    def _formation_error(self):
        east = sorted(point[0] for point in self.world.values())
        north = sorted(point[1] for point in self.world.values())
        line_east = 0.5 * (east[(len(east) - 1) // 2] + east[len(east) // 2])
        cross_track = max(abs(point[0] - line_east) for point in self.world.values())
        spacing_error = max(
            abs((north[index + 1] - north[index]) - FORMATION_SPACING_M)
            for index in range(len(north) - 1)
        )
        return cross_track, spacing_error

    def _report(self):
        if len(self.world) < 2:
            return
        t = (self.get_clock().now().nanoseconds - self.start_ns) / 1e9
        now_ns = self.get_clock().now().nanoseconds

        closest, pair = math.inf, None
        for a, b in itertools.combinations(sorted(self.world), 2):
            (ax, ay, _), (bx, by, _) = self.world[a], self.world[b]
            d = math.hypot(ax - bx, ay - by)
            if d < closest:
                closest, pair = d, (a, b)

        alts = ",".join(f"{self.world[ns][2]:.1f}" for ns in sorted(self.world))
        if not self.airborne_ns and all(point[2] >= 5.0 for point in self.world.values()):
            self.airborne_ns = now_ns

        cross_track, spacing_error = self._formation_error()
        formation_now = (
            cross_track <= FORMATION_CROSSTRACK_M
            and spacing_error <= FORMATION_SPACING_ERROR_M
        )
        if not self.evaluation_complete_ns:
            if formation_now:
                if not self.formation_candidate_ns:
                    self.formation_candidate_ns = now_ns
                dwell = (now_ns - self.formation_candidate_ns) / 1e9
                if not self.formation_ns and dwell >= FORMATION_DWELL_S:
                    self.formation_ns = now_ns
                    self.formation_worst = math.inf
                    self.formation_breaches = 0
                    self.formation_avoidance_samples = 0
                    self.formation_decision_samples = 0
                    elapsed = (
                        (now_ns - self.airborne_ns) / 1e9
                        if self.airborne_ns else t
                    )
                    print(
                        f"FORMATION established t={elapsed:.1f}s "
                        f"cross_track={cross_track:.2f}m "
                        f"spacing_error={spacing_error:.2f}m",
                        flush=True,
                    )
                if self.formation_ns and dwell >= (
                        FORMATION_DWELL_S + FORMATION_HOLD_S):
                    self.evaluation_complete_ns = now_ns
                    print("EVALUATION complete: continuous formation hold passed", flush=True)
            else:
                if self.formation_ns:
                    print(
                        f"FORMATION lost cross_track={cross_track:.2f}m "
                        f"spacing_error={spacing_error:.2f}m; restarting dwell",
                        flush=True,
                    )
                self.formation_candidate_ns = 0
                self.formation_ns = 0
                self.formation_worst = math.inf
                self.formation_breaches = 0
                self.formation_avoidance_samples = 0
                self.formation_decision_samples = 0

        if self.evaluation_complete_ns:
            tag = ""
        elif closest < HARD_FLOOR_M:
            tag = "HARD BREACH "
            self.hard_breaches += 1
        elif self.formation_ns and closest < self.floor_m:
            tag = "FORMATION BREACH "
            self.formation_breaches += 1
        else:
            tag = ""
        print(
            f"{tag}sep t={t:7.1f} min={closest:5.2f} m "
            f"pair=({pair[0]},{pair[1]}) line=({cross_track:.2f},"
            f"{spacing_error:.2f}) alt=[{alts}]",
            flush=True,
        )
        if self.formation_ns:
            self.formation_worst = min(self.formation_worst, closest)
        if closest < self.worst:
            self.worst, self.worst_pair = closest, pair
            print(f"NEW MIN {closest:.2f} m pair=({pair[0]},{pair[1]}) t={t:.1f}", flush=True)

    def summary(self):
        if self.worst_pair is None:
            print("SUMMARY no odometry received -- check that the fleet is up", flush=True)
            return
        now_ns = self.get_clock().now().nanoseconds
        formation_end_ns = self.evaluation_complete_ns or now_ns
        formation_hold = (
            (formation_end_ns - self.formation_ns) / 1e9 if self.formation_ns else 0.0
        )
        active_pct = 100.0 * self.avoidance_samples / max(1, self.decision_samples)
        formation_active_pct = (
            100.0 * self.formation_avoidance_samples
            / max(1, self.formation_decision_samples)
        )
        passed = (
            self.hard_breaches == 0
            and self.formation_breaches == 0
            and self.formation_ns != 0
            and self.evaluation_complete_ns != 0
            and formation_active_pct <= FORMATION_MAX_ORCA_ACTIVE_PCT
        )
        print(
            f"SUMMARY worst={self.worst:.2f} m pair=({self.worst_pair[0]},"
            f"{self.worst_pair[1]}) hard_breaches={self.hard_breaches} "
            f"formation_worst={self.formation_worst:.2f}m "
            f"formation_breaches={self.formation_breaches} "
            f"formation_hold={formation_hold:.1f}s "
            f"accel_p95={percentile(self.accelerations, 95):.2f}m/s2 "
            f"accel_max={max(self.accelerations, default=math.nan):.2f}m/s2 "
            f"jerk_p95={percentile(self.jerks, 95):.2f}m/s3 "
            f"orca_active={active_pct:.1f}% "
            f"formation_orca_active={formation_active_pct:.1f}% "
            f"-> {'PASS' if passed else 'FAIL'}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", type=Path, default=DEFAULT_FLEET)
    parser.add_argument("--floor", type=float, default=2.8)
    args = parser.parse_args()

    rclpy.init()
    node = SeparationMonitor(read_fleet(args.fleet), args.floor)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.summary()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
