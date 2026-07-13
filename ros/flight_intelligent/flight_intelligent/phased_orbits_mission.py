"""Phased-orbits mission — one node per drone, launch-relative frame.

Each PX4 instance's EKF local NED frame is anchored at its own spawn point
(verified: drone i reads its own vehicle_local_position as ~0,0 at rest). So if
every drone spawns on its hover spot (3 m apart) and flies the *identical*
launch-relative geometry (circle center 4.6 m downrange, R=4.6), the physical
spawn offsets reconstruct the world pattern. Drones differ only in phase phi_i =
phase0 + phase_step*i. No absolute setpoint is ever computed.

Operator command structure (qualifier rules section 221): two separate commands,
each a single shared instant delivered to all drones.

    /start_mission  (takeoff)        -> base controller arms + climbs; this node
                                        climbs vertically while yawing to the
                                        center bearing, transits to the circle
                                        center, and holds (the staged ready state)
    /begin_orbit    (begin circling) -> spiral out onto the orbit, then the drones
                                        auto-land near their origins

Takeoff goes up-then-over: climb in place while yawing to face the center, then
transit horizontally (no diagonal slide off the pad). The transit happens during
the hover phase, so by `/begin_orbit` every drone is settled at its center; the
ready signal is position-gated (within acceptance of center + altitude).
`/begin_orbit` sets a shared clock tau = now - gate, so all drones phase off one
instant (per-drone takeoff-timing variance would otherwise desync the phases).
Windows over tau (all setpoints in the drone's own ENU frame, one altitude):

    [0, spiral)     spiral insertion from center: r 0->R while spinning to phi_i (+235 deg)
    [.., +orbit)    revolutions in the locked phased orbit
    [.., +return)   staggered peel-off: each drone curls into its center in turn
    then            AUTO.RTL from the center line (parallel south, safe)

The synchronous +235 deg spiral (phased_orbit_insertion, validated ~2.13 m min
sep) keeps the four in phase the whole way out, so none grazes a neighbor parked
at its center; a staggered spiral-out makes this worse (measured ~0.6-2.1 m), so
the peel-off pattern does not transfer to the insertion. The peel-off return
(phased_orbit_peeloff, validated ~3.0 m) collapses the orbit one drone at a time.
Both keep the whole maneuver at a single altitude and never reverse rotational
sense.

A global origin is set once at link-up (set_global_origin over XRCE-DDS): with
only a local (EV) estimate, AUTO.RTL/Land and failsafes need it to exist.

A buffered Voronoi cell safety net runs under the open-loop choreography: each
drone broadcasts its world position and clips its commanded setpoint off the
peers, engaging only when a disturbance pushes a pair together. Mechanism and
sizing live in _bvc_clip.
"""
from __future__ import annotations

import math

import rclpy
from flight_interfaces.msg import UwbState
from std_msgs.msg import Bool

from flight_lib import (
    buffered_voronoi_clip,
    peeloff_duration,
    phased_orbit_insertion,
    phased_orbit_peeloff,
    phased_orbit_setpoint,
)
from px4_offboard.controller import OffboardController, wrap_pi

YAW_ACCEPTANCE_RAD = math.radians(10.0)  # climb-phase yaw alignment gate


def enu_to_ned_setpoint(position_enu, yaw_enu):
    east, north, up = position_enu
    return float(north), float(east), -float(up), wrap_pi(math.pi / 2.0 - float(yaw_enu))


class PhasedOrbitsMission(OffboardController):
    """Phased orbits, two-command: takeoff+hover, then circle and auto-land."""

    def __init__(self):
        super().__init__("phased_orbits_mission")

        self.declare_parameter("drone_index", 0)
        self.declare_parameter("drone_count", 4)
        self.declare_parameter("orbit_center_e_m", 0.0)
        self.declare_parameter("orbit_center_n_m", 4.6)
        self.declare_parameter("orbit_radius_m", 4.6)
        self.declare_parameter("orbit_speed_mps", 2.0)
        self.declare_parameter("orbit_revolutions", 10.0)
        self.declare_parameter("phase_step_deg", 90.0)
        self.declare_parameter("phase0_deg", -90.0)
        self.declare_parameter("orbit_mod_amp", [0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("orbit_mod_phase_deg", [0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("insertion_spin_deg", 235.0)
        self.declare_parameter("to_center_time_s", 3.0)
        self.declare_parameter("spiral_time_s", 10.0)
        self.declare_parameter("peel_lead_in_s", 0.5)
        self.declare_parameter("peel_stagger_s", 3.0)
        self.declare_parameter("peel_duration_s", 4.0)
        self.declare_parameter("peel_spin_deg", 90.0)
        self.declare_parameter("peel_order", "")
        self.declare_parameter("orbit_auto_start", False)
        self.declare_parameter("yaw_mode", "inward")
        self.declare_parameter("bvc_enabled", True)
        self.declare_parameter("bvc_safety_radius_m", 0.75)
        self.declare_parameter("bvc_lookahead_m", 0.3)
        self.declare_parameter("spawn_e_m", 0.0)
        self.declare_parameter("spawn_n_m", 0.0)
        self.declare_parameter("peer_namespaces", [""])
        self.declare_parameter("origin_lat", 42.2658783)
        self.declare_parameter("origin_lon", -83.7487304)
        self.declare_parameter("origin_alt", 0.0)
        # SITL's EV-only estimate has no global anchor, so the node sets one at
        # link-up. A real GPS estimate already has a GPS-derived origin; sending
        # SET_GPS_GLOBAL_ORIGIN then can shove the local frame (mid-air jump), so
        # the real config sets this false and lets GPS provide the origin.
        self.declare_parameter("set_global_origin_on_link", True)
        # The EKF local frame can jump metres at rotor spool-up (GPS multipath
        # under the M-Air net: 7.5 m XY reset observed on flight 010024). Anchor
        # the geometry to the position *believed after the climb* instead of the
        # origin: climb with zero horizontal correction (setpoint tracks the
        # estimate, so a mid-climb reset is never fought), then freeze that
        # belief and fly all subsequent geometry relative to it.
        self.declare_parameter("anchor_frame_after_climb", True)

        self.index = int(self.get_parameter("drone_index").value)
        self.count = int(self.get_parameter("drone_count").value)
        self.center = (
            float(self.get_parameter("orbit_center_e_m").value),
            float(self.get_parameter("orbit_center_n_m").value),
        )
        self.radius = float(self.get_parameter("orbit_radius_m").value)
        self.speed_mps = float(self.get_parameter("orbit_speed_mps").value)
        self.revolutions = float(self.get_parameter("orbit_revolutions").value)
        self.phase_step = math.radians(float(self.get_parameter("phase_step_deg").value))
        self.phase0 = math.radians(float(self.get_parameter("phase0_deg").value))
        self.mod_amp, self.mod_phase = self._mod_for_index(
            list(self.get_parameter("orbit_mod_amp").value),
            list(self.get_parameter("orbit_mod_phase_deg").value))
        self.spin = math.radians(float(self.get_parameter("insertion_spin_deg").value))
        self.to_center_time_s = float(self.get_parameter("to_center_time_s").value)
        self.spiral_time_s = float(self.get_parameter("spiral_time_s").value)
        self.peel_lead_in_s = float(self.get_parameter("peel_lead_in_s").value)
        self.peel_stagger_s = float(self.get_parameter("peel_stagger_s").value)
        self.peel_duration_s = float(self.get_parameter("peel_duration_s").value)
        self.peel_spin = math.radians(float(self.get_parameter("peel_spin_deg").value))
        self.peel_order = self._parse_order(str(self.get_parameter("peel_order").value))
        self.orbit_auto_start = bool(self.get_parameter("orbit_auto_start").value)
        self.yaw_mode = str(self.get_parameter("yaw_mode").value)
        self.bvc_enabled = bool(self.get_parameter("bvc_enabled").value)
        self.bvc_safety_radius = float(self.get_parameter("bvc_safety_radius_m").value)
        self.bvc_lookahead = float(self.get_parameter("bvc_lookahead_m").value)
        self.spawn_e = float(self.get_parameter("spawn_e_m").value)
        self.spawn_n = float(self.get_parameter("spawn_n_m").value)
        self.peer_namespaces = [p for p in self.get_parameter("peer_namespaces").value if p]
        self.origin = (
            float(self.get_parameter("origin_lat").value),
            float(self.get_parameter("origin_lon").value),
            float(self.get_parameter("origin_alt").value),
        )
        self.set_origin_on_link = bool(self.get_parameter("set_global_origin_on_link").value)
        self.anchor_after_climb = bool(self.get_parameter("anchor_frame_after_climb").value)
        self._anchor = (0.0, 0.0)  # ENU (east, north); latched during climb

        self.alt = self.takeoff_altitude_m
        self.omega = self.speed_mps / self.radius
        self.phi = self.phase0 + self.phase_step * self.index

        self.orbit_duration = self.revolutions * 2.0 * math.pi / self.omega
        self.return_duration = peeloff_duration(
            self.count, lead_in=self.peel_lead_in_s, stagger=self.peel_stagger_s,
            peel_duration=self.peel_duration_s)

        self._orbit_gate_us = 0
        self._orbit_logged = False
        self._return_logged = False
        self._climbed = False
        self._transit_us = 0
        self._center_logged = False
        self.create_subscription(Bool, "begin_orbit", self._orbit_gate_cb, 10)

        self._seq = 0
        self.peer_state: dict[int, UwbState] = {}
        self.bvc_mode = UwbState.MODE_NOMINAL
        self.bvc_active_ticks = 0
        self.bvc_total_ticks = 0
        self.bvc_max_divert = 0.0
        self._bvc_last_log_us = 0
        self.state_pub = self.create_publisher(UwbState, f"/{self.ns}/uwb/state", 20)
        for peer in self.peer_namespaces:
            self.create_subscription(UwbState, f"/{peer}/uwb/state", self._peer_state_cb, 20)

    def _parse_order(self, raw: str):
        raw = raw.strip()
        if not raw:
            return None
        return [int(tok) for tok in raw.replace(",", " ").split()]

    def _mod_for_index(self, amps, phases_deg):
        """This drone's (mod_amp, mod_phase) from the per-drone arrays; 0 if unset."""
        if not amps:
            return 0.0, 0.0
        if len(amps) != self.count or len(phases_deg) != self.count:
            raise ValueError(
                f"orbit_mod_amp/phase must have length drone_count={self.count}")
        return float(amps[self.index]), math.radians(float(phases_deg[self.index]))

    def _orbit_gate_cb(self, msg: Bool):
        if msg.data and self._orbit_gate_us == 0:
            self._orbit_gate_us = self._now_us()
            self.get_logger().info("begin_orbit received, starting circle choreography.")

    def _peer_state_cb(self, msg: UwbState):
        self.peer_state[int(msg.vehicle_id)] = msg

    def _world_xy(self):
        """Own position in the shared world ENU frame (spawn offset + local)."""
        return (self.spawn_e + self.y, self.spawn_n + self.x)  # NED y=east, x=north

    def _publish_world_state(self):
        msg = UwbState()
        msg.stamp = self.get_clock().now().to_msg()
        msg.sequence = self._seq
        self._seq += 1
        msg.vehicle_id = self.index
        msg.yaw_rad = float(self.yaw)
        we, wn = self._world_xy()
        msg.position_enu_m = [float(we), float(wn), float(-self.z)]
        msg.velocity_enu_mps = [float(self.vy), float(self.vx), float(-self.vz)]
        msg.mode = self.bvc_mode
        self.state_pub.publish(msg)

    def _bvc_clip(self, pos_enu):
        """Nudge a local-frame ENU setpoint off peers via the buffered Voronoi cell.

        Half the safety radius is each drone's exclusive disc, so a 0.75 m radius
        holds pairs >= 1.5 m apart. The clip applies to a bounded lookahead from
        the current position (own + a short step toward the setpoint), not to the
        absolute setpoint: on a fast orbit the setpoint can be a whole diameter
        from a lagging drone, and clipping that long cross-formation line grazes
        peers' bisectors and yields spurious metre-scale corrections (feedback
        runaway). The lookahead correction is then added back to the setpoint, so
        the clip only nudges the trajectory and leaves it untouched when the
        lookahead is already inside the cell (peers far)."""
        if not self.bvc_enabled or not self.peer_state:
            return pos_enu
        peer_xys = [(float(s.position_enu_m[0]), float(s.position_enu_m[1]))
                    for vid, s in self.peer_state.items() if vid != self.index]
        if not peer_xys:
            return pos_enu
        self.bvc_total_ticks += 1
        own_xy = self._world_xy()
        goal_xy = (self.spawn_e + pos_enu[0], self.spawn_n + pos_enu[1])
        dx, dy = goal_xy[0] - own_xy[0], goal_xy[1] - own_xy[1]
        dist = math.hypot(dx, dy)
        if dist > self.bvc_lookahead:
            look = (own_xy[0] + dx / dist * self.bvc_lookahead,
                    own_xy[1] + dy / dist * self.bvc_lookahead)
        else:
            look = goal_xy
        safe = buffered_voronoi_clip(own_xy, look, peer_xys, self.bvc_safety_radius)
        corr_x, corr_y = float(safe[0]) - look[0], float(safe[1]) - look[1]
        goal_xy = (goal_xy[0] + corr_x, goal_xy[1] + corr_y)  # nudge the setpoint
        diverted = math.hypot(corr_x, corr_y)
        if diverted > 0.02:
            self.bvc_active_ticks += 1
            self.bvc_max_divert = max(self.bvc_max_divert, diverted)
            self.bvc_mode = UwbState.MODE_DECONFLICT
            now = self._now_us()
            if now - self._bvc_last_log_us > 1_000_000:  # throttle 1 Hz, not latched
                self._bvc_last_log_us = now
                nearest = min(math.hypot(p[0] - own_xy[0], p[1] - own_xy[1]) for p in peer_xys)
                self.get_logger().warn(
                    f"BVC clip {diverted:.3f} m | own=({own_xy[0]:.2f},{own_xy[1]:.2f}) "
                    f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}) nearest_peer={nearest:.2f} m "
                    f"(active {self.bvc_active_ticks}/{self.bvc_total_ticks})")
        else:
            self.bvc_mode = UwbState.MODE_NOMINAL
        return (goal_xy[0] - self.spawn_e, goal_xy[1] - self.spawn_n, pos_enu[2])

    def on_link_acquired(self):
        if not self.set_origin_on_link:
            self.get_logger().info(
                "set_global_origin_on_link=false; relying on GPS for the EKF origin")
            return
        lat, lon, alt = self.origin
        self.set_global_origin(lat, lon, alt)
        self.get_logger().info(f"set EKF global origin: {lat:.7f}, {lon:.7f}, {alt:.1f}")

    def on_active_start(self):
        if self.orbit_auto_start and self._orbit_gate_us == 0:  # single-drone smoke test
            self._orbit_gate_us = self._now_us()
        self.get_logger().info(
            f"phased orbits active (hovering): index={self.index}/{self.count} "
            f"phi={math.degrees(self.phi):.0f}deg orbit_dur={self.orbit_duration:.0f}s "
            f"return_dur={self.return_duration:.0f}s"
        )

    def _hold(self, east, north, yaw_enu):
        return (east, north, self.alt), yaw_enu

    def _orbit_kw(self):
        return dict(
            spacing=0.0, downrange=self.center[1], base=(self.center[0], 0.0),
            altitude=self.alt, phase_step=self.phase_step, phase0=self.phase0,
            yaw_mode=self.yaw_mode,
        )

    def _pre_orbit_setpoint(self):
        """Takeoff -> hold-at-center, the staged ready state (pre begin_orbit).

        Climb vertically at the spawn point while yawing to the center bearing,
        then transit horizontally to the circle center and hold. The ready signal
        is position-gated (within acceptance of center + altitude), not timed.
        """
        ce, cn = self.center
        face_center = math.atan2(cn, ce)  # ENU yaw from spawn toward the center
        target_yaw_ned = wrap_pi(math.pi / 2.0 - face_center)

        if not self._climbed:
            if self.anchor_after_climb:
                # re-latch every tick: the setpoint tracks the estimate, so the
                # climb applies no horizontal correction and a frame reset
                # (spool-up multipath) moves the anchor instead of the drone
                self._anchor = (self.y, self.x)  # NED y=east, x=north
            alt_ok = abs(self.z - (-self.alt)) <= self.takeoff_acceptance_m
            yaw_ok = abs(wrap_pi(self.yaw - target_yaw_ned)) <= YAW_ACCEPTANCE_RAD
            if alt_ok and yaw_ok:
                self._climbed = True
                self._transit_us = self._now_us()
                ae, an = self._anchor
                self.get_logger().info(
                    f"climbed + yawed to center bearing, transiting. "
                    f"frame anchor ENU=({ae:+.2f},{an:+.2f})")
            return self._hold(0.0, 0.0, face_center)

        th = (self._now_us() - self._transit_us) / 1_000_000.0
        u = min(1.0, max(0.0, th) / max(1e-3, self.to_center_time_s))
        s = u * u * (3.0 - 2.0 * u)  # smoothstep transit, zero rate at both ends
        ae, an = self._anchor
        horiz = math.hypot(self.x - (an + cn), self.y - (ae + ce))  # NED x=north, y=east
        if u >= 1.0 and horiz <= self.takeoff_acceptance_m and not self._center_logged:
            self._center_logged = True
            self.get_logger().info("at center, holding (ready for orbit).")
        return self._hold(ce * s, cn * s, face_center)

    def compute_setpoint(self):
        self._publish_world_state()
        result = self._nominal_setpoint()
        if result is None:
            return None
        pos_enu, yaw = result
        ae, an = self._anchor  # whole choreography rides on the latched frame
        pos_enu = (pos_enu[0] + ae, pos_enu[1] + an, pos_enu[2])
        pos_enu = self._bvc_clip(pos_enu)
        return enu_to_ned_setpoint(pos_enu, float(yaw))

    def _nominal_setpoint(self):
        """The open-loop choreography setpoint (ENU, local frame) for this tick."""
        if self._orbit_gate_us == 0:
            return self._pre_orbit_setpoint()

        tau = max(0.0, (self._now_us() - self._orbit_gate_us) / 1_000_000.0)

        if tau < self.spiral_time_s:
            s = tau / max(1e-3, self.spiral_time_s)
            pos, yaw = phased_orbit_insertion(
                s, self.index, self.count, self.radius, spin=self.spin, **self._orbit_kw())
            return pos, float(yaw)

        orbit_t = tau - self.spiral_time_s
        if orbit_t < self.orbit_duration:
            if not self._orbit_logged:
                self._orbit_logged = True
                self.get_logger().info("orbit begins (locked phased orbit).")
            pos, yaw = phased_orbit_setpoint(
                orbit_t, self.index, self.count, self.radius, self.omega,
                mod_amp=self.mod_amp, mod_phase=self.mod_phase, **self._orbit_kw())
            return pos, float(yaw)

        rt = orbit_t - self.orbit_duration
        if rt < self.return_duration:
            if not self._return_logged:
                self._return_logged = True
                self.get_logger().info("orbit complete, peeling off to centers.")
            pos, yaw = phased_orbit_peeloff(
                rt, self.index, self.count, self.radius, self.omega,
                peel_order=self.peel_order, lead_in=self.peel_lead_in_s,
                stagger=self.peel_stagger_s, peel_duration=self.peel_duration_s,
                spin=self.peel_spin, **self._orbit_kw())
            return pos, float(yaw)

        pct = 100.0 * self.bvc_active_ticks / max(1, self.bvc_total_ticks)
        self.get_logger().info(
            f"at center, commanding RTL. BVC active {self.bvc_active_ticks}/{self.bvc_total_ticks} "
            f"ticks ({pct:.1f}%), max divert {self.bvc_max_divert:.3f} m.")
        self.begin_return()
        return None


def main(args=None):
    rclpy.init(args=args)
    node = PhasedOrbitsMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
