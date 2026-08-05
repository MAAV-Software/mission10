"""Phased-orbits mission — one node per drone, launch-relative frame.

Each PX4 instance's EKF local NED frame is anchored at its own spawn point
(verified: drone i reads its own vehicle_local_position as ~0,0 at rest). So if
every drone spawns on its hover spot (3 m apart) and flies the *identical*
launch-relative geometry (circle center 4.6 m downrange, R=4.6), the physical
spawn offsets reconstruct the world pattern. Drones differ only in phase phi_i =
phase0 + phase_step*i. No absolute setpoint is ever computed.

Operator commands (qualifier rules section 221), each a single shared instant
delivered to all drones.

    /start_mission  (takeoff)   -> base controller arms + climbs; this node climbs
                                   vertically while yawing to the center bearing,
                                   transits to the circle center, and holds (the
                                   staged ready state)
    /begin_orbit    (execute)   -> spiral out onto the orbit and circle there
    /end_mission    (come home) -> peel off, return to the launch anchor, settle,
                                   then AUTO.LAND
    /abort_mission  (abort)     -> AUTO.LAND in place, now (base controller)

The orbit has no timed end; it circles until /end_mission. on_return_home then
ends it on the next whole revolution, which is the only instant at which the
peel-off joins the orbit without a position step. The backstop against a lost
operator is the flight controller's own battery/RC/datalink failsafes and the
geofence, not a timer here.

Takeoff goes up-then-over: climb in place while yawing to face the center. In
random-spawn SITL, each drone then converges through ORCA onto its fixed 3 m line
slot before transiting horizontally to its circle center (no diagonal slide off
the pad). By `/begin_orbit` every drone is settled at its center; the ready
signal is position-gated (within acceptance of center + altitude).
`/begin_orbit` sets a shared clock tau = now - gate, so all drones phase off one
instant (per-drone takeoff-timing variance would otherwise desync the phases).
Windows over tau (all setpoints in the drone's own ENU frame, one altitude):

    [0, spiral)     spiral insertion from center: r 0->R while spinning to phi_i (+235 deg)
    [.., +orbit)    the locked phased orbit; unbounded until /end_mission fixes it
                    at a whole number of revolutions
    [.., +return)   staggered peel-off: each drone curls into its center in turn
    then            local Offboard return to launch, settle, then AUTO.LAND

The synchronous +235 deg spiral (phased_orbit_insertion, validated ~2.13 m min
sep) keeps the four in phase the whole way out, so none grazes a neighbor parked
at its center; a staggered spiral-out makes this worse (measured ~0.6-2.1 m), so
the peel-off pattern does not transfer to the insertion. The peel-off return
(phased_orbit_peeloff, validated ~3.0 m) collapses the orbit one drone at a time.
Both keep the whole maneuver at a single altitude and never reverse rotational
sense.

A global origin can be set once at link-up for simulations and global failsafes,
but the normal landing path stays local and does not require GNSS altitude.

Range-history localization and ORCA own the horizontal command whenever peers
are present.  Each peer's protected radius expands with localization uncertainty
and closing speed.  The mission's position path is converted to a preferred
velocity before avoidance, so the PX4 position loop cannot add a hidden
horizontal correction around the planner.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
try:
    from flight_interfaces.msg import AvoidanceDecision, RelativePeerState, UwbState
except ImportError:  # build-free, single-aircraft deployment on the companion
    AvoidanceDecision = None
    RelativePeerState = None
    UwbState = None
from std_msgs.msg import Bool

from flight_lib import (
    OrcaPeer,
    STATUS_TRACKING,
    closest_point_of_approach,
    orca_effective_radius,
    orca_velocity,
    peeloff_duration,
    phased_orbit_insertion,
    phased_orbit_peeloff,
    phased_orbit_setpoint,
)
from px4_offboard.controller import OffboardController, wrap_pi
from px4_offboard.gate_qos import MISSION_GATE_QOS

YAW_ACCEPTANCE_RAD = math.radians(10.0)  # climb-phase yaw alignment gate
MODE_NOMINAL = UwbState.MODE_NOMINAL if UwbState is not None else 0
MODE_DECONFLICT = UwbState.MODE_DECONFLICT if UwbState is not None else 3

# Range-only localization needs two-dimensional relative motion before it can
# supply ORCA with a bearing.  Each drone traces the same small circle with an
# identity-dependent phase, returning to its launch point every cycle.  A 0.25 m
# radius gives adjacent identities a 0.71 m relative baseline, comfortably over
# the estimator's 0.5 m acquisition threshold while each drone stays within
# 0.5 m of its random spawn.
LOCALIZATION_RADIUS_M = 0.25
LOCALIZATION_CYCLE_S = 6.0


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
        self.declare_parameter("excite_enabled", False)
        self.declare_parameter("pre_excite_hover_s", 0.0)
        self.declare_parameter("excite_amp_m", 1.0)
        self.declare_parameter("excite_freq_hz", 0.4)
        self.declare_parameter("excite_cycles", 4.0)
        self.declare_parameter("land_xy_acceptance_m", 0.35)
        self.declare_parameter("land_vxy_acceptance_mps", 0.20)
        self.declare_parameter("land_settle_time_s", 1.0)
        self.declare_parameter("orca_protected_radius_m", 1.25)
        self.declare_parameter("orca_response_time_s", 0.6)
        self.declare_parameter("orca_horizon_s", 5.0)
        self.declare_parameter("orca_max_speed_mps", 3.0)
        self.declare_parameter("spawn_e_m", 0.0)
        self.declare_parameter("spawn_n_m", 0.0)
        self.declare_parameter("stage_e_m", 0.0)
        self.declare_parameter("stage_n_m", 0.0)
        self.declare_parameter("staging_enabled", False)
        self.declare_parameter("staging_speed_mps", 2.0)
        self.declare_parameter("staging_acceptance_m", 0.35)
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
        self.excite_enabled = bool(self.get_parameter("excite_enabled").value)
        self.pre_excite_hover_s = max(
            0.0, float(self.get_parameter("pre_excite_hover_s").value)
        )
        self.excite_amp_m = float(self.get_parameter("excite_amp_m").value)
        self.excite_freq_hz = float(self.get_parameter("excite_freq_hz").value)
        self.excite_cycles = float(self.get_parameter("excite_cycles").value)
        self.land_xy_acceptance = float(self.get_parameter("land_xy_acceptance_m").value)
        self.land_vxy_acceptance = float(
            self.get_parameter("land_vxy_acceptance_mps").value
        )
        self.land_settle_time_s = float(self.get_parameter("land_settle_time_s").value)
        if self.excite_enabled and (self.excite_freq_hz <= 0.0 or self.excite_cycles <= 0.0):
            raise ValueError("enabled excitation requires positive frequency and cycle count")
        self.orca_protected_radius = float(
            self.get_parameter("orca_protected_radius_m").value)
        self.orca_response_time = float(
            self.get_parameter("orca_response_time_s").value)
        self.orca_horizon = float(self.get_parameter("orca_horizon_s").value)
        self.orca_max_speed = float(self.get_parameter("orca_max_speed_mps").value)
        if self.orca_protected_radius <= 0.0 or self.orca_response_time < 0.0:
            raise ValueError("ORCA protected radius must be positive and response time nonnegative")
        self.spawn_e = float(self.get_parameter("spawn_e_m").value)
        self.spawn_n = float(self.get_parameter("spawn_n_m").value)
        self.stage_e = float(self.get_parameter("stage_e_m").value)
        self.stage_n = float(self.get_parameter("stage_n_m").value)
        self.staging_enabled = bool(self.get_parameter("staging_enabled").value)
        self.staging_speed = float(self.get_parameter("staging_speed_mps").value)
        self.staging_acceptance = float(self.get_parameter("staging_acceptance_m").value)
        if self.staging_speed <= 0.0:
            raise ValueError("staging_speed_mps must be positive")
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

        # The orbit runs until /end_mission, at which point on_return_home() sets a
        # finite duration on a revolution boundary. Until then `orbit_t < inf`
        # holds and the peel-off branch is unreachable.
        self.orbit_duration = math.inf
        self.return_duration = peeloff_duration(
            self.count, lead_in=self.peel_lead_in_s, stagger=self.peel_stagger_s,
            peel_duration=self.peel_duration_s)

        self._orbit_gate_us = 0
        self._orbit_logged = False
        self._return_logged = False
        self._climbed = False
        self._transit_us = 0
        self._staging_complete = not self.staging_enabled
        self._stage_start_us = 0
        self._localization_complete = not (
            self.staging_enabled and bool(self.peer_namespaces))
        self._localization_start_us = 0
        self._localization_cycle = 0
        self._center_logged = False
        self._pre_excite_hover_us = 0
        self._excite_us = 0
        self._excited = not self.excite_enabled
        self._hold_yaw_enu = None
        self._land_settle_us = 0
        self._land_requested = False
        self._landing_logged = False
        # Set by on_return_home() when /end_mission arrives before the orbit does.
        self._home_us = 0
        self._home_from = None
        self._home_yaw = 0.0
        self._last_nominal = None
        self._last_goal_xy = None
        self._last_goal_us = 0
        self.create_subscription(
            Bool, "begin_orbit", self._orbit_gate_cb, MISSION_GATE_QOS
        )

        self._seq = 0
        self.peer_state: dict = {}
        self.relative_state: dict = {}
        self.avoidance_mode = MODE_NOMINAL
        self.avoidance_active_ticks = 0
        self.avoidance_total_ticks = 0
        self.avoidance_max_delta = 0.0
        self._avoidance_last_log_us = 0
        self.state_pub = None
        self.avoidance_pub = None
        if UwbState is not None:
            state_topic = f"/{self.ns}/uwb/state" if self.ns else "/uwb/state"
            self.state_pub = self.create_publisher(UwbState, state_topic, 20)
            for peer in self.peer_namespaces:
                self.create_subscription(
                    UwbState, f"/{peer}/uwb/state", self._peer_state_cb, 20
                )
            if self.peer_namespaces:
                relative_topic = (
                    f"/{self.ns}/uwb/relative_state" if self.ns
                    else "/uwb/relative_state")
                self.create_subscription(
                    RelativePeerState, relative_topic, self._relative_state_cb, 50)
                decision_topic = (
                    f"/{self.ns}/avoidance/active" if self.ns
                    else "/avoidance/active")
                self.avoidance_pub = self.create_publisher(
                    AvoidanceDecision, decision_topic, 20)
        elif self.peer_namespaces:
            raise RuntimeError(
                "flight_interfaces is unavailable; peers require relative localization"
            )

    def _parse_order(self, raw: str):
        raw = raw.strip()
        if not raw:
            return None
        return [int(tok) for tok in raw.replace(",", " ").split()]

    def _mod_for_index(self, amps, phases_deg):
        """This drone's (mod_amp, mod_phase) from the per-drone arrays; 0 if unset."""
        if not amps:
            return 0.0, 0.0
        if len(amps) < self.count or len(phases_deg) < self.count:
            raise ValueError(
                f"orbit_mod_amp/phase need at least drone_count={self.count} entries")
        return float(amps[self.index]), math.radians(float(phases_deg[self.index]))

    def _orbit_gate_cb(self, msg: Bool):
        # Refused after a come-home: the orbit gate would otherwise latch a clock
        # the home branch already shadows, and the log would claim an orbit that
        # never runs.
        if (msg.data and self._orbit_gate_us == 0 and not self._home_requested
                and not self._center_logged):
            self.get_logger().warn(
                "begin_orbit ignored: fleet has not reached the orbit-ready centers")
            return
        if msg.data and self._orbit_gate_us == 0 and not self._home_requested:
            self._orbit_gate_us = self._now_us()
            self.get_logger().info("begin_orbit received, starting circle choreography.")

    def _peer_state_cb(self, msg: UwbState):
        self.peer_state[int(msg.vehicle_id)] = msg

    def _relative_state_cb(self, msg: RelativePeerState):
        if int(msg.observer_id) == self.index:
            self.relative_state[int(msg.peer_id)] = msg

    def _world_xy(self):
        """Own position in the shared world ENU frame (spawn offset + local)."""
        return (self.spawn_e + self.y, self.spawn_n + self.x)  # NED y=east, x=north

    def _publish_world_state(self):
        if UwbState is None:
            return
        msg = UwbState()
        msg.stamp = self.get_clock().now().to_msg()
        msg.sequence = self._seq
        self._seq += 1
        msg.vehicle_id = self.index
        msg.yaw_rad = float(self.yaw)
        we, wn = self._world_xy()
        msg.position_enu_m = [float(we), float(wn), float(-self.z)]
        msg.velocity_enu_mps = [float(self.vy), float(self.vx), float(-self.vz)]
        msg.mode = self.avoidance_mode
        self.state_pub.publish(msg)

    def _preferred_velocity(self, goal_xy, now_us):
        """Finite-difference path feed-forward plus local position feedback."""
        goal = np.asarray(goal_xy, float)
        feed_forward = np.zeros(2)
        if self._last_goal_xy is not None and now_us > self._last_goal_us:
            dt = (now_us - self._last_goal_us) * 1e-6
            if dt <= 0.25:
                feed_forward = (goal - self._last_goal_xy) / dt
        self._last_goal_xy = goal.copy()
        self._last_goal_us = now_us

        own = np.array([self.y, self.x], float)  # local ENU east, north
        preferred = feed_forward + (goal - own)
        speed = float(np.linalg.norm(preferred))
        if speed > self.orca_max_speed:
            preferred *= self.orca_max_speed / speed
        return preferred

    def _avoidance_velocity(self, goal_xy):
        """Return ORCA's horizontal ENU command.

        The scalar range-rate check remains telemetry only.  It cannot choose an
        escape direction, and overwriting ORCA with zero velocity removes the
        lateral motion that actually resolves a collision.
        """
        now_us = self._now_us()
        preferred = self._preferred_velocity(goal_xy, now_us)
        own_velocity = np.array([self.vy, self.vx], float)
        peers = []
        limiting_peer = 255
        limiting_range = math.inf
        limiting_closing_speed = 0.0
        limiting_effective_radius = math.nan
        limiting_clearance = math.inf
        t_cpa = math.nan
        d_cpa = math.nan
        range_alert = False

        for peer_id, estimate in self.relative_state.items():
            range_m = float(estimate.range_m)
            if int(estimate.status) != STATUS_TRACKING:
                continue
            uncertainty = float(estimate.confidence_radius_95_m)
            range_rate = float(estimate.range_rate_mps)
            if not math.isfinite(uncertainty):
                continue
            closing_speed = (
                max(0.0, min(2.0 * self.orca_max_speed, -range_rate))
                if math.isfinite(range_rate) else 0.0)
            effective_radius = orca_effective_radius(
                self.orca_protected_radius,
                uncertainty,
                range_rate,
                self.orca_response_time,
                2.0 * self.orca_max_speed,
            )
            clearance = range_m - effective_radius
            if clearance < limiting_clearance:
                limiting_clearance = clearance
                limiting_peer = peer_id
                limiting_range = range_m
                limiting_closing_speed = closing_speed
                limiting_effective_radius = effective_radius
                t_cpa, d_cpa = closest_point_of_approach(
                    estimate.position_enu_m, estimate.velocity_enu_mps)
            peers.append(OrcaPeer(
                position=np.asarray(estimate.position_enu_m, float),
                velocity=own_velocity + np.asarray(estimate.velocity_enu_mps, float),
                combined_radius=effective_radius,
            ))

        range_alert = limiting_clearance <= 0.0

        safe = orca_velocity(
            preferred,
            own_velocity,
            peers,
            time_horizon_s=self.orca_horizon,
            max_speed_mps=self.orca_max_speed,
        ) if peers else preferred.copy()

        delta = float(np.linalg.norm(safe - preferred))
        self.avoidance_total_ticks += 1
        active = delta > 0.02
        self.avoidance_mode = MODE_DECONFLICT if active else MODE_NOMINAL
        if active:
            self.avoidance_active_ticks += 1
            self.avoidance_max_delta = max(self.avoidance_max_delta, delta)
            if now_us - self._avoidance_last_log_us > 1_000_000:
                self._avoidance_last_log_us = now_us
                self.get_logger().warn(
                    f"ORCA active: range_alert={range_alert} delta_v={delta:.2f}m/s "
                    f"tracked={len(peers)} limiting={limiting_range:.2f}m/"
                    f"{limiting_effective_radius:.2f}m")

        if self.avoidance_pub is not None:
            msg = AvoidanceDecision()
            msg.stamp = self.get_clock().now().to_msg()
            msg.vehicle_id = self.index
            msg.nominal_velocity_enu_mps = [float(v) for v in preferred]
            msg.safe_velocity_enu_mps = [float(v) for v in safe]
            msg.estimator_usable = bool(peers)
            msg.range_alert = range_alert
            msg.limiting_peer_id = limiting_peer
            msg.limiting_range_m = limiting_range
            msg.limiting_closing_speed_mps = limiting_closing_speed
            msg.limiting_effective_radius_m = limiting_effective_radius
            msg.t_cpa_s = t_cpa
            msg.d_cpa_m = d_cpa
            self.avoidance_pub.publish(msg)
        return safe

    def on_link_acquired(self):
        if not self.set_origin_on_link:
            self.get_logger().info(
                "set_global_origin_on_link=false; relying on GPS for the EKF origin")
            return
        lat, lon, alt = self.origin
        self.set_global_origin(lat, lon, alt)
        self.get_logger().info(f"set EKF global origin: {lat:.7f}, {lon:.7f}, {alt:.1f}")

    def on_active_start(self):
        # "hold" uses the operator-set heading latched immediately before arm,
        # so spool-up/takeoff disturbances are corrected instead of re-latched.
        hold_yaw_ned = self._launch_yaw if self._launch_yaw_latched else self.yaw
        self._hold_yaw_enu = wrap_pi(math.pi / 2.0 - hold_yaw_ned)
        if self.orbit_auto_start and self._orbit_gate_us == 0:  # single-drone smoke test
            self._orbit_gate_us = self._now_us()
        self.get_logger().info(
            f"phased orbits active (hovering): index={self.index}/{self.count} "
            f"phi={math.degrees(self.phi):.0f}deg return_dur={self.return_duration:.0f}s"
        )

    def on_return_home(self):
        """/end_mission: bring this drone home from wherever it is right now."""
        if self._orbit_gate_us:
            # The peel-off restarts the orbit angle at phi and assumes the speed
            # modulation has returned to zero. Both hold only on a whole revolution
            # (phased_orbits.py peel-off docstring, test_peeloff_joins_orbit_at_t0),
            # so end the orbit on the next one rather than wherever the command
            # landed. Starting off-grid steps the commanded position by up to 2R in
            # one tick, which PX4 answers with a dash straight across the formation.
            rev = 2.0 * math.pi / self.omega
            tau = max(0.0, (self._now_us() - self._orbit_gate_us) / 1_000_000.0)
            orbit_t = max(0.0, tau - self.spiral_time_s)
            self.orbit_duration = rev * math.ceil(orbit_t / rev - 1e-9)
            # During the spiral orbit_t is 0, so the duration is 0 too: the
            # insertion finishes and hands straight to the peel-off, which starts
            # at the same point the insertion ends.
            self.get_logger().info(
                f"come home: peeling off in {self.orbit_duration - orbit_t:.1f}s "
                f"(end of revolution {self.orbit_duration / rev:.0f})"
            )
            return

        if not self._climbed:
            # The anchor re-latches to the live position every climb tick, so home
            # is already here and only the land is left.
            self.get_logger().info("come home during climb; landing at the pad.")
            self.request_land()
            return

        # Transiting to the center, or holding there. Blend back from the last
        # commanded point. Pre-orbit every drone is commanded to the same point in
        # its own frame, so the fleet retreats as a rigid formation and the
        # peel-off stagger has nothing to deconflict.
        self._excited = True  # abandon any excitation bob
        self._home_from, self._home_yaw = self._last_nominal
        self._home_us = self._now_us()
        self.get_logger().info("come home before orbit; returning to the launch anchor.")

    def _hold(self, east, north, yaw_enu):
        return (east, north, self._altitude_up()), self._mission_yaw(yaw_enu)

    def _altitude_up(self):
        """Pad-relative ENU-up target; the controller restores launch-z."""
        return self.alt

    def _mission_yaw(self, fallback_yaw_enu):
        if self.yaw_mode == "hold" and self._hold_yaw_enu is not None:
            return self._hold_yaw_enu
        return float(fallback_yaw_enu)

    def _orbit_kw(self):
        yaw_mode = "fixed" if self.yaw_mode == "hold" else self.yaw_mode
        return dict(
            spacing=0.0, downrange=self.center[1], base=(self.center[0], 0.0),
            altitude=self._altitude_up(), phase_step=self.phase_step, phase0=self.phase0,
            yaw_mode=yaw_mode, fixed_yaw=self._mission_yaw(0.0),
        )

    @staticmethod
    def _bearing_resolved(estimate):
        return (
            int(estimate.status) == STATUS_TRACKING
            or (
                not bool(estimate.alternate_valid)
                and float(estimate.observability) >= 0.10
                and all(math.isfinite(v) for v in estimate.position_enu_m)
            )
        )

    def _all_peer_bearings_resolved(self):
        return (
            len(self.relative_state) >= len(self.peer_namespaces)
            and all(
                self._bearing_resolved(estimate)
                for estimate in self.relative_state.values()
            )
        )

    def _localization_setpoint(self, yaw_enu):
        """Run bounded horizontal excitation and return to the launch point.

        Staging cannot safely create traffic while any peer is still only a
        scalar range.  Repeat the same small cycle until every peer has a usable
        bearing; completing only at a cycle boundary keeps the staging seam
        position-continuous.
        """
        now = self._now_us()
        if self._localization_start_us == 0:
            self._localization_start_us = now
            self.get_logger().info(
                f"resolving peer bearings before staging: "
                f"radius={LOCALIZATION_RADIUS_M:.2f}m "
                f"cycle={LOCALIZATION_CYCLE_S:.1f}s")

        elapsed = max(0.0, (now - self._localization_start_us) * 1e-6)
        cycle = int(elapsed / LOCALIZATION_CYCLE_S)
        if cycle > self._localization_cycle:
            self._localization_cycle = cycle
            resolved = sum(
                self._bearing_resolved(estimate)
                for estimate in self.relative_state.values()
            )
            if self._all_peer_bearings_resolved():
                self._localization_complete = True
                self._stage_start_us = 0
                self.get_logger().info(
                    f"peer bearing sides resolved "
                    f"({resolved}/{len(self.peer_namespaces)}); "
                    "starting staging")
                return self._hold(0.0, 0.0, yaw_enu)
            self.get_logger().warn(
                f"peer bearings incomplete ({resolved}/{len(self.peer_namespaces)}); "
                "repeating localization cycle")

        angle = 2.0 * math.pi * (
            (elapsed % LOCALIZATION_CYCLE_S) / LOCALIZATION_CYCLE_S)
        identity_phase = 2.0 * math.pi * self.index / max(1, self.count)
        east = LOCALIZATION_RADIUS_M * (
            math.sin(angle + identity_phase) - math.sin(identity_phase))
        north = LOCALIZATION_RADIUS_M * (
            math.cos(angle + identity_phase) - math.cos(identity_phase))
        return self._hold(east, north, yaw_enu)

    def _staging_setpoint(self, yaw_enu):
        """Converge from a random spawn onto this drone's fixed line slot.

        The returned point is relative to ``_anchor`` because compute_setpoint
        adds that post-climb EKF anchor. On arrival the slot becomes the new
        choreography anchor and the normal center transit starts without a
        setpoint jump. Active ORCA owns the horizontal velocity around peers.
        """
        now = self._now_us()
        if self._stage_start_us == 0:
            self._stage_start_us = now
            distance = math.hypot(
                self.stage_e - self.spawn_e - self._anchor[0],
                self.stage_n - self.spawn_n - self._anchor[1])
            self.get_logger().info(
                f"staging to fixed line slot world=({self.stage_e:+.2f},"
                f"{self.stage_n:+.2f}) distance={distance:.2f}m")

        target_local = (self.stage_e - self.spawn_e, self.stage_n - self.spawn_n)
        delta = (target_local[0] - self._anchor[0], target_local[1] - self._anchor[1])
        distance = math.hypot(*delta)
        # smoothstep peaks at 1.5 * distance / duration
        duration = max(self.to_center_time_s, 1.5 * distance / self.staging_speed)
        u = min(1.0, max(0.0, (now - self._stage_start_us) / 1_000_000.0) / duration)
        smooth = u * u * (3.0 - 2.0 * u)

        own_e, own_n = self._world_xy()
        error = math.hypot(own_e - self.stage_e, own_n - self.stage_n)
        if u >= 1.0 and error <= self.staging_acceptance:
            # The old anchor + delta and the new anchor + zero are identical, so
            # switching frames here is position-continuous.
            self._anchor = target_local
            self._staging_complete = True
            self._transit_us = now
            self.get_logger().info(
                f"reached staging line slot: error={error:.2f}m; "
                "transiting to orbit center")
            return self._hold(0.0, 0.0, yaw_enu)

        return self._hold(delta[0] * smooth, delta[1] * smooth, yaw_enu)

    def _pre_orbit_setpoint(self):
        """Takeoff -> hold-at-center, the staged ready state (pre begin_orbit).

        Climb vertically at the spawn point while yawing to the center bearing,
        then transit horizontally to the circle center and hold. The ready signal
        is position-gated (within acceptance of center + altitude), not timed.
        """
        ce, cn = self.center
        face_center = math.atan2(cn, ce)  # ENU yaw from spawn toward the center
        desired_yaw_enu = self._mission_yaw(face_center)
        target_yaw_ned = wrap_pi(math.pi / 2.0 - desired_yaw_enu)

        if not self._climbed:
            if self.anchor_after_climb:
                # re-latch every tick: the setpoint tracks the estimate, so the
                # climb applies no horizontal correction and a frame reset
                # (spool-up multipath) moves the anchor instead of the drone
                self._anchor = (self.y, self.x)  # NED y=east, x=north
            alt_ok = abs(self.z - self._takeoff_target_z()) <= self.takeoff_acceptance_m
            yaw_ok = abs(wrap_pi(self.yaw - target_yaw_ned)) <= YAW_ACCEPTANCE_RAD
            if alt_ok and yaw_ok:
                self._climbed = True
                now = self._now_us()
                if self._localization_complete:
                    if self.staging_enabled:
                        self._stage_start_us = 0
                    else:
                        self._transit_us = now
                ae, an = self._anchor
                next_phase = (
                    "localizing" if not self._localization_complete
                    else "staging" if self.staging_enabled
                    else "transiting")
                self.get_logger().info(
                    f"climbed + yawed to center bearing, {next_phase}. "
                    f"frame anchor ENU=({ae:+.2f},{an:+.2f})")
            return self._hold(0.0, 0.0, face_center)

        if not self._localization_complete:
            return self._localization_setpoint(face_center)

        if not self._staging_complete:
            return self._staging_setpoint(face_center)

        th = (self._now_us() - self._transit_us) / 1_000_000.0
        u = min(1.0, max(0.0, th) / max(1e-3, self.to_center_time_s))
        s = u * u * (3.0 - 2.0 * u)  # smoothstep transit, zero rate at both ends
        ae, an = self._anchor
        horiz = math.hypot(self.x - (an + cn), self.y - (ae + ce))  # NED x=north, y=east
        at_center = u >= 1.0 and horiz <= self.takeoff_acceptance_m
        if not at_center and not self._excited:
            # The vibration reference is meant to be a continuous stationary
            # segment. Re-entering the center acceptance region starts it over.
            self._pre_excite_hover_us = 0
        if at_center and not self._excited:
            now = self._now_us()
            if self._pre_excite_hover_us == 0:
                self._pre_excite_hover_us = now
                if self.pre_excite_hover_s > 0.0:
                    self.get_logger().info(
                        f"starting steady pre-excitation hover: "
                        f"duration={self.pre_excite_hover_s:.1f}s"
                    )
            hover_s = (now - self._pre_excite_hover_us) / 1_000_000.0
            if hover_s < self.pre_excite_hover_s:
                return self._hold(ce, cn, face_center)
            if self._excite_us == 0:
                self._excite_us = now
                if self.pre_excite_hover_s > 0.0:
                    self.get_logger().info(
                        "steady pre-excitation hover complete; starting bob."
                    )
                self.get_logger().info(
                    f"starting position-only vertical excitation: "
                    f"amp={self.excite_amp_m:.2f}m freq={self.excite_freq_hz:.2f}Hz "
                    f"cycles={self.excite_cycles:g}"
                )
            te = (now - self._excite_us) / 1_000_000.0
            duration = self.excite_cycles / self.excite_freq_hz
            if te < duration:
                omega = 2.0 * math.pi * self.excite_freq_hz
                dz = 0.5 * self.excite_amp_m * (1.0 - math.cos(omega * te))
                # Deliberately return only position+yaw. Publishing the analytic
                # downward velocity tripped PX4's land detector, which enabled
                # EKF ground-effect baro rejection while still airborne.
                return (ce, cn, self._altitude_up() + dz), self._mission_yaw(face_center)
            self._excited = True
            self.get_logger().info("vertical excitation complete; holding at nominal altitude.")

        if at_center and not self._center_logged:
            self._center_logged = True
            self.get_logger().info("at center, holding (ready for orbit).")

        return self._hold(ce * s, cn * s, face_center)

    def compute_setpoint(self):
        self._publish_world_state()
        result = self._nominal_setpoint()
        if result is None:
            return None
        # Latched so on_return_home() can start its blend from the last commanded
        # point instead of re-deriving the transit smoothstep.
        self._last_nominal = result
        pos_enu, yaw = result
        ae, an = self._anchor  # whole choreography rides on the latched frame
        pos_enu = (pos_enu[0] + ae, pos_enu[1] + an, pos_enu[2])
        if self.peer_namespaces:
            safe_enu = self._avoidance_velocity(pos_enu[:2])
            # Mixed-axis Offboard: ORCA owns horizontal velocity while PX4 keeps
            # the mission altitude as a position target. NaN disables horizontal
            # position control, preventing it from bypassing the safe velocity.
            return (
                math.nan, math.nan, -float(pos_enu[2]),
                wrap_pi(math.pi / 2.0 - float(yaw)),
                float(safe_enu[1]), float(safe_enu[0]), math.nan,
            )
        return enu_to_ned_setpoint(pos_enu, float(yaw))

    def _nominal_setpoint(self):
        """The open-loop choreography setpoint (ENU, local frame) for this tick."""
        if self._home_us:  # come home issued before the orbit started
            home_t = (self._now_us() - self._home_us) / 1_000_000.0
            return self._home_setpoint(self._home_from, self._home_yaw, home_t)

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

        # Peel-off ends at the orbit center. Reverse the pre-orbit center transit
        # to the post-climb launch anchor while remaining in Offboard.
        home_t = rt - self.return_duration
        ce, cn = self.center
        _, final_yaw = phased_orbit_peeloff(
            self.return_duration, self.index, self.count, self.radius, self.omega,
            peel_order=self.peel_order, lead_in=self.peel_lead_in_s,
            stagger=self.peel_stagger_s, peel_duration=self.peel_duration_s,
            spin=self.peel_spin, **self._orbit_kw())
        return self._home_setpoint((ce, cn, self._altitude_up()), final_yaw, home_t)

    def _home_setpoint(self, from_enu, yaw, home_t):
        """Blend from `from_enu` back to the launch anchor, settle, then land.

        Shared by the post-peel-off return and by a come-home issued before the
        orbit starts; only the start point differs. The altitude term blends any
        excitation offset out instead of stepping it, and is a no-op when the
        start point is already at the nominal altitude.
        """
        u = min(1.0, max(0.0, home_t) / max(1e-3, self.to_center_time_s))
        s = u * u * (3.0 - 2.0 * u)
        fe, fn, fz = from_enu
        alt = self._altitude_up()
        anchor_setpoint = (
            fe * (1.0 - s),
            fn * (1.0 - s),
            fz + (alt - fz) * s,
        )

        if u < 1.0:
            self._land_settle_us = 0
            return anchor_setpoint, float(yaw)

        ae, an = self._anchor
        xy_error = math.hypot(self.x - an, self.y - ae)
        vxy = math.hypot(self.vx, self.vy)
        settled = (
            math.isfinite(xy_error)
            and math.isfinite(vxy)
            and xy_error <= self.land_xy_acceptance
            and vxy <= self.land_vxy_acceptance
        )
        now = self._now_us()
        if settled:
            if self._land_settle_us == 0:
                self._land_settle_us = now
            dwell = (now - self._land_settle_us) / 1_000_000.0
        else:
            self._land_settle_us = 0
            dwell = 0.0

        if not self._landing_logged:
            self._landing_logged = True
            pct = 100.0 * self.avoidance_active_ticks / max(1, self.avoidance_total_ticks)
            self.get_logger().info(
                f"at launch anchor; settling before local land. ORCA active "
                f"{self.avoidance_active_ticks}/{self.avoidance_total_ticks} ticks "
                f"({pct:.1f}%), max delta-v {self.avoidance_max_delta:.3f} m/s."
            )
        if dwell >= self.land_settle_time_s and not self._land_requested:
            self._land_requested = True
            self.get_logger().info(
                f"launch anchor settled: xy_error={xy_error:.2f}m "
                f"vxy={vxy:.2f}m/s dwell={dwell:.1f}s; requesting NAV_LAND."
            )
            self.request_land()
        return anchor_setpoint, float(yaw)


def main(args=None):
    rclpy.init(args=args)
    node = PhasedOrbitsMission()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
