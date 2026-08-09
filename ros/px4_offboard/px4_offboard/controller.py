"""Reusable PX4 offboard plumbing: namespacing, the offboard handshake,
vehicle-state tracking, link-health watchdog, and the operator gates.

Three gates arrive as `std_msgs/Bool` on global (un-namespaced) topics, so one
publish reaches every drone at once. `/start_mission` releases the handshake.
`/abort_mission` lands in place immediately, from any state. `/end_mission` means
come home: the base only latches it and calls `on_return_home()`, leaving the
mission to fly its own return, since what "home" means is choreography the base
knows nothing about. A mission that wants PX4 AUTO.RTL overrides that hook to
call `begin_return()`.

A mission subclasses `OffboardController` and overrides `compute_setpoint`,
returning the next NED position+yaw target each control tick (or None to hold
the takeoff point). Setpoints are PX4 NED (x north, y east, z down; yaw CW from
north); flight_lib emits z-up ENU, so the mission layer converts first.

`force_arm` defaults to False. Force-arm (param2 = 21196) bypasses PX4 pre-arm
checks, which suits SITL but stays an explicit opt-in.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.clock import Clock
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from px4_msgs.msg import (
    GotoSetpoint,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleCommand,
    VehicleCommandAck,
    VehicleGlobalPosition,
    VehicleLocalPosition,
    VehicleStatus,
)
from std_msgs.msg import Bool

from px4_offboard.gate_qos import MISSION_GATE_QOS

FORCE_ARM_MAGIC = 21196.0
ORIGIN_RESEND_INTERVAL_S = 0.5
ORIGIN_CONFIRM_TIMEOUT_S = 20.0
ARMING_STATE_ARMED = 2

WAIT_LINK = "wait_link"
WAIT_START = "wait_start"
PRESTREAM = "prestream"
TAKEOFF = "takeoff"
ENGAGE = "engage"
ACTIVE = "active"
RETURNING = "returning"
LAND_REQUESTED = "land_requested"
LANDING = "landing"
FAILSAFE = "failsafe"
DONE = "done"


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class OffboardController(Node):
    """Gets a vehicle armed + offboard and streams mission setpoints."""

    def __init__(self, node_name: str = "offboard_controller"):
        super().__init__(node_name)

        self.declare_parameter("vehicle_namespace", "")
        self.declare_parameter("setpoint_rate_hz", 20.0)
        self.declare_parameter("prestream_cycles", 10)
        self.declare_parameter("takeoff_altitude_m", 5.0)
        self.declare_parameter("takeoff_acceptance_m", 0.4)
        self.declare_parameter("takeoff_timeout_s", 30.0)
        self.declare_parameter("force_arm", False)
        self.declare_parameter("wait_for_start", False)
        self.declare_parameter("status_stale_timeout_s", 5.0)
        self.declare_parameter("launch_stability_s", 3.0)

        self.ns = self.get_parameter("vehicle_namespace").value.strip("/")
        self.rate_hz = float(self.get_parameter("setpoint_rate_hz").value)
        self.prestream_cycles = int(self.get_parameter("prestream_cycles").value)
        self.takeoff_altitude_m = float(self.get_parameter("takeoff_altitude_m").value)
        self.takeoff_acceptance_m = float(self.get_parameter("takeoff_acceptance_m").value)
        self.takeoff_timeout_s = float(self.get_parameter("takeoff_timeout_s").value)
        self.force_arm = bool(self.get_parameter("force_arm").value)
        self.wait_for_start = bool(self.get_parameter("wait_for_start").value)
        self.status_stale_timeout_s = float(self.get_parameter("status_stale_timeout_s").value)
        self.launch_stability_s = float(self.get_parameter("launch_stability_s").value)

        # PX4 publishes every uORB topic BEST_EFFORT over uXRCE-DDS. A RELIABLE
        # subscription is silently incompatible with it and receives nothing,
        # so subclasses take this profile for any /fmu/out topic they add.
        sensor_qos = self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._cmd_pub = self.create_publisher(VehicleCommand, self._topic("in/vehicle_command"), 10)
        self._offboard_pub = self.create_publisher(OffboardControlMode, self._topic("in/offboard_control_mode"), sensor_qos)
        self._traj_pub = self.create_publisher(TrajectorySetpoint, self._topic("in/trajectory_setpoint"), sensor_qos)
        self._goto_pub = self.create_publisher(GotoSetpoint, self._topic("in/goto_setpoint"), sensor_qos)

        # Keep legacy names for existing images and add the v1.18 message versions.
        self.create_subscription(VehicleStatus, self._topic("out/vehicle_status"), self._status_cb, sensor_qos)
        self.create_subscription(VehicleStatus, self._topic("out/vehicle_status_v1"), self._status_cb, sensor_qos)
        self.create_subscription(VehicleStatus, self._topic("out/vehicle_status_v4"), self._status_cb, sensor_qos)
        self.create_subscription(VehicleLocalPosition, self._topic("out/vehicle_local_position"), self._pos_cb, sensor_qos)
        self.create_subscription(VehicleLocalPosition, self._topic("out/vehicle_local_position_v1"), self._pos_cb, sensor_qos)
        self.create_subscription(VehicleAttitude, self._topic("out/vehicle_attitude"), self._att_cb, sensor_qos)
        self.create_subscription(VehicleCommandAck, self._topic("out/vehicle_command_ack"), self._ack_cb, sensor_qos)
        self.create_subscription(VehicleCommandAck, self._topic("out/vehicle_command_ack_v1"), self._ack_cb, sensor_qos)
        self.create_subscription(VehicleGlobalPosition, self._topic("out/vehicle_global_position"), self._gpos_cb, sensor_qos)
        self.create_subscription(VehicleGlobalPosition, self._topic("out/vehicle_global_position_v1"), self._gpos_cb, sensor_qos)

        self.create_subscription(
            Bool, "start_mission", self._start_cb, MISSION_GATE_QOS
        )
        self.create_subscription(Bool, "end_mission", self._end_cb, MISSION_GATE_QOS)
        self.create_subscription(
            Bool, "abort_mission", self._abort_cb, MISSION_GATE_QOS
        )

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_DISARMED
        self.failsafe = False
        self.x = self.y = self.z = 0.0  # NED metres
        self.vx = self.vy = self.vz = 0.0  # NED m/s
        self.yaw = 0.0
        self._launch_xy = None
        self._launch_z = 0.0
        self._launch_reference_latched = False
        self._launch_z_latched = False
        self._launch_yaw = 0.0
        self._launch_yaw_latched = False
        self._xy_valid = False
        self._v_xy_valid = False
        self._z_valid = False
        self._attitude_seen = False

        self._status_seen = False
        self._last_status_us = 0
        self._prestream_count = 0
        self._start_ok = not self.wait_for_start
        self._abort_requested = False
        self._home_requested = False
        self._last_log_us = 0
        self._last_command_us = 0
        self._takeoff_started_us = 0
        self._link_acquired_fired = False
        self._heartbeat_velocity = False
        self._goto_active = False
        self._goto_max_horizontal_speed = None
        self._goto_max_vertical_speed = None
        self._goto_max_heading_rate = None
        self._global_xy_valid = False
        self._global_alt_valid = False
        self._pending_origin = None
        self._origin_send_us = 0
        self._origin_start_us = 0
        self._origin_confirmed = False
        self._handoff_hold = None
        self._xy_reset_counter = None
        self._z_reset_counter = None
        self._heading_reset_counter = None
        self._last_local_reset_us = 0

        self.state = WAIT_LINK
        self._timer = self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f"OffboardController up: ns={self.ns or 'root'} rate={self.rate_hz}Hz "
            f"force_arm={self.force_arm} wait_for_start={self.wait_for_start}"
        )

    # hooks

    def compute_setpoint(self):
        """Return (x, y, z, yaw) in NED, or None to hold. Called each ACTIVE tick."""
        return None

    def request_land(self):
        """Land in place, at the current position. A mission calls this once its
        own return has already put the vehicle where it wants to touch down."""
        self._abort_requested = True

    def begin_return(self):
        if self.state not in (RETURNING, LAND_REQUESTED, LANDING, DONE):
            self._begin_return()

    def enable_goto_setpoints(
        self,
        *,
        max_horizontal_speed: float,
        max_vertical_speed: float | None = None,
        max_heading_rate: float | None = None,
    ) -> None:
        """Use PX4's jerk-limited GotoControl for subsequent position targets."""
        self._goto_active = True
        self._heartbeat_velocity = False
        self._goto_max_horizontal_speed = float(max_horizontal_speed)
        self._goto_max_vertical_speed = (
            None if max_vertical_speed is None else float(max_vertical_speed)
        )
        self._goto_max_heading_rate = (
            None if max_heading_rate is None else float(max_heading_rate)
        )

    def command_takeoff(self, altitude_m: float | None = None):
        altitude = self.takeoff_altitude_m if altitude_m is None else float(altitude_m)
        self._publish_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param7=altitude)

    def command_return(self):
        self._publish_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)

    def set_global_origin(self, lat, lon, alt=0.0):
        # EKF2 takes this vehicle_command directly (EKF2.cpp), source-agnostic, so
        # it works over XRCE-DDS without MAVLink. A global origin lets a local-only
        # (EV/mocap) estimate produce a global position, which the auto modes
        # (RTL/Land/Hold) and failsafes require. param5/6 are float64 (lat/lon).
        # Fire-and-forget races EKF2 init (a command sent before ekf2 is up is
        # dropped, leaving no global position and RTL unavailable), so latch it and
        # re-send from _tick until vehicle_global_position reports lat_lon_valid.
        self._pending_origin = (float(lat), float(lon), float(alt))
        self._origin_start_us = self._now_us()
        self._origin_send_us = 0
        self._origin_confirmed = False

    def _send_pending_origin(self):
        if self._pending_origin is None:
            return
        if self._global_xy_valid:
            self._pending_origin = None
            if not self._origin_confirmed:
                self._origin_confirmed = True
                self.get_logger().info("EKF global origin accepted (lat_lon_valid).")
            return
        elapsed = (self._now_us() - self._origin_start_us) / 1_000_000.0
        if elapsed > ORIGIN_CONFIRM_TIMEOUT_S:
            raise RuntimeError(
                f"EKF global origin not accepted after {elapsed:.0f}s "
                f"(vehicle_global_position.lat_lon_valid still false); global modes unavailable."
            )
        now = self._now_us()
        if now - self._origin_send_us < ORIGIN_RESEND_INTERVAL_S * 1_000_000:
            return
        self._origin_send_us = now
        lat, lon, alt = self._pending_origin
        self._publish_command(
            VehicleCommand.VEHICLE_CMD_SET_GPS_GLOBAL_ORIGIN,
            param5=lat, param6=lon, param7=alt,
        )

    def command_offboard_mode(self):
        self._publish_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def command_arm(self):
        arm_p2 = FORCE_ARM_MAGIC if self.force_arm else 0.0
        # PX4 commander only lets the 21196 force-arm magic bypass preflight
        # checks when the command is not marked as external.
        self._publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
            param2=arm_p2,
            from_external=not self.force_arm,
        )

    def command_disarm(self, force: bool = False):
        self._publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,
            param2=FORCE_ARM_MAGIC if force else 0.0,
        )

    def on_link_acquired(self):
        """Mission hook called once when PX4 telemetry first arrives (pre-arm)."""

    def on_active_start(self):
        """Mission hook called once when OFFBOARD setpoints become active."""

    def on_return_home(self):
        """Mission hook called once when /end_mission arrives.

        Deliberately empty rather than defaulting to `begin_return()`: PX4 AUTO.RTL
        climbs to RTL_RETURN_ALT (60 m by default) before it transits, which is a
        net strike under the ~12 m geofence the real-flight config assumes. A
        mission that wants RTL opts in by overriding this with `begin_return()`.
        """

    def on_local_frame_reset(self, delta_xy, delta_z, delta_heading):
        """Mission hook for rebasing state stored outside the base controller."""

    # plumbing

    def _topic(self, suffix: str) -> str:
        return f"/{self.ns}/fmu/{suffix}" if self.ns else f"/fmu/{suffix}"

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _status_cb(self, msg: VehicleStatus):
        self._status_seen = True
        self._last_status_us = self._now_us()
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state
        self.failsafe = msg.failsafe

    def _pos_cb(self, msg: VehicleLocalPosition):
        self._handle_local_frame_reset(msg)
        self.x, self.y, self.z = msg.x, msg.y, msg.z
        self.vx, self.vy, self.vz = msg.vx, msg.vy, msg.vz
        self._xy_valid = bool(msg.xy_valid)
        self._v_xy_valid = bool(msg.v_xy_valid)
        self._z_valid = bool(msg.z_valid)
        if (
            not self._launch_reference_latched
            and all(math.isfinite(v) for v in (msg.x, msg.y))
        ):
            self._launch_xy = (float(msg.x), float(msg.y))

    def _handle_local_frame_reset(self, msg: VehicleLocalPosition):
        xy_counter = int(msg.xy_reset_counter)
        z_counter = int(msg.z_reset_counter)
        heading_counter = int(msg.heading_reset_counter)
        if self._xy_reset_counter is None:
            self._xy_reset_counter = xy_counter
            self._z_reset_counter = z_counter
            self._heading_reset_counter = heading_counter
            self._last_local_reset_us = self._now_us()
            return

        delta_xy = (0.0, 0.0)
        delta_z = 0.0
        delta_heading = 0.0
        xy_changed = xy_counter != self._xy_reset_counter
        z_changed = z_counter != self._z_reset_counter
        heading_changed = heading_counter != self._heading_reset_counter
        if xy_changed:
            delta_xy = (float(msg.delta_xy[0]), float(msg.delta_xy[1]))
        if z_changed:
            delta_z = float(msg.delta_z)
        if heading_changed:
            delta_heading = float(msg.delta_heading)

        self._xy_reset_counter = xy_counter
        self._z_reset_counter = z_counter
        self._heading_reset_counter = heading_counter
        if not (xy_changed or z_changed or heading_changed):
            return

        self._last_local_reset_us = self._now_us()
        if self._launch_xy is not None:
            self._launch_xy = (
                self._launch_xy[0] + delta_xy[0],
                self._launch_xy[1] + delta_xy[1],
            )
        if self._launch_z_latched:
            self._launch_z += delta_z
        if self._launch_yaw_latched:
            self._launch_yaw = wrap_pi(self._launch_yaw + delta_heading)
        if self._handoff_hold is not None:
            x, y, z, yaw = self._handoff_hold
            self._handoff_hold = (
                x + delta_xy[0],
                y + delta_xy[1],
                z,
                wrap_pi(yaw + delta_heading),
            )
        self.on_local_frame_reset(delta_xy, delta_z, delta_heading)
        self.get_logger().warn(
            "PX4 local frame reset: "
            f"xy=({delta_xy[0]:+.3f}, {delta_xy[1]:+.3f}) m "
            f"z={delta_z:+.3f} m heading={math.degrees(delta_heading):+.2f} deg"
        )

    def _att_cb(self, msg: VehicleAttitude):
        w, x, y, z = msg.q  # PX4 order w, x, y, z
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        if math.isfinite(yaw) and sum(float(v) * float(v) for v in msg.q) > 0.25:
            self.yaw = yaw
            self._attitude_seen = True

    def _gpos_cb(self, msg: VehicleGlobalPosition):
        # A manually supplied origin can make latitude/longitude valid without
        # producing a trustworthy global altitude. Keep these separate: origin
        # confirmation only needs XY, but AUTO.RTL requires both.
        self._global_xy_valid = bool(msg.lat_lon_valid)
        self._global_alt_valid = bool(getattr(msg, "alt_valid", False))

    def _ack_cb(self, msg: VehicleCommandAck):
        if msg.result != VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED:
            self.get_logger().warn(f"command rejected: cmd={msg.command} result={msg.result}")
        else:
            self.get_logger().debug(f"ack cmd={msg.command} result={msg.result}")

    def _start_cb(self, msg: Bool):
        if msg.data and not self._start_ok:
            self._start_ok = True
            self.get_logger().info("start_mission received.")

    def _end_cb(self, msg: Bool):
        # Latched, and not as a safety gate: a mission's return is a scheduled
        # choreography, and re-entering it partway through would fling the vehicle
        # back to where the schedule starts. sitl.sh publishes --times 5, so the
        # repeats arrive as a matter of course.
        if msg.data and not self._home_requested:
            self._home_requested = True
            self.get_logger().info("end_mission received, returning home.")
            self.on_return_home()

    def _abort_cb(self, msg: Bool):
        if msg.data and not self._abort_requested:
            self._abort_requested = True
            self.get_logger().info("abort_mission received, landing in place.")

    @property
    def is_armed(self) -> bool:
        return int(self.arm_state) == ARMING_STATE_ARMED

    def _link_alive(self) -> bool:
        if not self._status_seen:
            return False
        stale_us = int(max(1.0, self.status_stale_timeout_s) * 1_000_000)
        return (self._now_us() - self._last_status_us) <= stale_us

    def _publish_command(self, command, from_external=True, **params):
        m = VehicleCommand()
        m.command = int(command)
        for i in range(1, 8):
            setattr(m, f"param{i}", float(params.get(f"param{i}", 0.0)))
        m.target_system = 0
        m.target_component = 1
        m.source_system = 1
        m.source_component = 1
        m.from_external = bool(from_external)
        m.timestamp = int(Clock().now().nanoseconds / 1000)
        self._cmd_pub.publish(m)

    def _publish_command_throttled(self, command, period_us: int = 1_000_000, **params):
        now = self._now_us()
        if now - self._last_command_us >= period_us:
            self._last_command_us = now
            self._publish_command(command, **params)

    def _publish_heartbeat(self):
        off = OffboardControlMode()
        off.timestamp = int(Clock().now().nanoseconds / 1000)
        off.position = not self._heartbeat_velocity
        off.velocity = self._heartbeat_velocity
        off.acceleration = False
        off.attitude = False
        off.body_rate = False
        self._offboard_pub.publish(off)

    def publish_position_setpoint(self, x, y, z, yaw=None, yawspeed=0.0):
        self._heartbeat_velocity = False
        traj = TrajectorySetpoint()
        traj.timestamp = int(Clock().now().nanoseconds / 1000)
        # Mission z is relative to the pad; PX4's local datum need not be zero
        # on the ground, so restore the launch offset at the publication seam.
        traj.position[0], traj.position[1], traj.position[2] = (
            float(x), float(y), float(z) + self._launch_z
        )
        for i in range(3):
            traj.velocity[i] = float("nan")
            traj.acceleration[i] = float("nan")
        traj.yaw = wrap_pi(self.yaw if yaw is None else float(yaw))
        traj.yawspeed = float(yawspeed)
        self._traj_pub.publish(traj)

    def publish_position_velocity_setpoint(self, x, y, z, vx, vy, vz, yaw=None, yawspeed=0.0):
        self._heartbeat_velocity = True
        traj = TrajectorySetpoint()
        traj.timestamp = int(Clock().now().nanoseconds / 1000)
        traj.position[0], traj.position[1], traj.position[2] = (
            float(x), float(y), float(z) + self._launch_z
        )
        traj.velocity[0], traj.velocity[1], traj.velocity[2] = float(vx), float(vy), float(vz)
        for i in range(3):
            traj.acceleration[i] = float("nan")
        traj.yaw = wrap_pi(self.yaw if yaw is None else float(yaw))
        traj.yawspeed = float(yawspeed)
        self._traj_pub.publish(traj)

    def publish_goto_setpoint(self, x, y, z, yaw=None):
        self._heartbeat_velocity = False
        goto = GotoSetpoint()
        goto.timestamp = int(Clock().now().nanoseconds / 1000)
        goto.position[0], goto.position[1], goto.position[2] = (
            float(x), float(y), float(z) + self._launch_z
        )
        goto.flag_control_heading = yaw is not None
        goto.heading = wrap_pi(self.yaw if yaw is None else float(yaw))
        goto.flag_set_max_horizontal_speed = True
        goto.max_horizontal_speed = self._goto_max_horizontal_speed
        goto.flag_set_max_vertical_speed = self._goto_max_vertical_speed is not None
        if self._goto_max_vertical_speed is not None:
            goto.max_vertical_speed = self._goto_max_vertical_speed
        goto.flag_set_max_heading_rate = self._goto_max_heading_rate is not None
        if self._goto_max_heading_rate is not None:
            goto.max_heading_rate = self._goto_max_heading_rate
        self._goto_pub.publish(goto)

    def _hold_setpoint(self):
        target = (
            *self._takeoff_hold(),
            self._launch_yaw if self._launch_yaw_latched else self.yaw,
        )
        if self._goto_active:
            self.publish_goto_setpoint(*target)
        else:
            self.publish_position_setpoint(*target)

    def _takeoff_target_z(self):
        return self._launch_z - abs(self.takeoff_altitude_m)

    def _takeoff_hold(self):
        lx, ly = self._launch_xy if self._launch_xy else (self.x, self.y)
        return lx, ly, -abs(self.takeoff_altitude_m)

    def _latch_launch_reference(self) -> bool:
        """Latch the converged pad altitude and operator-set heading pre-arm."""
        if self._launch_reference_latched:
            return True
        stable_s = (self._now_us() - self._last_local_reset_us) / 1_000_000.0
        if (
            stable_s >= self.launch_stability_s
            and self._xy_valid
            and self._z_valid
            and all(math.isfinite(v) for v in (self.x, self.y, self.z))
            and self._attitude_seen
        ):
            self._launch_xy = (float(self.x), float(self.y))
            self._launch_z = float(self.z)
            self._launch_reference_latched = True
            self._launch_z_latched = True
            self._launch_yaw = float(self.yaw)
            self._launch_yaw_latched = True
            self.get_logger().info(
                f"latched launch datum N {self._launch_xy[0]:+.3f} "
                f"E {self._launch_xy[1]:+.3f} Z {self._launch_z:+.3f} m; "
                f"holding operator-set heading {math.degrees(self._launch_yaw):.1f} deg NED"
            )
            return True
        return False

    def _capture_handoff_hold(self):
        self._heartbeat_velocity = False
        hold_yaw = self._launch_yaw if self._launch_yaw_latched else self.yaw
        self._handoff_hold = (
            float(self.x), float(self.y), float(self.z - self._launch_z), float(hold_yaw)
        )

    def _publish_handoff_hold(self):
        if self._handoff_hold is None:
            self._capture_handoff_hold()
        if self._goto_active:
            self.publish_goto_setpoint(*self._handoff_hold)
        else:
            self.publish_position_setpoint(*self._handoff_hold)

    def _log_throttled(self, msg: str, period_us: int = 1_000_000):
        now = self._now_us()
        if now - self._last_log_us > period_us:
            self._last_log_us = now
            self.get_logger().info(msg)

    # state machine

    def _tick(self):
        self._send_pending_origin()
        # OFFBOARD drops without an OffboardControlMode stream at >=2 Hz
        awaiting_rtl = (
            self.state == RETURNING
            and self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )
        if self.state in (PRESTREAM, TAKEOFF, ENGAGE, ACTIVE, LAND_REQUESTED) or awaiting_rtl:
            self._publish_heartbeat()

        if self.state == WAIT_LINK:
            if self._link_alive():
                if not self._link_acquired_fired:
                    self._link_acquired_fired = True
                    self.on_link_acquired()
                self.state = WAIT_START if not self._start_ok else PRESTREAM
            else:
                self._log_throttled("waiting for PX4 telemetry (MicroXRCEAgent/DDS up?)")

        elif self.state == WAIT_START:
            self._log_throttled("waiting for start_mission")
            if self._start_ok:
                self.state = PRESTREAM

        elif self.state == PRESTREAM:
            self._hold_setpoint()
            self._prestream_count += 1
            if self._prestream_count >= self.prestream_cycles:
                # Never arm without a valid pad-relative altitude datum.
                if not self._latch_launch_reference():
                    max_wait = self.prestream_cycles + int(
                        self.rate_hz * self.takeoff_timeout_s
                    )
                    if self._prestream_count > max_wait:
                        raise RuntimeError(
                            "launch reference never became stable and valid; refusing to arm"
                        )
                    self._log_throttled(
                        "waiting for a stable valid local pose before takeoff"
                    )
                    return
                self._takeoff_started_us = self._now_us()
                self.get_logger().info("commanding OFFBOARD takeoff.")
                self.command_offboard_mode()
                self._last_command_us = self._now_us()
                self.state = ENGAGE

        elif self.state == TAKEOFF:
            self._hold_setpoint()
            if self._abort_requested:
                self._begin_landing()
                return
            self.command_arm()
            self._publish_command_throttled(
                VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
                param7=self.takeoff_altitude_m,
            )
            target_z = self._takeoff_target_z()
            altitude_error = abs(self.z - target_z)
            elapsed_s = (self._now_us() - self._takeoff_started_us) / 1_000_000.0
            if altitude_error <= self.takeoff_acceptance_m or elapsed_s >= self.takeoff_timeout_s:
                self.state = ENGAGE
                self.get_logger().info(
                    f"AUTO.TAKEOFF complete enough: z={self.z:.2f}, "
                    f"target={target_z:.2f}, elapsed={elapsed_s:.1f}s."
                )
            else:
                self._log_throttled(
                    f"taking off: armed={self.is_armed} z={self.z:.2f} target={target_z:.2f}"
                )

        elif self.state == ENGAGE:
            self._hold_setpoint()
            if self._abort_requested:
                self._begin_landing()
                return
            self.command_offboard_mode()
            self.command_arm()
            if self.is_armed and self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.state = ACTIVE
                self.on_active_start()
                self.get_logger().info("armed + OFFBOARD, mission setpoints active.")
            else:
                self._log_throttled(f"engaging: armed={self.is_armed} nav_state={self.nav_state}")

        elif self.state == ACTIVE:
            if self._abort_requested:
                self._begin_landing()
                return
            if (
                self.failsafe
                or self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD
                or not (self._xy_valid and self._v_xy_valid and self._z_valid)
            ):
                # PX4 owns the response from this point. Stop the Offboard
                # stream so the old user intention cannot silently resume
                # after estimator validity returns, as happened in the
                # 2026-08-08 Drone 4 survey crash.
                self.state = FAILSAFE
                self.get_logger().error(
                    "mission interlock tripped: "
                    f"failsafe={self.failsafe} nav_state={self.nav_state} "
                    f"xy_valid={self._xy_valid} v_xy_valid={self._v_xy_valid} "
                    f"z_valid={self._z_valid}; yielding permanently to PX4"
                )
                return
            sp = self.compute_setpoint()
            if sp is None:
                self._hold_setpoint()
            elif self._goto_active:
                if len(sp) != 4:
                    raise ValueError("goto setpoint must be (x, y, z, yaw)")
                self.publish_goto_setpoint(*sp)
            else:
                if len(sp) == 4:
                    x, y, z, yaw = sp
                    self.publish_position_setpoint(x, y, z, yaw)
                elif len(sp) == 7:
                    x, y, z, yaw, vx, vy, vz = sp
                    self.publish_position_velocity_setpoint(x, y, z, vx, vy, vz, yaw)
                else:
                    raise ValueError("setpoint must be (x, y, z, yaw) or (x, y, z, yaw, vx, vy, vz)")

        elif self.state == RETURNING:
            if self._abort_requested:
                self._begin_landing()
                return
            if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                # Keep the last safe local setpoint alive until PX4 actually
                # accepts AUTO.RTL. Merely sending the command is not a mode
                # transition, and dropping this stream causes Offboard loss.
                self._publish_handoff_hold()
            self._publish_command_throttled(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
            if not self.is_armed:
                self.state = DONE
                self.get_logger().info("RTL complete and disarmed.")
            else:
                self._log_throttled(f"returning via RTL: nav_state={self.nav_state}")

        elif self.state == LAND_REQUESTED:
            if not self.is_armed:
                self.state = DONE
                self.get_logger().info("landed and disarmed.")
            elif self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                self.state = LANDING
                self.get_logger().info("PX4 accepted NAV_LAND; AUTO.LAND active.")
            else:
                # NAV_LAND can be temporarily rejected. Hold autonomously in
                # Offboard and retry instead of falling through to RC Position.
                self._publish_handoff_hold()
                self._publish_command_throttled(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self._log_throttled(
                    f"waiting for AUTO.LAND acceptance: nav_state={self.nav_state}"
                )

        elif self.state == LANDING:
            if not self.is_armed:
                self.state = DONE
                self.get_logger().info("landed and disarmed.")

        elif self.state == FAILSAFE:
            # Deliberately publish neither OffboardControlMode nor trajectory
            # setpoints. A fresh process and operator start are required after
            # any in-flight estimator or mode failure.
            if not self.is_armed:
                self.state = DONE
                self.get_logger().info("PX4 failsafe complete and disarmed.")

    def _begin_return(self):
        if not (self._global_xy_valid and self._global_alt_valid):
            self.get_logger().warn(
                "AUTO.RTL unavailable without valid global XY + altitude; "
                "requesting local NAV_LAND instead."
            )
            self._begin_landing()
            return
        self._capture_handoff_hold()
        self.state = RETURNING
        self.get_logger().info("commanding AUTO.RTL.")
        self.command_return()

    def _begin_landing(self):
        self._capture_handoff_hold()
        self.state = LAND_REQUESTED
        self._last_command_us = self._now_us()
        self.get_logger().info("requesting NAV_LAND while maintaining Offboard hold.")
        self._publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)


def main(args=None):
    rclpy.init(args=args)
    node = OffboardController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
