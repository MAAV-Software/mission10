"""Reusable PX4 offboard plumbing: namespacing, the offboard handshake,
vehicle-state tracking, link-health watchdog, and the start/end gate.

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
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from px4_msgs.msg import (
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
LANDING = "landing"
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

        self.ns = self.get_parameter("vehicle_namespace").value.strip("/")
        self.rate_hz = float(self.get_parameter("setpoint_rate_hz").value)
        self.prestream_cycles = int(self.get_parameter("prestream_cycles").value)
        self.takeoff_altitude_m = float(self.get_parameter("takeoff_altitude_m").value)
        self.takeoff_acceptance_m = float(self.get_parameter("takeoff_acceptance_m").value)
        self.takeoff_timeout_s = float(self.get_parameter("takeoff_timeout_s").value)
        self.force_arm = bool(self.get_parameter("force_arm").value)
        self.wait_for_start = bool(self.get_parameter("wait_for_start").value)
        self.status_stale_timeout_s = float(self.get_parameter("status_stale_timeout_s").value)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._cmd_pub = self.create_publisher(VehicleCommand, self._topic("in/vehicle_command"), 10)
        self._offboard_pub = self.create_publisher(OffboardControlMode, self._topic("in/offboard_control_mode"), sensor_qos)
        self._traj_pub = self.create_publisher(TrajectorySetpoint, self._topic("in/trajectory_setpoint"), sensor_qos)

        # legacy + _v1 names cover px4_msgs topic-version differences
        self.create_subscription(VehicleStatus, self._topic("out/vehicle_status"), self._status_cb, sensor_qos)
        self.create_subscription(VehicleStatus, self._topic("out/vehicle_status_v1"), self._status_cb, sensor_qos)
        self.create_subscription(VehicleLocalPosition, self._topic("out/vehicle_local_position"), self._pos_cb, sensor_qos)
        self.create_subscription(VehicleLocalPosition, self._topic("out/vehicle_local_position_v1"), self._pos_cb, sensor_qos)
        self.create_subscription(VehicleAttitude, self._topic("out/vehicle_attitude"), self._att_cb, sensor_qos)
        self.create_subscription(VehicleCommandAck, self._topic("out/vehicle_command_ack"), self._ack_cb, sensor_qos)
        self.create_subscription(VehicleGlobalPosition, self._topic("out/vehicle_global_position"), self._gpos_cb, sensor_qos)
        self.create_subscription(VehicleGlobalPosition, self._topic("out/vehicle_global_position_v1"), self._gpos_cb, sensor_qos)

        self.create_subscription(Bool, "start_mission", self._start_cb, 10)
        self.create_subscription(Bool, "end_mission", self._end_cb, 10)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_DISARMED
        self.failsafe = False
        self.x = self.y = self.z = 0.0  # NED metres
        self.vx = self.vy = self.vz = 0.0  # NED m/s
        self.yaw = 0.0
        self._launch_xy = None

        self._status_seen = False
        self._last_status_us = 0
        self._prestream_count = 0
        self._start_ok = not self.wait_for_start
        self._end_requested = False
        self._last_log_us = 0
        self._last_command_us = 0
        self._takeoff_started_us = 0
        self._link_acquired_fired = False
        self._heartbeat_velocity = False
        self._global_pos_valid = False
        self._pending_origin = None
        self._origin_send_us = 0
        self._origin_start_us = 0
        self._origin_confirmed = False

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
        self._end_requested = True

    def begin_return(self):
        if self.state not in (RETURNING, LANDING, DONE):
            self._begin_return()

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
        if self._global_pos_valid:
            self._pending_origin = None
            if not self._origin_confirmed:
                self._origin_confirmed = True
                self.get_logger().info("EKF global origin accepted (lat_lon_valid).")
            return
        elapsed = (self._now_us() - self._origin_start_us) / 1_000_000.0
        if elapsed > ORIGIN_CONFIRM_TIMEOUT_S:
            raise RuntimeError(
                f"EKF global origin not accepted after {elapsed:.0f}s "
                f"(vehicle_global_position.lat_lon_valid still false); RTL/Land unavailable."
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

    def on_link_acquired(self):
        """Mission hook called once when PX4 telemetry first arrives (pre-arm)."""

    def on_active_start(self):
        """Mission hook called once when OFFBOARD setpoints become active."""

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
        self.x, self.y, self.z = msg.x, msg.y, msg.z
        self.vx, self.vy, self.vz = msg.vx, msg.vy, msg.vz
        if self._launch_xy is None and all(math.isfinite(v) for v in (msg.x, msg.y)):
            self._launch_xy = (float(msg.x), float(msg.y))

    def _att_cb(self, msg: VehicleAttitude):
        w, x, y, z = msg.q  # PX4 order w, x, y, z
        self.yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _gpos_cb(self, msg: VehicleGlobalPosition):
        self._global_pos_valid = bool(msg.lat_lon_valid)

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
        if msg.data and not self._end_requested:
            self._end_requested = True
            self.get_logger().info("end_mission received, landing.")

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

    def _publish_heartbeat(self, position=True, velocity=None):
        if velocity is None:
            velocity = self._heartbeat_velocity
        off = OffboardControlMode()
        off.timestamp = int(Clock().now().nanoseconds / 1000)
        off.position = position
        off.velocity = velocity
        off.acceleration = False
        off.attitude = False
        off.body_rate = False
        self._offboard_pub.publish(off)

    def publish_position_setpoint(self, x, y, z, yaw=None, yawspeed=0.0):
        self._heartbeat_velocity = False
        traj = TrajectorySetpoint()
        traj.timestamp = int(Clock().now().nanoseconds / 1000)
        traj.position[0], traj.position[1], traj.position[2] = float(x), float(y), float(z)
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
        traj.position[0], traj.position[1], traj.position[2] = float(x), float(y), float(z)
        traj.velocity[0], traj.velocity[1], traj.velocity[2] = float(vx), float(vy), float(vz)
        for i in range(3):
            traj.acceleration[i] = float("nan")
        traj.yaw = wrap_pi(self.yaw if yaw is None else float(yaw))
        traj.yawspeed = float(yawspeed)
        self._traj_pub.publish(traj)

    def _hold_setpoint(self):
        lx, ly = self._launch_xy if self._launch_xy else (self.x, self.y)
        self.publish_position_setpoint(lx, ly, -abs(self.takeoff_altitude_m))

    def _log_throttled(self, msg: str, period_us: int = 1_000_000):
        now = self._now_us()
        if now - self._last_log_us > period_us:
            self._last_log_us = now
            self.get_logger().info(msg)

    # state machine

    def _tick(self):
        self._send_pending_origin()
        # OFFBOARD drops without an OffboardControlMode stream at >=2 Hz
        if self.state in (PRESTREAM, TAKEOFF, ENGAGE, ACTIVE):
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
                self._takeoff_started_us = self._now_us()
                self.get_logger().info("commanding OFFBOARD takeoff.")
                self.command_offboard_mode()
                self._last_command_us = self._now_us()
                self.state = ENGAGE

        elif self.state == TAKEOFF:
            self._hold_setpoint()
            if self._end_requested:
                self._begin_landing()
                return
            self.command_arm()
            self._publish_command_throttled(
                VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
                param7=self.takeoff_altitude_m,
            )
            target_z = -abs(self.takeoff_altitude_m)
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
            if self._end_requested:
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
            if self._end_requested:
                self._begin_landing()
                return
            sp = self.compute_setpoint()
            if sp is None:
                self._hold_setpoint()
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
            if self._end_requested:
                self._begin_landing()
                return
            self._publish_command_throttled(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
            if not self.is_armed:
                self.state = DONE
                self.get_logger().info("RTL complete and disarmed.")
            else:
                self._log_throttled(f"returning via RTL: nav_state={self.nav_state}")

        elif self.state == LANDING:
            if not self.is_armed:
                self.state = DONE
                self.get_logger().info("landed and disarmed.")

    def _begin_return(self):
        self.state = RETURNING
        self.get_logger().info("commanding AUTO.RTL.")
        self.command_return()

    def _begin_landing(self):
        self.state = LANDING
        self.get_logger().info("commanding NAV_LAND.")
        self._publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)


def main(args=None):
    rclpy.init(args=args)
    node = OffboardController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
