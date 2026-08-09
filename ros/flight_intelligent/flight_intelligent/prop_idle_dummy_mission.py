"""Bench-only one-shot motor-idle mission gated by ``/start_mission``.

Selecting this executable establishes the diagnostic context. It is a deliberate
exception to the normal ``/start_mission`` arm-and-climb contract in
``rfd-mission-execution.md`` and ``rfd-command-transport.md``.

The node waits for one true Bool gate, streams direct-actuator OFFBOARD for one
second, arms normally, and commands PX4 minimum output to Motors 1-4 for at
most one minute. ``/end_mission`` ends the interval early. The node keeps
stopped outputs active throughout cleanup, escalates from normal to forced
disarm when PX4 does not confirm the normal request, and exits after disarm.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool

from px4_msgs.msg import (
    ActuatorMotors,
    OffboardControlMode,
    VehicleCommand,
    VehicleCommandAck,
    VehicleStatus,
)
from px4_offboard.gate_qos import MISSION_GATE_QOS


WAIT_LINK = "wait_link"
WAIT_START = "wait_start"
PRESTREAM = "prestream"
WAIT_OFFBOARD = "wait_offboard"
WAIT_ARM = "wait_arm"
HOLD_MINIMUM = "hold_minimum"
NORMAL_DISARM = "normal_disarm"
FORCE_DISARM = "force_disarm"
DONE = "done"

ARMED = VehicleStatus.ARMING_STATE_ARMED
DISARMED = VehicleStatus.ARMING_STATE_DISARMED
OFFBOARD = VehicleStatus.NAVIGATION_STATE_OFFBOARD
FORCE_DISARM_MAGIC = 21196.0


class PropIdleDummyMission(Node):
    """Run one bounded minimum-output motor test after ``/start_mission``."""

    def __init__(self):
        super().__init__("prop_idle_dummy_mission")
        self.declare_parameter("vehicle_namespace", "")
        self.declare_parameter("rate_hz", 25.0)
        self.declare_parameter("prestream_seconds", 1.0)
        self.declare_parameter("armed_seconds", 60.0)
        self.declare_parameter("transition_timeout_seconds", 3.0)
        self.declare_parameter("status_stale_timeout_seconds", 1.0)
        self.declare_parameter("normal_disarm_timeout_seconds", 0.6)

        self.ns = str(self.get_parameter("vehicle_namespace").value).strip("/")
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.prestream_seconds = float(
            self.get_parameter("prestream_seconds").value
        )
        self.armed_seconds = float(self.get_parameter("armed_seconds").value)
        self.transition_timeout_seconds = float(
            self.get_parameter("transition_timeout_seconds").value
        )
        self.status_stale_timeout_seconds = float(
            self.get_parameter("status_stale_timeout_seconds").value
        )
        self.normal_disarm_timeout_seconds = float(
            self.get_parameter("normal_disarm_timeout_seconds").value
        )
        self._validate_parameters()

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.mode_pub = self.create_publisher(
            OffboardControlMode,
            self._topic("in/offboard_control_mode"),
            sensor_qos,
        )
        self.motor_pub = self.create_publisher(
            ActuatorMotors,
            self._topic("in/actuator_motors"),
            sensor_qos,
        )
        self.command_pub = self.create_publisher(
            VehicleCommand,
            self._topic("in/vehicle_command"),
            10,
        )
        for suffix in ("out/vehicle_status", "out/vehicle_status_v4"):
            self.create_subscription(
                VehicleStatus,
                self._topic(suffix),
                self._status_cb,
                sensor_qos,
            )
        for suffix in ("out/vehicle_command_ack", "out/vehicle_command_ack_v1"):
            self.create_subscription(
                VehicleCommandAck,
                self._topic(suffix),
                self._ack_cb,
                sensor_qos,
            )
        self.create_subscription(
            Bool, "start_mission", self._start_cb, MISSION_GATE_QOS
        )
        self.create_subscription(Bool, "end_mission", self._end_cb, MISSION_GATE_QOS)
        self.create_subscription(
            Bool, "abort_mission", self._abort_cb, MISSION_GATE_QOS
        )

        self.status = None
        self.last_status_us = 0
        self.start_requested = False
        self.attempted = False
        self.exit_on_done = True
        self.state = WAIT_LINK
        self.state_started_us = self._now_us()
        self.last_force_disarm_us = 0
        self._last_wait_log_us = 0
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().warn(
            "BENCH-ONLY prop-idle dummy mission ready: one true "
            "/start_mission gate commands minimum output to Motors 1-4"
        )

    def _validate_parameters(self):
        if not 5.0 <= self.rate_hz <= 100.0:
            raise ValueError("rate_hz must be between 5 and 100")
        if not 1.0 <= self.prestream_seconds <= 5.0:
            raise ValueError("prestream_seconds must be between 1 and 5")
        if not 0.1 <= self.armed_seconds <= 60.0:
            raise ValueError("armed_seconds must be between 0.1 and 60")
        if not 1.0 <= self.transition_timeout_seconds <= 10.0:
            raise ValueError("transition_timeout_seconds must be between 1 and 10")
        if not 0.5 <= self.status_stale_timeout_seconds <= 5.0:
            raise ValueError(
                "status_stale_timeout_seconds must be between 0.5 and 5"
            )
        if not 0.2 <= self.normal_disarm_timeout_seconds <= 2.0:
            raise ValueError(
                "normal_disarm_timeout_seconds must be between 0.2 and 2"
            )

    def _topic(self, suffix: str) -> str:
        return f"/{self.ns}/fmu/{suffix}" if self.ns else f"/fmu/{suffix}"

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    @property
    def is_armed(self) -> bool:
        return self.status is not None and self.status.arming_state == ARMED

    def _status_cb(self, msg: VehicleStatus):
        self.status = msg
        self.last_status_us = self._now_us()

    def _ack_cb(self, msg: VehicleCommandAck):
        if msg.command not in (
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
        ):
            return
        if msg.result == VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED:
            self.get_logger().info(
                f"ACK command={msg.command} result={msg.result}"
            )
        else:
            self.get_logger().warn(
                f"ACK command={msg.command} result={msg.result}"
            )

    def _start_cb(self, msg: Bool):
        if not msg.data:
            return
        if self.attempted or self.start_requested:
            self.get_logger().info(
                f"duplicate start_mission ignored in state {self.state}"
            )
            return
        self.start_requested = True
        self.get_logger().warn("start_mission received; prop-idle attempt latched")

    def _abort_cb(self, msg: Bool):
        if msg.data:
            self._end_or_abort("abort_mission received")

    def _end_cb(self, msg: Bool):
        if msg.data:
            self._end_or_abort("end_mission received")

    def _end_or_abort(self, reason: str):
        if self.state in (NORMAL_DISARM, FORCE_DISARM, DONE):
            return
        if self.state in (WAIT_LINK, WAIT_START):
            self.attempted = True
            self._finish(f"{reason} before start")
            return
        self._begin_cleanup(reason)

    def _set_state(self, state: str):
        self.state = state
        self.state_started_us = self._now_us()

    def _elapsed(self) -> float:
        return (self._now_us() - self.state_started_us) * 1e-6

    def _status_fresh(self) -> bool:
        return (
            self.status is not None
            and self._now_us() - self.last_status_us
            <= int(self.status_stale_timeout_seconds * 1_000_000)
        )

    def _publish_outputs(self, stopped: bool):
        timestamp = self._now_us()
        mode = OffboardControlMode()
        mode.timestamp = timestamp
        mode.direct_actuator = True
        self.mode_pub.publish(mode)

        motors = ActuatorMotors()
        motors.timestamp = timestamp
        motors.timestamp_sample = timestamp
        motors.reversible_flags = 0
        motors.control = [math.nan] * 12
        if not stopped:
            motors.control[:4] = [0.0] * 4
        self.motor_pub.publish(motors)

    def _command(self, command: int, param1: float, param2: float = 0.0):
        msg = VehicleCommand()
        msg.timestamp = self._now_us()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 0
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.command_pub.publish(msg)

    def _begin_attempt(self):
        self.attempted = True
        if self.status is None:
            self._finish("vehicle status disappeared before start")
            return
        if self.status.arming_state != DISARMED:
            self._finish("vehicle was already armed; refusing test")
            return
        if self.status.failsafe:
            self._finish("vehicle is in failsafe; refusing test")
            return
        self.get_logger().warn(
            f"PRESTREAM minimum outputs for {self.prestream_seconds:.1f}s; "
            f"preflight={self.status.pre_flight_checks_pass}"
        )
        self._set_state(PRESTREAM)

    def _begin_cleanup(self, reason: str):
        self.get_logger().warn(f"CLEANUP: {reason}")
        self._publish_outputs(stopped=True)
        if not self.is_armed:
            self._finish("disarm confirmed")
            return
        self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self._set_state(NORMAL_DISARM)

    def _begin_forced_disarm(self, reason: str):
        self.get_logger().error(f"CLEANUP: {reason}; using forced disarm")
        self.last_force_disarm_us = 0
        self._set_state(FORCE_DISARM)

    def _finish(self, message: str):
        self._set_state(DONE)
        self.get_logger().info(f"DONE: {message}; terminating dummy mission")
        if self.exit_on_done:
            self.context.shutdown()

    def _publish_for_state(self):
        if self.state in (PRESTREAM, WAIT_OFFBOARD, WAIT_ARM, HOLD_MINIMUM):
            self._publish_outputs(stopped=False)
        elif self.state in (NORMAL_DISARM, FORCE_DISARM):
            self._publish_outputs(stopped=True)

    def _tick(self):
        now = self._now_us()
        if self.state == WAIT_LINK:
            if self._status_fresh():
                self._set_state(WAIT_START)
            else:
                if now - self._last_wait_log_us >= 1_000_000:
                    self._last_wait_log_us = now
                    self.get_logger().info("waiting for vehicle_status")
                return

        if self.state == WAIT_START:
            if not self._status_fresh():
                self._set_state(WAIT_LINK)
                return
            if self.start_requested:
                self._begin_attempt()
            return

        if self.state == DONE:
            return

        self._publish_for_state()

        if not self._status_fresh():
            if self.state == FORCE_DISARM:
                self._send_forced_disarm(now)
                return
            self._begin_forced_disarm("vehicle_status became stale")
            return

        if self.state == PRESTREAM:
            if self._elapsed() >= self.prestream_seconds:
                self._command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                    1.0,
                    6.0,
                )
                self.get_logger().info("OFFBOARD requested")
                self._set_state(WAIT_OFFBOARD)
        elif self.state == WAIT_OFFBOARD:
            if self.status.nav_state == OFFBOARD:
                self._command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    1.0,
                )
                self.get_logger().warn("OFFBOARD accepted; normal arm requested")
                self._set_state(WAIT_ARM)
            elif self._elapsed() >= self.transition_timeout_seconds:
                self._begin_cleanup("OFFBOARD was not accepted")
        elif self.state == WAIT_ARM:
            if self.status.nav_state != OFFBOARD:
                self._begin_cleanup("mode changed before arm confirmation")
            elif self.is_armed:
                self.get_logger().warn(
                    f"ARMED: holding minimum outputs for "
                    f"{self.armed_seconds:.3f}s"
                )
                self._set_state(HOLD_MINIMUM)
            elif self._elapsed() >= self.transition_timeout_seconds:
                self._begin_cleanup("arm was not accepted")
        elif self.state == HOLD_MINIMUM:
            if not self.is_armed:
                self._finish("PX4 disarmed during the minimum-output hold")
            elif self.status.nav_state != OFFBOARD:
                self._begin_cleanup("mode changed while armed")
            elif self._elapsed() >= self.armed_seconds:
                self._begin_cleanup("minimum-output interval complete")
        elif self.state == NORMAL_DISARM:
            if not self.is_armed:
                self._finish("normal disarm confirmed")
            elif self._elapsed() >= self.normal_disarm_timeout_seconds:
                self._begin_forced_disarm("normal disarm was not confirmed")
        elif self.state == FORCE_DISARM:
            if not self.is_armed:
                self._finish("forced disarm confirmed")
            else:
                self._send_forced_disarm(now)

    def _send_forced_disarm(self, now_us: int):
        self._publish_outputs(stopped=True)
        if now_us - self.last_force_disarm_us >= 200_000:
            self.last_force_disarm_us = now_us
            self._command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                0.0,
                FORCE_DISARM_MAGIC,
            )

    def shutdown_cleanup(self):
        if not self.is_armed:
            return
        self.get_logger().error(
            "shutdown requested while armed; holding stopped outputs until disarm"
        )
        next_command_us = 0
        while rclpy.ok() and self.is_armed:
            now = self._now_us()
            self._publish_outputs(stopped=True)
            if now >= next_command_us:
                self._command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    0.0,
                    FORCE_DISARM_MAGIC,
                )
                next_command_us = now + 200_000
            rclpy.spin_once(self, timeout_sec=0.04)


def main():
    rclpy.init()
    node = PropIdleDummyMission()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown_cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
