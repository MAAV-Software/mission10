from unittest.mock import Mock

import rclpy
from std_msgs.msg import Bool

from flight_intelligent.prop_idle_dummy_mission import (
    DONE,
    FORCE_DISARM,
    HOLD_MINIMUM,
    NORMAL_DISARM,
    OFFBOARD,
    PRESTREAM,
    WAIT_ARM,
    WAIT_LINK,
    WAIT_OFFBOARD,
    WAIT_START,
    PropIdleDummyMission,
)
from px4_msgs.msg import VehicleCommand, VehicleStatus


class TestPropIdleDummyMission:
    @classmethod
    def setup_class(cls):
        rclpy.init()

    @classmethod
    def teardown_class(cls):
        rclpy.shutdown()

    def setup_method(self):
        self.node = PropIdleDummyMission()
        self.node.timer.cancel()
        self.node.exit_on_done = False
        self.now = 1_000_000
        self.node._now_us = Mock(side_effect=lambda: self.now)
        self.node._publish_outputs = Mock()
        self.node._command = Mock()
        self.status = VehicleStatus()
        self.status.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.status.nav_state = VehicleStatus.NAVIGATION_STATE_AUTO_LOITER
        self.status.failsafe = False
        self.status.pre_flight_checks_pass = True
        self.node.status = self.status
        self.node.last_status_us = self.now
        self.node.state = WAIT_START
        self.node.state_started_us = self.now

    def teardown_method(self):
        self.node.destroy_node()

    def advance(self, seconds):
        self.now += int(seconds * 1_000_000)
        self.node.last_status_us = self.now
        self.node._tick()

    def test_false_gate_does_not_start(self):
        self.node._start_cb(Bool(data=False))
        self.node._tick()
        assert self.node.state == WAIT_START
        assert not self.node.attempted

    def test_wait_link_is_passive_without_fresh_status(self):
        self.node.state = WAIT_LINK
        self.node.status = None
        self.node.last_status_us = 0

        self.node._tick()

        assert self.node.state == WAIT_LINK
        self.node._publish_outputs.assert_not_called()
        self.node._command.assert_not_called()

    def test_true_gate_runs_one_bounded_attempt_and_disarms(self):
        self.node._start_cb(Bool(data=True))
        self.node._tick()
        assert self.node.state == PRESTREAM

        self.advance(self.node.prestream_seconds)
        assert self.node.state == WAIT_OFFBOARD
        self.node._command.assert_called_with(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0
        )

        self.status.nav_state = OFFBOARD
        self.advance(0.04)
        assert self.node.state == WAIT_ARM
        self.node._command.assert_called_with(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0
        )

        self.status.arming_state = VehicleStatus.ARMING_STATE_ARMED
        self.advance(0.04)
        assert self.node.state == HOLD_MINIMUM

        self.advance(self.node.armed_seconds)
        assert self.node.state == NORMAL_DISARM
        self.node._command.assert_called_with(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0
        )

        self.status.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.advance(0.04)
        assert self.node.state == DONE

        self.node._start_cb(Bool(data=True))
        self.node._tick()
        assert self.node.state == DONE

    def test_mode_change_while_armed_starts_cleanup(self):
        self.node.attempted = True
        self.node.state = HOLD_MINIMUM
        self.node.state_started_us = self.now
        self.status.arming_state = VehicleStatus.ARMING_STATE_ARMED
        self.status.nav_state = VehicleStatus.NAVIGATION_STATE_AUTO_LOITER

        self.node._tick()

        assert self.node.state == NORMAL_DISARM
        self.node._publish_outputs.assert_called_with(stopped=True)

    def test_stale_status_while_armed_enters_forced_cleanup(self):
        self.node.attempted = True
        self.node.state = HOLD_MINIMUM
        self.node.state_started_us = self.now
        self.status.arming_state = VehicleStatus.ARMING_STATE_ARMED
        self.status.nav_state = OFFBOARD
        self.node.last_status_us = (
            self.now
            - int(self.node.status_stale_timeout_seconds * 1_000_000)
            - 1
        )

        self.node._tick()

        assert self.node.state == FORCE_DISARM

    def test_end_mission_stops_the_active_interval(self):
        self.node.attempted = True
        self.node.state = HOLD_MINIMUM
        self.node.state_started_us = self.now
        self.status.arming_state = VehicleStatus.ARMING_STATE_ARMED
        self.status.nav_state = OFFBOARD

        self.node._end_cb(Bool(data=True))

        assert self.node.state == NORMAL_DISARM
        self.node._publish_outputs.assert_called_with(stopped=True)
        self.node._command.assert_called_with(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0
        )
