from unittest.mock import Mock

import rclpy
from px4_msgs.msg import VehicleCommand, VehicleStatus
from std_msgs.msg import Bool

from px4_offboard.controller import (
    ACTIVE,
    DONE,
    LANDING,
    LAND_REQUESTED,
    RETURNING,
    OffboardController,
)


class TestControllerHandoff:
    @classmethod
    def setup_class(cls):
        rclpy.init()

    @classmethod
    def teardown_class(cls):
        rclpy.shutdown()

    def setup_method(self):
        self.node = OffboardController("test_controller_handoff")
        self.node._timer.cancel()
        self.node._send_pending_origin = Mock()

    def teardown_method(self):
        self.node.destroy_node()

    def test_launch_reference_holds_operator_heading(self):
        self.node._launch_xy = (1.0, 2.0)
        self.node.z = -0.28
        self.node.yaw = 0.42
        self.node._z_valid = True
        self.node._attitude_seen = True

        assert self.node._latch_launch_reference()
        self.node.publish_position_setpoint = Mock()
        self.node._hold_setpoint()

        self.node.publish_position_setpoint.assert_called_once_with(
            1.0, 2.0, -self.node.takeoff_altitude_m, 0.42
        )

    def test_rejected_land_keeps_offboard_hold_until_auto_land(self):
        self.node.state = LAND_REQUESTED
        self.node.arm_state = VehicleStatus.ARMING_STATE_ARMED
        self.node.nav_state = VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self.node._handoff_hold = (1.0, 2.0, -3.0, 0.4)
        self.node._publish_heartbeat = Mock()
        self.node._publish_handoff_hold = Mock()
        self.node._publish_command_throttled = Mock()

        self.node._tick()

        assert self.node.state == LAND_REQUESTED
        self.node._publish_heartbeat.assert_called_once()
        self.node._publish_handoff_hold.assert_called_once()
        self.node._publish_command_throttled.assert_called_once_with(
            VehicleCommand.VEHICLE_CMD_NAV_LAND
        )

        self.node.nav_state = VehicleStatus.NAVIGATION_STATE_AUTO_LAND
        self.node._tick()
        assert self.node.state == LANDING

    def test_rtl_without_global_altitude_falls_back_to_local_land(self):
        self.node._global_xy_valid = True
        self.node._global_alt_valid = False
        self.node._publish_command = Mock()

        self.node._begin_return()

        assert self.node.state == LAND_REQUESTED
        self.node._publish_command.assert_called_once_with(
            VehicleCommand.VEHICLE_CMD_NAV_LAND
        )

    def test_rejected_rtl_keeps_offboard_heartbeat_and_hold(self):
        self.node.state = RETURNING
        self.node.arm_state = VehicleStatus.ARMING_STATE_ARMED
        self.node.nav_state = VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self.node._publish_heartbeat = Mock()
        self.node._publish_handoff_hold = Mock()
        self.node._publish_command_throttled = Mock()

        self.node._tick()

        assert self.node.state == RETURNING
        self.node._publish_heartbeat.assert_called_once()
        self.node._publish_handoff_hold.assert_called_once()

        self.node.arm_state = VehicleStatus.ARMING_STATE_DISARMED
        self.node._tick()
        assert self.node.state == DONE

    def test_abort_lands_in_place_from_active_and_from_returning(self):
        # /abort_mission preempts whatever the mission is doing, including its own
        # return, because ACTIVE checks the flag before calling compute_setpoint.
        for state in (ACTIVE, RETURNING):
            node = OffboardController("test_controller_abort")
            node._timer.cancel()
            node._send_pending_origin = Mock()
            node.state = state
            node.arm_state = VehicleStatus.ARMING_STATE_ARMED
            node.nav_state = VehicleStatus.NAVIGATION_STATE_OFFBOARD
            node._publish_heartbeat = Mock()
            node._publish_command = Mock()
            node.compute_setpoint = Mock()

            node._abort_cb(Bool(data=True))
            node._tick()

            assert node.state == LAND_REQUESTED, state
            node._publish_command.assert_called_once_with(
                VehicleCommand.VEHICLE_CMD_NAV_LAND
            )
            node.compute_setpoint.assert_not_called()
            node.destroy_node()

    def test_come_home_calls_the_mission_hook_once(self):
        # Latched: re-entering a scheduled return partway through would fling the
        # vehicle back to where the schedule starts.
        self.node.on_return_home = Mock()

        self.node._end_cb(Bool(data=True))
        self.node._end_cb(Bool(data=True))

        self.node.on_return_home.assert_called_once()
        assert self.node._home_requested

    def test_come_home_does_not_land_in_place(self):
        # The whole point of the split: /end_mission must not reach _begin_landing.
        self.node.state = ACTIVE
        self.node.arm_state = VehicleStatus.ARMING_STATE_ARMED
        self.node._publish_heartbeat = Mock()
        self.node._publish_command = Mock()
        self.node.compute_setpoint = Mock(return_value=None)
        self.node._hold_setpoint = Mock()

        self.node._end_cb(Bool(data=True))
        self.node._tick()

        assert self.node.state == ACTIVE
        assert not self.node._abort_requested
        self.node._publish_command.assert_not_called()
