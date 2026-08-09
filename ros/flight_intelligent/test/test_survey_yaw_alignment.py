import math
from unittest.mock import Mock

import rclpy

from flight_intelligent.survey_mission import SurveyMission
from px4_offboard.controller import wrap_pi


class TestSurveyYawAlignment:
    @classmethod
    def setup_class(cls):
        rclpy.init()

    @classmethod
    def teardown_class(cls):
        rclpy.shutdown()

    def setup_method(self):
        self.node = SurveyMission()
        self.node._timer.cancel()
        self.node._send_pending_origin = Mock()
        self.node._anchor = (0.0, 0.0)
        self.node._climbed = True
        self.node._yaw_alignment_complete = True
        self.node._gate_us = 1_000_000

    def teardown_method(self):
        self.node.destroy_node()

    def test_post_takeoff_alignment_settles_at_cardinal_south(self):
        self.node.post_takeoff_align = True
        self.node.post_takeoff_yaw_ned = math.pi
        self.node.post_takeoff_yaw_enu = -math.pi / 2.0
        self.node.post_takeoff_yaw_tolerance = math.radians(5.0)
        self.node.post_takeoff_yaw_settle_s = 1.0
        self.node._yaw_alignment_complete = False
        self.node._yaw_cmd_enu = None
        self.node.yaw = math.radians(179.0)

        first = self.node._post_takeoff_alignment_setpoint(2_000_000)
        second = self.node._post_takeoff_alignment_setpoint(3_100_000)

        assert abs(wrap_pi(first[3] - math.pi)) < 1e-9
        assert abs(wrap_pi(second[3] - math.pi)) < 1e-9
        assert self.node._yaw_alignment_complete
        assert self.node._hold_yaw_enu == -math.pi / 2.0
