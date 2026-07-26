import math
from unittest.mock import Mock

import rclpy

from flight_intelligent.phased_orbits_mission import PhasedOrbitsMission


class TestPhasedOrbitsLanding:
    @classmethod
    def setup_class(cls):
        # The build-free companion deployment intentionally omits
        # These single-aircraft tests have no peers, so ORCA naturally no-ops.
        rclpy.init()

    @classmethod
    def teardown_class(cls):
        rclpy.shutdown()

    def setup_method(self):
        self.node = PhasedOrbitsMission()
        self.node._timer.cancel()

    def teardown_method(self):
        self.node.destroy_node()

    def test_excitation_is_position_only(self):
        self.node.center = (0.0, 0.0)
        self.node._anchor = (0.0, 0.0)
        self.node._climbed = True
        self.node._transit_us = 1_000_000
        self.node.x = self.node.y = 0.0
        self.node.excite_enabled = True
        self.node._excited = False
        self.node._excite_us = 4_000_000
        self.node._now_us = Mock(return_value=4_250_000)

        nominal = self.node._pre_orbit_setpoint()

        assert len(nominal) == 2
        position, _ = nominal
        assert len(position) == 3

        self.node._publish_world_state = Mock()
        self.node._nominal_setpoint = Mock(return_value=nominal)
        setpoint = self.node.compute_setpoint()
        assert len(setpoint) == 4

    def test_steady_hover_completes_before_excitation(self):
        self.node.center = (0.0, 0.0)
        self.node._anchor = (0.0, 0.0)
        self.node._climbed = True
        self.node._transit_us = 1_000_000
        self.node.x = self.node.y = 0.0
        self.node.excite_enabled = True
        self.node._excited = False
        self.node.pre_excite_hover_s = 30.0
        self.node._now_us = Mock(return_value=4_000_000)

        position, _ = self.node._pre_orbit_setpoint()
        assert position[2] == self.node._altitude_up()
        assert self.node._excite_us == 0

        self.node._now_us.return_value = 33_999_999
        position, _ = self.node._pre_orbit_setpoint()
        assert position[2] == self.node._altitude_up()
        assert self.node._excite_us == 0

        self.node._now_us.return_value = 34_000_000
        self.node._pre_orbit_setpoint()
        assert self.node._excite_us == 34_000_000

    def test_steady_hover_timer_resets_after_leaving_center(self):
        self.node.center = (0.0, 0.0)
        self.node._anchor = (0.0, 0.0)
        self.node._climbed = True
        self.node._transit_us = 1_000_000
        self.node.x = self.node.y = 0.0
        self.node.excite_enabled = True
        self.node._excited = False
        self.node.pre_excite_hover_s = 30.0
        self.node._now_us = Mock(return_value=4_000_000)

        self.node._pre_orbit_setpoint()
        assert self.node._pre_excite_hover_us == 4_000_000

        self.node.x = self.node.takeoff_acceptance_m + 0.1
        self.node._now_us.return_value = 20_000_000
        self.node._pre_orbit_setpoint()
        assert self.node._pre_excite_hover_us == 0

        self.node.x = 0.0
        self.node._now_us.return_value = 21_000_000
        self.node._pre_orbit_setpoint()
        assert self.node._pre_excite_hover_us == 21_000_000

    def test_return_to_anchor_requires_continuous_settle_before_land(self):
        self.node._orbit_gate_us = 1_000_000
        # The orbit is unbounded until commanded home, so stand in for a come-home
        # that ended it after two revolutions.
        self.node.orbit_duration = 2.0 * 2.0 * math.pi / self.node.omega
        self.node._anchor = (1.5, -0.5)  # ENU east, north
        self.node.x = -0.5
        self.node.y = 1.5
        self.node.vx = self.node.vy = 0.0
        self.node.request_land = Mock()

        end_s = (
            self.node.spiral_time_s
            + self.node.orbit_duration
            + self.node.return_duration
            + self.node.to_center_time_s
        )
        now_us = self.node._orbit_gate_us + math.ceil(end_s * 1_000_000) + 100_000
        self.node._now_us = Mock(return_value=now_us)

        first = self.node._nominal_setpoint()
        assert math.hypot(*first[0][:2]) < 1e-9
        self.node.request_land.assert_not_called()

        self.node._now_us.return_value = now_us + int(
            (self.node.land_settle_time_s + 0.1) * 1_000_000
        )
        second = self.node._nominal_setpoint()
        assert math.hypot(*second[0][:2]) < 1e-9
        self.node.request_land.assert_called_once()

    def test_the_orbit_has_no_timed_end(self):
        assert self.node.orbit_duration == math.inf

    def test_come_home_mid_orbit_keeps_the_peeloff_seam_continuous(self):
        # The peel-off restarts the orbit angle at phi and assumes the speed
        # modulation is back to zero, so it only joins the orbit on a whole
        # revolution. This fails if the come-home ends the orbit wherever the
        # command happened to land: the setpoint would step by up to 2R in a tick.
        self.node.mod_amp, self.node.mod_phase = 0.43, math.radians(204.0)
        gate = 1_000_000
        self.node._orbit_gate_us = gate
        rev = 2.0 * math.pi / self.node.omega
        orbit_t = 1.37 * rev  # a deliberately awkward instant
        self.node._now_us = Mock(
            return_value=gate + int((self.node.spiral_time_s + orbit_t) * 1e6)
        )

        self.node.on_return_home()

        assert self.node.orbit_duration >= orbit_t
        revs = self.node.orbit_duration / rev
        assert abs(revs - round(revs)) < 1e-9

        # Straddle the seam by 1 ms either side. Sampling any closer is defeated by
        # microsecond truncation in `end`, which leaves both samples on the orbit
        # side and makes the assertion vacuous. Over 2 ms the drone travels 4 mm at
        # orbit speed, well inside the tolerance.
        end = gate + int((self.node.spiral_time_s + self.node.orbit_duration) * 1e6)
        self.node._now_us.return_value = end - 1_000  # last orbit tick
        last_orbit, _ = self.node._nominal_setpoint()
        self.node._now_us.return_value = end + 1_000  # first peel-off tick
        first_peel, _ = self.node._nominal_setpoint()
        assert math.hypot(*(last_orbit[:2] - first_peel[:2])) < 0.01

    def test_come_home_during_the_spiral_ends_the_orbit_at_zero(self):
        gate = 1_000_000
        self.node._orbit_gate_us = gate
        self.node._now_us = Mock(
            return_value=gate + int(0.5 * self.node.spiral_time_s * 1e6)
        )

        self.node.on_return_home()
        assert self.node.orbit_duration == 0.0

        # The insertion finishes and hands straight to the peel-off, which starts
        # at the same point the insertion ends.
        end = gate + int(self.node.spiral_time_s * 1e6)
        self.node._now_us.return_value = end - 1_000
        last_spiral, _ = self.node._nominal_setpoint()
        self.node._now_us.return_value = end + 1_000
        first_peel, _ = self.node._nominal_setpoint()
        assert math.hypot(*(last_spiral[:2] - first_peel[:2])) < 0.01

    def test_come_home_during_the_climb_lands_at_the_pad(self):
        # The anchor re-latches to the live position every climb tick, so there is
        # nowhere to fly back to.
        self.node._climbed = False
        self.node.request_land = Mock()

        self.node.on_return_home()
        self.node.request_land.assert_called_once()

    def test_come_home_before_the_orbit_returns_to_the_anchor_and_lands(self):
        self.node.center = (0.0, 4.6)
        self.node._climbed = True
        self.node._transit_us = 1_000_000
        self.node._anchor = (0.0, 0.0)
        self.node.x = self.node.y = 0.0
        self.node.vx = self.node.vy = 0.0
        self.node.request_land = Mock()
        self.node._publish_world_state = Mock()

        # settled at the center: the staged ready state, before /begin_orbit
        now = self.node._transit_us + int(2.0 * self.node.to_center_time_s * 1e6)
        self.node._now_us = Mock(return_value=now)
        self.node.compute_setpoint()  # populates _last_nominal
        assert math.hypot(*self.node._last_nominal[0][:2]) > 1.0

        self.node.on_return_home()
        assert self.node._home_us == now

        arrive = now + int((self.node.to_center_time_s + 0.1) * 1e6)
        self.node._now_us.return_value = arrive
        landed, _ = self.node._nominal_setpoint()
        assert math.hypot(*landed[:2]) < 1e-9
        self.node.request_land.assert_not_called()

        self.node._now_us.return_value = arrive + int(
            (self.node.land_settle_time_s + 0.1) * 1e6
        )
        self.node._nominal_setpoint()
        self.node.request_land.assert_called_once()

    def test_begin_orbit_is_refused_after_a_come_home(self):
        from std_msgs.msg import Bool

        self.node._climbed = True
        self.node._home_requested = True

        self.node._orbit_gate_cb(Bool(data=True))
        assert self.node._orbit_gate_us == 0

    def test_random_spawn_stages_at_fixed_world_slot_before_center_transit(self):
        self.node.staging_enabled = True
        self.node._staging_complete = False
        self.node.spawn_e, self.node.spawn_n = 7.0, -8.0
        self.node.stage_e, self.node.stage_n = 1.0, 4.5
        self.node._anchor = (0.0, 0.0)
        self.node._climbed = True
        self.node.peer_namespaces = []
        self.node.x = self.node.y = 0.0
        self.node.vx = self.node.vy = 0.0
        start = 1_000_000
        self.node._now_us = Mock(return_value=start)

        position, _ = self.node._pre_orbit_setpoint()
        assert position[:2] == (0.0, 0.0)

        target_local = (self.node.stage_e - self.node.spawn_e,
                        self.node.stage_n - self.node.spawn_n)
        self.node.y, self.node.x = target_local  # local NED x=north, y=east
        duration = max(
            self.node.to_center_time_s,
            1.5 * math.hypot(*target_local) / self.node.staging_speed)
        settled_at = start + int((duration + 0.1) * 1e6)
        self.node._now_us.return_value = settled_at
        position, _ = self.node._pre_orbit_setpoint()

        assert self.node._staging_complete is True
        assert self.node._anchor == target_local
        assert position[:2] == (0.0, 0.0)
