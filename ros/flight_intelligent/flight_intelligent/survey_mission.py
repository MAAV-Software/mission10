"""Survey mission — single drone, serpentine coverage plus a cube circuit.

The coverage pattern the object-map layer is built around: serpentine lanes
over a rectangular field, a revisit gap, a cross-hatch second pass, then a
cube edge circuit (pure single-axis translations in E, N, and U) for VIO
translational excitation, and home. Geometry lives in ``flight_lib.survey``;
this node walks the schedule at the setpoint rate.

Operator gates (same shape as the phased-orbits mission):

    /start_mission  (takeoff)  -> arm + climb, hold at the launch anchor
    /begin_survey   (execute)  -> fly the schedule (also accepts /begin_orbit,
                                  so the existing sitl.sh gate works unchanged)
    /end_mission    (come home)-> abandon the remaining schedule, return to
                                  the launch anchor, settle, NAV_LAND
    /abort_mission  (abort)    -> AUTO.LAND in place, now (base controller)

Frame: launch-relative ENU on the post-climb anchor, exactly like the orbit
mission — the climb applies no horizontal correction, then the believed
position is frozen and all geometry rides on it. The real-flight profile
aligns to cardinal south after the climb and rotates the survey geometry to
that heading. Yaw tracks the direction of travel through a slew-rate limiter
(a serpentine reversal is a 180° step; PX4 gets a continuous ramp instead).

"""
from __future__ import annotations

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Bool

from flight_lib import (
    build_survey_schedule,
    rotate_schedule,
    schedule_duration,
    schedule_setpoint,
)
from px4_offboard.controller import OffboardController, wrap_pi


def enu_to_ned_setpoint(position_enu, yaw_enu):
    east, north, up = position_enu
    return float(north), float(east), -float(up), wrap_pi(math.pi / 2.0 - float(yaw_enu))


class SurveyMission(OffboardController):
    """Serpentine + cross-hatch + cube, two-command: takeoff+hover, then fly."""

    def __init__(self):
        super().__init__("survey_mission")

        self.declare_parameter("field_e0_m", 2.0)
        self.declare_parameter("field_n0_m", 2.0)
        self.declare_parameter("lane_axis_deg", 0.0)
        self.declare_parameter("field_length_m", 10.0)
        self.declare_parameter("field_width_m", 6.0)
        self.declare_parameter("lane_spacing_m", 2.0)
        self.declare_parameter("speed_mps", 1.5)
        self.declare_parameter("revisit_gap_s", 8.0)
        self.declare_parameter("crosshatch", True)
        self.declare_parameter("cube_side_m", 2.0)
        self.declare_parameter("cube_speed_mps", 1.0)
        self.declare_parameter("yaw_mode", "track")  # track | hold
        self.declare_parameter("yaw_rate_max_dps", 45.0)
        self.declare_parameter("to_home_time_s", 4.0)
        self.declare_parameter("land_xy_acceptance_m", 0.35)
        self.declare_parameter("land_vxy_acceptance_mps", 0.20)
        self.declare_parameter("land_settle_time_s", 1.0)
        self.declare_parameter("survey_auto_start", False)
        # Optional legacy alignment: rotate the whole schedule by the
        # operator's pre-arm heading.
        self.declare_parameter("align_to_launch_yaw", False)
        self.declare_parameter("post_takeoff_yaw_alignment", False)
        self.declare_parameter("post_takeoff_yaw_deg", 180.0)
        self.declare_parameter("post_takeoff_yaw_tolerance_deg", 5.0)
        self.declare_parameter("post_takeoff_yaw_settle_s", 1.0)
        self.declare_parameter("origin_lat", 42.2658783)
        self.declare_parameter("origin_lon", -83.7487304)
        self.declare_parameter("origin_alt", 0.0)
        # SITL's EV-only estimate has no global anchor, so the node sets one at
        # link-up; a real GPS estimate already has one, and sending the origin
        # then can shove the local frame (see phased_orbits_mission).
        self.declare_parameter("set_global_origin_on_link", True)
        # Same anchoring rationale as the orbit mission: the EKF local frame
        # can jump metres at rotor spool-up (GPS multipath under the net), so
        # climb with zero horizontal correction and freeze the post-climb
        # belief as the geometry frame.
        self.declare_parameter("anchor_frame_after_climb", True)

        self.alt = self.takeoff_altitude_m
        self.schedule = build_survey_schedule(
            field_e0_m=float(self.get_parameter("field_e0_m").value),
            field_n0_m=float(self.get_parameter("field_n0_m").value),
            lane_axis_deg=float(self.get_parameter("lane_axis_deg").value),
            field_length_m=float(self.get_parameter("field_length_m").value),
            field_width_m=float(self.get_parameter("field_width_m").value),
            lane_spacing_m=float(self.get_parameter("lane_spacing_m").value),
            altitude_m=self.alt,
            speed_mps=float(self.get_parameter("speed_mps").value),
            revisit_gap_s=float(self.get_parameter("revisit_gap_s").value),
            crosshatch=bool(self.get_parameter("crosshatch").value),
            cube_side_m=float(self.get_parameter("cube_side_m").value),
            cube_speed_mps=float(self.get_parameter("cube_speed_mps").value),
        )
        self.yaw_mode = str(self.get_parameter("yaw_mode").value)
        self.yaw_rate_max = math.radians(
            float(self.get_parameter("yaw_rate_max_dps").value))
        self.to_home_time_s = float(self.get_parameter("to_home_time_s").value)
        self.land_xy_acceptance = float(self.get_parameter("land_xy_acceptance_m").value)
        self.land_vxy_acceptance = float(
            self.get_parameter("land_vxy_acceptance_mps").value)
        self.land_settle_time_s = float(self.get_parameter("land_settle_time_s").value)
        self.auto_start = bool(self.get_parameter("survey_auto_start").value)
        self.anchor_after_climb = bool(
            self.get_parameter("anchor_frame_after_climb").value)
        self.post_takeoff_align = bool(
            self.get_parameter("post_takeoff_yaw_alignment").value)
        self.post_takeoff_yaw_ned = wrap_pi(math.radians(
            float(self.get_parameter("post_takeoff_yaw_deg").value)))
        self.post_takeoff_yaw_enu = wrap_pi(
            math.pi / 2.0 - self.post_takeoff_yaw_ned)
        self.post_takeoff_yaw_tolerance = math.radians(
            float(self.get_parameter("post_takeoff_yaw_tolerance_deg").value))
        self.post_takeoff_yaw_settle_s = float(
            self.get_parameter("post_takeoff_yaw_settle_s").value)
        self._anchor = (0.0, 0.0)  # ENU (east, north); latched during climb
        self._climbed = False
        self._yaw_alignment_complete = not self.post_takeoff_align
        self._yaw_alignment_settle_us = 0
        self._gate_us = 0
        self._done_logged = False
        self._label_logged = None
        self._hold_yaw_enu = None
        self._yaw_cmd_enu = None
        self._last_tick_us = 0
        self._home_us = 0
        self._home_from = None
        self._land_settle_us = 0
        self._land_requested = False
        self._landing_logged = False
        self.create_subscription(Bool, "begin_survey", self._gate_cb, 10)
        self.create_subscription(Bool, "begin_orbit", self._gate_cb, 10)

    def on_link_acquired(self):
        if not bool(self.get_parameter("set_global_origin_on_link").value):
            self.get_logger().info(
                "set_global_origin_on_link=false; relying on GPS for the EKF origin")
            return
        lat = float(self.get_parameter("origin_lat").value)
        lon = float(self.get_parameter("origin_lon").value)
        alt = float(self.get_parameter("origin_alt").value)
        self.set_global_origin(lat, lon, alt)
        self.get_logger().info(f"set EKF global origin: {lat:.7f}, {lon:.7f}, {alt:.1f}")

    def _gate_cb(self, msg: Bool):
        if not msg.data or self._gate_us or self._home_requested:
            return
        if not self._climbed or not self._yaw_alignment_complete:
            self.get_logger().warn(
                "begin_survey ignored: climb/yaw alignment still in progress")
            return
        self._gate_us = self._now_us()
        self.get_logger().info(
            f"begin_survey received: {len(self.schedule)} segments, "
            f"{schedule_duration(self.schedule):.0f}s planned")

    def on_active_start(self):
        hold_yaw_ned = self._launch_yaw if self._launch_yaw_latched else self.yaw
        self._hold_yaw_enu = wrap_pi(math.pi / 2.0 - hold_yaw_ned)
        self._yaw_cmd_enu = self._hold_yaw_enu
        if self.post_takeoff_align:
            self.schedule = rotate_schedule(
                self.schedule, self.post_takeoff_yaw_enu)
            self.get_logger().info(
                f"schedule aligned to post-takeoff cardinal heading "
                f"({math.degrees(self.post_takeoff_yaw_ned):.1f} deg NED)")
        elif bool(self.get_parameter("align_to_launch_yaw").value):
            self.schedule = rotate_schedule(self.schedule, self._hold_yaw_enu)
            self.get_logger().info(
                f"schedule rotated to launch heading "
                f"({math.degrees(self._hold_yaw_enu):.1f} deg ENU): "
                "the field lies straight out the nose")
        self.get_logger().info(
            f"survey active: {len(self.schedule)} segments, "
            f"{schedule_duration(self.schedule):.0f}s planned, alt={self.alt:.1f}m")

    def on_return_home(self):
        """/end_mission: abandon the rest of the schedule and come home."""
        if not self._climbed:
            self.get_logger().info("come home during climb; landing at the pad.")
            self.request_land()
            return
        if self._home_us == 0:
            self._home_from = self._current_nominal()[0]
            self._home_us = self._now_us()
            self.get_logger().info(
                "come home: leaving the schedule, returning to the launch anchor.")

    def _slewed_yaw(
        self,
        target_enu: float,
        now_us: int,
        force_target: bool = False,
    ) -> float:
        if (
            self.yaw_mode == "hold"
            and self._hold_yaw_enu is not None
            and not force_target
        ):
            return self._hold_yaw_enu
        if self._yaw_cmd_enu is None:
            self._yaw_cmd_enu = target_enu
        dt = (now_us - self._last_tick_us) * 1e-6 if self._last_tick_us else 0.0
        dt = min(max(dt, 0.0), 0.25)
        err = wrap_pi(target_enu - self._yaw_cmd_enu)
        step = self.yaw_rate_max * dt
        self._yaw_cmd_enu = wrap_pi(
            self._yaw_cmd_enu + max(-step, min(step, err)))
        return self._yaw_cmd_enu

    def _schedule_elapsed_s(self, now_us: int) -> float:
        return max(0.0, (now_us - self._gate_us) * 1e-6)

    def _current_nominal(self, now_us: int | None = None):
        """(position, yaw target) of the schedule at this instant, pre-slew."""
        now = self._now_us() if now_us is None else now_us
        t = self._schedule_elapsed_s(now)
        pos, direction, done = schedule_setpoint(self.schedule, t)
        if direction != (0.0, 0.0):
            yaw = math.atan2(direction[1], direction[0])
        else:
            yaw = self._hold_yaw_enu or 0.0
        return pos, yaw, done

    def _post_takeoff_alignment_setpoint(self, now_us: int):
        yaw_error = wrap_pi(self.post_takeoff_yaw_ned - self.yaw)
        if abs(yaw_error) <= self.post_takeoff_yaw_tolerance:
            if self._yaw_alignment_settle_us == 0:
                self._yaw_alignment_settle_us = now_us
            settled_s = (now_us - self._yaw_alignment_settle_us) * 1e-6
        else:
            self._yaw_alignment_settle_us = 0
            settled_s = 0.0

        if settled_s >= self.post_takeoff_yaw_settle_s:
            self._yaw_alignment_complete = True
            self._hold_yaw_enu = self.post_takeoff_yaw_enu
            self._yaw_cmd_enu = self.post_takeoff_yaw_enu
            self.get_logger().info(
                f"post-takeoff yaw aligned south: "
                f"error={math.degrees(yaw_error):+.1f}deg; "
                "holding for begin_survey")
            if self.auto_start and self._gate_us == 0:
                self._gate_us = now_us

        return self._emit(
            (0.0, 0.0, self.alt),
            self.post_takeoff_yaw_enu,
            now_us,
            force_yaw_target=True,
        )

    def compute_setpoint(self):
        now = self._now_us()
        try:
            if not self._climbed:
                if self.anchor_after_climb:
                    # re-latch every tick: zero horizontal correction during the
                    # climb; a frame reset moves the anchor, not the drone
                    self._anchor = (self.y, self.x)  # NED y=east, x=north
                if abs(self.z - self._takeoff_target_z()) <= self.takeoff_acceptance_m:
                    self._climbed = True
                    ae, an = self._anchor
                    self.get_logger().info(
                        f"climbed; "
                        f"{'aligning cardinal yaw' if self.post_takeoff_align else 'holding for begin_survey'}. "
                        f"frame anchor ENU=({ae:+.2f},{an:+.2f})")
                    if (
                        self.auto_start
                        and self._yaw_alignment_complete
                        and self._gate_us == 0
                    ):
                        self._gate_us = self._now_us()
                return self._emit((0.0, 0.0, self.alt), self._hold_yaw_enu or 0.0, now)

            if not self._yaw_alignment_complete:
                return self._post_takeoff_alignment_setpoint(now)

            if self._home_us:  # /end_mission path
                return self._emit(*self._home_setpoint(now), now)

            if self._gate_us == 0:  # staged ready state
                return self._emit((0.0, 0.0, self.alt), self._hold_yaw_enu or 0.0, now)

            pos, yaw, done = self._current_nominal(now)
            label = None
            t = self._schedule_elapsed_s(now)
            for seg in self.schedule:
                if t <= seg.duration_s:
                    label = seg.label
                    break
                t -= seg.duration_s
            if label and label != self._label_logged:
                self._label_logged = label
                self.get_logger().info(f"segment: {label}")
            if done:
                return self._emit(*self._settle_and_land(pos, yaw, now), now)
            return self._emit(pos, yaw, now)
        finally:
            self._last_tick_us = now

    def _emit(
        self,
        pos_enu,
        yaw_target_enu,
        now_us,
        force_yaw_target: bool = False,
    ):
        yaw = self._slewed_yaw(
            float(yaw_target_enu), now_us, force_target=force_yaw_target)
        ae, an = self._anchor
        return enu_to_ned_setpoint(
            (pos_enu[0] + ae, pos_enu[1] + an, pos_enu[2]), yaw)

    def _home_setpoint(self, now_us):
        """Blend from where the schedule was abandoned back to the anchor."""
        u = min(1.0, max(
            0.0, (now_us - self._home_us) * 1e-6) / max(1e-3, self.to_home_time_s))
        s = u * u * (3.0 - 2.0 * u)
        fe, fn, fz = self._home_from
        pos = (fe * (1.0 - s), fn * (1.0 - s), fz + (self.alt - fz) * s)
        yaw = self._yaw_cmd_enu if self._yaw_cmd_enu is not None else 0.0
        if u >= 1.0:
            return self._settle_and_land(pos, yaw, now_us)
        return pos, yaw

    def _settle_and_land(self, pos, yaw, now_us):
        if not self._done_logged:
            self._done_logged = True
            self.get_logger().info("schedule complete; settling at the anchor.")
        ae, an = self._anchor
        xy_error = math.hypot(self.x - an, self.y - ae)
        vxy = math.hypot(self.vx, self.vy)
        settled = (
            math.isfinite(xy_error) and math.isfinite(vxy)
            and xy_error <= self.land_xy_acceptance
            and vxy <= self.land_vxy_acceptance
        )
        if settled:
            if self._land_settle_us == 0:
                self._land_settle_us = now_us
            dwell = (now_us - self._land_settle_us) * 1e-6
        else:
            self._land_settle_us = 0
            dwell = 0.0
        if dwell >= self.land_settle_time_s and not self._land_requested:
            self._land_requested = True
            self.get_logger().info(
                f"anchor settled: xy_error={xy_error:.2f}m vxy={vxy:.2f}m/s; "
                "requesting NAV_LAND.")
            self.request_land()
        return pos, yaw


def main(args=None):
    rclpy.init(args=args)
    node = SurveyMission()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
