"""ROS rim for the mission engine: subscriptions in, setpoints out.

The rim owns transport and nothing else. It samples flight state, joins
detections to the pose at their image stamp, ticks `MissionEngine`, and
publishes the setpoint the engine returns. Every decision — phase changes,
aborts, dip policy, coverage — belongs to the pure core.

Operator gates follow the survey mission's shape, so one publish drives
either node:

    /start_mission   arm and climb to the survey altitude (base controller)
    /begin_survey    freeze the anchor and run the engine (also /begin_orbit)
    /end_mission     abandon the remaining lanes and egress
    /abort_mission   AUTO.LAND in place, now (base controller)

Two guards run here because they need sensors the core does not model, and
both trace to the 2026-07-24 wall encounters
(`reference/flight_bags/analyses/20260725_drone4_wall_impacts/REPORT.md`):

  - the tag anchor (`core/anchor.py`), which measures flight-layer drift
    against re-observed ground tags and aborts on sustained disagreement;
  - the horizontal fence and mission timeout, which the core applies to both
    the estimate and the command (`MissionConfig`).

The optional setpoint correction closes the mission-layer loop without
changing EKF2. It accepts only a fresh, geometrically consistent two-tag
measurement and holds each center return until that correction converges.
"""
from __future__ import annotations

import json
import math
import socket
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray

from px4_msgs.msg import DistanceSensor, VehicleAttitude, VehicleLocalPosition
from px4_offboard.controller import ACTIVE, OffboardController
from px4_offboard.gate_qos import MISSION_GATE_QOS

from mission_engine.core.anchor import (
    AnchorConfig,
    CorrectionConfig,
    SetpointCorrection,
    TagAnchorMap,
)
from mission_engine.core.config import CameraModel
from mission_engine.core.dumpproto import build_payload, encode_frame
from mission_engine.core.ingest import PoseHistory, PoseSnapshot, make_observation
from mission_engine.core.mission import (
    ABORT,
    DONE,
    DUMP,
    LAND,
    MissionConfig,
    MissionEngine,
    ROSETTE_HEADINGS_DEG,
)

TAG_PREFIX = "tag36h11:"
CRASH_DISARM_RANGE_M = 0.10
CRASH_DISARM_HOLD_S = 0.10
CRASH_DISARM_RETRY_US = 100_000


class EngineNode(OffboardController):
    """Streams `MissionEngine` setpoints and feeds it detections."""

    def __init__(self) -> None:
        super().__init__("mission_engine")

        # field geometry, relative to the frozen post-climb anchor
        self.declare_parameter("mission_pattern", "serpentine")
        self.declare_parameter("lane_length_m", 10.0)
        self.declare_parameter("n_lanes", 3)
        self.declare_parameter("lane_spacing_m", 3.0)
        self.declare_parameter("lane_heading_deg", 180.0)  # cardinal south
        self.declare_parameter("lanes_offset_ne", [0.0, -3.0])
        self.declare_parameter("rosette_radius_m", 1.5)
        self.declare_parameter("rosette_outer_hold_s", 0.5)
        self.declare_parameter("rosette_center_hold_s", 0.5)
        self.declare_parameter("rosette_petals", len(ROSETTE_HEADINGS_DEG))
        self.declare_parameter("survey_alt_m", 4.0)
        self.declare_parameter("lane_speed_mps", 1.0)
        self.declare_parameter("reach_tolerance_m", 0.25)
        self.declare_parameter("land_settle_s", 3.0)
        self.declare_parameter("land_speed_tolerance_mps", 0.20)
        # envelope
        self.declare_parameter("fence_radius_m", 12.0)
        self.declare_parameter("mission_timeout_s", 300.0)
        self.declare_parameter("max_dips", 0)
        self.declare_parameter("detector_silence_s", 0.0)  # 0 disables the guard
        # camera
        self.declare_parameter("detections_topic", "/detections/down")
        self.declare_parameter("camera_fx_px", 1298.69385194)
        self.declare_parameter("camera_width_px", 1640)
        self.declare_parameter("camera_height_px", 1232)
        self.declare_parameter("camera_roll_deg", 0.0)
        self.declare_parameter("camera_tilt_deg", 0.0)
        # anchor guard
        self.declare_parameter("anchor_gate_m", 1.5)
        self.declare_parameter("anchor_persist_s", 1.0)
        self.declare_parameter("anchor_max_radial_m", 3.0)
        # Which tags are survey markers at known fixed places. The anchor keys
        # its datum by tag id, so a second tag carrying an anchor id puts the
        # datum at the midpoint of the two, where no tag is, and moves it by
        # metres when either one leaves view. The mine log still ingests every
        # decode; only the anchor is restricted. Empty accepts any tag.
        self.declare_parameter("anchor_tag_ids", [6, 7])
        self.declare_parameter("anchor_aborts", True)
        # anchor correction. Off by default: this moves the airframe, so a
        # flight enables it only after an open-loop flight has characterised
        # the anchor on the day's tag layout.
        self.declare_parameter("anchor_corrects", False)
        self.declare_parameter("anchor_correction_rate_mps", 0.30)
        self.declare_parameter("anchor_correction_max_m", 5.0)
        self.declare_parameter("anchor_correction_fresh_s", 0.5)
        self.declare_parameter("anchor_correction_pair_error_m", 0.25)
        self.declare_parameter("anchor_correction_settle_m", 0.10)
        # dump
        self.declare_parameter("drone_id", "drone4")
        self.declare_parameter("mission_id", "")
        self.declare_parameter("dump_dir", "/tmp")
        self.declare_parameter("dump_host", "")
        self.declare_parameter("dump_port", 5010)

        # The base controller defaults `wait_for_start` to False, which arms
        # and climbs the moment the node starts. This node is gated on
        # /start_mission by design, so it holds the gate closed whatever the
        # parameter says: a missing line in a parameter file must not be able
        # to launch an aircraft.
        if not self.wait_for_start:
            self.get_logger().warn(
                "wait_for_start was false; the engine gates on /start_mission "
                "regardless and will not arm until it is published"
            )
            self.wait_for_start = True
            self._start_ok = False

        p = self.get_parameter
        self.cam = self._camera_model()
        self.camera_roll = math.radians(float(p("camera_roll_deg").value))
        self.detections_topic = str(p("detections_topic").value)
        self.anchor_aborts = bool(p("anchor_aborts").value)
        self.anchor_tag_ids = {
            f"{TAG_PREFIX}{int(i)}" for i in (p("anchor_tag_ids").value or [])
        }
        self.drone_id = str(p("drone_id").value)
        self.mission_id = str(p("mission_id").value) or time.strftime("%Y%m%d_%H%M%S")
        self.dump_dir = Path(str(p("dump_dir").value))

        self.anchor = TagAnchorMap(
            AnchorConfig(
                gate_m=float(p("anchor_gate_m").value),
                gate_persist_s=float(p("anchor_persist_s").value),
                max_radial_m=float(p("anchor_max_radial_m").value),
                max_agl_m=float(p("survey_alt_m").value) + 2.0,
            )
        )
        self.anchor_corrects = bool(p("anchor_corrects").value)
        self.anchor_correction_fresh_s = float(
            p("anchor_correction_fresh_s").value
        )
        self.anchor_correction_pair_error_m = float(
            p("anchor_correction_pair_error_m").value
        )
        self.anchor_correction_settle_m = float(
            p("anchor_correction_settle_m").value
        )
        self.correction = SetpointCorrection(
            CorrectionConfig(
                max_rate_mps=float(p("anchor_correction_rate_mps").value),
                max_correction_m=float(p("anchor_correction_max_m").value),
            )
        )
        self.poses = PoseHistory(horizon_s=5.0)
        self.engine: MissionEngine | None = None
        self.q = (1.0, 0.0, 0.0, 0.0)
        self._agl = None
        self._agl_t = 0.0
        self._range_airborne = False
        self._ground_range_t0 = None
        self._crash_disarm = False
        self._last_crash_disarm_us = 0
        self._anchor_ne = None
        self._begun = False
        self._dumped = False
        self._land_requested = False
        self._last_phase = ""

        self.create_subscription(
            Detection2DArray, self.detections_topic, self._detections_cb, 10
        )
        # The raw dToF, on PX4's BEST_EFFORT profile. The anchor audits EKF2,
        # so its height must not come from EKF2: `dist_bottom` is a fused
        # product of the estimator under test, this is the sensor.
        self.create_subscription(
            DistanceSensor,
            self._topic("out/distance_sensor"),
            self._dist_cb,
            self.sensor_qos,
        )
        self.create_subscription(
            Bool, "begin_survey", self._begin_cb, MISSION_GATE_QOS
        )
        self.create_subscription(Bool, "begin_orbit", self._begin_cb, MISSION_GATE_QOS)

        self.get_logger().info(
            f"mission_engine rim up: detections={self.detections_topic} "
            f"pattern={p('mission_pattern').value} "
            f"fence={p('fence_radius_m').value} m "
            f"anchor_gate={p('anchor_gate_m').value} m "
            f"anchor_tags={sorted(self.anchor_tag_ids) or 'any'} "
            f"anchor_aborts={self.anchor_aborts}"
        )

    # ------------------------------------------------------------ config

    def _camera_model(self) -> CameraModel:
        width = int(self.get_parameter("camera_width_px").value)
        fx = float(self.get_parameter("camera_fx_px").value)
        return CameraModel(
            width_px=width,
            height_px=int(self.get_parameter("camera_height_px").value),
            hfov_deg=math.degrees(2.0 * math.atan(0.5 * width / fx)),
            tilt_deg=float(self.get_parameter("camera_tilt_deg").value),
        )

    def _mission_config(self) -> MissionConfig:
        p = self.get_parameter
        offset = [float(v) for v in p("lanes_offset_ne").value]
        anchor = self._anchor_ne
        return MissionConfig(
            pattern=str(p("mission_pattern").value),
            lanes_origin=(anchor[0] + offset[0], anchor[1] + offset[1]),
            lane_length=float(p("lane_length_m").value),
            n_lanes=int(p("n_lanes").value),
            lane_spacing=float(p("lane_spacing_m").value),
            lane_heading_deg=float(p("lane_heading_deg").value),
            rosette_radius_m=float(p("rosette_radius_m").value),
            rosette_outer_hold_s=float(p("rosette_outer_hold_s").value),
            rosette_center_hold_s=float(p("rosette_center_hold_s").value),
            rosette_petals=int(p("rosette_petals").value),
            survey_alt_m=float(p("survey_alt_m").value),
            lane_speed=float(p("lane_speed_mps").value),
            reach_tol_m=float(p("reach_tolerance_m").value),
            egress_ne=(anchor[0], anchor[1]),
            land_settle_s=float(p("land_settle_s").value),
            land_speed_tol_mps=float(p("land_speed_tolerance_mps").value),
            max_dips=int(p("max_dips").value),
            detector_silence_s=float(p("detector_silence_s").value),
            fence_radius_m=float(p("fence_radius_m").value),
            mission_timeout_s=float(p("mission_timeout_s").value),
        )

    # ------------------------------------------------------------ inputs

    def _att_cb(self, msg: VehicleAttitude) -> None:
        super()._att_cb(msg)
        q = [float(v) for v in msg.q]  # PX4 order w, x, y, z, body -> NED
        if sum(v * v for v in q) > 0.25:
            self.q = (q[0], q[1], q[2], q[3])

    def _pos_cb(self, msg: VehicleLocalPosition) -> None:
        super()._pos_cb(msg)
        if bool(msg.dist_bottom_valid) and math.isfinite(msg.dist_bottom):
            self._agl = float(msg.dist_bottom)
            self._agl_t = self._now_us() * 1e-6
        if self.engine is not None:
            self.engine.note_reset_counter(int(msg.xy_reset_counter))
        self._push_pose()

    def on_local_frame_reset(self, delta_xy, delta_z, delta_heading) -> None:
        dn, de = delta_xy
        self.anchor.apply_frame_reset(dn, de)
        self.poses.clear()
        if self.engine is not None:
            self.correction.apply_frame_reset(dn, de)
            self.engine.apply_heading_reset(delta_heading)

    def _dist_cb(self, msg: DistanceSensor) -> None:
        d = float(msg.current_distance)
        now_us = self._now_us()
        now = now_us * 1e-6

        if self.is_armed and d >= msg.min_distance:
            self._range_airborne = True

        if self.is_armed and self._range_airborne and d <= CRASH_DISARM_RANGE_M:
            if self._ground_range_t0 is None:
                self._ground_range_t0 = now
            elif now - self._ground_range_t0 >= CRASH_DISARM_HOLD_S:
                if not self._crash_disarm:
                    self._crash_disarm = True
                    self.get_logger().error(
                        "dToF returned to ground after takeoff; cutting motor power"
                    )
        else:
            self._ground_range_t0 = None

        if (
            self._crash_disarm
            and self.is_armed
            and now_us - self._last_crash_disarm_us >= CRASH_DISARM_RETRY_US
        ):
            self._last_crash_disarm_us = now_us
            self.command_disarm(force=True)

        if msg.min_distance <= d <= msg.max_distance:
            # dToF measures along body -z; level it with the current attitude.
            tilt = self._tilt_from_level()
            self._agl = d * math.cos(tilt)
            self._agl_t = now

    def _tilt_from_level(self) -> float:
        """Angle between body -z and the local vertical, from the attitude."""
        w, x, y, z = self.q
        down_z = 1.0 - 2.0 * (x * x + y * y)  # body-down expressed in NED, z part
        return math.acos(max(-1.0, min(1.0, down_z)))

    def _agl_now(self) -> float:
        """Metres above ground; falls back to the pad-relative altitude."""
        now = self._now_us() * 1e-6
        if self._agl is not None and now - self._agl_t < 1.0:
            return self._agl
        return max(0.0, -(self.z - self._launch_z))

    def _push_pose(self) -> None:
        t = self._now_us() * 1e-6
        if not all(math.isfinite(v) for v in (self.x, self.y, self.z)):
            return
        snap = PoseSnapshot(
            t=t,
            pos=(float(self.x), float(self.y), float(self.z - self._launch_z)),
            q=self.q,
            agl=self._agl_now(),
        )
        try:
            self.poses.append(snap)
        except ValueError:
            pass  # a clock step; the next sample re-establishes the buffer

    def _unroll(self, u: float, v: float) -> tuple[float, float]:
        """Undo the camera's roll about the optical axis. Rotating a pixel
        about the principal point is exactly equivalent to rolling a pinhole
        camera, so the shared `CameraModel` needs no mount-specific field."""
        if self.camera_roll == 0.0:
            return (u, v)
        c, s = math.cos(-self.camera_roll), math.sin(-self.camera_roll)
        du, dv = u - self.cam.cx, v - self.cam.cy
        return (self.cam.cx + c * du - s * dv, self.cam.cy + s * du + c * dv)

    def _detections_cb(self, msg: Detection2DArray) -> None:
        t_img = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        snap = self.poses.nearest(t_img)
        if snap is None:
            return
        if self.engine is not None:
            self.engine.note_detector_alive(self._now_us() * 1e-6)
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0].hypothesis
            tag_id = str(det.id) if str(det.id).startswith(TAG_PREFIX) else None
            pixel = self._unroll(float(det.bbox.center.position.x), float(det.bbox.center.position.y))
            obs = make_observation(
                self.cam,
                snap,
                t_img,
                pixel,
                float(hyp.score),
                str(hyp.class_id),
                tag_id=tag_id,
            )
            if obs is None:
                continue
            if self.engine is not None:
                self.engine.log.ingest(obs)
            # Every decode reaches the mine log above. Only a known survey
            # marker is allowed to move the anchor's datum.
            if tag_id is not None and (
                not self.anchor_tag_ids or tag_id in self.anchor_tag_ids
            ):
                self._observe_anchor(t_img, tag_id, obs.ground_local, snap)

    def _observe_anchor(self, t_img, tag_id, fix, snap: PoseSnapshot) -> None:
        # The fix is back-projected through the raw flight-layer position, so
        # the anchor keeps measuring the true error even while the rim is
        # compensating for it. Correcting a setpoint does not move EKF2.
        nadir = (snap.pos[0], snap.pos[1])
        self.anchor.observe(t_img, tag_id, fix, nadir, snap.agl or 0.0)
        reason = self.anchor.disagreement(t_img)
        if reason is None or self.engine is None:
            return
        if self.engine.phase in (ABORT, LAND, DONE):
            return
        if self.anchor_corrects:
            # A drift being compensated is not a fault, so the guard moves to
            # the drift the rim has declined to compensate.
            if self.correction.saturated:
                over = f"drift exceeds the {self.correction.cfg.max_correction_m:.1f} m correction limit: {reason}"
                self.get_logger().error(f"ABORT: {over}")
                self.engine.request_abort(over)
            else:
                self._log_throttled(f"correcting: {reason}")
        elif self.anchor_aborts:
            self.get_logger().error(f"ABORT: {reason}")
            self.engine.request_abort(reason)
        else:
            self._log_throttled(f"anchor disagreement (guard disarmed): {reason}")

    def _begin_cb(self, msg: Bool) -> None:
        if not msg.data or self._begun:
            return
        if self.state != ACTIVE:
            self.get_logger().warn("begin_survey before the climb finished; ignored.")
            return
        self._begun = True
        self._anchor_ne = (float(self.x), float(self.y))
        self.engine = MissionEngine(
            self._mission_config(),
            initial_yaw=self._launch_yaw,
        )
        self.engine.start()
        self.get_logger().info(
            f"survey begun: anchor N {self._anchor_ne[0]:.2f} E {self._anchor_ne[1]:.2f}, "
            f"pattern={self.engine.cfg.pattern}"
        )

    def on_return_home(self) -> None:
        if self.engine is not None:
            self.engine.request_abort("operator end_mission")

    def on_active_start(self) -> None:
        self.get_logger().info("climb complete; waiting for begin_survey.")

    # ------------------------------------------------------------ output

    def compute_setpoint(self):
        if self.engine is None:
            return None
        t = self._now_us() * 1e-6
        # The offset converts both ways around the tick. Correcting only the
        # emitted setpoint would leave the engine reading a drifted position
        # and steering against its own correction.
        correction_measurement = None
        if self.anchor_corrects:
            correction_measurement = self.anchor.drift(
                t,
                min_tags=2,
                max_age_s=self.anchor_correction_fresh_s,
                max_pair_error_m=self.anchor_correction_pair_error_m,
            )
        self.correction.update(t, correction_measurement)
        anchor_ready = not self.anchor_corrects or (
            correction_measurement is not None
            and not self.correction.saturated
            and self.correction.pending_m <= self.anchor_correction_settle_m
        )
        n, e = self.correction.to_plan((float(self.x), float(self.y)))
        sp = self.engine.tick(
            t,
            (n, e, float(self.z - self._launch_z)),
            (float(self.vx), float(self.vy), float(self.vz)),
            horizontal_valid=self._xy_valid and self._v_xy_valid,
            survey_ready=anchor_ready,
        )
        self._on_phase(self.engine.phase)
        if sp is None:
            return None
        out_n, out_e = self.correction.to_flight((sp.pos[0], sp.pos[1]))
        if sp.vel is None:
            return (out_n, out_e, sp.pos[2], sp.yaw)
        return (out_n, out_e, sp.pos[2], sp.yaw, sp.vel[0], sp.vel[1], sp.vel[2])

    def _on_phase(self, phase: str) -> None:
        if phase != self._last_phase:
            self._last_phase = phase
            reason = self.engine.abort_reason
            self.get_logger().info(
                f"phase -> {phase}" + (f" ({reason})" if reason else "")
            )
        if phase == DUMP and not self._dumped:
            self._dump()
        if phase in (LAND, DONE) and not self._land_requested:
            if not (self._xy_valid and self._v_xy_valid):
                self._log_throttled(
                    "LAND blocked: horizontal position or velocity is invalid; "
                    "maintaining Offboard hold for safety-pilot takeover"
                )
                return
            # PX4 owns the touchdown; the engine's LAND descent is the fake's.
            self._land_requested = True
            self.get_logger().info("engine reached LAND; handing off to NAV_LAND.")
            self.request_land()

    def _dump(self) -> None:
        self._dumped = True
        payload = build_payload(
            self.engine.log,
            drone_id=self.drone_id,
            mission_id=self.mission_id,
            t_takeoff=self.engine.t_takeoff or 0.0,
            t_dump=self._now_us() * 1e-6,
            coverage=self.engine.coverage_report(),
            stats=dict(self.engine.stats(), anchor=self.anchor.report()),
        )
        path = self.dump_dir / f"minefield_{self.drone_id}_{self.mission_id}.json"
        try:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2))
            self.get_logger().info(f"dump written to {path}")
        except OSError as exc:
            self.get_logger().error(f"dump artifact not written: {exc}")
        ok = self._send_dump(encode_frame(payload))
        self.engine.notify_dump_result(ok)

    def _send_dump(self, frame: bytes) -> bool:
        """One connection, one frame, one ack (regroup dump protocol). With no
        master configured the local artifact is the delivery."""
        host = str(self.get_parameter("dump_host").value)
        if not host:
            return True
        port = int(self.get_parameter("dump_port").value)
        try:
            with socket.create_connection((host, port), timeout=3.0) as sock:
                sock.sendall(frame)
                ack = sock.recv(16)
            if ack.strip() == b"ok":
                self.get_logger().info(f"dump acked by {host}:{port}")
                return True
            self.get_logger().warn(f"dump refused by {host}:{port}: {ack!r}")
        except OSError as exc:
            self.get_logger().warn(f"dump to {host}:{port} failed: {exc}")
        return False


def main(args=None):
    rclpy.init(args=args)
    node = EngineNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
