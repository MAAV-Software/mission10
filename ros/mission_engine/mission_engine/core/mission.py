"""Mission state machine: takeoff -> survey -> egress -> dump -> settle ->
land, abort from anywhere (rfd-mission-execution).

Pure core: the engine is ticked with time + flight state and returns a
setpoint; the ROS rim (or the test fake) owns transport. Detection
ingest is not a state — it feeds the mine log continuously from takeoff
to land. Setpoints are always in the flight-layer local frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .geometry import Lane, Vec3, distance_to_polygon, point_in_polygon, polygon_serpentine, serpentine
from .minelog import CONFIRMED, DIPPED, Cluster, MineLog

# phases
PREFLIGHT = "PREFLIGHT"
OFFBOARD_SYNC = "OFFBOARD_SYNC"
TAKEOFF = "TAKEOFF"
LANE = "LANE"
VERIFY_DIP = "VERIFY_DIP"
EGRESS = "EGRESS"
SETTLE = "SETTLE"
DUMP = "DUMP"
LAND = "LAND"
ABORT = "ABORT"
DONE = "DONE"

ROSETTE_HEADINGS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)


@dataclass(frozen=True)
class Setpoint:
    pos: Vec3  # NED position target
    yaw: Optional[float] = None


@dataclass(frozen=True)
class MissionConfig:
    # survey pattern
    pattern: str = "serpentine"
    # lanes
    lanes_origin: Tuple[float, float] = (0.0, 0.0)
    lane_length: float = 20.0
    n_lanes: int = 3
    lane_spacing: float = 6.0
    lane_heading_deg: float = 0.0  # 0 runs the lanes north and steps them east
    survey_alt_m: float = 6.0  # AGL, positive
    lane_speed: float = 2.0
    # center-return rosette
    rosette_radius_m: float = 1.5
    rosette_outer_hold_s: float = 0.5
    rosette_center_hold_s: float = 0.5
    rosette_petals: int = len(ROSETTE_HEADINGS_DEG)
    # transit / vertical
    climb_speed: float = 1.5
    descent_speed: float = 0.8
    reach_tol_m: float = 0.5
    sync_duration_s: float = 1.2
    egress_ne: Tuple[float, float] = (0.0, 0.0)
    land_settle_s: float = 3.0
    land_speed_tol_mps: float = 0.20
    # verify-dip stub
    dip_hover_s: float = 2.0
    max_dips: int = 4
    # abort policy
    detector_silence_s: float = 5.0
    reset_storm_count: int = 3
    dump_retry_limit: int = 3
    # Horizontal envelope around the sync point, applied to both the estimate
    # and the commanded target; 0 disables. The estimate side catches an
    # estimator runaway, the command side catches a schedule that leaves the
    # configured field. Neither catches an estimate that is confidently wrong
    # while the airframe is elsewhere — the tag anchor owns that class
    # (core/anchor.py).
    fence_radius_m: float = 0.0
    # NED vertices in perimeter order. When present this is both the survey
    # geometry and the horizontal command/estimate fence.
    fence_polygon_ne: Tuple[Tuple[float, float], ...] = ()
    # Wall clock from takeoff; 0 disables. A survey that cannot finish must
    # come home rather than hold its setpoint indefinitely.
    mission_timeout_s: float = 0.0

    def __post_init__(self) -> None:
        if self.pattern not in ("serpentine", "center_return_rosette"):
            raise ValueError(f"unknown mission pattern {self.pattern!r}")
        if self.survey_alt_m <= 0.0 or self.lane_speed <= 0.0:
            raise ValueError("survey_alt_m and lane_speed must be positive")
        if self.rosette_radius_m <= 0.0:
            raise ValueError("rosette radius must be positive")
        if not 1 <= self.rosette_petals <= len(ROSETTE_HEADINGS_DEG):
            raise ValueError(
                f"rosette_petals must be between 1 and {len(ROSETTE_HEADINGS_DEG)}"
            )
        if self.rosette_outer_hold_s < 0.0 or self.rosette_center_hold_s < 0.0:
            raise ValueError("rosette holds must not be negative")
        if self.reach_tol_m <= 0.0:
            raise ValueError("reach_tol_m must be positive")
        if self.land_settle_s <= 0.0 or self.land_speed_tol_mps <= 0.0:
            raise ValueError("landing settle time and speed tolerance must be positive")
        if self.fence_radius_m < 0.0 or self.mission_timeout_s < 0.0:
            raise ValueError("fence_radius_m and mission_timeout_s must not be negative")
        if self.fence_polygon_ne and len(self.fence_polygon_ne) < 3:
            raise ValueError("fence_polygon_ne needs at least three vertices")


@dataclass
class CoveredInterval:
    lane: int
    s0: float
    s1: float


class MissionEngine:
    """Deterministic mission sequencer. Drive it with tick(t, pos);
    feed detections into `log` via the ingest module; read `phase`."""

    def __init__(
        self,
        cfg: MissionConfig,
        log: Optional[MineLog] = None,
        *,
        initial_yaw: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.log = log if log is not None else MineLog()
        self.lanes: List[Lane] = (
            polygon_serpentine(cfg.fence_polygon_ne, cfg.lane_spacing)
            if cfg.fence_polygon_ne and cfg.pattern == "serpentine"
            else serpentine(
                cfg.lanes_origin, cfg.lane_length, cfg.n_lanes,
                cfg.lane_spacing, math.radians(cfg.lane_heading_deg),
            )
        )
        self.phase = PREFLIGHT
        self.abort_reason: Optional[str] = None
        self.t_takeoff: Optional[float] = None
        self.dips_done = 0
        self.dump_attempts = 0
        self.dump_acked = False
        self.covered: List[CoveredInterval] = []
        self.ekf_resets_seen = 0
        self._sync_t0: Optional[float] = None
        self._lane_i = 0
        self._lane_s = 0.0
        self._home_ne: Optional[Tuple[float, float]] = None
        self._dip_target: Optional[Cluster] = None
        self._dip_t0: Optional[float] = None
        self._last_detector_t: Optional[float] = None
        self._last_reset_counter: Optional[int] = None
        self._last_yaw = initial_yaw
        self._petal_i = 0
        self._petal_stage = "center"
        self._petal_hold_t0: Optional[float] = None
        self._lane_entered = False
        self._settle_t0: Optional[float] = None
        self._settle_next_phase = DUMP
        self._settle_requires_survey_ready = False

    # ------------------------------------------------------------ inputs

    def start(self) -> None:
        """Operator mission-start ack (the arming gate lives in the rim)."""
        if self.phase == PREFLIGHT:
            self.phase = OFFBOARD_SYNC

    def note_detector_alive(self, t: float) -> None:
        self._last_detector_t = t

    def note_reset_counter(self, counter: int) -> None:
        if self._last_reset_counter is not None and counter != self._last_reset_counter:
            self.ekf_resets_seen += abs(counter - self._last_reset_counter)
        self._last_reset_counter = counter

    def apply_heading_reset(self, delta: float) -> None:
        self._last_yaw = math.atan2(
            math.sin(self._last_yaw + delta),
            math.cos(self._last_yaw + delta),
        )

    def operator_abort(self) -> None:
        self._abort("operator")

    def request_abort(self, reason: str) -> None:
        """Abort from a guard the rim owns (tag anchor, link health)."""
        self._abort(reason)

    def notify_dump_result(self, ok: bool) -> None:
        if self.phase != DUMP:
            return
        if ok:
            self.dump_acked = True
        else:
            self.dump_attempts += 1
            if self.dump_attempts >= self.cfg.dump_retry_limit:
                self._abort("dump retries exhausted")

    # ------------------------------------------------------------ state

    def _abort(self, reason: str) -> None:
        if self.phase in (ABORT, DONE):
            return
        self.abort_reason = reason
        self.phase = ABORT

    def _alt_target(self) -> float:
        return -self.cfg.survey_alt_m  # NED

    def _egress_ne(self) -> Tuple[float, float]:
        # A rosette's center is frozen in the engine's corrected planning
        # frame during OFFBOARD_SYNC. Using the pre-correction ROS parameter
        # here would apply the tag offset twice on return and landing.
        if self.cfg.pattern == "center_return_rosette" and self._home_ne is not None:
            return self._home_ne
        return self.cfg.egress_ne

    def _check_aborts(self, t: float, pos: Vec3) -> None:
        if self.phase in (
            PREFLIGHT,
            OFFBOARD_SYNC,
            ABORT,
            SETTLE,
            DUMP,
            LAND,
            DONE,
        ):
            return
        if (
            self.phase in (TAKEOFF, LANE, VERIFY_DIP)
            and self.cfg.detector_silence_s > 0.0
            and self._last_detector_t is not None
            and t - self._last_detector_t > self.cfg.detector_silence_s
        ):
            self._abort(f"detector silent > {self.cfg.detector_silence_s} s")
        if self.ekf_resets_seen >= self.cfg.reset_storm_count:
            self._abort("EKF reset storm")
        if self.cfg.mission_timeout_s > 0.0 and self.t_takeoff is not None:
            elapsed = t - self.t_takeoff
            if elapsed > self.cfg.mission_timeout_s:
                self._abort(f"mission timeout at {elapsed:.0f} s")
        violation = self._fence_violation(pos[0], pos[1])
        if violation is not None:
            self._abort(f"estimate {violation}")

    def _fence_radius(self, n: float, e: float) -> Optional[float]:
        """Distance beyond the fence, or None while inside it."""
        if self.cfg.fence_radius_m <= 0.0 or self._home_ne is None:
            return None
        r = math.hypot(n - self._home_ne[0], e - self._home_ne[1])
        return r if r > self.cfg.fence_radius_m else None

    def _fence_violation(self, n: float, e: float) -> Optional[str]:
        polygon = self.cfg.fence_polygon_ne
        if polygon:
            if not point_in_polygon((n, e), polygon):
                distance = distance_to_polygon((n, e), polygon)
                return f"{distance:.1f} m outside the field polygon"
            return None
        radius = self._fence_radius(n, e)
        if radius is not None:
            return f"{radius:.1f} m outside the {self.cfg.fence_radius_m:.0f} m radius fence"
        return None

    def _target(self, target: Vec3) -> Setpoint:
        """Return the end of the current leg; PX4 smooths the trajectory."""
        return Setpoint(pos=target, yaw=self._last_yaw)

    def _reached(self, pos: Vec3, target: Vec3) -> bool:
        return (
            math.hypot(target[0] - pos[0], target[1] - pos[1]) <= self.cfg.reach_tol_m
            and abs(target[2] - pos[2]) <= self.cfg.reach_tol_m
        )

    # ------------------------------------------------------------ tick

    def tick(
        self,
        t: float,
        pos: Vec3,
        vel: Vec3 = (0.0, 0.0, 0.0),
        *,
        horizontal_valid: bool = True,
        survey_ready: bool = True,
    ) -> Optional[Setpoint]:
        sp = self._tick(t, pos, vel, horizontal_valid, survey_ready)
        if sp is not None and self.phase in (TAKEOFF, LANE, VERIFY_DIP):
            violation = self._fence_violation(sp.pos[0], sp.pos[1])
            if violation is not None:
                self._abort(f"command {violation}")
                return self._tick(t, pos, vel, horizontal_valid, survey_ready)
        return sp

    def _tick(
        self,
        t: float,
        pos: Vec3,
        vel: Vec3,
        horizontal_valid: bool,
        survey_ready: bool,
    ) -> Optional[Setpoint]:
        self._check_aborts(t, pos)

        if self.phase == PREFLIGHT:
            return None

        if self.phase == OFFBOARD_SYNC:
            if self._sync_t0 is None:
                self._sync_t0 = t
                self._home_ne = (pos[0], pos[1])
            if t - self._sync_t0 >= self.cfg.sync_duration_s:
                self.phase = TAKEOFF
                self.t_takeoff = t
            return Setpoint(
                pos=(self._home_ne[0], self._home_ne[1], pos[2]),
                yaw=self._last_yaw,
            )

        if self.phase == TAKEOFF:
            target = (self._home_ne[0], self._home_ne[1], self._alt_target())
            if self._reached(pos, target):
                self.phase = LANE
                return self._tick_survey(t, pos, vel, horizontal_valid, survey_ready)
            return self._target(target)

        if self.phase == LANE:
            return self._tick_survey(t, pos, vel, horizontal_valid, survey_ready)

        if self.phase == VERIFY_DIP:
            return self._tick_dip(t, pos)

        if self.phase == EGRESS or self.phase == ABORT:
            egress = self._egress_ne()
            target = (egress[0], egress[1], self._alt_target())
            if self._reached(pos, target):
                if self.phase == EGRESS:
                    self.phase = DUMP
                else:
                    self._begin_settle(LAND, require_survey_ready=False)
                return Setpoint(pos=target, yaw=self._last_yaw)
            return self._target(target)

        if self.phase == SETTLE:
            egress = self._egress_ne()
            target = (egress[0], egress[1], self._alt_target())
            speed_ok = math.hypot(vel[0], vel[1]) <= self.cfg.land_speed_tol_mps
            ready = (
                horizontal_valid
                and speed_ok
                and self._reached(pos, target)
                and (survey_ready or not self._settle_requires_survey_ready)
            )
            if ready:
                if self._settle_t0 is None:
                    self._settle_t0 = t
                elif t - self._settle_t0 >= self.cfg.land_settle_s:
                    self.phase = self._settle_next_phase
            else:
                self._settle_t0 = None
            return Setpoint(pos=target, yaw=self._last_yaw)

        if self.phase == DUMP:
            if self.dump_acked:
                self._begin_settle(LAND, require_survey_ready=True)
            egress = self._egress_ne()
            return Setpoint(
                pos=(egress[0], egress[1], self._alt_target()),
                yaw=self._last_yaw,
            )

        if self.phase == LAND:
            egress = self._egress_ne()
            target = (egress[0], egress[1], 0.0)
            if self._reached(pos, target):
                self.phase = DONE
                return None
            return self._target(target)

        return None  # DONE

    def _begin_settle(self, next_phase: str, *, require_survey_ready: bool) -> None:
        self.phase = SETTLE
        self._settle_t0 = None
        self._settle_next_phase = next_phase
        self._settle_requires_survey_ready = require_survey_ready

    def _tick_survey(
        self,
        t: float,
        pos: Vec3,
        vel: Vec3,
        horizontal_valid: bool,
        survey_ready: bool,
    ) -> Setpoint:
        if self.cfg.pattern == "center_return_rosette":
            return self._tick_rosette(t, pos, vel, horizontal_valid, survey_ready)
        return self.tick_lane(pos, vel)

    def _tick_rosette(
        self,
        t: float,
        pos: Vec3,
        vel: Vec3,
        horizontal_valid: bool,
        survey_ready: bool,
    ) -> Setpoint:
        """Fly one short radial leg at a time and reacquire the tag anchor at
        center before every departure."""
        center = (self._home_ne[0], self._home_ne[1], self._alt_target())

        if self._petal_stage == "center":
            stable = (
                horizontal_valid
                and survey_ready
                and self._reached(pos, center)
                and math.hypot(vel[0], vel[1]) <= self.cfg.land_speed_tol_mps
            )
            if stable:
                if self._petal_hold_t0 is None:
                    self._petal_hold_t0 = t
                elif t - self._petal_hold_t0 >= self.cfg.rosette_center_hold_s:
                    self._petal_hold_t0 = None
                    if self._petal_i >= self.cfg.rosette_petals:
                        self.phase = DUMP
                    else:
                        self._petal_stage = "outbound"
            else:
                self._petal_hold_t0 = None
            return Setpoint(pos=center, yaw=self._last_yaw)

        heading = math.radians(ROSETTE_HEADINGS_DEG[self._petal_i])
        outer = (
            center[0] + self.cfg.rosette_radius_m * math.cos(heading),
            center[1] + self.cfg.rosette_radius_m * math.sin(heading),
            center[2],
        )

        if self._petal_stage == "outbound":
            if self._reached(pos, outer):
                self._petal_stage = "outer_hold"
                self._petal_hold_t0 = None
                return Setpoint(pos=outer, yaw=self._last_yaw)
            return self._target(outer)

        if self._petal_stage == "outer_hold":
            stable = (
                self._reached(pos, outer)
                and math.hypot(vel[0], vel[1]) <= self.cfg.land_speed_tol_mps
            )
            if stable:
                if self._petal_hold_t0 is None:
                    self._petal_hold_t0 = t
                elif t - self._petal_hold_t0 >= self.cfg.rosette_outer_hold_s:
                    self._petal_hold_t0 = None
                    self._petal_stage = "inbound"
            else:
                self._petal_hold_t0 = None
            return Setpoint(pos=outer, yaw=self._last_yaw)

        if self._reached(pos, center):
            self._petal_i += 1
            self._petal_stage = "center"
            self._petal_hold_t0 = None
            return Setpoint(pos=center, yaw=self._last_yaw)
        return self._target(center)

    def tick_lane(self, pos: Vec3, vel: Vec3) -> Setpoint:
        lane = self.lanes[self._lane_i]
        entry_n, entry_e = lane.point_at(self._lane_s)
        entry = (entry_n, entry_e, self._alt_target())
        stopped = math.hypot(vel[0], vel[1]) <= self.cfg.land_speed_tol_mps

        if not self._lane_entered:
            if not self._reached(pos, entry) or not stopped:
                return self._target(entry)
            self._lane_entered = True

        # dip trigger between progress steps, budget permitting
        if self.dips_done < self.cfg.max_dips:
            c = self.log.next_dip_target()
            if c is not None:
                self._dip_target = c
                self._dip_t0 = None
                self.phase = VERIFY_DIP
                return self._target(entry)

        along = (
            (pos[0] - lane.start[0]) * math.cos(lane.heading)
            + (pos[1] - lane.start[1]) * math.sin(lane.heading)
        )
        measured_s = max(0.0, min(along, lane.length))
        if measured_s > self._lane_s:
            self._record_coverage(lane.index, self._lane_s, measured_s)
            self._lane_s = measured_s

        end_n, end_e = lane.point_at(lane.length)
        target = (end_n, end_e, self._alt_target())
        if self._reached(pos, target) and stopped:
            self._record_coverage(lane.index, self._lane_s, lane.length)
            if self._lane_i + 1 < len(self.lanes):
                self._lane_i += 1
                self._lane_s = 0.0
                self._lane_entered = False
                next_lane = self.lanes[self._lane_i]
                n, e = next_lane.point_at(0.0)
                return self._target((n, e, self._alt_target()))
            else:
                self.phase = EGRESS
        return self._target(target)

    def _tick_dip(self, t: float, pos: Vec3) -> Setpoint:
        """v0 stub: hover over the cluster centroid at survey altitude,
        mark it dipped, resume the lane at the recorded progress point."""
        c = self._dip_target
        target = (c.centroid[0], c.centroid[1], self._alt_target())
        if not self._reached(pos, target):
            return self._target(target)
        if self._dip_t0 is None:
            self._dip_t0 = t
        if t - self._dip_t0 >= self.cfg.dip_hover_s:
            c.status = DIPPED
            self.dips_done += 1
            self._dip_target = None
            self._lane_entered = False
            self.phase = LANE
        return Setpoint(pos=target, yaw=self._last_yaw)

    def _record_coverage(self, lane_index: int, s0: float, s1: float) -> None:
        if s1 <= s0:
            return
        if (
            self.covered
            and self.covered[-1].lane == lane_index
            and abs(self.covered[-1].s1 - s0) < 1e-9
        ):
            self.covered[-1].s1 = s1
        else:
            self.covered.append(CoveredInterval(lane_index, s0, s1))

    # ------------------------------------------------------------ outputs

    def coverage_report(self) -> Dict:
        if self.cfg.pattern == "center_return_rosette":
            center = self._home_ne or self.cfg.egress_ne
            petals = []
            for heading_deg in ROSETTE_HEADINGS_DEG:
                heading = math.radians(heading_deg)
                petals.append(
                    [
                        list(center),
                        [
                            center[0] + self.cfg.rosette_radius_m * math.cos(heading),
                            center[1] + self.cfg.rosette_radius_m * math.sin(heading),
                        ],
                        list(center),
                    ]
                )
            return {
                "pattern": self.cfg.pattern,
                "petals": petals,
                "completed_petals": self._petal_i,
                "lanes": [],
                "gaps": [],
            }
        lanes = []
        gaps = []
        for iv in self.covered:
            lane = self.lanes[iv.lane]
            lanes.append([list(lane.point_at(iv.s0)), list(lane.point_at(iv.s1))])
        by_lane: Dict[int, List[CoveredInterval]] = {}
        for iv in self.covered:
            by_lane.setdefault(iv.lane, []).append(iv)
        for lane in self.lanes:
            ivs = sorted(by_lane.get(lane.index, []), key=lambda iv: iv.s0)
            cursor = 0.0
            for iv in ivs:
                if iv.s0 > cursor + 1e-6:
                    gaps.append([list(lane.point_at(cursor)), list(lane.point_at(iv.s0))])
                cursor = max(cursor, iv.s1)
            if cursor < lane.length - 1e-6:
                gaps.append([list(lane.point_at(cursor)), list(lane.point_at(lane.length))])
        return {"lanes": lanes, "gaps": gaps}

    def stats(self) -> Dict:
        return {
            "detections": self.log.n_ingested,
            "dips": self.dips_done,
            "ekf_resets": self.ekf_resets_seen,
        }
