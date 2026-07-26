"""Tag anchor: flight-layer drift measured against re-observed AprilTags.

A tag on the ground does not move. Its back-projected fix in the
flight-layer local frame therefore holds still for exactly as long as that
frame holds still, so the motion of the fix *is* the drift of the frame.
This is the induced loop closure: the anchor is relative, it needs no
surveyed tag position, and the first settled sighting defines the datum
every later sighting is measured against.

The measurement is the one the 2026-07-24 wall encounters were diagnosed
with offline
(`reference/flight_bags/analyses/20260725_drone4_wall_impacts`),
where the camera placed the drone 1.60 m from the pad and EKF2 placed it
6.99 m from the same anchor. Running it online turns that post-hoc finding
into the estimator/truth disagreement guard the accepted mission-execution
RFD requires of every survey (section 3, ABORT policy).

Tag translation is consumed and tag orientation never is (rfd-mission-
execution section 5): the fix comes from the tag centre ray, the dToF AGL,
and the EKF attitude, not from a single-tag PnP rotation.

Pure core: stdlib only, no ROS. The rim supplies pose snapshots and
`DetectionObs` values; the engine consumes `drift()` and `disagreement()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AnchorConfig:
    """Gates on which sightings are trustworthy and when they disagree."""

    min_agl_m: float = 0.5
    max_agl_m: float = 8.0
    # Radial distance of the fix from the camera nadir point. Fixes near the
    # footprint edge ride a grazing ray, so AGL and attitude error project
    # into them much harder than they do near image centre.
    max_radial_m: float = 3.0
    # A reference is the mean of the first `settle_obs` sightings taken within
    # `settle_window_s` of each other, accepted only if they agree to
    # `settle_spread_m`.
    settle_obs: int = 4
    settle_window_s: float = 5.0
    settle_spread_m: float = 0.60
    # Disagreement: residual beyond `gate_m`, sustained `gate_persist_s`.
    gate_m: float = 1.50
    gate_persist_s: float = 1.00
    # A residual older than this stops counting as evidence of anything.
    stale_s: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 < self.min_agl_m < self.max_agl_m:
            raise ValueError("need 0 < min_agl_m < max_agl_m")
        if self.settle_obs < 2:
            raise ValueError("settle_obs must be at least 2")
        if self.gate_m <= self.settle_spread_m:
            raise ValueError("gate_m must exceed settle_spread_m")


@dataclass(frozen=True)
class AnchorTransform:
    """The flight-layer frame's motion since the datum was set.

    Tags are placed by hand, so nothing absolute is known about them: not the
    separation, not the bearing, not the position. Every quantity here is
    therefore differential — the datum is whatever the first settled sighting
    saw, and only the change from it is consumed. Absolute placement never
    enters, which is what lets the layout be rough and change between flights.

    Yaw is not optional bookkeeping. A yaw drift of `dyaw` rotates every ground
    fix about the camera nadir point, so with two tags it appears as *opposite*
    residuals rather than a common one. Averaging the residuals would report a
    translation that no part of the frame actually made.
    """

    dn: float
    de: float
    dyaw: float  # radians; 0 when a single tag leaves rotation unobservable
    n_tags: int
    points: Tuple[Tuple[float, float], ...]  # the datum positions it was fit to

    def displacement_vector_at(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Where the frame has carried `point`, as a vector. The translation
        term is about the coordinate origin, not about the anchored place, so
        with any fitted yaw it is metres away from the local displacement
        whenever the tags sit far from the origin."""
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        moved_n = c * point[0] - s * point[1] + self.dn
        moved_e = s * point[0] + c * point[1] + self.de
        return (moved_n - point[0], moved_e - point[1])

    def displacement_at(self, point: Tuple[float, float]) -> float:
        """How far the frame has carried `point`. This, not the translation
        term, is the quantity that matters: under a rotation the translation
        term alone says nothing about how far any particular place moved."""
        return math.hypot(*self.displacement_vector_at(point))

    @property
    def centroid(self) -> Tuple[float, float]:
        """The middle of the anchored points — where the drift is measured,
        and so the only place a constant offset can claim to represent."""
        if not self.points:
            return (0.0, 0.0)
        n = len(self.points)
        return (
            sum(p[0] for p in self.points) / n,
            sum(p[1] for p in self.points) / n,
        )

    @property
    def max_displacement(self) -> float:
        """Worst displacement over the anchored points — the guard's measure,
        because the tags are the only places the drift is actually observed."""
        return max((self.displacement_at(p) for p in self.points), default=0.0)


@dataclass(frozen=True)
class AnchorFix:
    """One accepted sighting and what it says about the flight layer."""

    t: float
    tag_id: str
    fix: Tuple[float, float]  # back-projected ground fix, local NED
    residual: Tuple[float, float]  # fix - reference; the flight-layer drift
    agl_m: float
    radial_m: float  # fix distance from the camera nadir point
    settled: bool  # False while the reference is still being formed

    @property
    def residual_m(self) -> float:
        return math.hypot(self.residual[0], self.residual[1])


@dataclass
class _Reference:
    """The datum for one tag, and the sightings still forming it."""

    tag_id: str
    pending: List[Tuple[float, float, float]] = field(default_factory=list)
    origin: Optional[Tuple[float, float]] = None
    n_obs: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def settled(self) -> bool:
        return self.origin is not None


class TagAnchorMap:
    """Per-tag reference fixes and the drift they expose.

    `observe` takes a ground fix already back-projected into the flight-layer
    local frame (`ingest.make_observation` produces exactly this) plus the
    nadir point of the camera that saw it, and returns the fix's residual
    against that tag's datum. `drift` blends the residuals of every tag seen
    recently; `disagreement` reports the sustained-gate breach.
    """

    def __init__(self, cfg: Optional[AnchorConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else AnchorConfig()
        self.refs: Dict[str, _Reference] = {}
        self.fixes: List[AnchorFix] = []
        self.n_rejected = 0
        self._breach_t0: Optional[float] = None

    # ------------------------------------------------------------ ingest

    def observe(
        self,
        t: float,
        tag_id: str,
        fix: Tuple[float, float],
        nadir_ne: Tuple[float, float],
        agl_m: float,
    ) -> Optional[AnchorFix]:
        """Accept one tag ground fix. None = the sighting failed a gate."""
        if not (self.cfg.min_agl_m <= agl_m <= self.cfg.max_agl_m):
            self.n_rejected += 1
            return None
        radial = math.hypot(fix[0] - nadir_ne[0], fix[1] - nadir_ne[1])
        if radial > self.cfg.max_radial_m:
            self.n_rejected += 1
            return None
        if not all(math.isfinite(v) for v in fix):
            self.n_rejected += 1
            return None

        ref = self.refs.get(tag_id)
        if ref is None:
            ref = _Reference(tag_id=tag_id, first_seen=t)
            self.refs[tag_id] = ref
        ref.n_obs += 1
        ref.last_seen = t

        # A sighting that helps form the datum is measured against itself, so
        # its residual is near zero by construction. Marking the whole forming
        # burst unsettled keeps those zeros out of `drift`, which would
        # otherwise dilute the first real disagreement.
        was_settled = ref.settled
        if not was_settled:
            self._accumulate(ref, t, fix)

        origin = ref.origin if ref.settled else self._pending_mean(ref)
        out = AnchorFix(
            t=t,
            tag_id=tag_id,
            fix=fix,
            residual=(fix[0] - origin[0], fix[1] - origin[1]),
            agl_m=agl_m,
            radial_m=radial,
            settled=was_settled,
        )
        self.fixes.append(out)
        return out

    def _accumulate(self, ref: _Reference, t: float, fix: Tuple[float, float]) -> None:
        """Grow the settling window; promote it to a datum once it is tight."""
        window = [s for s in ref.pending if t - s[0] <= self.cfg.settle_window_s]
        window.append((t, fix[0], fix[1]))
        ref.pending = window
        if len(window) < self.cfg.settle_obs:
            return
        mean_n = sum(s[1] for s in window) / len(window)
        mean_e = sum(s[2] for s in window) / len(window)
        spread = max(math.hypot(s[1] - mean_n, s[2] - mean_e) for s in window)
        if spread <= self.cfg.settle_spread_m:
            ref.origin = (mean_n, mean_e)
            ref.pending = []
        else:
            # Too loose to be a datum: keep only the newest half and retry, so
            # a settling window spanning a real drift event does not freeze a
            # smeared origin into the map.
            ref.pending = window[len(window) // 2 :]

    @staticmethod
    def _pending_mean(ref: _Reference) -> Tuple[float, float]:
        n = len(ref.pending)
        if n == 0:
            return (0.0, 0.0)
        return (
            sum(s[1] for s in ref.pending) / n,
            sum(s[2] for s in ref.pending) / n,
        )

    # ------------------------------------------------------------ outputs

    def drift(self, t: float) -> Optional[AnchorTransform]:
        """Best current estimate of how the flight-layer frame has moved since
        the datum, or None when no settled tag has been seen recently.

        Each tag contributes one point correspondence — its datum against the
        median of its recent fixes, so a single bad frame cannot carry the
        answer. Two correspondences determine translation and rotation exactly;
        one leaves rotation unobservable and the solve reduces to a shift."""
        pairs = self._correspondences(t)
        if not pairs:
            return None
        if len(pairs) == 1:
            (origin, fix), = pairs
            return AnchorTransform(
                dn=fix[0] - origin[0],
                de=fix[1] - origin[1],
                dyaw=0.0,
                n_tags=1,
                points=(origin,),
            )
        return _rigid_fit(pairs)

    def _correspondences(
        self, t: float
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        per_tag: Dict[str, List[AnchorFix]] = {}
        for fx in self.fixes:
            if not fx.settled or t - fx.t > self.cfg.stale_s:
                continue
            per_tag.setdefault(fx.tag_id, []).append(fx)
        pairs = []
        for tag_id, group in sorted(per_tag.items()):
            origin = self.refs[tag_id].origin
            if origin is None:
                continue
            pairs.append(
                (
                    origin,
                    (
                        _median([f.fix[0] for f in group]),
                        _median([f.fix[1] for f in group]),
                    ),
                )
            )
        return pairs

    def disagreement(self, t: float) -> Optional[str]:
        """Reason string once the drift has exceeded the gate continuously for
        `gate_persist_s`; None otherwise. Latches nothing — the caller owns
        the abort, and this keeps reporting while the breach lasts."""
        d = self.drift(t)
        moved = 0.0 if d is None else d.max_displacement
        if d is None or moved <= self.cfg.gate_m:
            self._breach_t0 = None
            return None
        if self._breach_t0 is None:
            self._breach_t0 = t
            return None
        held = t - self._breach_t0
        if held < self.cfg.gate_persist_s:
            return None
        yaw = f", yaw {math.degrees(d.dyaw):+.1f}°" if d.n_tags > 1 else ""
        return (
            f"tag anchor disagrees with flight layer by {moved:.2f} m "
            f"(N {d.dn:+.2f}, E {d.de:+.2f}{yaw}) for {held:.1f} s"
        )

    def report(self) -> Dict:
        """Anchor summary for the dump and the debrief."""
        return {
            "tags": {
                tag_id: {
                    "settled": ref.settled,
                    "origin_ne": list(ref.origin) if ref.origin else None,
                    "n_obs": ref.n_obs,
                    "first_seen": ref.first_seen,
                    "last_seen": ref.last_seen,
                }
                for tag_id, ref in sorted(self.refs.items())
            },
            "n_fixes": len(self.fixes),
            "n_rejected": self.n_rejected,
            "max_residual_m": max(
                (fx.residual_m for fx in self.fixes if fx.settled), default=0.0
            ),
        }


def _rigid_fit(
    pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> AnchorTransform:
    """Least-squares rotation + translation taking datum points onto their
    current fixes (planar Kabsch). Scale is fixed at 1 deliberately: the tags
    are placed by hand, so a fitted scale would absorb placement error and
    dToF error indiscriminately instead of exposing either."""
    n = len(pairs)
    pn = sum(p[0][0] for p in pairs) / n
    pe = sum(p[0][1] for p in pairs) / n
    qn = sum(p[1][0] for p in pairs) / n
    qe = sum(p[1][1] for p in pairs) / n
    cross = sum((p[0][0] - pn) * (p[1][1] - qe) - (p[0][1] - pe) * (p[1][0] - qn) for p in pairs)
    dot = sum((p[0][0] - pn) * (p[1][0] - qn) + (p[0][1] - pe) * (p[1][1] - qe) for p in pairs)
    dyaw = math.atan2(cross, dot) if (cross or dot) else 0.0
    c, s = math.cos(dyaw), math.sin(dyaw)
    return AnchorTransform(
        dn=qn - (c * pn - s * pe),
        de=qe - (s * pn + c * pe),
        dyaw=dyaw,
        n_tags=n,
        points=tuple(p[0] for p in pairs),
    )


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


@dataclass(frozen=True)
class CorrectionConfig:
    """Limits on the correction the engine is allowed to apply."""

    # A re-acquisition after a long blind stretch arrives as a step. Slewing it
    # in makes the airframe walk to the corrected place instead of lunging.
    max_rate_mps: float = 0.30
    # Past this the estimate is not worth correcting; something else is wrong.
    max_correction_m: float = 5.0

    def __post_init__(self) -> None:
        if self.max_rate_mps <= 0.0:
            raise ValueError(f"max_rate_mps must be positive, got {self.max_rate_mps}")
        if self.max_correction_m <= 0.0:
            raise ValueError(
                f"max_correction_m must be positive, got {self.max_correction_m}"
            )


class SetpointCorrection:
    """The map->odom offset the engine applies when it commands a setpoint.

    A tag fix says where the flight-layer frame has carried a place that did
    not move, so `AnchorTransform.dn/de` *is* the flight layer's position
    error: `pos_flight = pos_true + drift`. The engine plans in the tag frame
    and PX4 consumes the flight frame, so the rim subtracts this offset from
    the position it reports to the engine and adds it back to the setpoint it
    emits (rfd-mission-execution section 4, map-layer goals converted at
    command time). EKF2 is never told anything, so no reset counter moves and
    no innovation gate can silently drop the correction.

    The offset holds through blind stretches instead of decaying. It is the
    error accumulated up to the last sighting, and that error does not go away
    because the tags went out of frame — decaying to zero would hand the known
    error back.
    """

    def __init__(self, cfg: Optional[CorrectionConfig] = None) -> None:
        self.cfg = cfg or CorrectionConfig()
        self.applied: Tuple[float, float] = (0.0, 0.0)
        self.target: Tuple[float, float] = (0.0, 0.0)
        self.saturated = False
        self._last_t: Optional[float] = None

    def update(
        self, t: float, transform: Optional[AnchorTransform]
    ) -> Tuple[float, float]:
        """Advance the offset toward the latest drift and return it. A None
        transform means no fresh fix, which holds the target where it is."""
        if transform is not None:
            self._retarget(transform)
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        self.applied = self._slew(dt)
        return self.applied

    def _retarget(self, transform: AnchorTransform) -> None:
        # The offset is the frame's displacement at the tags, not the fit's
        # translation term. Those differ by metres once the fit carries any
        # yaw and the tags sit away from the frame origin.
        #
        # The rotation is resolved at the tags and applied nowhere else. The
        # pair's baseline resolves yaw far worse than it resolves position, so
        # extrapolating it out to a distant waypoint would move that waypoint
        # much further than the measurement supports. The yaw stays in the
        # transform for the guard to read.
        dn, de = transform.displacement_vector_at(transform.centroid)
        mag = math.hypot(dn, de)
        self.saturated = mag > self.cfg.max_correction_m
        if self.saturated:
            scale = self.cfg.max_correction_m / mag
            dn, de = dn * scale, de * scale
        self.target = (dn, de)

    def _slew(self, dt: float) -> Tuple[float, float]:
        dn = self.target[0] - self.applied[0]
        de = self.target[1] - self.applied[1]
        gap = math.hypot(dn, de)
        step = self.cfg.max_rate_mps * dt
        if gap <= step or gap == 0.0:
            return self.target
        return (
            self.applied[0] + dn * step / gap,
            self.applied[1] + de * step / gap,
        )

    @property
    def pending_m(self) -> float:
        """Correction measured but not yet slewed in."""
        return math.hypot(
            self.target[0] - self.applied[0], self.target[1] - self.applied[1]
        )

    def to_plan(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        """Flight-layer position -> the frame the engine plans in."""
        return (pos[0] - self.applied[0], pos[1] - self.applied[1])

    def to_flight(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """A planned point -> the frame PX4 acts on."""
        return (point[0] + self.applied[0], point[1] + self.applied[1])
