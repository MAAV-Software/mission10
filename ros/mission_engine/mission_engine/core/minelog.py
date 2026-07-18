"""In-RAM mine log: online clustering of ground-hit observations.

Clustering runs in the flight-layer local frame (smooth, jump-free);
global lat/lon rides along per observation and is aggregated per pass —
averaging across passes is what improves the global fix, not more
frames within one (rfd-mission-execution, detection ingest section).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# status ladder; the engine moves clusters up it, never down
CANDIDATE = "candidate"
CONFIRMED = "confirmed"
DIPPED = "dipped"
VERIFIED = "verified"


@dataclass(frozen=True)
class DetectionObs:
    """One back-projected detection, ready for the log."""

    t: float
    ground_local: Tuple[float, float]  # (north, east), flight-layer frame
    conf: float
    class_id: str
    ll: Optional[Tuple[float, float]] = None  # derived mine lat/lon
    tag_id: Optional[str] = None


@dataclass
class Cluster:
    cluster_id: int
    centroid: Tuple[float, float]
    weight: float
    n_obs: int = 0
    n_passes: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    status: str = CANDIDATE
    tag_ids: List[str] = field(default_factory=list)
    ll_per_pass: List[Tuple[float, float]] = field(default_factory=list)
    _sq_dev_sum: float = 0.0  # weighted squared deviation about the centroid
    _pass_ll_sum: Tuple[float, float] = (0.0, 0.0)
    _pass_ll_n: int = 0

    @property
    def spread_m(self) -> float:
        if self.weight <= 0.0:
            return 0.0
        return math.sqrt(self._sq_dev_sum / self.weight)

    @property
    def ll(self) -> Optional[Tuple[float, float]]:
        """Across-pass mean of the per-pass lat/lon fixes."""
        if not self.ll_per_pass:
            return None
        n = len(self.ll_per_pass)
        return (
            sum(p[0] for p in self.ll_per_pass) / n,
            sum(p[1] for p in self.ll_per_pass) / n,
        )


class MineLog:
    """Gated nearest-cluster ingest. A ground hit joins the nearest cluster
    within `gate_m` (default well under lane spacing) or founds a new one.
    A gap of more than `pass_gap_s` since a cluster was last seen starts a
    new pass; per-pass lat/lon means accumulate in `ll_per_pass`."""

    def __init__(
        self,
        gate_m: float = 1.0,
        pass_gap_s: float = 5.0,
        confirm_obs: int = 5,
        confirm_passes: int = 2,
    ) -> None:
        self.gate_m = gate_m
        self.pass_gap_s = pass_gap_s
        self.confirm_obs = confirm_obs
        self.confirm_passes = confirm_passes
        self.clusters: List[Cluster] = []
        self.n_ingested = 0

    def _nearest(self, p: Tuple[float, float]) -> Tuple[Optional[Cluster], float]:
        best, best_d = None, math.inf
        for c in self.clusters:
            d = math.hypot(p[0] - c.centroid[0], p[1] - c.centroid[1])
            if d < best_d:
                best, best_d = c, d
        return best, best_d

    def _close_pass(self, c: Cluster) -> None:
        if c._pass_ll_n > 0:
            c.ll_per_pass.append(
                (c._pass_ll_sum[0] / c._pass_ll_n, c._pass_ll_sum[1] / c._pass_ll_n)
            )
        c._pass_ll_sum = (0.0, 0.0)
        c._pass_ll_n = 0

    def ingest(self, obs: DetectionObs) -> Cluster:
        self.n_ingested += 1
        w = max(obs.conf, 1e-6)
        c, d = self._nearest(obs.ground_local)
        if c is None or d > self.gate_m:
            c = Cluster(
                cluster_id=len(self.clusters),
                centroid=obs.ground_local,
                weight=0.0,
                first_seen=obs.t,
                last_seen=obs.t,
                n_passes=1,
            )
            self.clusters.append(c)
        elif obs.t - c.last_seen > self.pass_gap_s:
            self._close_pass(c)
            c.n_passes += 1

        # weighted incremental centroid + spread
        new_w = c.weight + w
        dn = obs.ground_local[0] - c.centroid[0]
        de = obs.ground_local[1] - c.centroid[1]
        c.centroid = (c.centroid[0] + dn * w / new_w, c.centroid[1] + de * w / new_w)
        c._sq_dev_sum += w * (dn * dn + de * de) * (c.weight / new_w)
        c.weight = new_w
        c.n_obs += 1
        c.last_seen = obs.t
        if obs.ll is not None:
            c._pass_ll_sum = (c._pass_ll_sum[0] + obs.ll[0], c._pass_ll_sum[1] + obs.ll[1])
            c._pass_ll_n += 1
        if obs.tag_id is not None:
            if obs.tag_id not in c.tag_ids:
                c.tag_ids.append(obs.tag_id)
            c.status = VERIFIED
        elif (
            c.status == CANDIDATE
            and c.n_obs >= self.confirm_obs
            and c.n_passes >= self.confirm_passes
        ):
            c.status = CONFIRMED
        return c

    def finalize(self) -> None:
        """Close every open pass (call once before building the dump)."""
        for c in self.clusters:
            self._close_pass(c)

    def next_dip_target(self) -> Optional[Cluster]:
        """Oldest confirmed, un-dipped, un-verified cluster — the dip
        trigger policy's queue (budget is the engine's business)."""
        for c in self.clusters:
            if c.status == CONFIRMED:
                return c
        return None
