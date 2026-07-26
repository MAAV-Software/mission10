"""Cross-pass association: per-pass tracklets onto persistent map tracks.

The measured design (rfd-object-map, association section; experiments A/B
in reference/object-map-experiments/ASSOCIATION_EXPERIMENTS_20260715.md):

- Tracklets first: detections aggregate per pass with a robust
  median-trimmed center before any cross-pass decision.
- SE(2) before gate (A): the common pass->map transform is estimated
  before gating, because pass error is common-mode with a yaw component;
  translation-only alignment leaves the 1 m gate fragmenting at the
  0.4-0.6 m knee.
- One-to-one assignment (B): Hungarian with dummy rows for birth and
  miss removes duplicate merges outright; class is a hard gate.
- Visibility-conditioned miss/birth costs: a track outside the pass's
  visibility footprint counts as unobserved, not missed — leaving it
  unmatched is free, and it does not compete for nearby tracklets on
  equal terms with tracks the pass actually looked at. The caller
  supplies visibility as a callable (see `Associator.ingest_pass`),
  typically `VisibilityLedger.query` closed over the pass window.

Existence thresholds are deliberately absent (E: the numbers do not
exist until a bag with real clutter); tracks carry per-pass hit/miss
counts for the existence layer to consume later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .minelog import DetectionObs

NE = Tuple[float, float]
_BIG = 1e18


# --------------------------------------------------------------- tracklets


@dataclass
class Tracklet:
    """One object as one pass saw it."""

    center: NE  # flight-layer local frame
    t0: float
    t1: float
    n_obs: int
    weight: float
    spread_m: float
    class_id: str
    tag_ids: List[str] = field(default_factory=list)


def form_tracklets(
    observations: Sequence[DetectionObs],
    gate_m: float = 1.0,
    trim_floor_m: float = 0.5,
) -> List[Tracklet]:
    """Cluster one pass's detections into tracklets.

    Greedy nearest-center gating within the pass (same discipline as
    MineLog ingest), then a robust center per cluster: component-wise
    median, drop points beyond max(trim_floor_m, p90 radius), and take
    the confidence-weighted mean of the rest — the aggregation the A/B
    harness measured. Class is a hard split: clusters never mix classes.
    """
    clusters: Dict[str, List[List[DetectionObs]]] = {}
    for obs in sorted(observations, key=lambda o: o.t):
        rows = clusters.setdefault(obs.class_id, [])
        best, best_d = None, math.inf
        for members in rows:
            cn = sum(m.ground_local[0] for m in members) / len(members)
            ce = sum(m.ground_local[1] for m in members) / len(members)
            d = math.hypot(obs.ground_local[0] - cn, obs.ground_local[1] - ce)
            if d < best_d:
                best, best_d = members, d
        if best is None or best_d > gate_m:
            rows.append([obs])
        else:
            best.append(obs)

    out: List[Tracklet] = []
    for class_id, rows in clusters.items():
        for members in rows:
            xs = sorted(m.ground_local[0] for m in members)
            ys = sorted(m.ground_local[1] for m in members)
            median = (xs[len(xs) // 2], ys[len(ys) // 2])
            radii = [
                math.hypot(m.ground_local[0] - median[0], m.ground_local[1] - median[1])
                for m in members
            ]
            cut = max(trim_floor_m, sorted(radii)[max(0, math.ceil(0.9 * len(radii)) - 1)])
            kept = [m for m, r in zip(members, radii) if r <= cut]
            w = sum(max(m.conf, 1e-6) for m in kept)
            cn = sum(m.ground_local[0] * max(m.conf, 1e-6) for m in kept) / w
            ce = sum(m.ground_local[1] * max(m.conf, 1e-6) for m in kept) / w
            spread = math.sqrt(
                sum(
                    max(m.conf, 1e-6)
                    * ((m.ground_local[0] - cn) ** 2 + (m.ground_local[1] - ce) ** 2)
                    for m in kept
                )
                / w
            )
            tags: List[str] = []
            for m in members:
                if m.tag_id is not None and m.tag_id not in tags:
                    tags.append(m.tag_id)
            out.append(
                Tracklet(
                    center=(cn, ce),
                    t0=members[0].t,
                    t1=members[-1].t,
                    n_obs=len(members),
                    weight=w,
                    spread_m=spread,
                    class_id=class_id,
                    tag_ids=tags,
                )
            )
    return out


# ---------------------------------------------------------------- geometry


def rigid_fit(
    src: Sequence[NE], dst: Sequence[NE], rotation: bool = True
) -> Tuple[float, NE]:
    """Least-squares SE(2) taking src onto dst (closed form in 2-D)."""
    n = len(src)
    if n == 0:
        return 0.0, (0.0, 0.0)
    sm = (sum(p[0] for p in src) / n, sum(p[1] for p in src) / n)
    dm = (sum(p[0] for p in dst) / n, sum(p[1] for p in dst) / n)
    theta = 0.0
    if rotation and n >= 2:
        num = den = 0.0
        for (sx, sy), (dx, dy) in zip(src, dst):
            ax, ay = sx - sm[0], sy - sm[1]
            bx, by = dx - dm[0], dy - dm[1]
            num += ax * by - ay * bx
            den += ax * bx + ay * by
        if num != 0.0 or den != 0.0:
            theta = math.atan2(num, den)
    c, s = math.cos(theta), math.sin(theta)
    return theta, (dm[0] - (c * sm[0] - s * sm[1]), dm[1] - (s * sm[0] + c * sm[1]))


def apply_se2(theta: float, trans: NE, p: NE) -> NE:
    c, s = math.cos(theta), math.sin(theta)
    return (c * p[0] - s * p[1] + trans[0], s * p[0] + c * p[1] + trans[1])


def hungarian(cost: List[List[float]]) -> List[int]:
    """Minimum-cost perfect matching on a square matrix; returns the
    column assigned to each row. O(n^3) shortest-augmenting-path."""
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = [0] * n
    for j in range(1, n + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


def blind_align(
    obs: Sequence[NE],
    ref: Sequence[NE],
    rotation: bool = True,
    coarse_gate_m: float = 3.0,
    iterations: int = 8,
) -> Tuple[float, NE]:
    """Correspondence-free SE(2) estimate: alternate Hungarian pairing at
    a coarse gate with a rigid fit until fixed point (the A harness's
    alignment loop)."""
    theta, trans = 0.0, (0.0, 0.0)
    if not obs or not ref:
        return theta, trans
    for _ in range(iterations):
        aligned = [apply_se2(theta, trans, p) for p in obs]
        n = max(len(obs), len(ref))
        cost = [
            [
                (aligned[i][0] - ref[j][0]) ** 2 + (aligned[i][1] - ref[j][1]) ** 2
                if i < len(obs) and j < len(ref)
                else coarse_gate_m**2
                for j in range(n)
            ]
            for i in range(n)
        ]
        assign = hungarian(cost)
        pairs = [
            (i, j)
            for i, j in enumerate(assign)
            if i < len(obs) and j < len(ref) and cost[i][j] <= coarse_gate_m**2
        ]
        # Two pairs minimum even for translation: a single correspondence
        # would absorb its whole residual into the transform, erasing the
        # innovation the gate is supposed to judge.
        if len(pairs) < 2:
            break
        nt, nx = rigid_fit(
            [obs[i] for i, _ in pairs], [ref[j] for _, j in pairs], rotation=rotation
        )
        if abs(nt - theta) < 1e-5 and math.hypot(nx[0] - trans[0], nx[1] - trans[1]) < 1e-4:
            theta, trans = nt, nx
            break
        theta, trans = nt, nx
    return theta, trans


# ------------------------------------------------------------- association

# visibility verdicts (kept string-compatible with visibility.py)
V_OBSERVED = "observed"
V_EDGE = "edge"
V_UNSEEN = "unseen"


@dataclass
class Track:
    track_id: int
    class_id: str
    centers: List[NE] = field(default_factory=list)  # one per pass that hit
    weights: List[float] = field(default_factory=list)
    n_passes_hit: int = 0
    n_passes_missed: int = 0  # in-footprint, no detection (existence input)
    first_seen: float = 0.0
    last_seen: float = 0.0
    tag_ids: List[str] = field(default_factory=list)

    @property
    def center(self) -> NE:
        w = sum(self.weights)
        return (
            sum(c[0] * wi for c, wi in zip(self.centers, self.weights)) / w,
            sum(c[1] * wi for c, wi in zip(self.centers, self.weights)) / w,
        )


@dataclass
class PassResult:
    theta: float
    trans: NE
    matches: List[Tuple[int, int]]  # (track_id, tracklet index)
    born: List[int]  # track_ids created this pass
    missed: List[int]  # in-footprint tracks left unmatched
    out_of_view: List[int]  # tracks the pass never looked at


class Associator:
    """Persistent map of tracks; one `ingest_pass` per pass-tracklet set.

    Costs: candidate pairs beyond `gate_m` are inadmissible (the measured
    1 m gate is hard); within the gate, birth and miss dummies split the
    gate budget so an in-footprint track competes for its tracklet while
    an out-of-footprint track concedes cheaply (miss cost 0 — leaving it
    unmatched is free, and a tracklet near it births instead unless it is
    a strictly better fit).
    """

    def __init__(self, gate_m: float = 1.0, coarse_gate_m: float = 3.0) -> None:
        self.gate_m = gate_m
        self.coarse_gate_m = coarse_gate_m
        self.tracks: List[Track] = []

    def _new_track(self, tracklet: Tracklet, center: NE) -> Track:
        track = Track(
            track_id=len(self.tracks),
            class_id=tracklet.class_id,
            first_seen=tracklet.t0,
        )
        self.tracks.append(track)
        self._hit(track, tracklet, center)
        return track

    def _hit(self, track: Track, tracklet: Tracklet, center: NE) -> None:
        track.centers.append(center)
        track.weights.append(tracklet.weight)
        track.n_passes_hit += 1
        track.last_seen = tracklet.t1
        for tag in tracklet.tag_ids:
            if tag not in track.tag_ids:
                track.tag_ids.append(tag)

    def ingest_pass(
        self,
        tracklets: Sequence[Tracklet],
        visibility: Optional[Callable[[NE], str]] = None,
    ) -> PassResult:
        """Associate one pass. `visibility(ne) -> observed|edge|unseen`
        answers whether this pass looked at a map-frame point — typically
        `lambda ne: ledger.query(ne, margin_m=sigma, window=(t0, t1))`.
        None means every track counts as observed (no ledger yet)."""
        if not self.tracks:
            born = [self._new_track(tk, tk.center).track_id for tk in tracklets]
            return PassResult(0.0, (0.0, 0.0), [], born, [], [])

        obs = [tk.center for tk in tracklets]
        ref = [t.center for t in self.tracks]
        theta, trans = blind_align(
            obs, ref, rotation=len(obs) >= 2, coarse_gate_m=self.coarse_gate_m
        )
        aligned = [apply_se2(theta, trans, p) for p in obs]

        verdicts = [
            visibility(t.center) if visibility is not None else V_OBSERVED
            for t in self.tracks
        ]
        g2 = self.gate_m**2
        birth_cost = g2 / 2.0
        miss_cost = {V_OBSERVED: g2 / 2.0, V_EDGE: g2 / 4.0, V_UNSEEN: 0.0}

        n_obs, n_trk = len(tracklets), len(self.tracks)
        n = n_obs + n_trk
        cost = [[_BIG] * n for _ in range(n)]
        for i, tk in enumerate(tracklets):
            for j, track in enumerate(self.tracks):
                if tk.class_id != track.class_id:
                    continue
                d2 = (aligned[i][0] - ref[j][0]) ** 2 + (aligned[i][1] - ref[j][1]) ** 2
                if d2 <= g2:
                    cost[i][j] = d2
            cost[i][n_trk + i] = birth_cost  # birth dummy for this tracklet
        for j in range(n_trk):
            cost[n_obs + j][j] = miss_cost[verdicts[j]]  # miss dummy
            for i in range(n_obs):
                cost[n_obs + j][n_trk + i] = 0.0  # dummy-dummy: free

        assign = hungarian(cost)
        matches: List[Tuple[int, int]] = []
        born: List[int] = []
        matched_tracks = set()
        for i, tk in enumerate(tracklets):
            j = assign[i]
            if j < n_trk and cost[i][j] < _BIG:
                track = self.tracks[j]
                self._hit(track, tk, aligned[i])
                matches.append((track.track_id, i))
                matched_tracks.add(j)
            else:
                born.append(self._new_track(tk, aligned[i]).track_id)

        missed: List[int] = []
        out_of_view: List[int] = []
        for j, track in enumerate(self.tracks[: n_trk]):
            if j in matched_tracks:
                continue
            if verdicts[j] == V_UNSEEN:
                out_of_view.append(track.track_id)
            else:
                track.n_passes_missed += 1
                missed.append(track.track_id)
        return PassResult(theta, trans, matches, born, missed, out_of_view)
