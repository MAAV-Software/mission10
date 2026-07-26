"""Visibility ledger: when each ground cell was actually in view.

Three consumers (rfd-object-map, association section): existence miss
counting (a missed pass is evidence of absence only where the pass
genuinely looked), transform-conditioned miss/birth costs in assignment
(query a track's position mapped through the candidate transform, with the
transform's positional uncertainty as the margin), and seam-coverage
accounting at regroup (union of footprints across drones).

Pass structure is resolved per cell, not per drone: a cell accumulates
*visits* — maximal in-view intervals separated by more than
`revisit_gap_s`, the same gap convention as MineLog's per-cluster pass
counting. A drone-global segmentation would be wrong here: the pose
stream is continuous across a serpentine, yet adjacent lanes revisit the
same ground minutes apart, and it is those revisits that are independent
visibility opportunities. Consumers with their own notion of a pass (a
tracklet's time span) query with that window.

Cells are visited by projection (world -> pixel), not by rasterizing a
footprint polygon: each candidate ground cell's center is projected into
the image and kept when it lands inside the frame, the ray descends at
least `min_depression_deg` below horizontal, and the slant range is within
`max_range_m`. Camera tilt, vehicle attitude, and the horizon are handled
uniformly, including forward-tilted mounts whose upper pixels never meet
the ground. The ground plane per frame is z = pos.z + agl, the same flat-
ground model as detection ingest.

Drive `note_pose` from the rim's pose stream — not only from frames with
detections; recording where the camera looked while nothing was detected
is the point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .backproject import _cam_to_body
from .config import CameraModel
from .geometry import quat_rotate
from .ingest import PoseSnapshot

# query verdicts
OBSERVED = "observed"  # every cell within the margin was observed
EDGE = "edge"  # some but not all cells within the margin were observed
UNSEEN = "unseen"  # no cell within the margin was observed

Cell = Tuple[int, int]


@dataclass
class Visit:
    """One maximal in-view interval for one cell."""

    t0: float
    t1: float
    n_frames: int = 1


class VisibilityLedger:
    """Accumulates per-cell visit intervals from a pose stream.

    Cells index the flight-layer local frame directly: cell (i, j) spans
    [i*cell_m, (i+1)*cell_m) north x [j*cell_m, (j+1)*cell_m) east, so
    ledgers from different drones share a grid once query points are
    mapped through the candidate inter-drone transform.
    """

    def __init__(
        self,
        cam: CameraModel,
        cell_m: float = 0.5,
        revisit_gap_s: float = 5.0,
        min_depression_deg: float = 20.0,
        max_range_m: float = 15.0,
        edge_margin_px: float = 0.0,
    ) -> None:
        if cell_m <= 0.0:
            raise ValueError(f"cell_m must be positive, got {cell_m}")
        self.cam = cam
        self.cell_m = cell_m
        self.revisit_gap_s = revisit_gap_s
        self.min_depression_deg = min_depression_deg
        self.max_range_m = max_range_m
        self.edge_margin_px = edge_margin_px
        self.cells: Dict[Cell, List[Visit]] = {}
        self.n_frames = 0
        self._last_t: Optional[float] = None
        self._tan_min_dep = math.tan(math.radians(min_depression_deg))

    def cell_of(self, ne: Tuple[float, float]) -> Cell:
        return (math.floor(ne[0] / self.cell_m), math.floor(ne[1] / self.cell_m))

    # ------------------------------------------------------------ ingest

    def note_pose(self, snap: PoseSnapshot) -> None:
        """Fold one pose into the ledger."""
        if self._last_t is not None and snap.t < self._last_t:
            raise ValueError(f"non-monotonic pose time {snap.t} < {self._last_t}")
        self._last_t = snap.t
        self.n_frames += 1

        agl = snap.agl if snap.agl is not None else -snap.pos[2]
        if agl <= 0.0:
            return
        ground_z = snap.pos[2] + agl

        # camera axes in NED, once per frame
        axes_body = _cam_to_body(self.cam.tilt_deg)
        x_ax, y_ax, z_ax = (quat_rotate(snap.q, a) for a in axes_body)

        # candidate radius: depression floor and slant range both cap it
        r_dep = agl / self._tan_min_dep if self._tan_min_dep > 0.0 else math.inf
        r_rng = math.sqrt(max(self.max_range_m**2 - agl * agl, 0.0))
        r = min(r_dep, r_rng)
        if r <= 0.0:
            return

        pn, pe, pz = snap.pos
        cm = self.cell_m
        f = self.cam.focal_px
        u_lo, u_hi = self.edge_margin_px, self.cam.width_px - 1 - self.edge_margin_px
        v_lo, v_hi = self.edge_margin_px, self.cam.height_px - 1 - self.edge_margin_px
        i0 = math.floor((pn - r) / cm)
        i1 = math.floor((pn + r) / cm)
        j0 = math.floor((pe - r) / cm)
        j1 = math.floor((pe + r) / cm)
        dz = ground_z - pz
        max_r2 = self.max_range_m**2
        t = snap.t
        gap = self.revisit_gap_s
        cells = self.cells
        for i in range(i0, i1 + 1):
            dn = (i + 0.5) * cm - pn
            for j in range(j0, j1 + 1):
                de = (j + 0.5) * cm - pe
                h2 = dn * dn + de * de
                if h2 + dz * dz > max_r2:
                    continue
                if dz * dz < h2 * self._tan_min_dep**2:
                    continue
                x_c = x_ax[0] * dn + x_ax[1] * de + x_ax[2] * dz
                y_c = y_ax[0] * dn + y_ax[1] * de + y_ax[2] * dz
                z_c = z_ax[0] * dn + z_ax[1] * de + z_ax[2] * dz
                if z_c <= 0.0:
                    continue
                u = self.cam.cx + f * x_c / z_c
                if not (u_lo <= u <= u_hi):
                    continue
                v = self.cam.cy + f * y_c / z_c
                if not (v_lo <= v <= v_hi):
                    continue
                visits = cells.get((i, j))
                if visits is None:
                    cells[(i, j)] = [Visit(t, t)]
                elif t - visits[-1].t1 > gap:
                    visits.append(Visit(t, t))
                else:
                    visits[-1].t1 = t
                    visits[-1].n_frames += 1

    # ------------------------------------------------------------- query

    def visits(self, ne: Tuple[float, float]) -> List[Visit]:
        """The cell's visibility opportunities (existence miss counting)."""
        return self.cells.get(self.cell_of(ne), [])

    def _cell_seen(
        self,
        cell: Cell,
        min_frames: int,
        window: Optional[Tuple[float, float]],
    ) -> bool:
        for v in self.cells.get(cell, ()):  # visits are time-ordered
            if v.n_frames < min_frames:
                continue
            if window is None or (v.t0 <= window[1] and v.t1 >= window[0]):
                return True
        return False

    def query(
        self,
        ne: Tuple[float, float],
        margin_m: float = 0.0,
        min_frames: int = 1,
        window: Optional[Tuple[float, float]] = None,
    ) -> str:
        """Was this local-frame point observed (during `window`, if given)?

        Considers every cell whose center lies within `margin_m` of the
        point (always at least the containing cell). All observed at
        `min_frames` -> OBSERVED; some -> EDGE; none -> UNSEEN. Pass the
        candidate transform's positional uncertainty as the margin so that
        near-boundary misses degrade to EDGE instead of counting as
        evidence of absence; pass the tracklet's time span as the window
        for per-pass miss/birth costs.
        """
        cm = self.cell_m
        k = math.ceil(margin_m / cm) if margin_m > 0.0 else 0
        ci, cj = self.cell_of(ne)
        seen = unseen = 0
        for i in range(ci - k, ci + k + 1):
            for j in range(cj - k, cj + k + 1):
                if (i, j) != (ci, cj):
                    dn = (i + 0.5) * cm - ne[0]
                    de = (j + 0.5) * cm - ne[1]
                    if dn * dn + de * de > margin_m * margin_m:
                        continue
                if self._cell_seen((i, j), min_frames, window):
                    seen += 1
                else:
                    unseen += 1
        if seen == 0:
            return UNSEEN
        return OBSERVED if unseen == 0 else EDGE

    def coverage_cells(self, min_frames: int = 1) -> Dict[Cell, int]:
        """cell -> number of qualifying visits. Seam-coverage input."""
        out: Dict[Cell, int] = {}
        for cell, visits in self.cells.items():
            n = sum(1 for v in visits if v.n_frames >= min_frames)
            if n:
                out[cell] = n
        return out

    def to_dict(self) -> Dict:
        return {
            "cell_m": self.cell_m,
            "revisit_gap_s": self.revisit_gap_s,
            "min_depression_deg": self.min_depression_deg,
            "max_range_m": self.max_range_m,
            "edge_margin_px": self.edge_margin_px,
            "n_frames": self.n_frames,
            "cells": sorted(
                [i, j, [[v.t0, v.t1, v.n_frames] for v in visits]]
                for (i, j), visits in self.cells.items()
            ),
        }
