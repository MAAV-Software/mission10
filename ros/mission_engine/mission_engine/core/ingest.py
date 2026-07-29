"""Detection ingest: join a detection to flight state at the image stamp,
back-project to a ground fix, and derive the global anchor.

The join discipline is the load-bearing part (rfd-mission-execution,
detection ingest section): every detection is paired with the flight
state nearest its *image* timestamp, never arrival time.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .backproject import AboveHorizon, ground_point
from .config import CameraModel
from .geometry import Quat, Vec3
from .minelog import DetectionObs

_M_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class PoseSnapshot:
    """Flight state at one instant, as the ROS rim samples it."""

    t: float
    pos: Vec3  # vehicle_local_position NED
    q: Quat  # body -> NED
    reset_counter: int = 0
    ll: Optional[Tuple[float, float]] = None  # vehicle_global_position
    agl: Optional[float] = None  # metres above ground; None = derive from -pos.z


class PoseHistory:
    """Small time-ordered buffer of snapshots with nearest-stamp lookup."""

    def __init__(self, horizon_s: float = 5.0) -> None:
        self.horizon_s = horizon_s
        self._ts: list = []
        self._snaps: list = []

    def append(self, snap: PoseSnapshot) -> None:
        if self._ts and snap.t < self._ts[-1]:
            raise ValueError(f"non-monotonic snapshot time {snap.t} < {self._ts[-1]}")
        self._ts.append(snap.t)
        self._snaps.append(snap)
        cutoff = snap.t - self.horizon_s
        drop = bisect.bisect_left(self._ts, cutoff)
        if drop > 0:
            del self._ts[:drop]
            del self._snaps[:drop]

    def clear(self) -> None:
        self._ts.clear()
        self._snaps.clear()

    def nearest(self, t: float) -> Optional[PoseSnapshot]:
        if not self._ts:
            return None
        i = bisect.bisect_left(self._ts, t)
        if i == 0:
            return self._snaps[0]
        if i == len(self._ts):
            return self._snaps[-1]
        before, after = self._snaps[i - 1], self._snaps[i]
        return before if t - before.t <= after.t - t else after


def mine_ll(
    drone_ll: Tuple[float, float], d_north_m: float, d_east_m: float
) -> Tuple[float, float]:
    """Drone lat/lon + local metric offset -> mine lat/lon (small-offset
    equirectangular shift; centimetre-exact at arena scale)."""
    lat = drone_ll[0] + d_north_m / _M_PER_DEG_LAT
    lon = drone_ll[1] + d_east_m / (_M_PER_DEG_LAT * math.cos(math.radians(drone_ll[0])))
    return (lat, lon)


def make_observation(
    cam: CameraModel,
    snap: PoseSnapshot,
    t_img: float,
    bbox_center_px: Tuple[float, float],
    conf: float,
    class_id: str,
    tag_id: Optional[str] = None,
) -> Optional[DetectionObs]:
    """Back-project one detection through the flight state at its image
    stamp. None = the ray missed the ground (logged upstream, not a fix)."""
    agl = snap.agl if snap.agl is not None else -snap.pos[2]
    if agl <= 0.0:
        return None
    ground_z = snap.pos[2] + agl
    try:
        n, e = ground_point(cam, snap.pos, snap.q, bbox_center_px, ground_z=ground_z)
    except AboveHorizon:
        return None
    ll = None
    if snap.ll is not None:
        ll = mine_ll(snap.ll, n - snap.pos[0], e - snap.pos[1])
    return DetectionObs(
        t=t_img,
        ground_local=(n, e),
        conf=conf,
        class_id=class_id,
        ll=ll,
        tag_id=tag_id,
    )
