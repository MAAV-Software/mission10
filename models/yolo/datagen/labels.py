"""YOLO boxes by projecting mine corners through the runtime camera model.

The old script computed boxes from nadir-camera trigonometry over flat
ground; here every corner goes through mission_engine's project_raw —
the tilt, the yaw and the perspective are exactly what the drone sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from mission_engine.core.backproject import BehindCamera, project_raw
from mission_engine.core.config import CameraModel
from mission_engine.core.geometry import Quat, Vec3

from .scatter import MinePose

CLASS_MINE = 0


@dataclass(frozen=True)
class YoloBox:
    cls: int
    cx: float  # all normalized to [0, 1]
    cy: float
    w: float
    h: float
    visible_frac: float  # clipped/raw bbox area (1.0 = fully in frame)

    def line(self) -> str:
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def mine_corners(m: MinePose, dims: Tuple[float, float, float]) -> List[Vec3]:
    """8 corners of the mine's oriented box sitting on the ground
    (local NED: ground at z=0, top face at z=-height)."""
    hl, hw = dims[0] / 2.0, dims[1] / 2.0
    c, s = math.cos(m.yaw), math.sin(m.yaw)
    corners: List[Vec3] = []
    for dx, dy in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw)):
        north = m.north + dx * c - dy * s
        east = m.east + dx * s + dy * c
        for z in (0.0, -dims[2]):
            corners.append((north, east, z))
    return corners


def yolo_box(
    cam: CameraModel,
    pos: Vec3,
    q: Quat,
    mine: MinePose,
    dims: Tuple[float, float, float],
    *,
    min_visible_frac: float,
    min_box_px: float,
) -> Optional[YoloBox]:
    """Axis-aligned image box over the projected corners, clipped to the
    frame. None = not (usefully) visible from this station."""
    us: List[float] = []
    vs: List[float] = []
    for corner in mine_corners(mine, dims):
        try:
            u, v = project_raw(cam, pos, q, corner)
        except BehindCamera:
            return None  # degenerate station/mine pairing
        us.append(u)
        vs.append(v)
    u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
    raw_area = (u1 - u0) * (v1 - v0)
    if raw_area <= 0.0:
        return None
    cu0, cu1 = max(u0, 0.0), min(u1, float(cam.width_px))
    cv0, cv1 = max(v0, 0.0), min(v1, float(cam.height_px))
    if cu1 <= cu0 or cv1 <= cv0:
        return None
    frac = ((cu1 - cu0) * (cv1 - cv0)) / raw_area
    if frac < min_visible_frac:
        return None
    if (cu1 - cu0) < min_box_px or (cv1 - cv0) < min_box_px:
        return None
    return YoloBox(
        cls=CLASS_MINE,
        cx=(cu0 + cu1) / 2.0 / cam.width_px,
        cy=(cv0 + cv1) / 2.0 / cam.height_px,
        w=(cu1 - cu0) / cam.width_px,
        h=(cv1 - cv0) / cam.height_px,
        visible_frac=frac,
    )
