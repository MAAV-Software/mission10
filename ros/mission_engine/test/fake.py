"""Kinematic fake (test-only): a point that chases setpoints at capped
velocity/acceleration + a scripted minefield that emits synthetic
detections through the real wire shapes (pixel bbox centers via
project_raw, inverted later by the real ingest)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from mission_engine.core.backproject import BehindCamera, project_raw
from mission_engine.core.config import CameraModel
from mission_engine.core.geometry import Quat, Vec3
from mission_engine.core.mission import Setpoint


class KinematicPoint:
    def __init__(self, pos: Vec3, v_max: float = 3.0, a_max: float = 6.0) -> None:
        self.pos = pos
        self.vel = (0.0, 0.0, 0.0)
        self.v_max = v_max
        self.a_max = a_max

    def step(self, dt: float, sp: Optional[Setpoint]) -> None:
        if sp is None:
            return
        d = tuple(sp.pos[i] - self.pos[i] for i in range(3))
        dist = math.sqrt(sum(x * x for x in d))
        speed = min(self.v_max, dist / dt) if dt > 0 else 0.0
        want = tuple(x / dist * speed if dist > 1e-9 else 0.0 for x in d)
        dv = tuple(want[i] - self.vel[i] for i in range(3))
        dv_n = math.sqrt(sum(x * x for x in dv))
        cap = self.a_max * dt
        if dv_n > cap:
            dv = tuple(x / dv_n * cap for x in dv)
        self.vel = tuple(self.vel[i] + dv[i] for i in range(3))
        self.pos = tuple(self.pos[i] + self.vel[i] * dt for i in range(3))


class ScriptedMinefield:
    """Emits (center_px, conf) for every mine inside the frame."""

    def __init__(self, cam: CameraModel, mines: List[Tuple[float, float]]) -> None:
        self.cam = cam
        self.mines = mines

    def detect(self, pos: Vec3, q: Quat) -> List[Tuple[Tuple[float, float], float]]:
        out = []
        for n, e in self.mines:
            try:
                u, v = project_raw(self.cam, pos, q, (n, e, 0.0))
            except BehindCamera:
                continue
            if 0.0 <= u < self.cam.width_px and 0.0 <= v < self.cam.height_px:
                out.append(((u, v), 0.9))
        return out
