"""Camera stations along the serpentine — the same lane generator the
mission engine flies (mission_engine.core.geometry.serpentine)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from mission_engine.core.geometry import Quat, Vec3, quat_from_yaw, serpentine

from .config import GenConfig


@dataclass(frozen=True)
class Station:
    pos: Vec3  # local NED, z = -alt
    q: Quat  # FRD body -> NED (yaw-only; level flight assumed)
    lane: int
    s: float  # arc length along the lane


def stations(
    cfg: GenConfig, alt_rng: random.Random, rng: random.Random
) -> List[Station]:
    # altitude is drawn per station, not per scene: scale must vary
    # independently of everything else that is per-scene (minefield layout,
    # ground variant, lighting), and stations are independent training
    # images, not a flyable trajectory
    width = cfg.east_extent[1] - cfg.east_extent[0]
    span = (cfg.n_lanes - 1) * cfg.lane_spacing_m
    origin = (cfg.north_extent[0], cfg.east_extent[0] + (width - span) / 2.0)
    lanes = serpentine(
        origin,
        cfg.north_extent[1] - cfg.north_extent[0],
        cfg.n_lanes,
        cfg.lane_spacing_m,
    )
    jit = math.radians(cfg.yaw_jitter_deg)
    out: List[Station] = []
    for lane in lanes:
        s = 0.0
        while s <= lane.length + 1e-9:
            north, east = lane.point_at(s)
            yaw = lane.heading + rng.uniform(-jit, jit)
            alt = alt_rng.uniform(*cfg.alt_range_m)
            out.append(
                Station(pos=(north, east, -alt), q=quat_from_yaw(yaw), lane=lane.index, s=s)
            )
            s += cfg.station_interval_m
    return out
