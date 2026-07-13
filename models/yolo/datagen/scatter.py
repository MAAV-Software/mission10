"""Seeded mine placement on the ground plane."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from .config import GenConfig

_MAX_ATTEMPTS = 1000


class ScatterFailed(Exception):
    """Could not place all mines under the separation constraint."""


@dataclass(frozen=True)
class MinePose:
    north: float
    east: float
    yaw: float  # NED yaw of the long axis


def scatter(cfg: GenConfig, rng: random.Random) -> List[MinePose]:
    n = rng.randint(cfg.mines_min, cfg.mines_max)
    m = cfg.edge_margin_m
    n_lo, n_hi = cfg.north_extent
    e_lo, e_hi = cfg.east_extent
    placed: List[MinePose] = []
    for _ in range(n):
        for _attempt in range(_MAX_ATTEMPTS):
            north = rng.uniform(n_lo + m, n_hi - m)
            east = rng.uniform(e_lo + m, e_hi - m)
            if all(
                (p.north - north) ** 2 + (p.east - east) ** 2
                >= cfg.min_separation_m**2
                for p in placed
            ):
                placed.append(MinePose(north, east, rng.uniform(0.0, 2 * math.pi)))
                break
        else:
            raise ScatterFailed(
                f"placed {len(placed)}/{n} mines after {_MAX_ATTEMPTS} attempts "
                f"(min_separation_m={cfg.min_separation_m})"
            )
    return placed
