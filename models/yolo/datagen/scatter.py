"""Seeded mine placement on the ground plane."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
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
    tag_layout: str = "none"  # physical prop layout: both / one / none
    tag_up: bool = False  # landing flip; meaningful to visibility for one-face only
    tag_visible: bool = field(init=False)  # derived, never sampled directly

    def __post_init__(self) -> None:
        if self.tag_layout not in {"both", "one", "none"}:
            raise ValueError(f"bad tag_layout {self.tag_layout!r}")
        object.__setattr__(
            self,
            "tag_visible",
            self.tag_layout == "both" or (self.tag_layout == "one" and self.tag_up),
        )


def _tag_layout(cfg: GenConfig, rng: random.Random) -> str:
    weights = (cfg.p_tag_both, cfg.p_tag_one, cfg.p_tag_none)
    return rng.choices(("both", "one", "none"), weights=weights)[0]


def scatter(
    cfg: GenConfig, rng: random.Random, tag_rng: random.Random | None = None
) -> List[MinePose]:
    """Place mines, then attach independently sampled tag layout/flip state.

    build_scene supplies a dedicated per-scene tag_rng, keeping these new draws
    from perturbing the established placement/flightpath RNG stream.  The
    fallback preserves the convenient standalone scatter(cfg, rng) API.
    """
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
    tags = tag_rng if tag_rng is not None else rng
    return [
        MinePose(
            p.north,
            p.east,
            p.yaw,
            tag_layout=_tag_layout(cfg, tags),
            tag_up=tags.random() < cfg.tag_up_prob,
        )
        for p in placed
    ]
