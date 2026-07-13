"""Generation parameters — every magic number lives here, validated."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from mission_engine.core.config import CameraModel


@dataclass(frozen=True)
class GenConfig:
    seed: str = "m10"  # dataset identity; per-scene rngs derive from it
    n_scenes: int = 50

    camera: CameraModel = field(default_factory=CameraModel)

    # field extent (local NED ground plane, z = 0)
    north_extent: Tuple[float, float] = (0.0, 25.0)
    east_extent: Tuple[float, float] = (0.0, 15.0)

    # mines (PFM-1 replica footprint)
    mines_min: int = 4
    mines_max: int = 12
    mine_dims_m: Tuple[float, float, float] = (0.12, 0.061, 0.020)
    min_separation_m: float = 1.0
    edge_margin_m: float = 0.5

    # camera stations (serpentine, lanes centered across the east extent)
    n_lanes: int = 3
    lane_spacing_m: float = 6.0
    station_interval_m: float = 1.0
    alt_range_m: Tuple[float, float] = (4.0, 8.0)  # sampled per scene
    yaw_jitter_deg: float = 3.0

    # labels
    min_visible_frac: float = 0.25  # clipped/raw bbox area at frame edges
    min_box_px: float = 4.0

    # render randomization (consumed by the bpy adapter only)
    sun_elevation_deg: Tuple[float, float] = (25.0, 80.0)
    sun_azimuth_deg: Tuple[float, float] = (0.0, 360.0)
    sun_strength: Tuple[float, float] = (2.0, 6.0)
    mine_hue_jitter: float = 0.05
    render_samples: int = 16

    def __post_init__(self) -> None:
        if self.n_scenes < 1:
            raise ValueError(f"n_scenes must be >= 1, got {self.n_scenes}")
        if not (1 <= self.mines_min <= self.mines_max):
            raise ValueError(f"bad mine count range {self.mines_min}..{self.mines_max}")
        for lo, hi, name in (
            (*self.north_extent, "north_extent"),
            (*self.east_extent, "east_extent"),
            (*self.alt_range_m, "alt_range_m"),
        ):
            if hi <= lo:
                raise ValueError(f"{name} not increasing: ({lo}, {hi})")
        if self.station_interval_m <= 0.0 or self.min_separation_m <= 0.0:
            raise ValueError("intervals/separations must be positive")
        span = (self.n_lanes - 1) * self.lane_spacing_m
        width = self.east_extent[1] - self.east_extent[0]
        if span > width:
            raise ValueError(f"lane span {span} exceeds field width {width}")
        if not (0.0 < self.min_visible_frac <= 1.0):
            raise ValueError(f"bad min_visible_frac {self.min_visible_frac}")
