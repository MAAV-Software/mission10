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
    # sampled per scene; the survey cruises anywhere from near-ground dips to
    # max mapping altitude, so train across the full envelope
    alt_range_m: Tuple[float, float] = (1.0, 8.0)
    yaw_jitter_deg: float = 3.0
    # camera tilt from nadir, sampled per scene: the physical CM2 mount is
    # nadir; the range covers in-flight body pitch plus viewpoint diversity
    tilt_range_deg: Tuple[float, float] = (0.0, 15.0)

    # labels
    min_visible_frac: float = 0.25  # clipped/raw bbox area at frame edges
    min_box_px: float = 4.0

    # ground-surface domain randomization (sampled once per scene)
    surface_materials: Tuple[str, ...] = (
        "grass",
        "dirt",
        "gravel",
        "pavement",
        "concrete",
    )
    mixed_surface_prob: float = 0.25
    mixed_strip_width_m: Tuple[float, float] = (2.0, 5.0)

    # AprilTag layout weights (relative; need not sum to 1). Equal both/one
    # weights deliberately do not bake in the addendum's unresolved
    # one-face-vs-both question; small p_none keeps untagged props rare.
    p_tag_both: float = 0.495
    p_tag_one: float = 0.495
    p_tag_none: float = 0.01
    tag_up_prob: float = 0.5  # guess: real resting orientation may be biased

    # render randomization (consumed by the bpy adapter only)
    # tallest blade length of the painter layer, sampled per scene: it sets
    # how much real blade occlusion mine silhouettes get. The managed arena
    # ground plausibly spans mowed stubble to shin-high August growth at the
    # edges; the per-area brush-weight noise then varies actual height (and
    # hue) well below the draw, so short cover appears within tall scenes too
    grass_blade_m: Tuple[float, float] = (0.05, 0.35)
    # per-scene multiplier on the strand count: thin patchy cover through
    # full density, never lush-only
    grass_density: Tuple[float, float] = (0.35, 1.0)
    sun_elevation_deg: Tuple[float, float] = (25.0, 80.0)
    sun_azimuth_deg: Tuple[float, float] = (0.0, 360.0)
    # clear-sky daylight is sun-dominated (~4:1 over sky fill); the bpy
    # adapter auto-exposes per scene, so this range varies shadow contrast
    # and color temperature rather than overall frame brightness
    sun_strength: Tuple[float, float] = (3.0, 8.0)
    # sky-fill ambience drawn per scene: low = hard clear-sky sun, high =
    # hazy/overcast fill. Too much fill washes translucent grass blades pale
    # (the AE holds total brightness, so only the sun:sky ratio shows)
    sky_strength: Tuple[float, float] = (0.08, 0.25)
    # per-scene AE metering error in EV on top of the auto-exposure: real AE
    # is imperfect, and the positive skew lets some frames run slightly hot
    # (clipped highlights) instead of every frame sitting at the same midtone
    exposure_jitter_ev: Tuple[float, float] = (-0.4, 0.9)
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

        if self.mixed_surface_prob > 0.0 and len(self.surface_materials) < 2:
            raise ValueError("mixed surfaces require at least two surface_materials")
        # silent-failure guards: a bad probability skews the dataset without
        # raising anywhere downstream
        for name in ("mixed_surface_prob", "tag_up_prob"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"bad {name} {v}")
        weights = (self.p_tag_both, self.p_tag_one, self.p_tag_none)
        if min(weights) < 0.0 or sum(weights) <= 0.0:
            raise ValueError(f"bad tag layout weights {weights}")
        if not (0.0 < self.grass_blade_m[0] <= self.grass_blade_m[1]):
            raise ValueError(f"bad grass_blade_m {self.grass_blade_m}")
        if not (0.0 < self.grass_density[0] <= self.grass_density[1] <= 1.0):
            raise ValueError(f"bad grass_density {self.grass_density}")
        if not (0.0 <= self.tilt_range_deg[0] <= self.tilt_range_deg[1]):
            raise ValueError(f"bad tilt_range_deg {self.tilt_range_deg}")
        if self.exposure_jitter_ev[1] < self.exposure_jitter_ev[0]:
            raise ValueError(f"bad exposure_jitter_ev {self.exposure_jitter_ev}")
        if not (0.0 < self.sky_strength[0] <= self.sky_strength[1]):
            raise ValueError(f"bad sky_strength {self.sky_strength}")
