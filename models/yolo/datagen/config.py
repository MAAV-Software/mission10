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

    # Mine-filament domain randomization. Pick one dominant batch color per
    # scene, then vary each mine slightly around it. The lime tail covers our
    # brighter replica without making it the synthetic dataset's main prior.
    mine_color_names: Tuple[str, ...] = ("lime", "green", "muddy_olive")
    mine_color_palette_srgb: Tuple[Tuple[float, float, float], ...] = (
        (0x76 / 255, 0xA8 / 255, 0x2B / 255),  # #76A82B
        (0x4F / 255, 0x7D / 255, 0x36 / 255),  # #4F7D36
        (0x55 / 255, 0x57 / 255, 0x37 / 255),  # #555737
    )
    mine_color_weights: Tuple[float, ...] = (0.10, 0.45, 0.45)
    mine_color_hue_jitter_deg: float = 4.0
    mine_color_saturation_scale: Tuple[float, float] = (0.90, 1.10)
    mine_color_value_scale: Tuple[float, float] = (0.85, 1.15)

    # Grass-primary scenes are usually sparse, with rare deliberately hard
    # dense/tall scenes. A balanced deterministic schedule realizes this
    # fraction without small-shard Bernoulli variance. These are absolute GG
    # Grass Painter density inputs, sampled by the pure scene model so the
    # choice is manifest-auditable.
    grass_dense_prob: float = 0.10
    grass_sparse_blade_m: Tuple[float, float] = (0.12, 0.35)
    grass_sparse_density: Tuple[float, float] = (210.0, 600.0)
    grass_dense_blade_m: Tuple[float, float] = (0.50, 0.55)
    grass_dense_density: Tuple[float, float] = (1800.0, 2500.0)

    # render randomization (consumed by the bpy adapter only)
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
    render_samples: int = 16
    eevee_render_samples: int = 8
    png_compression: int = 15

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
        for name in ("mixed_surface_prob", "tag_up_prob", "grass_dense_prob"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"bad {name} {v}")
        weights = (self.p_tag_both, self.p_tag_one, self.p_tag_none)
        if min(weights) < 0.0 or sum(weights) <= 0.0:
            raise ValueError(f"bad tag layout weights {weights}")
        palette_lengths = (
            len(self.mine_color_names),
            len(self.mine_color_palette_srgb),
            len(self.mine_color_weights),
        )
        if min(palette_lengths) == 0 or len(set(palette_lengths)) != 1:
            raise ValueError(f"bad mine color palette lengths {palette_lengths}")
        if (
            min(self.mine_color_weights) < 0.0
            or sum(self.mine_color_weights) <= 0.0
        ):
            raise ValueError(f"bad mine color weights {self.mine_color_weights}")
        if any(
            len(rgb) != 3 or any(channel < 0.0 or channel > 1.0 for channel in rgb)
            for rgb in self.mine_color_palette_srgb
        ):
            raise ValueError(f"bad mine sRGB palette {self.mine_color_palette_srgb}")
        if self.mine_color_hue_jitter_deg < 0.0:
            raise ValueError(
                f"bad mine color hue jitter {self.mine_color_hue_jitter_deg}"
            )
        for scale, name in (
            (self.mine_color_saturation_scale, "mine_color_saturation_scale"),
            (self.mine_color_value_scale, "mine_color_value_scale"),
        ):
            if not (0.0 < scale[0] <= scale[1]):
                raise ValueError(f"bad {name} {scale}")
        for values, name in (
            (self.grass_sparse_blade_m, "grass_sparse_blade_m"),
            (self.grass_dense_blade_m, "grass_dense_blade_m"),
            (self.grass_sparse_density, "grass_sparse_density"),
            (self.grass_dense_density, "grass_dense_density"),
        ):
            if not (0.0 < values[0] <= values[1]):
                raise ValueError(f"bad {name} {values}")
        if self.render_samples < 1 or self.eevee_render_samples < 1:
            raise ValueError(
                "render sample counts must be positive: "
                f"{self.render_samples}, {self.eevee_render_samples}"
            )
        if not 0 <= self.png_compression <= 100:
            raise ValueError(f"bad png_compression {self.png_compression}")
        if not (0.0 <= self.tilt_range_deg[0] <= self.tilt_range_deg[1]):
            raise ValueError(f"bad tilt_range_deg {self.tilt_range_deg}")
        if self.exposure_jitter_ev[1] < self.exposure_jitter_ev[0]:
            raise ValueError(f"bad exposure_jitter_ev {self.exposure_jitter_ev}")
        if not (0.0 < self.sky_strength[0] <= self.sky_strength[1]):
            raise ValueError(f"bad sky_strength {self.sky_strength}")
