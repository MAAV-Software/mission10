"""One scene = one minefield + one survey flight + its labels."""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .config import GenConfig
from .flightpath import Station, stations
from .labels import YoloBox, yolo_box
from .scatter import MinePose, scatter


@dataclass(frozen=True)
class SurfaceChoice:
    primary: str
    secondary: Optional[str] = None
    strip_axis: Optional[str] = None  # direction the secondary strip runs
    strip_center_m: Optional[float] = None
    strip_width_m: Optional[float] = None


@dataclass(frozen=True)
class MineAppearance:
    """Auditable material choice for one mine, expressed in display sRGB."""

    color_family: str
    color_srgb: Tuple[float, float, float]


@dataclass(frozen=True)
class GrassChoice:
    """Auditable GG Grass Painter inputs for one grass-primary scene."""

    profile: str
    blade_m: float
    density: float


@dataclass(frozen=True)
class Scene:
    index: int
    mines: List[MinePose]
    stations: List[Station]
    mine_appearances: List[MineAppearance] = field(default_factory=list)
    surface: SurfaceChoice = field(default_factory=lambda: SurfaceChoice("grass"))
    grass: Optional[GrassChoice] = None
    tilt: float = 0.0  # camera tilt from nadir, degrees

    def __post_init__(self) -> None:
        if len(self.mine_appearances) != len(self.mines):
            raise ValueError(
                f"{len(self.mine_appearances)} mine appearances for "
                f"{len(self.mines)} mines"
            )


def _sample_surface(cfg: GenConfig, rng: random.Random) -> SurfaceChoice:
    primary = rng.choice(cfg.surface_materials)
    if rng.random() >= cfg.mixed_surface_prob:
        return SurfaceChoice(primary)

    secondary = rng.choice(
        tuple(name for name in cfg.surface_materials if name != primary)
    )
    axis = rng.choice(("north", "east"))
    width = rng.uniform(*cfg.mixed_strip_width_m)
    extent = cfg.east_extent if axis == "north" else cfg.north_extent
    half_width = width / 2.0
    if extent[1] - extent[0] >= width:
        center = rng.uniform(extent[0] + half_width, extent[1] - half_width)
    else:
        center = (extent[0] + extent[1]) / 2.0
    return SurfaceChoice(primary, secondary, axis, center, width)


def _sample_mine_appearances(
    cfg: GenConfig, count: int, rng: random.Random
) -> List[MineAppearance]:
    """Choose one filament batch per scene, then add mild per-mine variation."""
    palette_index = rng.choices(
        range(len(cfg.mine_color_names)), weights=cfg.mine_color_weights
    )[0]
    family = cfg.mine_color_names[palette_index]
    anchor = cfg.mine_color_palette_srgb[palette_index]
    base_h, base_s, base_v = colorsys.rgb_to_hsv(*anchor)
    hue_jitter = cfg.mine_color_hue_jitter_deg / 360.0

    appearances = []
    for _ in range(count):
        h = (base_h + rng.uniform(-hue_jitter, hue_jitter)) % 1.0
        s = min(
            1.0,
            max(
                0.0,
                base_s * rng.uniform(*cfg.mine_color_saturation_scale),
            ),
        )
        v = min(
            1.0,
            max(0.0, base_v * rng.uniform(*cfg.mine_color_value_scale)),
        )
        appearances.append(MineAppearance(family, colorsys.hsv_to_rgb(h, s, v)))
    return appearances


def _sample_grass(
    cfg: GenConfig,
    surface: SurfaceChoice,
    dense: bool,
    rng: random.Random,
) -> Optional[GrassChoice]:
    if surface.primary != "grass":
        return None
    # Reserve the former Bernoulli slot so changing to a balanced profile
    # schedule does not shift the established blade/density draws.
    rng.random()
    if dense:
        profile = "dense"
        blade_range = cfg.grass_dense_blade_m
        density_range = cfg.grass_dense_density
    else:
        profile = "sparse"
        blade_range = cfg.grass_sparse_blade_m
        density_range = cfg.grass_sparse_density
    return GrassChoice(
        profile=profile,
        blade_m=rng.uniform(*blade_range),
        density=rng.uniform(*density_range),
    )


def _dense_grass_for_ordinal(probability: float, ordinal: int) -> bool:
    """Low-discrepancy profile schedule with at most one-scene count error."""
    phase = 1.0 - probability
    epsilon = 1e-12
    before = math.floor(ordinal * probability + phase + epsilon)
    after = math.floor((ordinal + 1) * probability + phase + epsilon)
    return after > before


def _grass_primary_ordinal(cfg: GenConfig, index: int) -> int:
    """Zero-based position among grass-primary scenes before this index."""
    return sum(
        _sample_surface(
            cfg, random.Random(f"{cfg.seed}:{prior}:surface")
        ).primary
        == "grass"
        for prior in range(index)
    )


def build_scene(cfg: GenConfig, index: int) -> Scene:
    if not (0 <= index < cfg.n_scenes):
        raise ValueError(f"scene index {index} outside 0..{cfg.n_scenes - 1}")
    # string seeding is stable across runs and platforms
    rng = random.Random(f"{cfg.seed}:{index}")
    mines = scatter(cfg, rng, random.Random(f"{cfg.seed}:{index}:tags"))
    sts = stations(cfg, random.Random(f"{cfg.seed}:{index}:alts"), rng)
    surface = _sample_surface(cfg, random.Random(f"{cfg.seed}:{index}:surface"))
    # own stream: adding the draw must not shift mine/station sampling
    tilt = random.Random(f"{cfg.seed}:{index}:tilt").uniform(*cfg.tilt_range_deg)
    appearances = _sample_mine_appearances(
        cfg, len(mines), random.Random(f"{cfg.seed}:{index}:mine-colors")
    )
    grass_ordinal = (
        _grass_primary_ordinal(cfg, index)
        if surface.primary == "grass"
        else 0
    )
    grass = _sample_grass(
        cfg,
        surface,
        _dense_grass_for_ordinal(cfg.grass_dense_prob, grass_ordinal),
        random.Random(f"{cfg.seed}:{index}:grass-profile"),
    )
    return Scene(
        index=index,
        mines=mines,
        stations=sts,
        mine_appearances=appearances,
        surface=surface,
        grass=grass,
        tilt=tilt,
    )


def image_stem(cfg: GenConfig, scene: Scene, station_idx: int) -> str:
    return f"{cfg.seed}_s{scene.index:04d}_k{station_idx:04d}"


def scene_labels(cfg: GenConfig, scene: Scene) -> Dict[str, List[YoloBox]]:
    """stem -> boxes for every station (empty list = negative example)."""
    out: Dict[str, List[YoloBox]] = {}
    cam = replace(cfg.camera, tilt_deg=scene.tilt)
    for k, st in enumerate(scene.stations):
        boxes: List[YoloBox] = []
        for mine in scene.mines:
            box = yolo_box(
                cam,
                st.pos,
                st.q,
                mine,
                cfg.mine_dims_m,
                min_visible_frac=cfg.min_visible_frac,
                min_box_px=cfg.min_box_px,
            )
            if box is not None:
                boxes.append(box)
        out[image_stem(cfg, scene, k)] = boxes
    return out
