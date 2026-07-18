"""One scene = one minefield + one survey flight + its labels."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

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
class Scene:
    index: int
    mines: List[MinePose]
    stations: List[Station]
    surface: SurfaceChoice = field(default_factory=lambda: SurfaceChoice("grass"))
    tilt: float = 0.0  # camera tilt from nadir, degrees


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
    return Scene(
        index=index, mines=mines, stations=sts, surface=surface, tilt=tilt
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
