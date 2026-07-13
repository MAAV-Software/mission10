"""One scene = one minefield + one survey flight + its labels."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from .config import GenConfig
from .flightpath import Station, stations
from .labels import YoloBox, yolo_box
from .scatter import MinePose, scatter


@dataclass(frozen=True)
class Scene:
    index: int
    alt: float
    mines: List[MinePose]
    stations: List[Station]


def build_scene(cfg: GenConfig, index: int) -> Scene:
    if not (0 <= index < cfg.n_scenes):
        raise ValueError(f"scene index {index} outside 0..{cfg.n_scenes - 1}")
    # string seeding is stable across runs and platforms
    rng = random.Random(f"{cfg.seed}:{index}")
    alt = rng.uniform(*cfg.alt_range_m)
    mines = scatter(cfg, rng)
    sts = stations(cfg, alt, rng)
    return Scene(index=index, alt=alt, mines=mines, stations=sts)


def image_stem(cfg: GenConfig, scene: Scene, station_idx: int) -> str:
    return f"{cfg.seed}_s{scene.index:04d}_k{station_idx:04d}"


def scene_labels(cfg: GenConfig, scene: Scene) -> Dict[str, List[YoloBox]]:
    """stem -> boxes for every station (empty list = negative example)."""
    out: Dict[str, List[YoloBox]] = {}
    for k, st in enumerate(scene.stations):
        boxes: List[YoloBox] = []
        for mine in scene.mines:
            box = yolo_box(
                cfg.camera,
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
