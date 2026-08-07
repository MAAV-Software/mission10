"""Dataset manifest: enough to reproduce or audit any image from its stem."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Sequence

from .config import GenConfig
from .labels import YoloBox
from .scene import Scene, image_stem

SCHEMA = "minefield-datagen/7"
READABLE_SCHEMAS = frozenset(("minefield-datagen/6", SCHEMA))
OCCLUSION_SCHEMA = "minefield-occlusion/2"


def scene_manifest(
    cfg: GenConfig,
    scene: Scene,
    labels: Dict[str, List[YoloBox]],
    station_indices: Sequence[int],
) -> dict:
    positive = sum(bool(labels[image_stem(cfg, scene, k)]) for k in station_indices)
    return {
        "schema": SCHEMA,
        "seed": cfg.seed,
        "scene": scene.index,
        "tilt_deg": scene.tilt,
        "config": asdict(cfg),
        "surface": asdict(scene.surface),
        "grass": asdict(scene.grass) if scene.grass is not None else None,
        "tag_visible_fraction": (
            sum(m.tag_visible for m in scene.mines) / len(scene.mines)
        ),
        "mines": [
            {**asdict(mine), "appearance": asdict(appearance)}
            for mine, appearance in zip(
                scene.mines, scene.mine_appearances, strict=True
            )
        ],
        "selection": {
            "candidate_stations": len(scene.stations),
            "selected_stations": len(station_indices),
            "selected_positive_stations": positive,
            "selected_negative_stations": len(station_indices) - positive,
            "analytic_negative_keep": cfg.negative_frame_keep,
        },
        "stations": [
            {
                "station_index": k,
                "stem": image_stem(cfg, scene, k),
                "pos": list(st.pos),
                "q_wxyz": list(st.q),
                "lane": st.lane,
                "s": st.s,
                "labels": [b.line() for b in labels[image_stem(cfg, scene, k)]],
            }
            for k in station_indices
            for st in (scene.stations[k],)
        ],
    }
