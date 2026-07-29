"""Dataset manifest: enough to reproduce or audit any image from its stem."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .config import GenConfig
from .labels import YoloBox
from .scene import Scene, image_stem

SCHEMA = "minefield-datagen/4"


def scene_manifest(
    cfg: GenConfig, scene: Scene, labels: Dict[str, List[YoloBox]]
) -> dict:
    return {
        "schema": SCHEMA,
        "seed": cfg.seed,
        "scene": scene.index,
        "tilt_deg": scene.tilt,
        "config": asdict(cfg),
        "surface": asdict(scene.surface),
        "tag_visible_fraction": (
            sum(m.tag_visible for m in scene.mines) / len(scene.mines)
        ),
        "mines": [
            {**asdict(mine), "appearance": asdict(appearance)}
            for mine, appearance in zip(
                scene.mines, scene.mine_appearances, strict=True
            )
        ],
        "stations": [
            {
                "stem": image_stem(cfg, scene, k),
                "pos": list(st.pos),
                "q_wxyz": list(st.q),
                "lane": st.lane,
                "s": st.s,
                "labels": [b.line() for b in labels[image_stem(cfg, scene, k)]],
            }
            for k, st in enumerate(scene.stations)
        ],
    }
