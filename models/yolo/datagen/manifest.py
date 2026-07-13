"""Dataset manifest: enough to reproduce or audit any image from its stem."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .config import GenConfig
from .labels import YoloBox
from .scene import Scene, image_stem

SCHEMA = "minefield-datagen/1"


def scene_manifest(
    cfg: GenConfig, scene: Scene, labels: Dict[str, List[YoloBox]]
) -> dict:
    return {
        "schema": SCHEMA,
        "seed": cfg.seed,
        "scene": scene.index,
        "alt_m": scene.alt,
        "config": asdict(cfg),
        "mines": [asdict(m) for m in scene.mines],
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
