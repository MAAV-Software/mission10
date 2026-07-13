"""Write labels + manifests WITHOUT Blender (inspection / label-only runs).

    python3 -m datagen.dump --out /tmp/ds --scenes 0:5

generate.py (the bpy adapter) calls write_scene too, so rendered images
and label-only dumps are identical by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import GenConfig
from .manifest import scene_manifest
from .scene import build_scene, scene_labels


def parse_range(spec: str) -> range:
    lo, _, hi = spec.partition(":")
    if not hi:
        return range(int(lo), int(lo) + 1)
    return range(int(lo), int(hi))


def write_scene(cfg: GenConfig, index: int, out: Path) -> dict:
    scene = build_scene(cfg, index)
    labels = scene_labels(cfg, scene)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for stem, boxes in labels.items():
        (out / "labels" / f"{stem}.txt").write_text(
            "".join(b.line() + "\n" for b in boxes)
        )
    man = scene_manifest(cfg, scene, labels)
    (out / f"{cfg.seed}_s{scene.index:04d}.manifest.json").write_text(
        json.dumps(man, indent=2)
    )
    return man


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", default="0:1", help="LO:HI half-open, or one index")
    p.add_argument("--seed", default=None, help="override GenConfig.seed")
    ns = p.parse_args(argv)
    cfg = GenConfig(seed=ns.seed) if ns.seed is not None else GenConfig()
    out = Path(ns.out)
    total = 0
    for i in parse_range(ns.scenes):
        man = write_scene(cfg, i, out)
        n_boxes = sum(len(st["labels"]) for st in man["stations"])
        total += n_boxes
        print(
            f"scene {i}: {len(man['mines'])} mines, "
            f"{len(man['stations'])} stations, {n_boxes} boxes"
        )
    print(f"total label boxes: {total}")


if __name__ == "__main__":
    main()
