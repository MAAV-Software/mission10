"""Render one scene/station through the real datagen helpers and paint the
YOLO boxes on a copy — the overlay tripwire as a repeatable bench script.

    flatpak run --unset-env=PYTHONPATH org.blender.Blender \
        -b assets/m10-base.blend --python tools/smoke_station.py -- \
        --scene 2 --station 41
"""
import argparse
import random
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

YOLO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(YOLO_DIR))
sys.path.insert(0, str(YOLO_DIR.parents[1] / "ros" / "mission_engine"))

from datagen.config import GenConfig
from datagen.scene import build_scene, image_stem
from datagen.generate import (
    _append_mine,
    _apply_surface,
    _configure_camera,
    _configure_render,
    _grass_grid,
    _grass_template,
    _move_grass_grid,
    _place_mines,
    _randomize_sun,
    blender_camera_matrix,
)
from datagen.dump import write_scene

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument("--scene", type=int, default=2)
p.add_argument("--station", type=int, default=41)
p.add_argument("--out", default=str(YOLO_DIR / "dataset" / "smoke" / "images"))
p.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
ns = p.parse_args(argv)

cfg = GenConfig()
out = Path(ns.out)
out.mkdir(parents=True, exist_ok=True)

_configure_camera(bpy, cfg)
_configure_render(bpy, cfg, ns.engine)
template = _append_mine(bpy, YOLO_DIR / "assets" / "m10-mine.blend", "IARC_PFM-1_mine")

grass_patch = _grass_template(bpy)

scene = build_scene(cfg, ns.scene)
_randomize_sun(bpy, cfg, random.Random(f"{cfg.seed}:{ns.scene}:render:sun"))
_apply_surface(bpy, cfg, scene, "Ground")
grass = _grass_grid(
    bpy,
    grass_patch,
    scene,
    random.Random(f"{cfg.seed}:{ns.scene}:render:grass"),
)
_place_mines(bpy, template, scene, cfg,
             random.Random(f"{cfg.seed}:{ns.scene}:render:mines"))

st = scene.stations[ns.station]
cam = bpy.context.scene.camera
cam.matrix_world = Matrix(blender_camera_matrix(st, scene.tilt))
t = cam.matrix_world.translation
_move_grass_grid(scene, grass, t.x, t.y)
stem = image_stem(cfg, scene, ns.station)
img_path = out / f"{stem}.png"
bpy.context.scene.render.filepath = str(img_path)
bpy.ops.render.render(write_still=True)

# overlay: paint box borders in red on a copy
img = bpy.data.images.load(str(img_path))
w, h = img.size
px = list(img.pixels)


def paint(u, v):
    if 0 <= u < w and 0 <= v < h:
        i = 4 * ((h - 1 - v) * w + u)  # image rows start at the bottom
        px[i:i + 3] = (1.0, 0.0, 0.0)


write_scene(cfg, ns.scene, out.parent)  # labels/<stem>.txt via the real pipeline
label_file = out.parent / "labels" / f"{stem}.txt"
boxes = [tuple(float(t) for t in line.split()[1:])
         for line in label_file.read_text().splitlines() if line.strip()]
for (cx, cy, bw, bh) in boxes:
    x0, x1 = int((cx - bw / 2) * w), int((cx + bw / 2) * w)
    y0, y1 = int((cy - bh / 2) * h), int((cy + bh / 2) * h)
    for u in range(x0, x1 + 1):
        for v in (y0, y1):
            paint(u, v)
    for v in range(y0, y1 + 1):
        for u in (x0, x1):
            paint(u, v)

img.pixels = px
img.filepath_raw = str(out / f"{stem}_overlay.png")
img.file_format = "PNG"
img.save()
print(f"SMOKE {img_path} boxes={len(boxes)} surface={scene.surface}")
