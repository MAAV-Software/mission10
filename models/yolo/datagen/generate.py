"""Blender adapter — renders images matching the pure pipeline's labels.

UNTESTED SKELETON (no Blender on this machine yet); bench list at bottom.

    blender -b assets/Grass.blend -P datagen/generate.py -- \\
        --out out/ds1 --scenes 0:10

The pure pipeline decides placement, stations and labels (datagen.scene);
this file mirrors the scene into Blender and renders. Labels are NEVER
derived from bpy state — dump.write_scene emits them, so a rendered run
and a label-only run are identical by construction.

Frames: the pure core is local NED (north, east, down); Blender world is
Z-up right-handed. Mapping: (n, e, d) -> (x, y, z) = (e, n, -d). Blender
cameras look along local -Z with +Y up; the runtime optical frame is
+z forward, +x right (u), +y down (v) — so the Blender camera basis is
(x_opt, -y_opt, -z_opt), expressed in Blender world coordinates.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

from mission_engine.core.geometry import quat_rotate

from .config import GenConfig
from .dump import parse_range, write_scene
from .flightpath import Station
from .scene import build_scene, image_stem


def ned_to_blender(p):
    """NED (n, e, d) -> Blender Z-up (x, y, z). Linear; works for
    directions too."""
    return (p[1], p[0], -p[2])


def camera_basis_ned(tilt_deg: float, q):
    """Optical-frame axes as NED vectors (camera -> body columns match
    mission_engine.core.backproject._cam_to_body, then body -> NED by q)."""
    c = math.cos(math.radians(tilt_deg))
    s = math.sin(math.radians(tilt_deg))
    x_body = (0.0, 1.0, 0.0)
    y_body = (-c, 0.0, s)
    z_body = (s, 0.0, c)
    return tuple(quat_rotate(q, v) for v in (x_body, y_body, z_body))


def blender_camera_matrix(st: Station, tilt_deg: float):
    """4x4 matrix_world rows for the Blender camera at a station."""
    x_n, y_n, z_n = camera_basis_ned(tilt_deg, st.q)
    bx = ned_to_blender(x_n)
    by = tuple(-v for v in ned_to_blender(y_n))  # blender cam +Y is image-up
    bz = tuple(-v for v in ned_to_blender(z_n))  # blender cam looks along -Z
    t = ned_to_blender(st.pos)
    return [
        [bx[0], by[0], bz[0], t[0]],
        [bx[1], by[1], bz[1], t[1]],
        [bx[2], by[2], bz[2], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mine_blender_z_rotation(yaw_ned: float) -> float:
    """NED yaw (from north toward east) -> Blender z rotation (CCW from
    +X = east), assuming the mine template's long axis lies along +X."""
    return math.pi / 2.0 - yaw_ned


# --------------------------------------------------------------- bpy side


def _configure_camera(bpy, cfg: GenConfig) -> None:
    cam = bpy.context.scene.camera.data
    cam.sensor_fit = "HORIZONTAL"
    cam.angle_x = math.radians(cfg.camera.hfov_deg)
    render = bpy.context.scene.render
    render.resolution_x = cfg.camera.width_px
    render.resolution_y = cfg.camera.height_px
    render.resolution_percentage = 100


def _configure_render(bpy, cfg: GenConfig) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = cfg.render_samples
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    # device left at blender's default; pick METAL/CUDA/OPTIX on the bench


def _append_mine(bpy, blend: Path, object_name: str):
    if not blend.is_file():
        raise FileNotFoundError(f"mine blend not found: {blend}")
    bpy.ops.wm.append(
        filepath=str(blend / "Object" / object_name),
        directory=str(blend / "Object") + "/",
        filename=object_name,
    )
    template = bpy.data.objects[object_name]
    template.hide_render = True
    return template


def _randomize_sun(bpy, cfg: GenConfig, rng: random.Random) -> None:
    suns = [o for o in bpy.context.scene.objects if o.type == "LIGHT" and o.data.type == "SUN"]
    if not suns:
        raise RuntimeError("base scene has no sun light to randomize")
    sun = suns[0]
    sun.rotation_euler = (
        math.radians(90.0 - rng.uniform(*cfg.sun_elevation_deg)),
        0.0,
        math.radians(rng.uniform(*cfg.sun_azimuth_deg)),
    )
    sun.data.energy = rng.uniform(*cfg.sun_strength)


def _place_mines(bpy, template, scene, cfg: GenConfig, rng: random.Random):
    placed = []
    for m in scene.mines:
        obj = template.copy()
        obj.data = template.data.copy()
        obj.hide_render = False
        obj.location = ned_to_blender((m.north, m.east, 0.0))
        obj.rotation_euler = (0.0, 0.0, mine_blender_z_rotation(m.yaw))
        # hue jitter hook: vary the template material per mine within
        # cfg.mine_hue_jitter (bench: depends on the material node tree)
        bpy.context.collection.objects.link(obj)
        placed.append(obj)
    return placed


def _remove(bpy, objs) -> None:
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="render synthetic minefield scenes")
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", default="0:1")
    p.add_argument("--seed", default=None)
    p.add_argument(
        "--mine-blend",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "pfm1-mine-grass.blend"),
    )
    p.add_argument("--mine-object", default="IARC_PFM-1_mine")
    p.add_argument("--no-render", action="store_true", help="labels/manifest only")
    ns = p.parse_args(argv)

    import bpy  # the ONLY bpy import in datagen

    cfg = GenConfig(seed=ns.seed) if ns.seed is not None else GenConfig()
    out = Path(ns.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    _configure_camera(bpy, cfg)
    _configure_render(bpy, cfg)
    template = _append_mine(bpy, Path(ns.mine_blend), ns.mine_object)

    for index in parse_range(ns.scenes):
        write_scene(cfg, index, out)  # labels + manifest, pure pipeline
        if ns.no_render:
            continue
        scene = build_scene(cfg, index)  # deterministic: same objects
        rng = random.Random(f"{cfg.seed}:{index}:render")
        _randomize_sun(bpy, cfg, rng)
        mines = _place_mines(bpy, template, scene, cfg, rng)
        cam_obj = bpy.context.scene.camera
        for k, st in enumerate(scene.stations):
            cam_obj.matrix_world = blender_camera_matrix(st, cfg.camera.tilt_deg)
            bpy.context.scene.render.filepath = str(
                out / "images" / f"{image_stem(cfg, scene, k)}.png"
            )
            bpy.ops.render.render(write_still=True)
        _remove(bpy, mines)


# BENCH LIST (first Blender session):
# - Grass.blend is saved by Blender 5.0 (header v050); pfm1 blend by 4.3.
#   Pick ONE blender version for the pipeline and re-save; update assets.lock.
# - Verify the mine template object name and long-axis orientation
#   (mine_blender_z_rotation assumes the long axis lies along +X).
# - Retexture the mine to match the IARC prop, not a real PFM-1: matte
#   3D-printed PLA (no metal cylinder, no gloss), "IARC" engraving, 1-inch
#   AprilTag (family/IDs + one-or-both-faces per resource addendum; the tag
#   is a designed-in signature the detector SHOULD learn). See ../README.md
#   "The IARC prop is its own target class".
# - Verify FOV mapping (angle_x + sensor_fit=HORIZONTAL) with a ruler scene:
#   render a 1 m grid at known alt, check pixel extents vs project_raw.
# - Terrain relief vs flat-ground labels: labels assume ground z=0. Keep
#   relief small at survey alt, or extend MinePose with per-mine elevation
#   sampled from the terrain (pure core already takes corner z).
# - Spot-check overlays (draw label boxes onto renders) BEFORE any training
#   run — that's the image/label-mismatch tripwire this design enables.
# - GPU device: METAL on macOS, CUDA/OPTIX on linux; nixGL for GUI runs.

if __name__ == "__main__":
    main()
