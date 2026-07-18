"""Blender adapter — renders images matching the pure pipeline's labels.

Bench list at bottom.

    flatpak run --unset-env=PYTHONPATH org.blender.Blender \\
        -b assets/m10-base.blend -P datagen/generate.py -- \\
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
import colorsys
import json
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
    # grass blades are flat ribbons (~0.2 mm thick), not cylinders
    scene.cycles_curves.shape = "RIBBONS"
    # consecutive stations render an unchanged scene from a new camera pose;
    # without this every render invocation rebuilds and re-uploads the whole
    # multi-million-vert grass grid from scratch
    scene.render.use_persistent_data = True
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
    suns = [
        o
        for o in bpy.context.scene.objects
        if o.type == "LIGHT" and o.data.type == "SUN"
    ]
    if not suns:
        raise RuntimeError("base scene has no sun light to randomize")
    sun = suns[0]
    elevation = rng.uniform(*cfg.sun_elevation_deg)
    azimuth = rng.uniform(*cfg.sun_azimuth_deg)
    sun.rotation_euler = (
        math.radians(90.0 - elevation),
        0.0,
        math.radians(azimuth),
    )
    sun.data.energy = rng.uniform(*cfg.sun_strength)
    # per-scene fill: clear hard sun through hazy overcast (blade colors wash
    # out when fill dominates, so the low end matters for the grass reading)
    sky_strength = rng.uniform(*cfg.sky_strength)
    world = bpy.context.scene.world
    for node in world.node_tree.nodes if world and world.node_tree else ():
        if node.bl_idname == "ShaderNodeTexSky":
            # sky ambience must agree with the lamp or shadows contradict
            # the sky gradient; the lamp stays the direct light (disc off)
            node.sun_elevation = math.radians(elevation)
            node.sun_rotation = math.radians(azimuth)
        elif node.bl_idname == "ShaderNodeBackground":
            node.inputs["Strength"].default_value = sky_strength
    # auto-exposure, like the real camera's AE: hold ground irradiance at a
    # fixed display level so sun draws change shadows and color, not overall
    # frame brightness. 2.6 approximates the multiple-scattering sky's ambient
    # irradiance per unit background strength; 2.5 is the reference that keeps
    # midtones where the sky-fill calibration looked right.
    irradiance = sun.data.energy * math.sin(math.radians(elevation))
    irradiance += 2.6 * sky_strength
    exposure = math.log2(3.0 / irradiance)  # ref up from 2.5: 2.5 read dim
    # imperfect AE: per-scene metering error, skewed so some frames run hot
    exposure += rng.uniform(*cfg.exposure_jitter_ev)
    bpy.context.scene.view_settings.exposure = exposure


def _jitter_rgba_hue(rgba, amount: float, rng: random.Random):
    h, s, v = colorsys.rgb_to_hsv(*rgba[:3])
    r, g, b = colorsys.hsv_to_rgb((h + rng.uniform(-amount, amount)) % 1.0, s, v)
    return (r, g, b, rgba[3])


def _jitter_mine_hue(obj, amount: float, rng: random.Random) -> None:
    """Copy and tint body materials; AprilTag materials stay black/white."""
    if amount == 0.0:
        return
    for slot in obj.material_slots:
        source = slot.material
        if source is None or "tag" in source.name.lower():
            continue
        material = source.copy()
        material.diffuse_color = _jitter_rgba_hue(material.diffuse_color, amount, rng)
        if material.use_nodes:
            for node in material.node_tree.nodes:
                base_color = node.inputs.get("Base Color")
                if base_color is not None:
                    base_color.default_value = _jitter_rgba_hue(
                        base_color.default_value, amount, rng
                    )
        slot.material = material


def _assign_tag_id(bpy, obj, rng: random.Random) -> None:
    """Real mines carry unique AprilTag ids; pick one per instance from the
    tag36h11 pool packed into the template blend."""
    pool = sorted(
        img.name for img in bpy.data.images if img.name.startswith("tag36_11_")
    )
    if not pool:
        return
    name = rng.choice(pool)
    for slot in obj.material_slots:
        source = slot.material
        if source is None or "tag" not in source.name.lower():
            continue
        material = source.copy()
        for node in material.node_tree.nodes:
            if node.bl_idname == "ShaderNodeTexImage":
                node.image = bpy.data.images[name]
        slot.material = material


def _place_mines(bpy, template, scene, cfg: GenConfig, rng: random.Random):
    placed = []
    for m in scene.mines:
        obj = template.copy()
        obj.data = template.data.copy()
        obj.hide_render = False
        # m10-mine.blend's origin is the bbox center (so the tag-down flip is
        # symmetric); resting on the ground puts it at half thickness up
        obj.location = ned_to_blender((m.north, m.east, -cfg.mine_dims_m[2] / 2.0))
        # The camera only observes the derived bit: rotate the one-tag
        # template about its local long axis to put the tagged face down.
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (
            0.0 if m.tag_visible else math.pi,
            0.0,
            mine_blender_z_rotation(m.yaw),
        )
        _jitter_mine_hue(obj, cfg.mine_hue_jitter, rng)
        _assign_tag_id(bpy, obj, rng)
        bpy.context.collection.objects.link(obj)
        placed.append(obj)
    return placed


def _named_material(bpy, name: str):
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    lowered = name.casefold()
    for candidate in bpy.data.materials:
        if candidate.name.casefold() == lowered:
            return candidate
    raise RuntimeError(
        f"surface material {name!r} is missing from the base blend; "
        "create materials matching GenConfig.surface_materials"
    )


def _set_object_material(obj, material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


# Overrun the strip past the mine-field extents on each end so edge stations,
# which see ground beyond the extents at survey altitude, don't catch the strip
# terminating mid-frame.
_STRIP_OVERSHOOT_M = 15.0


# packed Poly Haven set stem, physical size (m), AE albedo compensation (EV):
# the auto-exposure in _randomize_sun meters only sun+sky irradiance, so a
# bright cured field would render near-white where a real camera's AE meters
# the ground and pulls down. The comp approximates that metering.
_GRASS_GROUND_VARIANTS = (
    ("aerial_grass_rock", 15.0, 0.0),
    ("sparse_grass", 2.0, -0.2),
    ("withered_grass", 2.0, -0.9),
)


def _randomize_grass_ground(bpy, cfg: GenConfig, scene) -> None:
    """Pick this scene's real-photo grass ground set (terrain palette draw).
    No-op on a blend that predates tools/prep_grass_ground.py."""
    material = bpy.data.materials.get("grass")
    if material is None or material.node_tree is None:
        return
    nodes = material.node_tree.nodes
    if not any(n.get("maav_map") for n in nodes):
        return
    rng = random.Random(f"{cfg.seed}:{scene.index}:render:groundtex")
    stem, size, ev_comp = rng.choice(_GRASS_GROUND_VARIANTS)
    if scene.surface.primary == "grass":  # comp only when the set is visible
        bpy.context.scene.view_settings.exposure += ev_comp
    for node in nodes:
        tag = node.get("maav_map")
        if tag:
            node.image = bpy.data.images[f"{stem}_{tag}_2k.jpg"]
        elif node.bl_idname == "ShaderNodeMapping":
            node.inputs["Scale"].default_value = (1.0 / size, 1.0 / size, 1.0)


def _apply_surface(bpy, cfg: GenConfig, scene, ground_object_name: str):
    """Assign the primary ground material and add an optional crossing strip."""
    ground = bpy.data.objects.get(ground_object_name)
    if ground is None or ground.type != "MESH":
        raise RuntimeError(f"ground mesh object {ground_object_name!r} not found")
    _randomize_grass_ground(bpy, cfg, scene)
    _set_object_material(ground, _named_material(bpy, scene.surface.primary))
    if scene.surface.secondary is None:
        return []

    n_mid = sum(cfg.north_extent) / 2.0
    e_mid = sum(cfg.east_extent) / 2.0
    n_len = cfg.north_extent[1] - cfg.north_extent[0] + 2.0 * _STRIP_OVERSHOOT_M
    e_len = cfg.east_extent[1] - cfg.east_extent[0] + 2.0 * _STRIP_OVERSHOOT_M
    if scene.surface.strip_axis == "north":  # strip runs along north
        center_ned = (n_mid, scene.surface.strip_center_m, -0.002)
        # blender axes: x = east = strip width, y = north = strip length
        scale = (scene.surface.strip_width_m / 2.0, n_len / 2.0, 1.0)
    else:  # strip runs along east
        center_ned = (scene.surface.strip_center_m, e_mid, -0.002)
        scale = (e_len / 2.0, scene.surface.strip_width_m / 2.0, 1.0)
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=ned_to_blender(center_ned))
    strip = bpy.context.object
    strip.name = "DatagenMixedSurfaceStrip"
    strip.scale = scale
    _set_object_material(strip, _named_material(bpy, scene.surface.secondary))
    return [strip]


# --- blade-grass layer: real blade occlusion over the photo-PBR ground ---
# The vendored GG Grass Painter template in m10-base (DatagenGrassPatch,
# tools/prep_grass_painter.py): a geometry-nodes generator emitting tapered
# curve blades with a procedural material. The archive Grass.blend particle
# path is retired (2026-07-17).
# A world-aligned 3x3 grid of copies follows the camera per station, so only
# the footprint pays the blade cost; a copy overlapping a non-grass strip is
# hidden rather than clipped, leaving the ground material to carry that band.
_GRASS_PATCH_M = 6.8
# GG calibration (probed 2026-07-17 at vendored Thickness 0.15): blade-tip
# height is ~0.22 m per Height unit, linear through Height 3.2 = 70 cm
# (re-probe if Thickness changes; below ~0.1 it also shrinks height). The
# blade draw sets the TALLEST blades — the brush-weight field scales height,
# density, and hue below it per area. Density 6000 with slim blades reads as
# finer-scale continuous cover, near the GG author's showcase look.
_GG_HEIGHT_M_PER_UNIT = 0.22
_GG_DENSITY = 6000.0


def _gg_idents(mod) -> dict:
    """Geometry-nodes inputs are keyed by socket identifier, not name."""
    return {
        item.name: item.identifier
        for item in mod.node_group.interface.items_tree
        if item.item_type == "SOCKET" and item.in_out == "INPUT"
    }


def _grass_template(bpy):
    patch = bpy.data.objects.get("DatagenGrassPatch")
    if patch is None:
        raise RuntimeError(
            "DatagenGrassPatch not in the blend — run tools/prep_grass_painter.py"
        )
    return patch


def _grass_grid(bpy, patch, scene, cfg: GenConfig, rng: random.Random):
    """Per-scene blade length + nine grid copies (positioned per station)."""
    # a grass strip on another surface is narrower than a patch; only a grass
    # primary gets the blade layer
    if patch is None or scene.surface.primary != "grass":
        return []
    # set per-scene inputs on the template before copying so the grid
    # inherits them; draw order predates the painter — do not reorder
    mod = patch.modifiers[0]
    idents = _gg_idents(mod)
    blade = rng.uniform(*cfg.grass_blade_m)
    mod[idents["Height"]] = blade / _GG_HEIGHT_M_PER_UNIT
    mod[idents["Density"]] = _GG_DENSITY * rng.uniform(*cfg.grass_density)
    # coverage gaps: the range fields are patchy turf, not lawn
    mod[idents["Patchiness"]] = rng.uniform(0.45, 0.65)
    mod[idents["Patch Scale"]] = rng.uniform(0.25, 0.5)
    # brush-strength field: painter weight shortens AND thins blades, and the
    # blade material shifts hue with relative length — the look hand-brushed
    # strokes give. One noise fill per scene on the shared template mesh
    # (~44 cm pitch, weights interpolate across faces); every grid copy
    # inherits it, so the pattern repeats per cell — accepted, the smooth
    # mottling tiles far less visibly than blade clumps would
    lo = rng.uniform(0.3, 0.6)
    vg = patch.vertex_groups["brush"]
    for v in patch.data.vertices:
        vg.add([v.index], rng.uniform(lo, 1.0), "REPLACE")
    copies = []
    # 3x3 camera-following grid, plus a flush row along each strip edge (the
    # grid rows snap off the strip, leaving up to a patch-width gap beside it)
    for _ in range(9 if scene.surface.secondary is None else 15):
        o = patch.copy()  # shares mesh + particle settings
        o.hide_render = False  # the template itself stays hidden
        bpy.context.collection.objects.link(o)
        copies.append(o)
    return copies


def _snap_out_of_strip(scene, x: float, y: float):
    """Blades must not overhang a non-grass strip: shift an overlapping patch
    along the strip normal until its edge sits flush with the strip edge (the
    overlap with the neighboring patch just reads as denser grass)."""
    surface = scene.surface
    if surface.secondary is None:
        return x, y
    half = (_GRASS_PATCH_M + surface.strip_width_m) / 2.0
    if surface.strip_axis == "north":  # strip spans east = blender x
        d = x - surface.strip_center_m
        if abs(d) < half:
            x = surface.strip_center_m + math.copysign(half, d or 1.0)
    else:  # strip spans north = blender y
        d = y - surface.strip_center_m
        if abs(d) < half:
            y = surface.strip_center_m + math.copysign(half, d or 1.0)
    return x, y


def _move_grass_grid(scene, copies, cam_x: float, cam_y: float) -> None:
    """Snap the 3x3 grid to world-aligned cells around the camera. Cell
    indices also seed the strands, so overlapping stations see identical
    grass and cells never pop between frames."""
    if not copies:
        return
    s = _GRASS_PATCH_M
    ci, cj = math.floor(cam_x / s), math.floor(cam_y / s)
    placements = [
        (_snap_out_of_strip(scene, (i + 0.5) * s, (j + 0.5) * s), (i * 131 + j) % 100003)
        for i in (ci - 1, ci, ci + 1)
        for j in (cj - 1, cj, cj + 1)
    ]
    surface = scene.surface
    if surface.secondary is not None:
        # flush row along each strip edge, tracking the camera along the strip
        off = (surface.strip_width_m + s) / 2.0
        along = (cj - 1, cj, cj + 1) if surface.strip_axis == "north" else (ci - 1, ci, ci + 1)
        for side, sign in enumerate((1.0, -1.0)):
            edge = surface.strip_center_m + sign * off
            for cell in along:
                a = (cell + 0.5) * s
                xy = (edge, a) if surface.strip_axis == "north" else (a, edge)
                placements.append((xy, (cell * 131 + 977 * (side + 1)) % 100003))
    for o, ((x, y), seed) in zip(copies, placements):
        if (o.location.x, o.location.y) != (x, y):
            # transform writes auto-tag; moving a patch never re-realizes
            # its blades (GG distribution is object-local, probed)
            o.location.x, o.location.y = x, y
        mod = o.modifiers[0]
        ident = _gg_idents(mod)["Patch Seed"]
        if mod[ident] != float(seed):
            # modifier-input writes do NOT auto-tag, hence update_tag — and
            # tagging re-runs the whole painter on CPU (~1M verts/patch), so
            # skip it whenever the cell seed is already right: the grid only
            # changes cells when the camera crosses a 6.8 m boundary, letting
            # ~85% of stations render with fully cached blades
            mod[ident] = float(seed)
            o.update_tag()


def _visible_fracs(bpy, mines) -> list:
    """Occlusion-only visible fraction per mine at the current camera pose:
    the area-x-view-cosine-weighted share of camera-facing faces whose first
    scene hit is the face itself. The painter blades are realized mesh, so
    they block rays like any geometry. Out-of-frame samples are skipped —
    frame-edge clipping is the pure pipeline's business, and counting it
    here would double-dip. ~256 systematic face samples per mine."""
    from bpy_extras.object_utils import world_to_camera_view

    scn = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    cam = scn.camera
    origin = cam.matrix_world.translation
    fracs = []
    for mine in mines:
        ev = mine.evaluated_get(dg)
        mw = ev.matrix_world
        rot = mw.to_3x3()
        polys = list(ev.data.polygons)
        step = max(1, len(polys) // 256)
        total = visible = 0.0
        for poly in polys[::step]:
            center = mw @ poly.center
            ray = center - origin
            dist = ray.length
            direction = ray / dist
            cos = -(rot @ poly.normal).normalized().dot(direction)
            if cos <= 0.0:
                continue  # back-facing
            ndc = world_to_camera_view(scn, cam, center)
            if ndc.z <= 0.0 or not (0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0):
                continue
            weight = poly.area * cos
            total += weight
            hit, loc, _n, _i, _o, _m = scn.ray_cast(dg, origin, direction)
            # first-hit-is-the-sample-point beats object identity: it also
            # handles grazing self-hits on the mine's own body correctly
            if hit and (loc - center).length < 1e-3:
                visible += weight
        fracs.append(visible / total if total > 0.0 else 0.0)
    return fracs


def _remove(bpy, objs) -> None:
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)


def _purge_orphans(bpy) -> None:
    """Drop the mesh/material datablocks left behind by removed per-scene
    objects. Object removal does NOT free the data.copy()/material.copy()
    datablocks, so without this they accumulate across a long render run
    (scenes * mines * slots) and eventually OOM."""
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=False, do_recursive=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="render synthetic minefield scenes")
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", default="0:1")
    p.add_argument("--seed", default=None)
    p.add_argument(
        "--mine-blend",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "m10-mine.blend"),
    )
    p.add_argument("--mine-object", default="IARC_PFM-1_mine")
    p.add_argument("--ground-object", default="Ground")
    p.add_argument("--no-render", action="store_true", help="labels/manifest only")
    ns = p.parse_args(argv)

    import bpy  # the ONLY bpy import in datagen
    from mathutils import Matrix

    cfg = GenConfig(seed=ns.seed) if ns.seed is not None else GenConfig()
    out = Path(ns.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    _configure_camera(bpy, cfg)
    _configure_render(bpy, cfg)
    template = _append_mine(bpy, Path(ns.mine_blend), ns.mine_object)
    grass_patch = None if ns.no_render else _grass_template(bpy)

    for index in parse_range(ns.scenes):
        write_scene(cfg, index, out)  # labels + manifest, pure pipeline
        if ns.no_render:
            continue
        scene = build_scene(cfg, index)  # deterministic: same objects
        _randomize_sun(
            bpy, cfg, random.Random(f"{cfg.seed}:{index}:render:sun")
        )
        surface_objs = _apply_surface(bpy, cfg, scene, ns.ground_object)
        grass = _grass_grid(
            bpy, grass_patch, scene, cfg,
            random.Random(f"{cfg.seed}:{index}:render:grass"),
        )
        mines = _place_mines(
            bpy,
            template,
            scene,
            cfg,
            random.Random(f"{cfg.seed}:{index}:render:mines"),
        )
        cam_obj = bpy.context.scene.camera
        occlusion = {}
        for k, st in enumerate(scene.stations):
            # bare list assignment fills the matrix column-major (transposed);
            # Matrix() applies the intended row-major reading
            cam_obj.matrix_world = Matrix(blender_camera_matrix(st, scene.tilt))
            t = cam_obj.matrix_world.translation
            _move_grass_grid(scene, grass, t.x, t.y)
            occlusion[image_stem(cfg, scene, k)] = {
                str(i): round(f, 4)
                for i, f in enumerate(_visible_fracs(bpy, mines))
            }
            bpy.context.scene.render.filepath = str(
                out / "images" / f"{image_stem(cfg, scene, k)}.png"
            )
            bpy.ops.render.render(write_still=True)
        # render-side sidecar: labels/manifest stay purely analytic, so the
        # measured occlusion lives beside them (datagen.materialize applies it)
        occ_dir = out / "occlusion"
        occ_dir.mkdir(parents=True, exist_ok=True)
        (occ_dir / f"{cfg.seed}_s{index:04d}.json").write_text(
            json.dumps(
                {
                    "schema": "minefield-occlusion/1",
                    "seed": cfg.seed,
                    "scene": index,
                    "visible_frac": occlusion,
                },
                indent=2,
            )
        )
        _remove(bpy, mines + surface_objs + grass)
        _purge_orphans(bpy)


# BENCH LIST — first session ran 2026-07-17 on Blender 5.1.2 (flatpak).
# Done that session (tools/prep_mine_template.py built assets/m10-mine.blend;
# assets/m10-base.blend derived from Grass.blend):
# - Pipeline pinned to Blender 5.1.x; both m10-* files saved by 5.1.2.
# - Template name + long-axis (+X) verified; junk transforms (0.001 non-uniform
#   scale, baked ~199 deg yaw) applied away; origin moved to the bbox center
#   (the pi-flip about local X used to sink the mesh and shift it ~2 cm).
# - Base scene: legacy Grass.blend had 13 render-enabled unlabeled mines
#   (dataset poison) and an animated camera whose fcurves overrode
#   matrix_world; m10-base has a bare 80 m Ground at z=0, the five named
#   surface materials, cleared camera/sun animation. Strip assignment, edge,
#   shadows, z-fighting all verified in renders.
# - Overlay tripwire ran and caught a real bug: matrix_world = list-of-rows
#   fills column-major (transposed camera). All boxes now land on mines.
# Still open:
# - Mine retextured (tools/prep_mine_material.py): matte PLA in PFM-1 green
#   (hue-jittered per instance), 1 in^2 tag36h11 decal on the body-top plateau,
#   all 24 tag images packed; _assign_tag_id picks a per-mine id from the pool.
#   tag_visible now actually changes pixels. Family + one-or-both-faces are
#   assumptions until the IARC resource addendum lands; green->brown variation
#   needs more than hue jitter (value/sat too) if the addendum shows brown.
# - Hue jitter: verify visually that jitter reads on the retextured prop
#   (tag material is named mine_tag and is skipped).
# - Grass is two layers: the photo-PBR ground material (see the real-grass
#   entry below; an interim synthetic nadir bake and its tools/
#   bake_grass_tile.py are deleted), and _grass_grid/_move_grass_grid put
#   REAL blades under the camera footprint for occlusion of mine silhouettes
#   (blade length = per-scene grass_blade_m knob on the
#   seed:scene:render:grass stream).
#   tools/smoke_station.py re-renders one scene/station + overlay to verify.
#   In the GUI viewport the strand layer draws as stripes — display_percentage
#   subsets particles by index and JIT order walks the patch in x-bands;
#   renders use all strands and are uniform (verified numerically + k41).
# - World is a physical sky (tools/prep_world_sky.py): the old background was
#   0.05 gray, so sun-only frames rendered dark with crushed shadows. Sky
#   Texture MULTIPLE_SCATTERING, sun disc off (the lamp stays the direct
#   light), background strength 0.35; _randomize_sun syncs sky angles to the
#   lamp draw and auto-exposes per scene (sun_strength now 3-8, sun-dominated
#   ~4:1 like clear-sky daylight; AE keeps frame brightness constant while
#   shadow contrast and color temperature vary). White balance 7500K on.
# - Surface mapping is world-space (tools/prep_surface_mapping.py rewired
#   m10-base's noise/dirt textures from Generated coords to Geometry.Position,
#   scales /80): Generated coords stretched ~11x along the non-uniformly
#   scaled strip plane. Verified s2/k41 (gravel) + s6/k30 (dirt on concrete).
# - Gravel/pavement/concrete are real Poly Haven CC0 PBR sets, meters-true
#   world-space mapping, packed into m10-base (tools/prep_surface_textures.py;
#   download URLs + md5 provenance in assets/textures/SOURCES.json).
# - alt_range_m widened to (1, 8) for the full cruise envelope. At 8 m the
#   footprint's far corner can reach ~7.3 m from the camera vs the 6.8 m the
#   3x3 grass grid guarantees: a rare bare-of-strands corner, which shows the
#   (view-consistent) baked grass — acceptable, revisit only if it reads.
# - Camera mount is nadir (CameraModel tilt_deg default 0, decision closed
#   2026-07-17); datagen draws per-scene tilt from tilt_range_deg (0-15) on
#   its own RNG stream, threaded through Scene.tilt into both the label
#   projection and the Blender camera — they cannot disagree.
# - Grass de-lushed round one: grass_blade_m (0.015, 0.07) + per-scene
#   grass_density (0.35-1.0) count multiplier (baked count remembered in the
#   maav_base_count ID prop so scenes scale from the original). Range map
#   (doc/range-map.pdf): all nine candidate sites are managed agricultural
#   ground, so dirt/stubble should dominate.
# - Grass ground is REAL photo PBR now (tools/prep_grass_ground.py): three
#   packed Poly Haven variants spanning the terrain palette (aerial_grass_rock
#   patchy green / sparse_grass thin-over-dirt / withered_grass cured August),
#   drawn per scene by _randomize_grass_ground on the :render:groundtex
#   stream; the synthetic baked tile (grass_baked.png, bake_grass_tile.py) is
#   deleted outright. Gotcha: unreferenced packed images need
#   use_fake_user or the save garbage-collects them. STILL OPEN: strand layer
#   tint should follow the drawn variant (bright-green strands over withered
#   ground currently read as regrowth — plausible but always-green).
# - Blade radius: the archive strands were ~2 cm wide tubes (showcase asset);
#   the adapter now overrides root/tip radius to real blade width (~3.6 mm).
# - Blade layer is the GG Grass Painter now (tools/prep_grass_painter.py
#   vendors the geometry-nodes group + procedural blade material into m10-base
#   as DatagenGrassPatch; author shared the blend for free use). Real tapered
#   curve blades replace the particle ribbons; per-scene Height/Density map
#   from the same grass stream draws, plus new Patchiness/Patch Scale draws
#   for coverage gaps. Distribution is object-local (probed), so the per-cell
#   seed goes through the Patch Seed input — same determinism scheme as the
#   particle path, which remains as a fallback for a pre-painter blend.
#   Realized mesh blades also rasterize in EEVEE (no hair artifacts).
#   Calibration constants _GG_HEIGHT_M_PER_UNIT/_GG_DENSITY probed 2026-07-17.
#   Legacy particle path fully retired same day (--grass-blend arg gone;
#   _grass_template errors on a blend without the vendored patch).
# - Per-variant AE albedo comp in _GRASS_GROUND_VARIANTS: the AE meters only
#   sun+sky, so the bright withered set rendered near-white under hot jitter
#   draws; a real camera meters the ground. Withered -0.9 EV, sparse -0.2.
# - Sky fill is a per-scene draw now (cfg.sky_strength 0.08-0.25, was fixed
#   0.35 in the blend): strong fill washed the translucent painter blades
#   mint-pale — the AE holds brightness, so only the sun:sky ratio shows.
#   Draw sits mid-sun-stream (exposure needs it), which shifted the jitter
#   draws of prior scenes; fine pre-dataset, don't do it again post-shard.
# - Brush-weight field (2026-07-17): the painter's Paint Group weight acts as
#   brush strength — probed: height AND density scale ~linearly with weight,
#   and hue shifts with it via the material's lngth_nrm ramps. The template
#   is subdivided (289 verts, ~44 cm pitch) with a "brush" vertex group;
#   _grass_grid fills it with per-scene noise (floor drawn 0.3-0.6), giving
#   hand-brushed height/hue/density mottling. grass_blade_m raised to
#   (0.05, 0.35) — the draw is now the TALLEST blades (mowed stubble through
#   shin-high August growth; 70 cm tried, too tall for a managed arena),
#   weight varies below it. Height slope 0.22 m/unit verified linear
#   through Height 3.2 = 70 cm. Weight pattern repeats per grid cell (mesh is
#   shared) — accepted, smooth mottling tiles far less visibly than clumps.
#   Tall draws make the buried-mine policy question below MORE acute.
# - AE reference 2.5 -> 3.0 plus per-scene exposure_jitter_ev (-0.4, +0.9):
#   frames were uniformly midtone-dark; the hot-skewed metering error brings
#   back occasional near-clipped bright frames.
# - Training tiling (datagen.materialize --tiles): the tile grid geometry
#   lives in mission_engine.core.tiles so onboard tiled inference and the
#   training materializer cannot drift — 640 px tiles, 192 px min overlap
#   (> the largest projected mine at 1 m alt, so every mine appears whole in
#   some tile). Box survival is occlusion x clip product >= --min-frac;
#   tiles a visible mine crosses without keeping a label are skipped
#   (unlabeled mine pixels teach suppression); empty tiles subsampled;
#   a seeded frame slice ships whole for the untiled inference mode with
#   min_box_px scaled to the 640 letterbox. Grid jitter is train-only.
#   Tile image cropping needs Pillow (RunPod; --no-images for local runs).
# - Occlusion validation renders: scene 4/2 sightings measure 0.75-1.0 under
#   current draws; forcing 35 cm full-density cover craters s0002 m4 at
#   k0034 to 0.02 with the render showing solid grass
#   (assets/_visfrac_probe.py --tall --render reruns it). tag_visible is
#   deliberately not occlusion-refined: nothing consumes it, and real decode
#   questions get answered by running the real AprilTag detector.
# - Hard negatives: place real distractor assets (the arena's decoys are
#   deliberately mine-like inert objects, rules v3.1.2 §85) once assets exist;
#   procedural primitives were tried and cut as not mine-like enough to help.
# - Verify _purge_orphans across a long (hundreds of scenes) run: bpy.data
#   meshes/materials must stay flat, not grow per scene. The template
#   data.copy() per mine is covered by the same purge.
# - Verify FOV mapping (angle_x + sensor_fit=HORIZONTAL) with a ruler scene:
#   render a 1 m grid at known alt, check pixel extents vs project_raw.
# - Confirm _STRIP_OVERSHOOT_M clears the widest FOV at max alt from edge
#   stations (smoke frames only sampled two stations).
# - GPU: this host's flatpak Cycles sees no CUDA/OPTIX (CPU ~9 s/frame, so a
#   full 50-scene set is ~10 h local); bulk renders go to RunPod. Run this
#   overlay spot-check again on RunPod's first shard.
# - EEVEE frame profile (assets/_gen_eevee.py, 78-frame run): grid move ~0 ms
#   (Patch-Seed caching in _move_grass_grid; occlusion sidecars byte-identical
#   before/after the caching change), visfrac ray-cast ~0.9 s, the render op
#   itself ~4.4 s even at 16 samples with use_persistent_data — per-invocation
#   bpy.ops.render overhead (scene sync / shadow bake / PNG encode) dominates,
#   not our scene mutation. NEXT GOAL: animation-mode restructure — drive
#   camera pose + grass Patch Seeds from a frame_change_pre handler and issue
#   ONE render op per scene (frame range = stations) so Blender amortizes the
#   sync; that is the identified path to sub-second frames.

if __name__ == "__main__":
    main()
