"""Render the controlled mine color/scale acceptance diagnostic with Cycles.

The diagnostic contains exactly 18 images:

* 15 positives: five exact palette anchors by 30, 60, and 120 projected
  pixels, all on one fixed grass background; and
* three mine-free plates: one each for grass, dirt, and concrete.

Pose, lighting, and surface placement are fixed. Only the declared mine color
and camera altitude vary among positives. The three negative plates share the
60-pixel reference camera pose. Each positive gets an analytic centered box,
a one-line YOLO label, and matching manifest metadata. Plates have no label
file and declare an empty ground-truth list. The AprilTag faces the ground and
is also made transparent, so it is never part of the comparison.

The case model and projection calculations are pure Python for tests and
review. Run the Blender adapter with:

    blender -b assets/m10-base.blend -P tools/render_color_scale_matrix.py -- \
        --out /tmp/mine-color-scale

This is an acceptance diagnostic, not training data. The output directory
must be absent or empty so stale cases cannot contaminate an audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

# Blender executes --python files outside their package context.
YOLO_DIR = Path(__file__).resolve().parents[1]
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))
MISSION_ENGINE = YOLO_DIR.parents[1] / "ros" / "mission_engine"
if str(MISSION_ENGINE) not in sys.path:
    sys.path.insert(0, str(MISSION_ENGINE))

from datagen.config import GenConfig


SCHEMA = "mine-color-scale-diagnostic/1"
TARGET_WIDTHS_PX = (30, 60, 120)
POSITIVE_BACKGROUND = "grass"
BACKGROUND_PLATES = ("grass", "dirt", "concrete")
PLATE_REFERENCE_WIDTH_PX = 60
TEMPLATE_DIMENSION_TOLERANCE_M = 0.0005


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    color_family: str
    color_srgb: tuple[float, float, float]
    target_width_px: int
    altitude_m: float
    background: str = POSITIVE_BACKGROUND


@dataclass(frozen=True)
class BackgroundPlate:
    case_id: str
    altitude_m: float
    background: str


def altitude_for_projected_width(
    target_width_px: int,
    focal_px: float,
    mine_length_m: float,
) -> float:
    """Pinhole altitude for a nadir, image-horizontal mine of a given width."""
    if target_width_px <= 0:
        raise ValueError("target width must be positive")
    if focal_px <= 0.0 or mine_length_m <= 0.0:
        raise ValueError("focal length and mine length must be positive")
    return focal_px * mine_length_m / target_width_px


def _validate_backgrounds(cfg: GenConfig, backgrounds: Sequence[str]) -> None:
    if not backgrounds:
        raise ValueError("diagnostic needs at least one background")
    if len(set(backgrounds)) != len(backgrounds):
        raise ValueError("backgrounds must be unique")
    unknown = set(backgrounds) - set(cfg.surface_materials)
    if unknown:
        raise ValueError(f"unknown background materials: {sorted(unknown)}")


def diagnostic_cases(
    cfg: GenConfig | None = None,
    target_widths_px: Sequence[int] = TARGET_WIDTHS_PX,
    mine_dims_m: Sequence[float] | None = None,
) -> tuple[DiagnosticCase, ...]:
    """Build the stable five-color by three-size positive matrix."""
    cfg = cfg or GenConfig()
    dims = tuple(cfg.mine_dims_m if mine_dims_m is None else mine_dims_m)
    if len(dims) != 3 or any(dimension <= 0.0 for dimension in dims):
        raise ValueError(f"invalid mine dimensions: {dims}")
    if not target_widths_px:
        raise ValueError("diagnostic needs at least one target width")
    if len(set(target_widths_px)) != len(target_widths_px):
        raise ValueError("target widths must be unique")
    _validate_backgrounds(cfg, (POSITIVE_BACKGROUND,))

    cases = []
    for family, color in zip(
        cfg.mine_color_names, cfg.mine_color_palette_srgb, strict=True
    ):
        for width in target_widths_px:
            cases.append(
                DiagnosticCase(
                    case_id=f"{family}__{width:03d}px__{POSITIVE_BACKGROUND}",
                    color_family=family,
                    color_srgb=color,
                    target_width_px=width,
                    altitude_m=altitude_for_projected_width(
                        width, cfg.camera.focal_px, dims[0]
                    ),
                )
            )
    return tuple(cases)


def diagnostic_plates(
    cfg: GenConfig | None = None,
    mine_dims_m: Sequence[float] | None = None,
    backgrounds: Sequence[str] = BACKGROUND_PLATES,
) -> tuple[BackgroundPlate, ...]:
    """Build the three mine-free plates at one fixed reference camera pose."""
    cfg = cfg or GenConfig()
    dims = tuple(cfg.mine_dims_m if mine_dims_m is None else mine_dims_m)
    if len(dims) != 3 or any(dimension <= 0.0 for dimension in dims):
        raise ValueError(f"invalid mine dimensions: {dims}")
    _validate_backgrounds(cfg, backgrounds)
    altitude = altitude_for_projected_width(
        PLATE_REFERENCE_WIDTH_PX, cfg.camera.focal_px, dims[0]
    )
    return tuple(
        BackgroundPlate(
            case_id=f"background_plate__{background}",
            altitude_m=altitude,
            background=background,
        )
        for background in backgrounds
    )


def centered_ground_truth(
    cfg: GenConfig,
    target_width_px: int,
    mine_dims_m: Sequence[float] | None = None,
) -> dict:
    """Return the analytic centered box in pixel and normalized YOLO forms."""
    dims = tuple(cfg.mine_dims_m if mine_dims_m is None else mine_dims_m)
    if len(dims) != 3 or dims[0] <= 0.0 or dims[1] <= 0.0:
        raise ValueError(f"invalid mine dimensions: {dims}")
    if target_width_px <= 0:
        raise ValueError("target width must be positive")
    width_px = float(target_width_px)
    height_px = width_px * dims[1] / dims[0]
    image_width = float(cfg.camera.width_px)
    image_height = float(cfg.camera.height_px)
    if width_px > image_width or height_px > image_height:
        raise ValueError("ground-truth box exceeds the diagnostic image")
    yolo = (0.5, 0.5, width_px / image_width, height_px / image_height)
    return {
        "class_id": 0,
        "class_name": "mine",
        "xyxy_px": [
            (image_width - width_px) / 2.0,
            (image_height - height_px) / 2.0,
            (image_width + width_px) / 2.0,
            (image_height + height_px) / 2.0,
        ],
        "yolo_xywhn": list(yolo),
        "yolo_line": "0 " + " ".join(f"{value:.9f}" for value in yolo),
    }


def validate_template_dimensions(
    actual_m: Sequence[float],
    expected_m: Sequence[float],
    tolerance_m: float = TEMPLATE_DIMENSION_TOLERANCE_M,
) -> tuple[float, float, float]:
    """Reject an asset whose evaluated dimensions drift from GenConfig."""
    actual = tuple(float(value) for value in actual_m)
    expected = tuple(float(value) for value in expected_m)
    if len(actual) != 3 or len(expected) != 3 or tolerance_m < 0.0:
        raise ValueError("dimension vectors need three values and nonnegative tolerance")
    if any(
        not math.isfinite(value) or value <= 0.0 for value in actual + expected
    ):
        raise ValueError("mine dimensions must be finite and positive")
    errors = tuple(abs(got - want) for got, want in zip(actual, expected, strict=True))
    if any(error > tolerance_m for error in errors):
        raise RuntimeError(
            f"evaluated mine dimensions {actual} differ from configured "
            f"{expected} by {errors}; tolerance={tolerance_m} m"
        )
    return actual


def _prepare_output(out: Path) -> tuple[Path, Path]:
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):
            raise ValueError(f"output directory is not empty: {out}")
    else:
        out.mkdir(parents=True)
    images = out / "images"
    labels = out / "labels"
    images.mkdir()
    labels.mkdir()
    return images, labels


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cycles-device",
        choices=("auto", "cpu", "cuda", "optix"),
        default="auto",
    )
    parser.add_argument("--samples", type=int, default=64)
    ns = parser.parse_args(argv)
    if ns.samples < 1:
        parser.error("--samples must be positive")
    return ns


def _fixed_lighting(bpy) -> dict:
    """Set the one declared lighting state used by every diagnostic image."""
    elevation_deg = 55.0
    azimuth_deg = 135.0
    sun_strength = 5.0
    sky_strength = 0.15
    suns = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "LIGHT" and obj.data.type == "SUN"
    ]
    if not suns:
        raise RuntimeError("base scene has no sun light")
    sun = suns[0]
    sun.rotation_euler = (
        math.radians(90.0 - elevation_deg),
        0.0,
        math.radians(azimuth_deg),
    )
    sun.data.energy = sun_strength

    world = bpy.context.scene.world
    for node in world.node_tree.nodes if world and world.node_tree else ():
        if node.bl_idname == "ShaderNodeTexSky":
            node.sun_elevation = math.radians(elevation_deg)
            node.sun_rotation = math.radians(azimuth_deg)
        elif node.bl_idname == "ShaderNodeBackground":
            node.inputs["Strength"].default_value = sky_strength

    irradiance = sun_strength * math.sin(math.radians(elevation_deg))
    irradiance += 2.6 * sky_strength
    exposure_ev = math.log2(3.0 / irradiance)
    bpy.context.scene.view_settings.exposure = exposure_ev
    return {
        "sun_elevation_deg": elevation_deg,
        "sun_azimuth_deg": azimuth_deg,
        "sun_strength": sun_strength,
        "sky_strength": sky_strength,
        "exposure_ev": exposure_ev,
    }


def _hide_apriltag_materials(obj) -> None:
    """Make tag faces transparent as a second guard behind the tag-down pose."""
    for slot in obj.material_slots:
        material = slot.material
        if material is None or "tag" not in material.name.casefold():
            continue
        hidden = material.copy()
        hidden.diffuse_color = (0.0, 0.0, 0.0, 0.0)
        if hidden.use_nodes:
            nodes = hidden.node_tree.nodes
            nodes.clear()
            output = nodes.new("ShaderNodeOutputMaterial")
            transparent = nodes.new("ShaderNodeBsdfTransparent")
            hidden.node_tree.links.new(
                transparent.outputs["BSDF"], output.inputs["Surface"]
            )
        slot.material = hidden


def _set_nadir_camera(
    cam,
    Matrix,
    cfg: GenConfig,
    altitude_m: float,
    mine_height_m: float,
) -> None:
    n_mid = sum(cfg.north_extent) / 2.0
    e_mid = sum(cfg.east_extent) / 2.0
    from datagen.generate import ned_to_blender

    # Local camera -Z points down. Identity rotation fixes nadir view and maps
    # the mine template's +X long axis to horizontal image pixels.
    cam.matrix_world = Matrix.Translation(
        ned_to_blender(
            (
                n_mid,
                e_mid,
                -(altitude_m + mine_height_m / 2.0),
            )
        )
    )


def render(ns: argparse.Namespace) -> dict:
    """Blender adapter for the pure diagnostic description."""
    import bpy
    from mathutils import Matrix

    from datagen.generate import (
        _append_mine,
        _configure_camera,
        _configure_render,
        _named_material,
        _set_mine_color,
        _set_object_material,
        ned_to_blender,
    )

    cfg = GenConfig(render_samples=ns.samples)
    images, labels = _prepare_output(ns.out)

    _configure_camera(bpy, cfg)
    device = _configure_render(
        bpy, cfg, engine="cycles", cycles_device=ns.cycles_device
    )
    if bpy.context.scene.render.engine != "CYCLES":
        raise RuntimeError("diagnostic acceptance requires the Cycles engine")
    lighting = _fixed_lighting(bpy)

    ground = bpy.data.objects.get("Ground")
    if ground is None or ground.type != "MESH":
        raise RuntimeError("Ground mesh is missing from the base blend")
    # Exclude optional procedural grass blades and stale generated objects.
    for obj in bpy.context.scene.objects:
        if obj.name.startswith(("DatagenGrass", "DatagenMine")):
            obj.hide_render = True

    template = _append_mine(
        bpy, YOLO_DIR / "assets" / "m10-mine.blend", "IARC_PFM-1_mine"
    )
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_template = template.evaluated_get(depsgraph)
    actual_dims_m = validate_template_dimensions(
        evaluated_template.dimensions, cfg.mine_dims_m
    )
    cases = diagnostic_cases(cfg, mine_dims_m=actual_dims_m)
    plates = diagnostic_plates(cfg, mine_dims_m=actual_dims_m)
    if len(cases) != 15 or len(plates) != 3:
        raise RuntimeError(
            "diagnostic contract requires exactly 15 positives and 3 plates"
        )

    mine = template.copy()
    mine.data = template.data.copy()
    mine.name = "DatagenMineColorScaleDiagnostic"
    mine.hide_render = False
    n_mid = sum(cfg.north_extent) / 2.0
    e_mid = sum(cfg.east_extent) / 2.0
    mine.location = ned_to_blender((n_mid, e_mid, -actual_dims_m[2] / 2.0))
    mine.rotation_mode = "XYZ"
    mine.rotation_euler = (math.pi, 0.0, 0.0)
    bpy.context.collection.objects.link(mine)
    _hide_apriltag_materials(mine)

    cam = bpy.context.scene.camera
    positive_records = []
    _set_object_material(ground, _named_material(bpy, POSITIVE_BACKGROUND))
    for case in cases:
        mine.hide_render = False
        _set_mine_color(mine, case.color_srgb)
        _set_nadir_camera(
            cam, Matrix, cfg, case.altitude_m, actual_dims_m[2]
        )
        output = images / f"{case.case_id}.png"
        label = labels / f"{case.case_id}.txt"
        truth = centered_ground_truth(cfg, case.target_width_px, actual_dims_m)
        label.write_text(truth["yolo_line"] + "\n")
        bpy.context.scene.render.filepath = str(output)
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
        positive_records.append(
            {
                **asdict(case),
                "kind": "positive",
                "image": str(output.relative_to(ns.out)),
                "label": str(label.relative_to(ns.out)),
                "ground_truth": [truth],
            }
        )

    mine.hide_render = True
    plate_records = []
    for plate in plates:
        _set_object_material(ground, _named_material(bpy, plate.background))
        _set_nadir_camera(
            cam, Matrix, cfg, plate.altitude_m, actual_dims_m[2]
        )
        output = images / f"{plate.case_id}.png"
        bpy.context.scene.render.filepath = str(output)
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
        plate_records.append(
            {
                **asdict(plate),
                "kind": "background_plate",
                "image": str(output.relative_to(ns.out)),
                "label": None,
                "ground_truth": [],
            }
        )

    manifest = {
        "schema": SCHEMA,
        "renderer": {
            "engine": "CYCLES",
            "device": device,
            "samples": ns.samples,
            "blender_version": bpy.app.version_string,
        },
        "camera": asdict(cfg.camera),
        "mine": {
            "configured_dimensions_m": list(cfg.mine_dims_m),
            "evaluated_dimensions_m": list(actual_dims_m),
            "dimension_tolerance_m": TEMPLATE_DIMENSION_TOLERANCE_M,
            "pose": "centered, nadir, image-horizontal, AprilTag hidden",
        },
        "lighting": lighting,
        "positive_background": POSITIVE_BACKGROUND,
        "target_widths_px": list(TARGET_WIDTHS_PX),
        "plate_reference_width_px": PLATE_REFERENCE_WIDTH_PX,
        "positive_cases": positive_records,
        "background_plates": plate_records,
        "acceptance": {
            "required_positive_detections": len(positive_records),
            "allowed_background_plate_detections": 0,
        },
    }
    (ns.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    manifest = render(_arguments(argv))
    print(
        f"rendered {len(manifest['positive_cases'])} positives and "
        f"{len(manifest['background_plates'])} background plates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
