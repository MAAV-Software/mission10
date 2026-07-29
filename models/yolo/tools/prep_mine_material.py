"""Retexture the mine template to match the IARC prop: matte 3D-printed PLA
in PFM-1 green (rules v3.1.2: "green/brown material"; the pure scene model
selects a bounded filament color) plus the rulebook's 1 in^2 AprilTag decal on
the flat top of the body. The decal is a separate mine_tag material (appearance
tint skips *tag* names) joined into the mesh so per-mine copy and the tag-down
flip work unchanged. All 24 tag36h11 images (assets/textures/apriltags/,
provenance in SOURCES.json) are packed so the render adapter can swap tag ids
per mine while the blend stays self-contained; the family is an assumption
until the IARC resource addendum names one.

    flatpak run --unset-env=PYTHONPATH org.blender.Blender \\
        -b <abs>/assets/m10-mine.blend --python tools/prep_mine_material.py
"""
from pathlib import Path

import bpy
from mathutils import Vector

TAGS = Path(__file__).resolve().parents[1] / "assets" / "textures" / "apriltags"
TAG_SIZE_M = 0.0254  # rulebook: 1 square inch decal
BODY_GREEN = (0.040, 0.075, 0.030, 1.0)  # linear; dark olive, PFM-1-like

if "mine_tag" in bpy.data.materials:
    raise SystemExit("mine_tag already exists — template is already retextured")

mine = bpy.data.objects["IARC_PFM-1_mine"]
body = bpy.data.materials["mine_body"]
body.diffuse_color = BODY_GREEN
for node in body.node_tree.nodes:
    if node.bl_idname == "ShaderNodeBsdfPrincipled":
        node.inputs["Base Color"].default_value = BODY_GREEN
        node.inputs["Roughness"].default_value = 0.65

# pack the whole tag pool; the adapter picks one per mine instance by name
images = []
for png in sorted(TAGS.glob("tag36_11_*.png")):
    img = bpy.data.images.load(str(png), check_existing=True)
    img.pack()
    images.append(img)
if not images:
    raise SystemExit(f"no tag images under {TAGS}")

tag_mat = bpy.data.materials.new("mine_tag")
nt = tag_mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
bsdf.inputs["Roughness"].default_value = 0.45
tex = nt.nodes.new("ShaderNodeTexImage")
tex.location = (-300, 0)
tex.image = images[0]
tex.interpolation = "Closest"  # the canonical renders are 10x10 px
tex.extension = "EXTEND"
nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

# the body top is a flat plateau around x=-30 mm; find its height by raycast
# and float the decal a hair above it to avoid z-fighting
deps = bpy.context.evaluated_depsgraph_get()
hit, loc, nrm, _ = mine.ray_cast(
    Vector((-0.030, 0.0, 0.05)), Vector((0, 0, -1)), depsgraph=deps
)
assert hit and nrm.z > 0.99, "no flat top plateau found at x=-30 mm"
bpy.ops.mesh.primitive_plane_add(
    size=TAG_SIZE_M, location=(-0.030, 0.0, loc.z + 0.0002)
)
decal = bpy.context.active_object
decal.data.materials.append(tag_mat)

# join drops UV layers absent from the target mesh; give the mine a matching
# UVMap so the decal's coordinates survive (body faces get zeros, unused)
if not mine.data.uv_layers:
    mine.data.uv_layers.new(name=decal.data.uv_layers[0].name)

bpy.ops.object.select_all(action="DESELECT")
decal.select_set(True)
mine.select_set(True)
bpy.context.view_layer.objects.active = mine
bpy.ops.object.join()

bpy.ops.wm.save_mainfile()
print(
    f"MINE RETEXTURED dims={tuple(round(v, 4) for v in mine.dimensions)} "
    f"slots={[s.material.name for s in mine.material_slots]} tags={len(images)}"
)
