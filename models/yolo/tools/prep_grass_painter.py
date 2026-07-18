"""Vendor the GG Grass Painter geometry-nodes generator into m10-base.blend,
replacing the archive particle-strand patch as the blade layer. The group
generates real tapered curve blades (procedural material, no image deps);
author shared it for free use. The template object this creates:

  DatagenGrassPatch — flat 6.8 m plane (subdivided 17x17 verts), transparent
  base material (the real photo-PBR ground below shows through; only blades
  render), NODES modifier with the painter group, and a "brush" vertex group
  wired as Paint Group. Weight acts as brush strength: lower weight = shorter
  + sparser blades with a hue shift, so the adapter fills the group with
  per-scene noise for hand-brushed-looking variation. hide_render=True: the
  adapter copies the patch into the camera-following grid and drives
  Height/Density/Patch Seed per scene.

    flatpak run --unset-env=PYTHONPATH org.blender.Blender \\
        -b <abs>/assets/m10-base.blend --python tools/prep_grass_painter.py
"""
from pathlib import Path

import bmesh
import bpy

GG_BLEND = Path.home() / "Downloads" / "GG_Grass_Painter_v1_2.blend"
GROUP = "GG Grass Painter"
PATCH_M = 6.8  # matches _GRASS_PATCH_M in datagen/generate.py

if "DatagenGrassPatch" in bpy.data.objects:
    raise SystemExit("DatagenGrassPatch already present — nothing to do")

with bpy.data.libraries.load(str(GG_BLEND)) as (_, dst):
    dst.node_groups = [GROUP]
group = bpy.data.node_groups[GROUP]

# the blade material rides in as a dependency named "Grass"; rename so it
# can't be confused with the (lowercase) "grass" ground material
blades = bpy.data.materials.get("Grass")
if blades is not None:
    blades.name = "gg_grass_blades"

bpy.ops.mesh.primitive_plane_add(size=PATCH_M, location=(0.0, 0.0, 0.001))
patch = bpy.context.object
patch.name = "DatagenGrassPatch"

# ~44 cm vertex pitch so the brush-weight field can hold noise; weights
# interpolate across faces, so this doubles as the noise scale
bm = bmesh.new()
bm.from_mesh(patch.data)
bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=15, use_grid_fill=True)
bm.to_mesh(patch.data)
bm.free()
vg = patch.vertex_groups.new(name="brush")
vg.add(list(range(len(patch.data.vertices))), 1.0, "REPLACE")

# base plane must not hide the ground material under it
base = bpy.data.materials.new("gg_patch_base")
base.use_nodes = True
nt = base.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
nt.links.new(transparent.outputs["BSDF"], out.inputs["Surface"])
patch.data.materials.append(base)

mod = patch.modifiers.new("GG Grass Painter", "NODES")
mod.node_group = group
idents = {
    item.name: item.identifier
    for item in group.interface.items_tree
    if item.item_type == "SOCKET" and item.in_out == "INPUT"
}
constants = {
    "Viewport": 4,
    "Render": 8,
    "Thickness": 0.15,  # near the author's look; couples into blade height
    "Random Rotation": 1.0,
    "Paint Group": "brush",  # per-scene noise weights, filled by the adapter
    "Animation": 0.0,  # wind sways are motion; stills don't want them
    "Strength": 0.0,
}
for name, value in constants.items():
    mod[idents[name]] = value
patch.hide_render = True

bpy.ops.wm.save_mainfile()
print("GRASS PAINTER vendored:", GROUP, "-> DatagenGrassPatch")
