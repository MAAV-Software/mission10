"""Rewire the procedural surface materials in m10-base.blend to world-space
mapping. Their noise/image textures previously used Generated coordinates
(the object's 0-1 bounding box), which stretched ~11x along the mixed-surface
strip because generate.py scales that plane non-uniformly. Geometry.Position
is world meters on any object, so the pattern scale matches between the 80 m
Ground and any strip; the /80 rescale preserves the look the scales were
tuned against on Ground.

    flatpak run --unset-env=PYTHONPATH org.blender.Blender \
        -b <abs>/assets/m10-base.blend --python tools/prep_surface_mapping.py
"""
import bpy

GROUND_M = 80.0

for name in ("gravel", "pavement", "concrete"):
    nt = bpy.data.materials[name].node_tree
    if any(n.bl_idname == "ShaderNodeNewGeometry" for n in nt.nodes):
        raise SystemExit(f"{name} already world-mapped — refusing to rescale twice")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    geo.location = (-700, 0)
    for n in nt.nodes:
        if n.bl_idname == "ShaderNodeTexNoise":
            nt.links.new(geo.outputs["Position"], n.inputs["Vector"])
            n.inputs["Scale"].default_value /= GROUND_M

nt = bpy.data.materials["dirt"].node_tree
if any(n.bl_idname == "ShaderNodeNewGeometry" for n in nt.nodes):
    raise SystemExit("dirt already world-mapped — refusing to rescale twice")
geo = nt.nodes.new("ShaderNodeNewGeometry")
geo.location = (-1100, 0)
for n in nt.nodes:
    if n.bl_idname == "ShaderNodeMapping":
        nt.links.new(geo.outputs["Position"], n.inputs["Vector"])
        n.inputs["Scale"].default_value[0] /= GROUND_M
        n.inputs["Scale"].default_value[1] /= GROUND_M
for n in [n for n in nt.nodes if n.bl_idname == "ShaderNodeTexCoord"]:
    nt.nodes.remove(n)

bpy.ops.wm.save_mainfile()
print("SURFACES world-mapped:", bpy.data.filepath)
