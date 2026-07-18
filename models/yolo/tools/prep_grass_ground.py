"""Replace the synthetic baked-grass ground material with real Poly Haven CC0
photo PBR, three variants spanning the Huntsville range's terrain palette
(doc/range-map.pdf: all candidate sites are managed agricultural ground):
aerial_grass_rock (patchy green over bare ground, 15 m aerial scan),
sparse_grass (thin grass over dirt), withered_grass (cured/dry August case).
All three sets are packed; the render adapter picks one per scene and swaps
the tagged image nodes (maav_map custom prop) plus the mapping scale. The
particle-strand layer on top is unchanged. Mapping is world-space
(Geometry.Position) at each set's real physical size.

    flatpak run --unset-env=PYTHONPATH org.blender.Blender \\
        -b <abs>/assets/m10-base.blend --python tools/prep_grass_ground.py
"""
from pathlib import Path

import bpy

TEX = Path(__file__).resolve().parents[1] / "assets" / "textures"
VARIANTS = {  # asset stem -> physical tile size in m
    "aerial_grass_rock": 15.0,
    "sparse_grass": 2.0,
    "withered_grass": 2.0,
}
DEFAULT = "aerial_grass_rock"

# pack every variant's maps; the adapter swaps between them by image name.
# fake_user keeps the sets the material doesn't currently reference from
# being garbage-collected on save. Idempotent: only missing images load.
packed = {}
for stem in VARIANTS:
    for tag in ("diff", "rough", "nor"):
        name = f"{stem}_{tag}_2k.jpg"
        img = bpy.data.images.get(name)
        if img is None:
            img = bpy.data.images.load(str(TEX / name), check_existing=True)
        if tag != "diff":
            img.colorspace_settings.name = "Non-Color"
        img.use_fake_user = True
        if not img.packed_file:
            img.pack()
        packed[(stem, tag)] = img

mat = bpy.data.materials["grass"]
nt = mat.node_tree
if any(n.get("maav_map") for n in nt.nodes):
    bpy.ops.wm.save_mainfile()
    raise SystemExit("material already built — variant packs topped up, saved")
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
out.location = (600, 0)
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
bsdf.location = (300, 0)
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
geo = nt.nodes.new("ShaderNodeNewGeometry")
geo.location = (-900, 0)
mapping = nt.nodes.new("ShaderNodeMapping")
mapping.location = (-700, 0)
s = 1.0 / VARIANTS[DEFAULT]
mapping.inputs["Scale"].default_value = (s, s, 1.0)
nt.links.new(geo.outputs["Position"], mapping.inputs["Vector"])
for i, tag in enumerate(("diff", "rough", "nor")):
    node = nt.nodes.new("ShaderNodeTexImage")
    node.location = (-450, 300 - 300 * i)
    node.image = packed[(DEFAULT, tag)]
    node["maav_map"] = tag  # the adapter finds swap targets by this prop
    nt.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    if tag == "diff":
        nt.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    elif tag == "rough":
        nt.links.new(node.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-150, -300)
        nt.links.new(node.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

bpy.ops.wm.save_mainfile()
print("GRASS GROUND retextured; variants packed:", ", ".join(VARIANTS))
