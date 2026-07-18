"""Replace the procedural gravel/pavement/concrete placeholders in
m10-base.blend with Poly Haven CC0 PBR sets (downloaded to assets/textures/,
provenance in SOURCES.json there). Mapping is world-space (Geometry.Position)
at each texture's real physical size, so the pattern is meters-true on the
80 m Ground and the non-uniformly scaled strip alike. Images are packed so
the blend stays self-contained for RunPod.

    flatpak run --unset-env=PYTHONPATH org.blender.Blender \
        -b <abs>/assets/m10-base.blend --python tools/prep_surface_textures.py
"""
from pathlib import Path

import bpy

TEX = Path(__file__).resolve().parents[1] / "assets" / "textures"
SETS = {  # material -> (asset stem, physical tile size in m)
    "gravel": ("gravel_road", 2.0),
    "pavement": ("asphalt_02", 3.0),
    "concrete": ("concrete_floor_02", 2.0),
}

for mat_name, (stem, size_m) in SETS.items():
    mat = bpy.data.materials[mat_name]
    nt = mat.node_tree
    if any(n.bl_idname == "ShaderNodeTexImage" for n in nt.nodes):
        raise SystemExit(f"{mat_name} already has image textures")
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
    mapping.inputs["Scale"].default_value = (1.0 / size_m, 1.0 / size_m, 1.0)
    nt.links.new(geo.outputs["Position"], mapping.inputs["Vector"])
    for i, (tag, non_color) in enumerate(
        (("diff", False), ("rough", True), ("nor", True))
    ):
        img = bpy.data.images.load(str(TEX / f"{stem}_{tag}_2k.jpg"),
                                   check_existing=True)
        if non_color:
            img.colorspace_settings.name = "Non-Color"
        img.pack()
        node = nt.nodes.new("ShaderNodeTexImage")
        node.location = (-450, 300 - 300 * i)
        node.image = img
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
print("SURFACE TEXTURES wired:", ", ".join(SETS))
