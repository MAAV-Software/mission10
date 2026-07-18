"""Give m10-base.blend a physical sky. The world background was a 0.05 gray —
no sky fill at all — so renders were lit by the sun lamp alone: dim frames
with crushed black shadows. A Sky Texture (multiple-scattering) supplies the
daylight ambience; the sun disc stays off because the lamp remains the direct
light (generate.py randomizes it and syncs the sky to the same angles).

    flatpak run --unset-env=PYTHONPATH org.blender.Blender \
        -b <abs>/assets/m10-base.blend --python tools/prep_world_sky.py
"""
import math

import bpy

world = bpy.context.scene.world
nt = world.node_tree
if any(n.bl_idname == "ShaderNodeTexSky" for n in nt.nodes):
    raise SystemExit("world already has a sky texture")

bg = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBackground")
sky = nt.nodes.new("ShaderNodeTexSky")
sky.location = (bg.location.x - 300, bg.location.y)
sky.sky_type = "MULTIPLE_SCATTERING"
sky.sun_disc = False
sky.sun_elevation = math.radians(50.0)
nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = 1.0

bpy.ops.wm.save_mainfile()
print("SKY added:", bpy.data.filepath)
