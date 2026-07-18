"""Prep a clean mine template for the datagen pipeline.

From pfm1-mine-grass.blend: keep only the IARC_PFM-1_mine template, zero the
baked junk rotation (settle tilt + 199 deg yaw), apply the ~0.001 non-uniform
scale, set the origin to the geometric bbox center (so the tag-down pi-flip
about local X is symmetric and never sinks the mesh), rename the material to
mine_body (hue jitter excludes materials named *tag*), save as m10-mine.blend.
"""
import bpy

SRC = "/home/muku/Projects/MAAV/mission10/models/yolo/assets/pfm1-mine-grass.blend"
DST = "/home/muku/Projects/MAAV/mission10/models/yolo/assets/m10-mine.blend"

bpy.ops.wm.open_mainfile(filepath=SRC)
keep = bpy.data.objects["IARC_PFM-1_mine"]
for o in list(bpy.data.objects):
    if o is not keep:
        bpy.data.objects.remove(o, do_unlink=True)

keep.hide_render = False
keep.rotation_euler = (0.0, 0.0, 0.0)  # back to the authored frame: long axis +X
keep.location = (0.0, 0.0, 0.0)
bpy.ops.object.select_all(action="DESELECT")
keep.select_set(True)
bpy.context.view_layer.objects.active = keep
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
keep.location = (0.0, 0.0, 0.0)

for slot in keep.material_slots:
    if slot.material:
        slot.material.name = "mine_body"

# orphan everything the deleted showcase objects referenced
bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=False, do_recursive=True)

d = tuple(round(v, 4) for v in keep.dimensions)
print(f"TEMPLATE dims={d} scale={tuple(keep.scale)} rot={tuple(keep.rotation_euler)}")
assert d == (0.12, 0.061, 0.02), d

bpy.ops.wm.save_as_mainfile(filepath=DST)
print("SAVED", DST)
