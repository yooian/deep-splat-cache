"""
Builds the M1a test scene from nothing and saves it as scene.blend.
You do not need to open Blender's UI or model anything.

    blender --background --python make_scene.py

Produces: a red sphere on a grey ground plane, one sun, one camera,
black world. Deliberately boring, because every extra feature is an
extra source of error you'd have to rule out on day one.
"""
import math
import os

import bpy
from mathutils import Vector

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.blend")


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def matte(name, color, roughness):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Metallic"].default_value = 0.0
    # Blender 4.0 renamed this socket; handle both spellings.
    for key in ("Specular IOR Level", "Specular"):
        if key in b.inputs:
            b.inputs[key].default_value = 0.5
            break
    return m


def main():
    clear()
    scene = bpy.context.scene

    # --- geometry -----------------------------------------------------------
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0),
                                         segments=64, ring_count=32)
    sphere = bpy.context.active_object
    sphere.name = "Sphere"
    bpy.ops.object.shade_smooth()
    sphere.data.materials.append(matte("SphereMat", (0.8, 0.25, 0.18), 0.25))

    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0, 0, -1))
    plane = bpy.context.active_object
    plane.name = "Ground"
    plane.data.materials.append(matte("GroundMat", (0.5, 0.5, 0.5), 0.25))

    # --- sun ----------------------------------------------------------------
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 4.0          # W/m^2 irradiance
    sun_data.color = (1.0, 0.95, 0.9)
    sun_data.angle = 0.0           # perfectly directional -> hard shadows
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_mode = "XYZ"
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(30))   # position A

    # --- camera -------------------------------------------------------------
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (0.0, -6.0, 1.6)
    direction = Vector((0, 0, 0.2)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    # --- world: pure black --------------------------------------------------
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0, 0, 0, 1)
    bg.inputs[1].default_value = 0.0
    scene.world = world

    scene.render.engine = "CYCLES"
    bpy.ops.wm.save_as_mainfile(filepath=OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
