"""
Blender script for M1a. Renders your scene twice:

    A) with the sun in its ORIGINAL position  -> AOVs (P, N, albedo, roughness)
    B) with the sun MOVED                     -> ground truth to beat

and writes a scene.json describing camera + sun B for relight.py.

Run headless:
    blender scene.blend --background --python render_aovs.py -- --stage 1 --out ./out

Stages (run them in this order, do not skip):
    1  direct light only, shadows OFF  -> should match almost exactly
    2  direct light only, shadows ON   -> difference is the visibility term
    3  full GI, shadows ON             -> extra difference is indirect light
"""
import argparse
import json
import os
import sys

import bpy
from mathutils import Vector


def get_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--out", default="./out")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--samples", type=int, default=512)
    # Sun B: where you MOVE the light to. Euler XYZ in degrees.
    p.add_argument("--sun-b", type=float, nargs=3, default=[50.0, 0.0, 120.0])
    return p.parse_args(argv)


def setup(scene, args):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False          # denoise noise != render noise
    scene.cycles.max_bounces = 0 if args.stage < 3 else 8
    scene.cycles.diffuse_bounces = 0 if args.stage < 3 else 4
    scene.cycles.glossy_bounces = 0 if args.stage < 3 else 4

    scene.render.resolution_x = scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True        # gives us a clean alpha mask

    ifs = scene.render.image_settings
    ifs.file_format = "OPEN_EXR_MULTILAYER"
    ifs.color_depth = "32"
    ifs.color_mode = "RGBA"
    ifs.exr_codec = "ZIP"

    vl = bpy.context.view_layer
    vl.use_pass_combined = True
    vl.use_pass_position = True                 # world-space P. No unprojection.
    vl.use_pass_normal = True                   # world-space N
    vl.use_pass_diffuse_color = True            # albedo

    # Black world, or ambient light contaminates the comparison.
    world = scene.world
    if world and world.use_nodes:
        for n in world.node_tree.nodes:
            if n.type == "BACKGROUND":
                n.inputs[0].default_value = (0, 0, 0, 1)
                n.inputs[1].default_value = 0.0


def add_roughness_aov():
    """Optional. For stage 1 just use one constant roughness and skip this."""
    try:
        vl = bpy.context.view_layer
        if "roughness" not in [a.name for a in vl.aovs]:
            aov = vl.aovs.add()
            aov.name, aov.type = "roughness", "VALUE"
        for mat in bpy.data.materials:
            if not mat.use_nodes:
                continue
            nt = mat.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if not bsdf or any(n.type == "OUTPUT_AOV" for n in nt.nodes):
                continue
            out = nt.nodes.new("ShaderNodeOutputAOV")
            out.name = "roughness"
            rin = bsdf.inputs["Roughness"]
            if rin.is_linked:
                nt.links.new(rin.links[0].from_socket, out.inputs["Value"])
            else:
                v = nt.nodes.new("ShaderNodeValue")
                v.outputs[0].default_value = rin.default_value
                nt.links.new(v.outputs[0], out.inputs["Value"])
        return True
    except Exception as e:
        print(f"[warn] roughness AOV skipped: {e}")
        return False


def only_sun():
    suns = [o for o in bpy.data.objects if o.type == "LIGHT" and o.data.type == "SUN"]
    if not suns:
        raise SystemExit("Add exactly one SUN light. Point lights need unit "
                         "conversion you do not want to debug on day one.")
    for o in bpy.data.objects:
        if o.type == "LIGHT" and o not in suns[:1]:
            o.hide_render = True
    return suns[0]


def sun_direction(sun):
    """Direction the light TRAVELS (lamps shine down their local -Z)."""
    return (sun.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    scene = bpy.context.scene
    setup(scene, args)
    has_rough = add_roughness_aov()

    sun = only_sun()
    sun.data.angle = 0.0                        # perfectly directional
    # BOTH must be set. `use_shadow` is EEVEE's switch; Cycles reads
    # `cycles.cast_shadow`. Setting only the first silently leaves shadows
    # ON in Cycles, which makes stage 1 look like a catastrophic failure.
    shadows = (args.stage >= 2)
    for obj, attr in ((sun.data, "use_shadow"), (sun.data.cycles, "cast_shadow")):
        try:
            setattr(obj, attr, shadows)
        except (AttributeError, TypeError):
            pass

    cam = scene.camera
    if cam is None:
        raise SystemExit("Scene has no active camera.")

    # --- Render A: original sun, harvested for AOVs ---
    scene.render.filepath = os.path.join(args.out, "A_aovs.exr")
    bpy.ops.render.render(write_still=True)

    # --- Render B: sun moved. This is the ground truth. ---
    sun.rotation_mode = "XYZ"
    sun.rotation_euler = [__import__("math").radians(d) for d in args.sun_b]
    bpy.context.view_layer.update()
    scene.render.filepath = os.path.join(args.out, "B_truth.exr")
    bpy.ops.render.render(write_still=True)

    d = sun_direction(sun)
    cd = cam.data
    scene_json = {
        "stage": args.stage,
        "camera": {
            "position": list(cam.matrix_world.translation),
            "forward": list((cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()),
            "lens_mm": cd.lens,
            "sensor_mm": cd.sensor_width,
            "sensor_fit": cd.sensor_fit,
            # Only used if the roughness AOV is unavailable:
            "default_roughness": 0.25,
        },
        "new_sun": {
            "direction": [d.x, d.y, d.z],
            "color": list(sun.data.color),
            "strength": sun.data.energy,
        },
        "metallic": 0.0,
        "specular": 0.5,
        "roughness_aov": has_rough,
    }
    with open(os.path.join(args.out, "scene.json"), "w") as f:
        json.dump(scene_json, f, indent=2)

    print(f"\nstage {args.stage}: wrote A_aovs.exr, B_truth.exr, scene.json to {args.out}")
    print("next:  python relight.py A_aovs.exr scene.json mine.exr")
    print("then:  python diff.py mine.exr B_truth.exr --mask A_aovs.exr")


if __name__ == "__main__":
    main()
