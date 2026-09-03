"""
M1a core: build one splat per pixel from render AOVs, then shade every splat
analytically under a NEW sun light and write the result.

Stage 0 deliberately does NOT rasterise. Because there is one splat per pixel
and we are re-rendering from the same camera, each splat maps back to the pixel
it came from. This isolates the question "does a splat carrying exact PBR
attributes relight correctly?" from the entirely separate question "does my
rasteriser work?". Do not conflate the two on your first run.

Usage:
    python relight.py aovs.exr scene.json out.exr
"""
import json
import sys

import numpy as np

from exrio import read_exr, write_exr, focal_px, normalize


# ----------------------------------------------------------------------------
# 1. Splat construction
# ----------------------------------------------------------------------------

def build_splats(aov, cam):
    """AOV dict -> dict of flat per-splat arrays, one splat per valid pixel."""
    P = aov["Position"][..., :3]
    N = normalize(aov["Normal"][..., :3])
    albedo = aov["DiffCol"][..., :3]

    h, w = P.shape[:2]

    # Roughness: from a custom AOV if present, else a constant you authored.
    if "roughness" in aov:
        rough = aov["roughness"][..., 0]
    else:
        rough = np.full((h, w), cam.get("default_roughness", 0.3), np.float32)

    # Background pixels have no surface. Blender leaves Position at 0 there.
    # Use the alpha channel of Combined if available, else a position test.
    if "Combined" in aov and aov["Combined"].shape[-1] == 4:
        valid = aov["Combined"][..., 3] > 0.5
    else:
        valid = np.linalg.norm(P, axis=-1) > 1e-6

    cam_pos = np.array(cam["position"], np.float32)
    forward = normalize(np.array(cam["forward"], np.float32))

    # View vector: surface -> camera.
    V = normalize(cam_pos[None, None, :] - P)

    # Splat footprint. Unused in stage 0, needed the moment you rasterise.
    # World size of one pixel at this depth, widened at grazing angles.
    f = focal_px(cam["lens_mm"], cam["sensor_mm"], w, h, cam.get("sensor_fit", "AUTO"))
    depth = np.abs(np.einsum("ijk,k->ij", P - cam_pos[None, None, :], forward))
    radius = (depth / f) / np.clip(np.abs(np.sum(N * V, -1)), 0.2, 1.0)

    m = valid
    return {
        "P": P[m],
        "N": N[m],
        "V": V[m],
        "albedo": albedo[m],
        "roughness": np.clip(rough[m], 0.01, 1.0),
        "radius": radius[m],
        "pixel_index": np.flatnonzero(m.ravel()),
        "shape": (h, w),
    }


# ----------------------------------------------------------------------------
# 2. Shading: Cook-Torrance GGX, matched to Blender's Principled BSDF
# ----------------------------------------------------------------------------

def shade_sun(splats, sun_dir, sun_color, sun_strength, metallic=0.0, specular=0.5):
    """
    Direct lighting from a single sun (distant, no falloff), no visibility.

    Blender's sun 'Strength' is irradiance in W/m^2 on a surface facing the
    light, so outgoing radiance is simply BRDF * strength * max(N.L, 0).
    Using a sun rather than a point light removes an entire class of
    unit-conversion bugs. Do stage 1 with a sun.
    """
    N, V = splats["N"], splats["V"]
    L = normalize(-np.asarray(sun_dir, np.float32))[None, :]   # surface -> light
    H = normalize(L + V)

    NoL = np.clip(np.sum(N * L, -1, keepdims=True), 0.0, 1.0)
    NoV = np.clip(np.sum(N * V, -1, keepdims=True), 1e-4, 1.0)
    NoH = np.clip(np.sum(N * H, -1, keepdims=True), 0.0, 1.0)
    VoH = np.clip(np.sum(V * H, -1, keepdims=True), 0.0, 1.0)

    base = splats["albedo"]
    # Blender/Disney convention: GGX alpha = roughness^2
    a = (splats["roughness"] ** 2)[:, None]
    a2 = a * a

    # GGX normal distribution
    denom = NoH * NoH * (a2 - 1.0) + 1.0
    D = a2 / np.maximum(np.pi * denom * denom, 1e-9)

    # Height-correlated Smith visibility (already folded in the 1/(4 NoL NoV))
    lv = NoL * np.sqrt(NoV * NoV * (1 - a2) + a2)
    ll = NoV * np.sqrt(NoL * NoL * (1 - a2) + a2)
    Vis = 0.5 / np.maximum(lv + ll, 1e-9)

    # Fresnel. Dielectric F0 = 0.08 * specular (0.04 at Blender's default 0.5)
    F0 = (1.0 - metallic) * (0.08 * specular) + metallic * base
    F = F0 + (1.0 - F0) * np.power(1.0 - VoH, 5.0)

    diffuse = (1.0 - metallic) * base / np.pi
    spec = D * Vis * F

    irradiance = np.asarray(sun_color, np.float32)[None, :] * sun_strength
    return (diffuse + spec) * irradiance * NoL


# ----------------------------------------------------------------------------
# 3. Scatter back to an image (stage 0: no rasterisation)
# ----------------------------------------------------------------------------

def to_image(splats, rgb):
    h, w = splats["shape"]
    img = np.zeros((h * w, 3), np.float32)
    img[splats["pixel_index"]] = rgb
    return img.reshape(h, w, 3)


def main():
    aov_path, scene_path, out_path = sys.argv[1:4]
    aov = read_exr(aov_path)
    scene = json.load(open(scene_path))

    print("passes found:", sorted(aov.keys()))
    splats = build_splats(aov, scene["camera"])
    print(f"built {len(splats['P']):,} splats "
          f"(median radius {np.median(splats['radius']):.5f} world units)")

    sun = scene["new_sun"]
    rgb = shade_sun(splats, sun["direction"], sun["color"], sun["strength"],
                    metallic=scene.get("metallic", 0.0),
                    specular=scene.get("specular", 0.5))

    write_exr(out_path, to_image(splats, rgb))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
