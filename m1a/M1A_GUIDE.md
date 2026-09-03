# M1a — Does a splat carrying exact PBR attributes relight correctly?

The whole hypothesis of the project, reduced to something you can finish in a
week. No machine learning. No CUDA. No deep EXR.

---

## What a splat actually is

Strip away the training machinery and a 3D Gaussian splat is a small record:

| Field | Meaning | Where yours comes from |
|---|---|---|
| μ (3) | centre in world space | the `Position` AOV, exactly |
| Σ (3×3) | covariance — an ellipsoid, stored as rotation q + scale s | oriented by `Normal`, sized by pixel footprint |
| α | opacity | 1 for opaque surfaces |
| colour | normally spherical harmonics, view-dependent | **replaced by albedo + roughness + normal** |

That last row is the entire idea. Standard 3DGS stores *baked appearance* as SH
coefficients — a memory of how the surface looked, which cannot be relit. You
store *material properties*, which can be shaded under any light.

Rendering is: project each ellipsoid to a 2D ellipse, sort back-to-front,
evaluate a Gaussian falloff, alpha blend. **You do not need any of that for
M1a.** Because you build one splat per pixel and re-render from the same
camera, each splat lands back on the pixel it came from. Shade it and write it
there. That isolates "does relighting work" from "does my rasteriser work" —
two failures you must never debug simultaneously.

---

## Unprojection

Blender's `Position` pass is world-space P per pixel, so for M1a there is no
unprojection at all. Read the pass, that's your splat centres.

You'll need the real thing for your own renderer, so here it is. For a pinhole
camera with focal length in pixels `f = lens_mm * res_x / sensor_mm`:

```
d_cam  = normalize( ((i + 0.5 - w/2) / f,  -(j + 0.5 - h/2) / f,  -1) )
d_world = R_cam @ d_cam                       # camera rotation matrix
P       = cam_pos + d_world * t
```

The trap is `t`. Some renderers store Z as distance along the ray, others as
perpendicular distance to the camera plane. Do not guess. Put a sphere at a
known distance, render, and read the pixel value. When you have both a
`Position` pass and a `Z` pass you can verify your convention against ground
truth in about five minutes — do that once, write down the answer.

---

## Run it

Scene: keep it stupid. A sphere and a ground plane. One **sun** lamp — not a
point light, because sun strength is irradiance in W/m² and needs no distance
or unit conversion. One Principled BSDF per object, metallic 0, **roughness
around 0.25** (see the caveat below). Black world background.

```bash
pip install OpenEXR numpy

# Stage 1 — direct light, shadows OFF
blender scene.blend -b -P render_aovs.py -- --stage 1 --out ./s1
python relight.py s1/A_aovs.exr s1/scene.json s1/mine.exr
python diff.py s1/mine.exr s1/B_truth.exr --mask s1/A_aovs.exr

# Stage 2 — direct light, shadows ON
blender scene.blend -b -P render_aovs.py -- --stage 2 --out ./s2
# ... same two commands against ./s2

# Stage 3 — full GI
blender scene.blend -b -P render_aovs.py -- --stage 3 --out ./s3
```

Each stage renders twice: once with the sun where you left it (harvested for
AOVs, which are lighting-independent) and once with the sun moved (the ground
truth). Your code never sees the second render.

---

## What you are looking for

**This is the part that matters. The summary percentage is not the result —
the structure of the error image is the result.** Open `diff.exr` and look at
it every single time.

### Stage 1 — direct, no shadows

Your analytic shading against Cycles' direct lighting, nothing else in play.

*Expected:* under ~2% mean relative error. Smooth, structureless residual.

*If it's under 2%:* the core hypothesis is confirmed. Exact material
attributes on a point relight correctly under a light the data has never seen.

*If it's wildly off,* it is a bug, not a discovery. In order of likelihood:
normals in the wrong space or not normalised; sun direction sign flipped
(lamps shine down local −Z, and the shading vector L points from the surface
*toward* the light); a view transform baked into your EXR (check raw pixel
values, not the viewport); `DiffCol` containing lighting because you rendered
it wrong; roughness→alpha remap (Blender uses α = roughness²).

*Expect a small honest residual at high roughness.* Cycles uses multi-scatter
GGX; the single-scatter model in `relight.py` loses energy above roughness
~0.4. That's a known model gap, not a failure of the idea. Keep roughness low
for stage 1, then deliberately raise it and watch the error grow — knowing the
shape of that curve is useful.

### Stage 2 — shadows on

*Expected:* stage-1 error plus a sharp, spatially coherent, strongly negative
region wherever the sphere occludes the plane. Your shading has no visibility
term, so you are over-lighting every shadowed pixel.

**That difference image is your visibility target.** Measure it: what fraction
of pixels are affected, and how much energy is in the error? That number sizes
the transfer term you'll have to bake or learn later.

### Stage 3 — full GI

*Expected:* an additional smooth, low-frequency, low-magnitude difference —
colour bleeding from the red sphere onto the plane, light filling the shadow.

**This is the go/no-go observation for the whole project.** If the indirect
residual is smooth and low-frequency, a handful of per-splat spherical
harmonic coefficients will capture it cheaply and the architecture works. If
it's high-frequency, high-contrast, or full of sharp caustic structure, the
learned term has to be far more expressive and the idea gets much harder.

Run stage 3 on two scenes: one with dull, well-separated objects, and one
deliberately nasty (a bright saturated wall next to a white object, a mirror,
a glass ball). The gap between those two tells you the honest operating range.

---

## When stage 3 is done

Add visibility and re-run stage 2. You have the source mesh, so you can ray
trace shadow rays against the actual geometry rather than the splat cloud —
bake a per-splat visibility value per light direction, fit a low-order SH, and
watch the shadow error collapse. If it does, you've built a working
relightable cache with zero neural networks in it, which is a genuinely
interesting result and a far better talk than a half-trained model.

*Only then* is it worth touching `gsplat`, and only for the residual term.
Do that shading in PyTorch first and accept a 100× slowdown. Correctness on
one frame beats speed on none.

---

## Files

- `exrio.py` — EXR reading, handles Blender's multilayer channel naming
- `render_aovs.py` — Blender-side render + scene export
- `relight.py` — build splats, shade with GGX, write image
- `diff.py` — scene-linear relative-error comparison
- `selftest.py` — synthetic sphere+plane, verifies the pipeline round-trips
  to zero before you trust it on real data. Run this first.
