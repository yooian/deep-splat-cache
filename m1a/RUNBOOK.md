# M1a Runbook

Every command below runs in your **normal terminal**. You never open Blender's
UI. You never paste anything into Blender's scripting tab.

---

## The thing that was confusing you: there are two Pythons

| | Which Python | How it runs |
|---|---|---|
| `make_scene.py` | Blender's built-in Python | `blender --background --python make_scene.py` |
| `render_aovs.py` | Blender's built-in Python | `blender scene.blend -b -P render_aovs.py -- ...` |
| `selftest.py` | Your system Python | `python3 selftest.py` |
| `relight.py` | Your system Python | `python3 relight.py ...` |
| `diff.py` | Your system Python | `python3 diff.py ...` |

The first two need `bpy`, which only exists inside Blender. The last three need
`numpy` and `OpenEXR`, which you install with pip and which Blender's Python
does not have. They can never be mixed. That is the whole distinction.

Everything is still one terminal — `blender` is just another command you type.

---

## Step 0 — Install (once)

```bash
# Blender: download from blender.org, or on Ubuntu/Debian:
sudo apt install blender
# macOS: brew install --cask blender

blender --version          # expect 4.0 or newer
```

If `blender --version` says "command not found" on macOS, the binary is buried
in the app bundle. Add this to your `~/.zshrc`:

```bash
alias blender="/Applications/Blender.app/Contents/MacOS/Blender"
```

Then the Python side:

```bash
cd /path/to/m1a
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy OpenEXR
```

Activate that venv in every new terminal before running `python3` commands.

---

## Step 1 — Prove the plumbing works before touching Blender

```bash
python3 selftest.py
python3 relight.py test_aovs.exr test_scene.json test_mine.exr
python3 diff.py test_mine.exr test_truth.exr --mask test_aovs.exr
```

`selftest.py` builds a synthetic sphere and plane analytically, writes fake
AOVs, and computes what the answer must be. It never touches Blender.

**Expected: `relative L1  0.00 %`.**

If it isn't zero, stop. Your numpy/OpenEXR install is broken, and no amount of
Blender debugging will help. Nothing downstream is trustworthy until this
prints zero.

---

## Step 2 — Build the test scene

```bash
blender --background --python make_scene.py
```

Writes `scene.blend`: a red sphere on a grey plane, one sun, one camera, black
world. You do not model anything. If you want to look at it, open
`scene.blend` in Blender normally — but you don't need to.

---

## Step 3 — Stage 1: direct light, shadows OFF

```bash
blender scene.blend -b -P render_aovs.py -- --stage 1 --out ./s1 --res 256 --samples 128
python3 relight.py s1/A_aovs.exr s1/scene.json s1/mine.exr
python3 diff.py s1/mine.exr s1/B_truth.exr --mask s1/A_aovs.exr --out s1/diff.exr
```

**Read the `--` carefully.** Everything before it is for Blender; everything
after it is for the script. Omit it and Blender will try to interpret
`--stage` itself and fail confusingly.

Each run renders twice: once with the sun where `make_scene.py` left it
(harvested for AOVs, which carry no lighting information) and once with the
sun rotated to `--sun-b` (the ground truth). Your relighting code only ever
sees the first render.

**Expected — I ran this exact command:**

```
energy ratio       0.9966   (1.0 = perfect)
relative L1         0.49 %  <- headline
  lit pixels     relative L1   0.42 %
```

Under 2% is a pass. This is the result that confirms the core hypothesis:
material attributes attached to points relight correctly under a light the
data has never seen.

*Not matching?* See "When stage 1 fails" below.

---

## Step 4 — Stage 2: shadows ON

```bash
blender scene.blend -b -P render_aovs.py -- --stage 2 --out ./s2 --res 256 --samples 128
python3 relight.py s2/A_aovs.exr s2/scene.json s2/mine.exr
python3 diff.py s2/mine.exr s2/B_truth.exr --mask s2/A_aovs.exr --out s2/diff.exr
```

You change nothing but `--stage` and `--out`. The script flips the shadow
switch for you. **Do not touch the Blender UI to do this** — the whole point
of scripting it is that the two runs differ in exactly one variable.

**Expected — my actual run:**

```
energy ratio       1.2696
relative L1        27.88 %
  unlit pixels   15,552 (38.5%), holding 97.6% of all error
  lit pixels     relative L1   0.70 %
LF fraction         93.3 %  -> smooth residual. Low-order SH should capture it.
```

Read that carefully, because it's the most informative output in the whole
experiment. Lit pixels are still at 0.7% — your shading is fine. But 97.6% of
all error now sits in shadowed pixels, because you have no visibility term and
are lighting things the sphere is blocking. **That error is the visibility
term you'll have to bake.** The 93.3% LF fraction says it's spatially smooth,
which is the good outcome.

---

## Step 5 — Stage 3: full global illumination

```bash
blender scene.blend -b -P render_aovs.py -- --stage 3 --out ./s3 --res 256 --samples 256
python3 relight.py s3/A_aovs.exr s3/scene.json s3/mine.exr
python3 diff.py s3/mine.exr s3/B_truth.exr --mask s3/A_aovs.exr --out s3/diff.exr
```

Use more samples here; GI is noisier and noise pollutes the comparison.

**Expected — my actual run:**

```
relative L1        33.02 %
  lit pixels     relative L1   4.30 %
LF fraction         92.7 %  -> smooth residual. Low-order SH should capture it.
```

Lit-pixel error rose from 0.7% to 4.3%: that's bounce light landing on
surfaces you shaded with direct light only. **This is the go/no-go
observation.** LF fraction above ~70% means a handful of per-splat spherical
harmonic coefficients can represent the missing light cheaply, and the
architecture works. Below ~40% would mean the residual is high-frequency and
the learned term needs to be far more expressive.

One caveat on my numbers: this scene is trivially simple. Re-run stage 3 on
something deliberately nasty — a saturated wall next to a white object, a
mirror, a glass ball — and watch the LF fraction. The gap between easy and
nasty is your honest operating range, and it's a more interesting result than
either number alone.

---

## Changing the light position

`--sun-b` is the rotation of sun B in degrees (Euler XYZ). Default is
`50 0 120`; the scene starts the sun at `50 0 30`, so the default moves it 90°
in azimuth.

```bash
# a much smaller move
blender scene.blend -b -P render_aovs.py -- --stage 2 --out ./s2_small --sun-b 50 0 45

# a low raking angle, which stresses shadows and grazing specular
blender scene.blend -b -P render_aovs.py -- --stage 2 --out ./s2_low --sun-b 15 0 200
```

Sweeping this is a genuinely useful experiment: does error grow smoothly with
angular distance from the original light, or is there a cliff? Plot it.

---

## When stage 1 fails

Stage 1 is your calibration. If it's above ~2%, it is a bug, not a discovery.
In order of how often it happens:

**Huge error concentrated in dark pixels, lit pixels fine.** Shadows are still
on. This bit me while testing: Blender has *two* shadow properties, and
`use_shadow` is EEVEE's while Cycles reads `cycles.cast_shadow`. The script now
sets both. If you modify it, keep both.

**Error everywhere, roughly a constant factor.** Light units. Confirm you're
using a SUN and not a point light, and that `scene.json` picked up the right
strength.

**Error follows surface curvature.** Normals — wrong space, not normalised, or
you're reading a screen-space normal pass.

**Everything looks washed out or crushed.** A view transform leaked into your
EXR. Check raw pixel values with numpy, never the viewport.

**Specular is off but diffuse is fine.** The roughness→alpha remap. Blender
uses α = roughness². Also expect an honest residual above roughness ~0.4:
Cycles uses multi-scatter GGX and `relight.py` is single-scatter. Test at 0.25
first, then raise roughness deliberately and watch the error curve.

**Error only at object silhouettes.** Antialiased edge pixels, where one pixel
straddles two surfaces so `Position` and `Normal` are blended nonsense. Real
problem, not your fault. For now ignore it; later, either render AA-off for
AOVs or reject pixels whose neighbours disagree sharply on depth.

---

## What to record for each run

Before you move on, save: the command line, the four numbers from `diff.py`,
and the diff image. You will want these in three months and you will not
remember them. See the blog section — this is the raw material.
