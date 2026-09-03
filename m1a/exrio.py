"""
Minimal scene-linear EXR I/O for the M1a experiment.

Handles Blender multilayer EXRs, where channels arrive named like
'ViewLayer.Combined.R' or 'ViewLayer.Position.X'. The OpenEXR reader
auto-groups RGB/RGBA triples but leaves XYZ triples split, so we
normalise both cases here.
"""
import numpy as np
import OpenEXR


def read_exr(path):
    """Return {pass_name: HxWxC float32 array}. Layer prefixes are stripped."""
    out = {}
    with OpenEXR.File(path) as f:
        raw = {k: v.pixels for k, v in f.channels().items()}

    # Group any channels the reader left split (X/Y/Z, or bare R/G/B).
    scalar_parts = {}
    for key, arr in raw.items():
        parts = key.split(".")
        if len(parts) > 1 and parts[-1] in ("X", "Y", "Z", "R", "G", "B", "A"):
            base, comp = ".".join(parts[:-1]), parts[-1]
            scalar_parts.setdefault(base, {})[comp] = arr
        else:
            out[key] = arr if arr.ndim == 3 else arr[..., None]

    for base, comps in scalar_parts.items():
        for order in (("X", "Y", "Z"), ("R", "G", "B", "A"), ("R", "G", "B")):
            if all(c in comps for c in order):
                out[base] = np.stack([comps[c] for c in order], axis=-1)
                break
        else:  # single stray channel
            out[base] = np.stack(list(comps.values()), axis=-1)

    # Strip the view-layer prefix: 'ViewLayer.Position' -> 'Position'
    stripped = {}
    for key, arr in out.items():
        stripped[key.split(".")[-1] if "." in key else key] = arr.astype(np.float32)
    return stripped


def write_exr(path, rgb):
    """Write an HxWx3 scene-linear float32 array."""
    rgb = np.ascontiguousarray(rgb.astype(np.float32))
    channels = {"RGB": rgb}
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION}, channels).write(path)


def focal_px(lens_mm, sensor_mm, res_x, res_y, sensor_fit="AUTO"):
    """Blender camera -> focal length in pixels (square pixels assumed)."""
    if sensor_fit == "VERTICAL" or (sensor_fit == "AUTO" and res_y > res_x):
        return lens_mm * res_y / sensor_mm
    return lens_mm * res_x / sensor_mm


def normalize(v, eps=1e-12):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)
