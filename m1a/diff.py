"""
Compare a relit result against a ground-truth re-render, in scene-linear.

Per-pixel relative error explodes wherever ground truth is near zero (shadow
interiors, terminators), so it is useless as a headline number. The metrics
here are scale-robust:

  ENERGY RATIO   total light you produced vs total light Cycles produced
  RELATIVE L1    sum|mine - truth| / sum(truth) -- the number to report
  LF FRACTION    how much of the residual survives an 8x blur. This is the
                 stage-3 go/no-go: a high number means the missing light is
                 smooth and a few per-splat SH coefficients will capture it.

Usage:
    python diff2.py mine.exr truth.exr --mask aovs.exr --out diff.exr
"""
import argparse

import numpy as np

from exrio import read_exr, write_exr

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def load_rgb(path):
    d = read_exr(path)
    for key in ("Combined", "RGB", "RGBA"):
        if key in d:
            return d[key][..., :3]
    return next(iter(d.values()))[..., :3]


def lowfreq_fraction(residual, mask, k=8):
    """Fraction of residual energy retained after a k x k box blur."""
    r = residual * mask[..., None]
    h, w = r.shape[:2]
    hh, ww = (h // k) * k, (w // k) * k
    r = r[:hh, :ww]
    blocks = r.reshape(hh // k, k, ww // k, k, 3).mean(axis=(1, 3))
    smooth = np.repeat(np.repeat(blocks, k, 0), k, 1)
    total = float((r ** 2).sum())
    return float((smooth ** 2).sum()) / total if total > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mine")
    ap.add_argument("truth")
    ap.add_argument("--mask", help="AOV exr, to exclude background pixels")
    ap.add_argument("--out", default="diff.exr")
    args = ap.parse_args()

    a, b = load_rgb(args.mine), load_rgb(args.truth)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"

    if args.mask:
        aov = read_exr(args.mask)
        if "Combined" in aov and aov["Combined"].shape[-1] == 4:
            m = aov["Combined"][..., 3] > 0.5
        else:
            m = np.linalg.norm(aov["Position"][..., :3], axis=-1) > 1e-6
    else:
        m = np.ones(a.shape[:2], bool)

    am, bm = a[m], b[m]
    res = am - bm

    sum_t = float(np.abs(bm).sum())
    print(f"valid pixels     {m.sum():,}")
    print(f"energy ratio     {am.sum() / max(sum_t, 1e-9):8.4f}   (1.0 = perfect)")
    print(f"relative L1      {np.abs(res).sum() / max(sum_t, 1e-9) * 100:7.2f} %  <- headline")

    # Split by whether the ground truth is essentially unlit. In shadowed
    # regions relative error is meaningless, so report absolute energy there.
    lum = bm @ LUMA
    unlit = lum < 0.02 * max(lum.max(), 1e-9)
    if unlit.any():
        share = np.abs(res[unlit]).sum() / max(np.abs(res).sum(), 1e-9)
        print(f"  unlit pixels   {unlit.sum():,} ({unlit.mean() * 100:.1f}%), "
              f"holding {share * 100:.1f}% of all error")
    if (~unlit).any():
        lit_rel = np.abs(res[~unlit]).sum() / max(np.abs(bm[~unlit]).sum(), 1e-9)
        print(f"  lit pixels     relative L1 {lit_rel * 100:6.2f} %")

    full = np.zeros_like(a)
    full[m] = res
    lf = lowfreq_fraction(full, m, k=8)
    rel_l1 = np.abs(res).sum() / max(sum_t, 1e-9)
    print(f"LF fraction      {lf * 100:7.1f} %  of residual energy survives an 8x blur")
    if rel_l1 < 0.02:
        print("                 -> ignore this; the residual is just sampling noise,")
        print("                    which is high-frequency by nature. Only read the")
        print("                    LF fraction once the residual is substantial.")
    elif lf > 0.7:
        print("                 -> smooth residual. Low-order SH should capture it.")
    elif lf > 0.4:
        print("                 -> partly smooth. SH will help but not suffice alone.")
    else:
        print("                 -> high-frequency residual. Hard for a cheap term.")

    write_exr(args.out, full)
    print(f"\nwrote signed difference to {args.out} -- open it and LOOK at it.")


if __name__ == "__main__":
    main()
