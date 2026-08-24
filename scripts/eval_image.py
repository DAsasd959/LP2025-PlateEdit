#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-fidelity metrics: FID, full-image LPIPS, and masked-region LPIPS.

    python scripts/eval_image.py \
        --gen_dir outputs/lp2025_test \
        --real_dir data/test/filtered_plate \
        --mask_dir data/test/partial_masks

Two things about these numbers are easy to get wrong, so they are enforced here:

1. FID is not comparable across sample sizes. It falls as N grows — on one
   fixed image set we measured 7.28 at N=200 against 3.85 at N=1000. Only
   compare FID values computed at the same N, and report N alongside.

2. Region LPIPS is a quality measure only when the target text equals the
   original text (reconstruction). When characters were deliberately replaced,
   the edited region is supposed to differ from the source, so a higher value
   carries no quality information. Pass --edit to have that stated in the
   output rather than silently reported as if it were a quality score.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

SIZE = 512


def to_tensor(pil):
    a = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1


def mask_bbox(mask_pil):
    m = np.array(mask_pil.convert("L").resize((SIZE, SIZE)))
    ys, xs = np.where(m > 127)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if x1 - x0 < 8:
        x1 = min(SIZE, x0 + 8)
    if y1 - y0 < 8:
        y1 = min(SIZE, y0 + 8)
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--mask_dir", default=None,
                    help="omit to skip region LPIPS")
    ap.add_argument("--edit", action="store_true",
                    help="targets differ from the source text")
    ap.add_argument("--no_fid", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import lpips
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    net = lpips.LPIPS(net="alex").to(dev).eval()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(a.gen_dir)
                   if f.endswith(".png") and not f.endswith("_compare.png"))
    # FID scans a directory non-recursively for images, so it would also pick up
    # the *_compare.png strips sitting next to the outputs. Both sides therefore
    # go into dedicated directories holding exactly the scored pairs.
    real_aligned = os.path.join(a.gen_dir, "_fid_real")
    gen_aligned = os.path.join(a.gen_dir, "_fid_gen")
    if not a.no_fid:
        for d in (real_aligned, gen_aligned):
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                os.unlink(os.path.join(d, f))

    full, region, missing = [], [], 0
    with torch.no_grad():
        for stem in stems:
            rp = os.path.join(a.real_dir, stem + ".jpg")
            if not os.path.exists(rp):
                rp = os.path.join(a.real_dir, stem + ".png")
            gp = os.path.join(a.gen_dir, stem + ".png")
            if not os.path.exists(rp):
                missing += 1
                continue
            real = Image.open(rp).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
            gen = Image.open(gp).convert("RGB")
            if gen.size != (SIZE, SIZE):
                gen = gen.resize((SIZE, SIZE), Image.BICUBIC)
            if not a.no_fid:
                real.save(os.path.join(real_aligned, stem + ".png"))
                gen.save(os.path.join(gen_aligned, stem + ".png"))
            ta, tb = to_tensor(real).to(dev), to_tensor(gen).to(dev)
            full.append(float(net(ta, tb)))

            if a.mask_dir:
                mp = os.path.join(a.mask_dir, stem + ".png")
                if os.path.exists(mp):
                    bb = mask_bbox(Image.open(mp))
                    if bb:
                        x0, y0, x1, y1 = bb
                        ra, rb = ta[:, :, y0:y1, x0:x1], tb[:, :, y0:y1, x0:x1]
                        if min(ra.shape[-2:]) < 64:      # LPIPS backbone needs size
                            ra = torch.nn.functional.interpolate(
                                ra, (64, 64), mode="bilinear", align_corners=False)
                            rb = torch.nn.functional.interpolate(
                                rb, (64, 64), mode="bilinear", align_corners=False)
                        region.append(float(net(ra, rb)))

    n = len(full)
    if n == 0:
        raise SystemExit("nothing scored — check --gen_dir and --real_dir")
    se2 = lambda v: 2 * np.std(v, ddof=1) / np.sqrt(len(v))
    print(f"n = {n}" + (f"   (skipped {missing} without a source image)" if missing else ""))
    print(f"Full LPIPS   = {np.mean(full):.4f} +/- {se2(full):.4f} (2SE)")
    if region:
        note = "  [not a quality measure for edits]" if a.edit else ""
        print(f"Region LPIPS = {np.mean(region):.4f} +/- {se2(region):.4f} (2SE){note}")

    if not a.no_fid:
        del net
        torch.cuda.empty_cache()
        from pytorch_fid import fid_score
        v = fid_score.calculate_fid_given_paths([real_aligned, gen_aligned], 50, dev, 2048)
        print(f"FID          = {v:.4f}   (N={n}; do not compare across N)")


if __name__ == "__main__":
    main()
