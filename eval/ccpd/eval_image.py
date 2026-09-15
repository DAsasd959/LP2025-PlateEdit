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

Region LPIPS follows the definition used for the published numbers: both images
are multiplied by the mask and LPIPS is taken over the full 512x512 result, so
everything outside the mask is identical black in both inputs. This is not the
same as cropping to the mask and comparing the crops — cropping gave 0.22 on the
released outputs where the masked-image definition gives 0.062, because a crop
removes the large identical region that otherwise dominates the score. Two papers
reporting "region LPIPS" can therefore differ by 3x on identical images; the
number is only meaningful alongside its definition.
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


def load_mask(mask_pil, dev):
    m = np.array(mask_pil.convert("L").resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    return torch.from_numpy(m)[None, None].to(dev).repeat(1, 3, 1, 1)


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
                    m = load_mask(Image.open(mp), dev)
                    if float(m.sum()) > 0:
                        region.append(float(net(ta * m, tb * m)))

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
