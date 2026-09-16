#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score a CCPD run's Chinese and alphanumeric cells separately, from one set of
outputs. No regeneration needed.

A mixed span like cells 0,5,6 edits the province character and two alphanumerics
at once, and a single Region LPIPS over that mask averages the two. Splitting the
mask -- cell 0 alone, then cells 1..6 alone -- scores each part of the same
generated image on its own, so "how well does it do Chinese" and "how well does it
do alphanumerics" become separate numbers from one run.

Only Region LPIPS decomposes this way. FID and full-image LPIPS score the whole
frame and cannot be attributed to part of a mask.

    python eval/ccpd/region_split_ccpd.py \
        --gen_dir <output with cells.txt> --plate_dir <native-resolution plates>

Masks are rebuilt exactly as they were at generation time: per-cell union, inset 2,
from the cell set recorded in cells.txt.
"""
import argparse, importlib.util, os, sys
import numpy as np, torch
from PIL import Image
import cv2
from pathlib import Path

HERE = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, HERE)

_s = importlib.util.spec_from_file_location(
    "cce", os.path.join(HERE, "eval", "ccpd", "ccpd_cn_edit.py"))
CCE = importlib.util.module_from_spec(_s); sys.modules["cce"] = CCE; _s.loader.exec_module(CCE)
_c = importlib.util.spec_from_file_location(
    "cse", os.path.join(HERE, "eval", "ccpd", "cellset_edit.py"))
CSE = importlib.util.module_from_spec(_c); sys.modules["cse"] = CSE; _c.loader.exec_module(CSE)

SIZE = 512


def to_tensor(pil):
    a = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1


def cells_of(d):
    out = {}
    for line in open(os.path.join(d, "cells.txt"), encoding="utf-8"):
        stem, cells, tgt = line.rstrip("\n").split("\t")
        out[stem] = [int(c) for c in cells.split(",")]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--plate_dir", required=True)
    ap.add_argument("--inset", type=int, default=2)
    a = ap.parse_args()

    import lpips
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lpips.LPIPS(net="alex").to(dev).eval()

    rec = cells_of(a.gen_dir)
    both, cn, al = [], [], []
    n_miss = 0
    with torch.no_grad():
        for stem, cs in rec.items():
            rp = os.path.join(a.plate_dir, stem + ".jpg")
            gp = os.path.join(a.gen_dir, "plates", stem + ".png")
            if not (os.path.exists(rp) and os.path.exists(gp)):
                n_miss += 1
                continue
            real = Image.open(rp).convert("RGB")
            W, H = real.size
            ta = to_tensor(real.resize((SIZE, SIZE), Image.BILINEAR)).to(dev)
            gen = Image.open(gp).convert("RGB")
            if gen.size != (SIZE, SIZE):
                gen = gen.resize((SIZE, SIZE), Image.BILINEAR)
            tb = to_tensor(gen).to(dev)
            quad, _, _ = CCE.parse_ccpd_filename(stem)

            def score(subset):
                if not subset:
                    return None
                m = CSE.build_mask(quad, subset, W, H, a.inset)
                m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
                mt = torch.from_numpy(m.astype(np.float32) / 255.0)[None, None].to(dev)
                mt = mt.repeat(1, 3, 1, 1)
                if float(mt.sum()) == 0:
                    return None
                return float(net(ta * mt, tb * mt))

            v = score(cs)
            if v is not None:
                both.append(v)
            # cell 0 is the province character; 1..6 are alphanumeric
            v = score([c for c in cs if c == 0])
            if v is not None:
                cn.append(v)
            v = score([c for c in cs if c >= 1])
            if v is not None:
                al.append(v)

    se2 = lambda v: 2 * np.std(v, ddof=1) / np.sqrt(len(v))
    name = os.path.basename(a.gen_dir.rstrip("/"))
    print(f"{name}   n = {len(both):,}" + (f"   (缺檔 {n_miss})" if n_miss else ""))
    for label, v in (("whole mask        ", both),
                     ("Chinese cell only ", cn),
                     ("alphanumeric only ", al)):
        if v:
            print(f"  Region LPIPS  {label} = {np.mean(v):.4f} +/- {se2(v):.4f}   (n={len(v)})")
        else:
            print(f"  Region LPIPS  {label} = —   (no samples)")


if __name__ == "__main__":
    main()
