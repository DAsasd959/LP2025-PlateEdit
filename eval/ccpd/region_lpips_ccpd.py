#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用論文的定義重算 CCPD 的 Region LPIPS
=====================================
先前我用「裁切到編輯格再比對」，論文（test_original/eval_lpips.py）用的是
「兩張圖都乘上遮罩，再對整張 512x512 算 LPIPS」。遮罩外兩者都是全黑、完全相同，
所以分數被大幅拉低 —— 同一批 LP 影像上兩種定義差 3.5 倍（0.2205 vs 0.0648）。
不統一定義，CCPD 與 LP 的數字並排就是錯的。

遮罩必須重建成生成當下那一份：逐格聯集 + inset=2，由 cells.txt 記錄的格子集合決定。
"""
import argparse, importlib.util, json, os, sys
import numpy as np, torch
from PIL import Image
import cv2
from pathlib import Path

HERE = str(Path(__file__).resolve().parents[2])          # repository root
sys.path.insert(0, HERE)
SC = os.path.dirname(os.path.abspath(__file__))
SIZE = 512

_s = importlib.util.spec_from_file_location(
    "cce", os.path.join(HERE, "eval", "ccpd", "ccpd_cn_edit.py"))
CCE = importlib.util.module_from_spec(_s); sys.modules["cce"] = CCE; _s.loader.exec_module(CCE)
_c = importlib.util.spec_from_file_location(
    "cse", os.path.join(HERE, "eval", "ccpd", "cellset_edit.py"))
CSE = importlib.util.module_from_spec(_c); sys.modules["cse"] = CSE; _c.loader.exec_module(CSE)


def to_tensor(pil):
    a = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1


def cells_of(d):
    out = {}
    for line in open(os.path.join(d, "cells.txt")):
        stem, cells, tgt = line.rstrip("\n").split("\t")
        out[stem] = [int(c) for c in cells.split(",")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--plate_dir", required=True)
    ap.add_argument("--inset", type=int, default=2)
    a = ap.parse_args()

    import lpips
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lpips.LPIPS(net="alex").to(dev).eval()

    rec = cells_of(a.gen_dir)
    crop_v, mask_v, n_miss = [], [], 0
    with torch.no_grad():
        for stem, cs in rec.items():
            rp = os.path.join(a.plate_dir, stem + ".jpg")
            gp = os.path.join(a.gen_dir, "plates", stem + ".png")
            if not (os.path.exists(rp) and os.path.exists(gp)):
                n_miss += 1; continue
            real = Image.open(rp).convert("RGB")
            W, H = real.size
            ta = to_tensor(real.resize((SIZE, SIZE), Image.BILINEAR)).to(dev)
            gen = Image.open(gp).convert("RGB")
            if gen.size != (SIZE, SIZE):
                gen = gen.resize((SIZE, SIZE), Image.BILINEAR)
            tb = to_tensor(gen).to(dev)

            quad, _, _ = CCE.parse_ccpd_filename(stem)
            m = CSE.build_mask(quad, cs, W, H, a.inset)          # 生成當下那份遮罩
            m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            mt = torch.from_numpy(m.astype(np.float32) / 255.0)[None, None].to(dev)
            mt = mt.repeat(1, 3, 1, 1)
            if float(mt.sum()) == 0:
                continue
            mask_v.append(float(net(ta * mt, tb * mt)))          # 論文定義

            ys, xs = np.where(m > 127)                            # 我先前的定義，供對照
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            ra, rb = ta[:, :, y0:y1, x0:x1], tb[:, :, y0:y1, x0:x1]
            if min(ra.shape[-2:]) < 64:
                ra = torch.nn.functional.interpolate(ra, (64, 64), mode="bilinear", align_corners=False)
                rb = torch.nn.functional.interpolate(rb, (64, 64), mode="bilinear", align_corners=False)
            crop_v.append(float(net(ra, rb)))

    se2 = lambda v: 2 * np.std(v, ddof=1) / np.sqrt(len(v))
    print(f"{os.path.basename(a.gen_dir)}   n = {len(mask_v):,}"
          + (f"   (缺檔 {n_miss})" if n_miss else ""))
    print(f"  Region LPIPS（論文定義：遮罩相乘）= {np.mean(mask_v):.4f} +/- {se2(mask_v):.4f}")
    print(f"  Region LPIPS（先前：裁切外接框）   = {np.mean(crop_v):.4f} +/- {se2(crop_v):.4f}")


if __name__ == "__main__":
    main()
