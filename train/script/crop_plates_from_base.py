#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 ccpd_base 的完整場景圖裁出車牌（新檔，只讀來源、只寫新目錄）
================================================================
為什麼需要這一步：
    train_8000/plates、test_1000/plates 存的是「已裁切的車牌」（如 240x91），
    而 ccpd_base 是完整場景圖（720x1160）。
    CnPlateDataset.parse（cn_preprocess.py:114-115）會把 quad 減掉最小值 ——
    那是為裁切圖設計的；直接餵完整場景圖會把遮罩平移到左上角（實測落在
    16x29 的角落區塊），整個訓練目標會錯掉。

裁切規則（比對 train_8000/plates 五筆驗證通過）：
    x0,y0,x1,y1 = 檔名第 4 欄 quad 的 bbox
    img.crop((x0, y0, x1, y1))      # PIL 右下不含 -> 尺寸剛好等於 train_8000

用法：
  python crop_plates_from_base.py --list stage2_base_cap100.txt \
      --out_root data/ccpd/crops --verify 10
"""
import argparse
import os

import numpy as np
from PIL import Image
from pathlib import Path

BASE = os.environ.get("CCPD_BASE", "data/ccpd/ccpd_base")   # full-scene CCPD images


def crop_one(fname, out_path):
    q = np.array([[int(v) for v in p.split("&")] for p in fname.split("-")[3].split("_")],
                 dtype=np.float32)
    x0, y0 = int(q[:, 0].min()), int(q[:, 1].min())
    x1, y1 = int(q[:, 0].max()), int(q[:, 1].max())
    Image.open(os.path.join(BASE, fname)).convert("RGB").crop((x0, y0, x1, y1)) \
        .save(out_path, quality=95)
    return x1 - x0, y1 - y0


def verify(out_root, n):
    """把 CnPlateDataset 實際算出的遮罩與 glyph 疊回裁切圖，確認落在車牌上。
    紅 = 遮罩，綠 = glyph。上一版的 bug 就是靠這張圖才會被抓到。"""
    import importlib.util
    import sys
    sp = importlib.util.spec_from_file_location(
        "cnp", str(Path(__file__).resolve().parent / "cn_preprocess.py"))
    CNP = importlib.util.module_from_spec(sp)
    sys.modules["cnp"] = CNP
    sp.loader.exec_module(CNP)

    ds = CNP.CnPlateDataset(os.path.join(out_root, "plates"),
                            province_ratio=1.0, max_k=1, limit=None)
    tiles, stats = [], []
    for i in range(n):
        d = ds[i]
        img = (d["image"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        msk = (d["hint"][0].numpy() * 255).astype(np.uint8)
        gly = (d["condition"][0].numpy() * 255).astype(np.uint8)
        vis = img.copy()
        vis[..., 0] = np.maximum(vis[..., 0], msk // 2)
        vis[..., 1] = np.maximum(vis[..., 1], gly)
        tiles.append(Image.fromarray(vis).resize((300, 300)))
        ys, xs = np.where(msk > 127)
        H, W = msk.shape
        stats.append((d["description"], msk.mean() / 255,
                      (xs.min() / W, xs.max() / W, ys.min() / H, ys.max() / H)))
    sheet = Image.new("RGB", (300 * len(tiles), 300), "white")
    for j, t in enumerate(tiles):
        sheet.paste(t, (j * 300, 0))
    p = os.path.join(out_root, "verify_overlay.png")
    sheet.save(p)
    print(f"\n驗證疊圖（紅=遮罩 綠=glyph）-> {p}")
    print(f"{'目標':<6}{'遮罩面積':>9}{'x 範圍(相對)':>18}{'y 範圍(相對)':>18}")
    for t, area, (xa, xb, ya, yb) in stats:
        print(f"{t:<6}{area:>8.1%}   {xa:.2f}-{xb:.2f}        {ya:.2f}-{yb:.2f}")
    print("\n判準：省份格應落在 x 約 0.02-0.16、y 約 0.08-0.92（canonical 第 0 格 + PAD_Y）")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--verify", type=int, default=10)
    args = ap.parse_args()

    rows = [l.rstrip("\n").split("\t") for l in open(args.list) if l.strip()]
    for split, sub in (("train", "plates"), ("val", "plates_val")):
        d = os.path.join(args.out_root, sub)
        os.makedirs(d, exist_ok=True)
        sizes, n = [], 0
        for fn, prov, sp in rows:
            if sp != split:
                continue
            sizes.append(crop_one(fn, os.path.join(d, fn)))
            n += 1
        w = [s[0] for s in sizes]
        h = [s[1] for s in sizes]
        print(f"{split:6s} -> {d}   {n} 張   "
              f"寬 {min(w)}-{max(w)}（中位 {int(np.median(w))}）  "
              f"高 {min(h)}-{max(h)}（中位 {int(np.median(h))}）")

    if args.verify:
        verify(args.out_root, args.verify)


if __name__ == "__main__":
    main()
