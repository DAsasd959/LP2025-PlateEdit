#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LP-2025（台灣車牌）編輯推論（新檔，不修改任何既有檔案）
=======================================================
用途：驗證同一份權重除了 CCPD 之外，在 LP-2025 上也能編輯英數字。

資料格式（來自 src/data/data_real.py:21-26 與實測標註檔）：
    filtered_plate/<stem>.jpg        車牌影像
    partial_labels_txt/<stem>.txt    第一個 token = 被遮罩片段的文字，
                                     後面 8 個數字 = 該片段的四個角點
    partial_glyphs/<stem>.png        預先渲染好的字形（GT 用）
    partial_masks/<stem>.png         預先算好的遮罩

本腳本兩種模式：
    --target_mode gt    用現成的 partial_glyphs/partial_masks 做重建
    --target_mode edit  用標註的四角自行渲染「不同的」英數字，做真正的編輯
                        （字型用 TWGen7_V1.ttf，即 CCPD 腳本裡的 DEF_LATIN_FONT）

用法：
  python lp_edit.py --limit 3 --target_mode edit \
      --config runs_cn_v2_stage1/<ts>/config.yaml \
      --lora   runs_cn_v2_stage1/<ts>/ckpt/20000/adapter_model.safetensors \
      --out /tmp/S1_lp
"""
import argparse
import importlib.util
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageFont
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
# lp_edit reuses the CCPD editor's machinery by monkeypatching it.
_spec = importlib.util.spec_from_file_location(
    "cce", os.path.join(ROOT, "eval", "ccpd", "ccpd_cn_edit.py"))
CCE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CCE)      # 幾何 / glyph 繪製 / 模型載入 / debug 圖全部沿用

LP_ROOT = os.environ.get("LP_ROOT", "data/lp")
ALNUM = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")   # 與 CCPD 腳本一致，排除 I/O


def parse_label(path):
    """回傳 (被遮罩片段文字, quad[4,2])"""
    parts = open(path, encoding="utf-8").readline().strip().split()
    text = parts[0]
    nums = [float(v) for v in parts[1:9]]
    quad = np.array(nums, dtype=np.float32).reshape(4, 2)
    return text, quad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp_root", default=LP_ROOT)
    ap.add_argument("--config", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--flux_dir", default=os.path.join(ROOT, "weights", "flux_base"))
    ap.add_argument("--target_mode", choices=["gt", "edit"], default="edit")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max_span", type=int, default=3,
                    help="只取被遮罩片段長度 <= 這個值的樣本，避免一次改太多字")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    lab_dir = os.path.join(args.lp_root, "partial_labels_txt")
    img_dir = os.path.join(args.lp_root, "filtered_plate")
    gly_dir = os.path.join(args.lp_root, "partial_glyphs")
    msk_dir = os.path.join(args.lp_root, "partial_masks")

    # 挑出四件套齊全、且片段長度合適的樣本
    stems = []
    for f in sorted(os.listdir(lab_dir)):
        if not f.endswith(".txt"):
            continue
        s = f[:-4]
        if not os.path.exists(os.path.join(img_dir, s + ".jpg")):
            continue
        try:
            t, _ = parse_label(os.path.join(lab_dir, f))
        except Exception:
            continue
        if 1 <= len(t) <= args.max_span:
            stems.append(s)
        if len(stems) >= args.limit:
            break
    print(f"[lp_edit] 取用 {len(stems)} 張（片段長度 <= {args.max_span}）  模式={args.target_mode}")

    os.makedirs(os.path.join(args.out, "debug"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "plates"), exist_ok=True)
    lat = ImageFont.truetype(CCE.DEF_LATIN_FONT, 60)
    cap = ImageFont.truetype(CCE.DEF_CJK_FONT, 15)

    plan = []
    for i, s in enumerate(stems):
        orig, quad = parse_label(os.path.join(lab_dir, s + ".txt"))
        rng = random.Random(f"{args.seed}-{s}")
        if args.target_mode == "gt":
            tgt = orig
        else:
            tgt = "".join(rng.choice([c for c in ALNUM if c != ch]) for ch in orig)
        plan.append((s, orig, tgt, quad))
        print(f"  {s[:28]:30s} 原片段='{orig}' -> 目標='{tgt}'")

    if args.dry_run:
        print("dry_run，未載入模型")
        return

    pipe, cfg = CCE.load_model(args.config, args.lora, args.flux_dir)
    prng = random.Random(args.seed)
    recs = []
    for i, (s, orig, tgt, quad) in enumerate(tqdm(plan)):
        scene = Image.open(os.path.join(img_dir, s + ".jpg")).convert("RGB")
        W, H = scene.size
        if args.target_mode == "gt":
            mask_pil = Image.open(os.path.join(msk_dir, s + ".png")).convert("L")
            glyph_pil = Image.open(os.path.join(gly_dir, s + ".png")).convert("L")
        else:
            q = CCE.order_points(quad)
            mnp = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mnp, [q.astype(np.int32)], 255)
            mask_pil = Image.fromarray(mnp)
            g = CCE.draw_glyph2(lat, tgt, q, W, H)
            glyph_pil = Image.fromarray((g[..., 0] * 255).astype(np.uint8))

        gen, prompt = CCE.run_single(pipe, cfg, scene, glyph_pil, mask_pil, tgt,
                                     prng, args.seed + i)
        gen.save(os.path.join(args.out, "plates", s + ".png"))
        CCE.debug_grid(scene, mask_pil, glyph_pil, gen,
                       f"LP  '{orig}' -> '{tgt}'",
                       os.path.join(args.out, "debug", s + "_debug.jpg"), caption_font=cap)
        recs.append((s, orig, tgt))

    with open(os.path.join(args.out, "labels.txt"), "w") as f:
        for s, orig, tgt in recs:
            f.write(f"{s}.png\t{orig}\t{tgt}\n")
    print(f"完成 {len(recs)} 張 -> {args.out}")


if __name__ == "__main__":
    main()
