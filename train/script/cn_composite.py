#!/usr/bin/env python3
"""
省份均衡的 pseudo-GT 合成器（探針實驗專用，新檔，不修改任何既有檔案）
=====================================================================

問題：CCPD 12,000 張裡 95.8% 是「皖」，510 張非皖散在 26 省（20 省 <= 8 張，
      7 個省份完全沒有樣本），無法用真實資料訓練全省份。

作法：拿真實 CCPD 車牌，把省份格的原字擦掉、換上指定的省份字，並做風格匹配
      （前景/背景色、模糊、雜訊都從同一張車牌的鄰近字元估計），產生省份均衡的
      pseudo ground truth。背景與退化全部來自真實影像。

輸出檔名沿用 CCPD 格式，只把 indices 欄位的第一個索引換成新省份，
因此 cn_preprocess.py 不需要任何修改就能正確解析。

用法：
  python train/script/cn_composite.py --src data/ccpd/train_8000/plates \
      --out data/ccpd/balanced_train --per_province 60
"""
import argparse
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
             "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
             "青", "宁", "新", "警", "学", "O"]
# 只用 31 個真實省份簡稱（排除 警/学/O）
REAL_PROV_IDX = list(range(31))

CELLS = [(12, 69), (69, 126), (140, 197), (197, 254), (254, 311), (311, 368), (368, 425)]
PAD_Y = 12
CJK_FONT = os.environ.get("CJK_FONT",
                          str(Path(__file__).resolve().parents[2] / "font" / "正黑體.ttf"))


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    xs = pts[np.argsort(pts[:, 0]), :]
    left, right = xs[:2, :], xs[2:, :]
    left = left[np.argsort(left[:, 1]), :]
    right = right[np.argsort(right[:, 1]), :]
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def parse_quad(filename):
    parts = filename.split("-")
    quad = np.array([[int(v) for v in p.split("&")] for p in parts[3].split("_")], dtype=np.float32)
    quad[:, 0] -= np.min(quad[:, 0])
    quad[:, 1] -= np.min(quad[:, 1])
    return quad


def cell_quad(quad, s, k=1):
    q = order_points(quad)
    src = np.array([[0, 0], [440, 0], [440, 140], [0, 140]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, q.astype(np.float32))
    x0, x1 = CELLS[s][0], CELLS[s + k - 1][1]
    canon = np.array([[x0, PAD_Y], [x1, PAD_Y], [x1, 140 - PAD_Y], [x0, 140 - PAD_Y]],
                     dtype=np.float32)
    return cv2.perspectiveTransform(canon.reshape(-1, 1, 2), M).reshape(4, 2).astype(np.float32)


def estimate_style(img_bgr, quad):
    """從後 5 碼字元區估計前景(白字)色、背景(藍底)色、模糊程度與雜訊強度"""
    mq = cell_quad(quad, 2, 5)                      # 後 5 碼整段
    m = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.fillPoly(m, [mq.astype(np.int32)], 255)
    region = img_bgr[m > 0]
    if len(region) < 20:
        return np.array([230, 230, 230], np.float32), np.array([120, 60, 20], np.float32), 1.0, 4.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gvals = gray[m > 0]
    thr = np.clip(np.percentile(gvals, 60), 1, 254)
    fg = region[gvals >= thr]
    bg = region[gvals < thr]
    fg_c = fg.mean(0).astype(np.float32) if len(fg) else np.array([230, 230, 230], np.float32)
    bg_c = bg.mean(0).astype(np.float32) if len(bg) else np.array([120, 60, 20], np.float32)
    # 車牌一律是亮字暗底；若估反了就交換（非藍底/曝光異常的車牌會出現）
    if fg_c.mean() < bg_c.mean():
        fg_c, bg_c = bg_c, fg_c

    # 模糊：Laplacian variance 越低越糊 -> 換成 blur sigma
    lap = cv2.Laplacian(cv2.bitwise_and(gray, gray, mask=m), cv2.CV_64F).var()
    sigma = float(np.clip(2.4 - 0.35 * np.log1p(lap), 0.3, 2.5))
    # 雜訊：背景像素的標準差
    noise = float(np.clip(bg.std() if len(bg) else 4.0, 1.0, 18.0))
    return fg_c, bg_c, sigma, noise


def render_glyph_alpha(font, ch, polygon, W, H, scale_jitter, offset):
    """把中文字渲染成 anti-aliased alpha，warp 進省份格"""
    p0, p1, p2, _ = polygon
    w = max(2, int(np.linalg.norm(p0 - p1)))
    h = max(2, int(np.linalg.norm(p1 - p2)))

    big = Image.new("L", (400, 400), 0)
    f = font.font_variant(size=260)
    ImageDraw.Draw(big).text((70, 40), ch, font=f, fill=255)
    a = np.array(big)
    ys, xs = np.where(a > 0)
    if len(xs) == 0:
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    tw = max(2, int(w * scale_jitter))
    th = max(2, int(h * 0.86 * scale_jitter))        # 中文字比拉丁字略扁一點
    a = cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)

    pad = np.zeros((h, w), np.uint8)
    y0 = np.clip((h - th) // 2 + offset[1], 0, max(0, h - th))
    x0 = np.clip((w - tw) // 2 + offset[0], 0, max(0, w - tw))
    pad[y0:y0 + th, x0:x0 + tw] = a[:min(th, h - y0), :min(tw, w - x0)]

    M = cv2.getPerspectiveTransform(
        np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32), polygon.astype(np.float32))
    return cv2.warpPerspective(pad, M, (W, H), borderValue=0).astype(np.float32) / 255.0


def composite(img_bgr, quad, ch, font, rng):
    H, W = img_bgr.shape[:2]
    pq = cell_quad(quad, 0, 1)
    fg_c, bg_c, sigma, noise = estimate_style(img_bgr, quad)

    # 1. 擦掉原本的省份字：用 inpaint 讓周圍背景自然延伸，
    #    避免平填單一背景色在有光照漸層的車牌上露出矩形補丁
    #    （若露出方塊，模型會學著畫方塊，評測就不可信）
    cell_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(cell_mask, [pq.astype(np.int32)], 255)
    m3 = (cell_mask > 0)[..., None]

    inpainted = cv2.inpaint(img_bgr, cell_mask, 3, cv2.INPAINT_TELEA).astype(np.float32)
    # inpaint 會過度平滑，補回與背景相符的雜訊
    inpainted = inpainted + np.random.normal(0, noise * 0.8, inpainted.shape) * m3
    out = np.where(m3, inpainted, img_bgr.astype(np.float32))

    # 2. 渲染新省份字（含尺度與位置抖動，避免變成條件圖的像素級複製）
    alpha = render_glyph_alpha(font, ch, pq, W, H,
                               scale_jitter=rng.uniform(0.90, 1.04),
                               offset=(rng.randint(-1, 1), rng.randint(-1, 1)))
    if alpha is None:
        return None
    # 3. 匹配模糊程度
    ks = int(max(1, round(sigma * 2)) * 2 + 1)
    alpha = cv2.GaussianBlur(alpha, (ks, ks), sigma)
    alpha = np.clip(alpha, 0, 1)[..., None]
    alpha = alpha * m3                                # 嚴格限制在省份格內

    # 4. 前景色改用「相對對比」而非絕對顏色：
    #    inpaint 後的局部底色可能偏亮，直接套 fg_c 會導致白字沒對比而看不見。
    #    改成讓貼上的字相對其局部背景，維持與真實字元相同的對比。
    local_bg = out[cell_mask > 0].reshape(-1, 3).mean(0).astype(np.float32)
    contrast = fg_c - bg_c                                  # 真實字元的對比向量
    fg_use = local_bg + contrast
    # 保底：亮度差至少 45，否則字會埋在背景裡
    if fg_use.mean() - local_bg.mean() < 45:
        fg_use = local_bg + (contrast / max(1e-3, contrast.mean())) * 45
    fg_use = np.clip(fg_use, 0, 255)

    fg = np.zeros_like(out)
    fg[:] = fg_use
    fg += np.random.normal(0, noise * 0.6, out.shape)
    out = out * (1 - alpha) + fg * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def rewrite_filename(stem, new_prov_idx):
    """把 CCPD indices 欄位的第一個索引換成新省份，其餘欄位保持不變"""
    parts = stem.split("-")
    idx = parts[4].split("_")
    idx[0] = str(new_prov_idx)
    parts[4] = "_".join(idx)
    return "-".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_province", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"輸出目錄已存在且非空，請換一個：{args.out}")
    os.makedirs(args.out, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    font = ImageFont.truetype(CJK_FONT, 260)

    src = Path(args.src)
    files = sorted(f.name for f in src.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng.shuffle(files)
    print(f"來源 {len(files)} 張，目標每省 {args.per_province} 張 -> "
          f"{len(REAL_PROV_IDX) * args.per_province} 張")

    cnt, made, fi = Counter(), 0, 0
    targets = [(p, i) for i in range(args.per_province) for p in REAL_PROV_IDX]
    rng.shuffle(targets)

    for prov_idx, _ in targets:
        # 逐一取用來源車牌（可重複使用不同來源配不同省份）
        for _ in range(len(files)):
            fn = files[fi % len(files)]
            fi += 1
            stem = Path(fn).stem
            try:
                quad = parse_quad(stem)
            except Exception:
                continue
            img = cv2.imread(str(src / fn))
            if img is None or img.shape[0] < 40 or img.shape[1] < 80:
                continue
            res = composite(img, quad, PROVINCES[prov_idx], font, rng)
            if res is None:
                continue
            new_stem = rewrite_filename(stem, prov_idx)
            out_p = os.path.join(args.out, new_stem + ".jpg")
            if os.path.exists(out_p):
                continue
            cv2.imwrite(out_p, res, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cnt[PROVINCES[prov_idx]] += 1
            made += 1
            break

    print(f"\n完成 {made} 張 -> {args.out}")
    print("省份分布:", dict(sorted(cnt.items(), key=lambda kv: -kv[1])))
    print(f"涵蓋省份數 = {len(cnt)}")


if __name__ == "__main__":
    main()
