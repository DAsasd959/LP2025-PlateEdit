#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the three inference conditions for a directory of CCPD plate crops:

    partial_masks/<stem>.png       the region to repaint, white on black
    partial_glyphs/<stem>.png      the target text warped into that region
    partial_labels_txt/<stem>.txt  target text, then the region's four corners

Nothing needs to be detected. CCPD encodes its own annotation in the filename:
field 3 is the plate's four vertices, field 4 the character indices. That is why
this dataset needs no text detector while LP-2025 does.

    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
                                \\________ four vertices ________/ \\__ characters __/

The plate is treated as seven cells: one Chinese province character followed by
six alphanumerics. `--province_ratio` sets how often the masked span starts at
cell 0, i.e. how often the province character is part of the edit:

    --province_ratio 0     never  -- alphanumeric only (the original behaviour)
    --province_ratio 0.5   half   -- what the released stage-1 cache was built with
    --province_ratio 1     always -- province character in every sample

Spans are contiguous and may cross the province/alphanumeric boundary, so a span
of length 2 starting at cell 0 covers one Chinese character and one letter. This
matches eval/ccpd/mixed_span_edit.py --span_mode cross.

Pure CPU; no model is loaded. Reproducible: per-image seed is `--seed + index`.

Run from the repository root:

    python eval/ccpd/build_ccpd_conditions.py \
        --src data/ccpd/test_1000/plates --out data/ccpd/test_1000 \
        --province_ratio 0.5 --max_k 3

The geometry and rendering below are copied from train/script/cn_preprocess.py so this
stays dependency-light (no diffusers, no torch). If you change the cell layout or
glyph rendering in one file, change it in the other too, or the conditions will
stop matching what the model was trained on.
"""
import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
             "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
             "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R",
             "S", "T", "U", "V", "W", "X", "Y", "Z", "O"]
ADS = ALPHABETS[:-1] + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]

# Seven cell boundaries on the CCPD canonical 440x140 plate. The separator dot
# falls in the natural 126..140 gap, which is why cell 2 starts at 140.
CELLS = [(12, 69), (69, 126), (140, 197), (197, 254), (254, 311), (311, 368), (368, 425)]
PAD_Y = 12

_REPO = Path(__file__).resolve().parents[2]
CJK_FONT = os.environ.get("CJK_FONT", str(_REPO / "font" / "正黑體.ttf"))
LATIN_FONT = os.environ.get("LATIN_FONT", str(_REPO / "font" / "TWGen7_V1.ttf"))


def is_cjk(s):
    return any("一" <= ch <= "鿿" for ch in s)


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    xs = pts[np.argsort(pts[:, 0]), :]
    left, right = xs[:2, :], xs[2:, :]
    left = left[np.argsort(left[:, 1]), :]
    right = right[np.argsort(right[:, 1]), :]
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def parse_filename(filename):
    """CCPD filename -> (quad in crop coordinates, seven characters)."""
    parts = filename.split("-")
    quad = np.array([[int(v) for v in p.split("&")] for p in parts[3].split("_")],
                    dtype=np.float32)
    quad[:, 0] -= np.min(quad[:, 0])
    quad[:, 1] -= np.min(quad[:, 1])
    idx = list(map(int, parts[4].split("_")))
    chars = [PROVINCES[idx[0]], ALPHABETS[idx[1]]] + [ADS[i] for i in idx[2:7]]
    return quad, chars


def mask_quad(quad, s, k):
    q = order_points(quad)
    src = np.array([[0, 0], [440, 0], [440, 140], [0, 140]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, q.astype(np.float32))
    x0, x1 = CELLS[s][0], CELLS[s + k - 1][1]
    canon = np.array([[x0, PAD_Y], [x1, PAD_Y], [x1, 140 - PAD_Y], [x0, 140 - PAD_Y]],
                     dtype=np.float32)
    return cv2.perspectiveTransform(canon.reshape(-1, 1, 2), M).reshape(4, 2).astype(np.float32)


def find_best_font_size(font, text, w, h):
    if not text:
        return 60
    canvas = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(canvas).text((w, h), text, font=font.font_variant(size=100), fill=255)
    rows = np.any(np.array(canvas) > 0, axis=1)
    if not np.any(rows):
        return 60
    return max(10, min(int(100 * (h * 0.9) / (np.sum(rows) + 1e-6)), 600))


def draw_glyph(text, polygon, width, height, cjk_font, latin_font):
    blank = np.zeros((height, width, 1), dtype=np.float32)
    if not text:
        return blank
    font = cjk_font if is_cjk(text) else latin_font
    p0, p1, p2, _ = polygon
    w = int(np.linalg.norm(p0 - p1))
    h = int(np.linalg.norm(p1 - p2))
    if w <= 1 or h <= 1:
        return blank
    nf = font.font_variant(size=find_best_font_size(font, text, w, h))
    big = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(big).text((w, h), text, font=nf, fill=255)
    arr = np.array(big)
    rows, cols = np.any(arr > 0, axis=1), np.any(arr > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return blank
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    resized = cv2.resize(arr[r0:r1 + 1, c0:c1 + 1], (w, h), interpolation=cv2.INTER_LINEAR)
    M = cv2.getPerspectiveTransform(
        np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32), polygon.astype(np.float32))
    return cv2.warpPerspective(resized, M, (width, height),
                               borderValue=0)[..., None].astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory of CCPD plate crops")
    ap.add_argument("--out", required=True,
                    help="output root; the three condition directories are created inside")
    ap.add_argument("--province_ratio", type=float, default=0.0,
                    help="probability the span starts at cell 0, i.e. includes the "
                         "Chinese province character (0 = alphanumeric only)")
    ap.add_argument("--max_k", type=int, default=3, help="maximum span length in cells (1..7)")
    ap.add_argument("--seed", type=int, default=42, help="per-image seed is this + index")
    ap.add_argument("--limit", type=int, default=0, help="0 = every image")
    ap.add_argument("--labels", default=None,
                    help="optional '<filename> <full-plate-text>' file; the full text is "
                         "written to line 2 of each label for traceability")
    ap.add_argument("--cjk_font", default=CJK_FONT)
    ap.add_argument("--latin_font", default=LATIN_FONT)
    a = ap.parse_args()

    if not 1 <= a.max_k <= 7:
        raise SystemExit("--max_k must be between 1 and 7")
    for f in (a.cjk_font, a.latin_font):
        if not os.path.isfile(f):
            raise SystemExit(f"font not found: {f}\n"
                             "Put it in font/, or pass --cjk_font / --latin_font.")
    cjk = ImageFont.truetype(a.cjk_font, size=60)
    latin = ImageFont.truetype(a.latin_font, size=60)

    src = Path(a.src)
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")
    out = Path(a.out)
    dirs = {k: out / k for k in ("partial_masks", "partial_glyphs", "partial_labels_txt")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(f.name for f in src.iterdir() if f.suffix.lower() in exts)
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"no images found in {src}")

    full = {}
    if a.labels:
        for line in Path(a.labels).read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 2:
                full[p[0]] = p[1]

    print(f"{len(files)} images  province_ratio={a.province_ratio}  max_k={a.max_k}")
    done = skipped = with_cn = 0
    for i, fn in enumerate(files):
        try:
            quad, chars = parse_filename(fn)
        except (IndexError, ValueError):
            skipped += 1
            continue
        rng = random.Random(a.seed + i)
        k = rng.randint(1, a.max_k)
        # A span starting at cell 0 covers the province character; one starting at
        # cell 2 or later is alphanumeric only. Cell 1 is a letter either way.
        s = 0 if rng.random() < a.province_ratio else rng.randint(2, 7 - k)
        text = "".join(chars[s:s + k])
        if not text:
            skipped += 1
            continue
        if is_cjk(text):
            with_cn += 1

        stem = Path(fn).stem
        img = Image.open(src / fn).convert("RGB")
        W, H = img.size
        mq = mask_quad(quad, s, k)

        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [mq.astype(np.int32)], 255)
        Image.fromarray(m, "L").save(dirs["partial_masks"] / f"{stem}.png")

        g = draw_glyph(text, mq, W, H, cjk, latin)
        Image.fromarray((g[..., 0] * 255).astype(np.uint8)).save(
            dirs["partial_glyphs"] / f"{stem}.png")

        coords = " ".join(str(int(v)) for pt in mq for v in pt)
        (dirs["partial_labels_txt"] / f"{stem}.txt").write_text(
            f"{text} {coords}\n{full.get(fn, ''.join(chars))} {s} {k}\n", encoding="utf-8")
        done += 1

    print(f"done: {done} written, {skipped} skipped, {with_cn} include the province character")
    for k, d in dirs.items():
        print(f"  {k:20s} {len(list(d.iterdir()))}")


if __name__ == "__main__":
    main()
