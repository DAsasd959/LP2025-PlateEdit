#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 of the LP-2025 annotation pipeline: turn DeepSolo++ detections into the
three conditions the model trains on.

    partial_masks/<stem>.png       region to repaint, white on black
    partial_glyphs/<stem>.png      target text warped into that region
    partial_labels_txt/<stem>.txt  target text, then the region's four corners

Stage 1 is DeepSolo++ itself (see docs/ANNOTATION.md); it writes one text file
per plate, each line `<recognised text> x1 y1 x2 y2 x3 y3 x4 y4`. Point --det at
that directory.

Why a detector at all: LP-2025 ships the plate string in the filename but no
geometry, so the text quadrilateral has to be found. CCPD needs none of this —
its filename carries the four vertices outright.

WHAT THIS SCRIPT DOES

1. Polygon selection. A detector returns several polygons per crop. Sort them by
   area and take index len//2 -- the upper median. Verified against all 3,258
   released test annotations: the released mask falls inside this polygon 3258/3258.

2. Fuzzy matching. Raw recognition is unreliable ('AXS9956' comes back as
   'AXS-99556'), so it is aligned to the ground-truth string from the filename by
   sliding-window edit distance under a visual-confusion dictionary -- B and 8,
   I and 1, O and 0 count as equal. The alignment says which substring of the
   plate the polygon actually covers.

3. Context-preserving mask.
     Partial coverage -- the polygon covers only part of the string: use it as
       the mask directly and render that substring.
     Full coverage -- the polygon spans the whole string: masking all of it would
       leave no style context, so the polygon is split into n equal segments
       (n = number of alphanumerics recognised) and a contiguous run of k < n is
       sampled. Splitting on the recognised count rather than the ground-truth
       count matches the released annotations far better: 72.2% against 38.2%
       at a 2% tolerance.

Reconstructed geometry checked against the 3,258 released annotations: mean mask
IoU 0.9612, median 0.9953, 87.8% above 0.9. Exact reproduction of *which* span
was drawn is not possible -- the original run sampled it randomly and the seed
was not recorded.

The labels this produces are weakly supervised and contain noise: matching
failures, boundary shifts, occasional oversized polygons. The thesis reports the
training framework tolerates this, and the released checkpoint was trained on
exactly such labels.

Run from the repository root:

    python train/script/annotate_lp.py \
        --det DeepSolo/DeepSolo++/test_res --images data/lp/test/filtered_plate \
        --out data/lp/test --font font/TWGen7_V1.ttf
"""
import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.lp.prepare_sample import order_points, span_quad, draw_glyph  # noqa: E402

# Characters a text recogniser routinely swaps on a plate. Members of a group are
# treated as identical during alignment, and only during alignment -- the target
# text always comes from the ground-truth string, never from the recogniser.
CONFUSABLE = ["O0DQ", "I1L", "B8", "S5", "Z2", "G6", "U V"]
_EQ = {}
for _g in CONFUSABLE:
    for _c in _g.replace(" ", ""):
        _EQ[_c] = _g.replace(" ", "")[0]


def _canon(s):
    return "".join(_EQ.get(c, c) for c in s.upper() if c.isalnum())


def _dist(a, b):
    """Edit distance where confusable characters cost nothing to swap."""
    a, b = _canon(a), _canon(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match_span(recognised, plate):
    """Which substring of `plate` the recognised text covers.

    Returns (start, end, normalised_distance). Every substring whose length is
    within two of the recognised length is scored; the closest wins, ties going
    to the longer span so that a full-coverage detection is not mistaken for a
    partial one.
    """
    n = len(plate)
    rl = len(_canon(recognised))
    best = None
    for i in range(n):
        for j in range(i + 1, n + 1):
            if abs((j - i) - rl) > 2:
                continue
            d = _dist(recognised, plate[i:j]) / max(1, j - i)
            if best is None or (d, -(j - i)) < (best[2], -(best[1] - best[0])):
                best = (i, j, d)
    return best if best else (0, n, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True, help="directory of DeepSolo++ .txt outputs")
    ap.add_argument("--images", required=True, help="directory of plate crops")
    ap.add_argument("--out", required=True, help="output root for the three condition dirs")
    ap.add_argument("--font", default="font/TWGen7_V1.ttf")
    ap.add_argument("--max_dist", type=float, default=0.5,
                    help="reject a plate whose best alignment is worse than this")
    ap.add_argument("--min_area", type=float, default=200.0,
                    help="reject polygons smaller than this many pixels")
    ap.add_argument("--seed", type=int, default=42, help="per-image seed is this + index")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default=None, help="write a per-plate TSV of what happened")
    a = ap.parse_args()

    if not os.path.isfile(a.font):
        raise SystemExit(f"font not found: {a.font}")
    font = ImageFont.truetype(a.font, 60)
    dirs = {k: os.path.join(a.out, k) for k in
            ("partial_masks", "partial_glyphs", "partial_labels_txt")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    stems = sorted(f[:-4] for f in os.listdir(a.det) if f.endswith(".txt"))
    if a.limit:
        stems = stems[:a.limit]

    rows = []
    n_empty = n_reject = n_partial = n_full = 0
    for idx, stem in enumerate(stems):
        dets = []
        for line in open(os.path.join(a.det, stem + ".txt"), encoding="utf-8"):
            t = line.split()
            if len(t) >= 9:
                q = np.array([float(v) for v in t[1:9]], dtype=np.float32).reshape(4, 2)
                if cv2.contourArea(q) >= a.min_area:
                    dets.append((t[0], q))
        if not dets:
            n_empty += 1
            rows.append((stem, "no-detection", "", "", ""))
            continue

        dets.sort(key=lambda d: cv2.contourArea(d[1]))
        rec, poly = dets[len(dets) // 2]          # upper median by area
        poly = order_points(poly)

        plate = stem.split("_")[-1].upper()
        s, e, dist = match_span(rec, plate)
        if dist > a.max_dist:
            n_reject += 1
            rows.append((stem, "match-failed", rec, plate, f"{dist:.3f}"))
            continue

        rng = random.Random(a.seed + idx)
        if e - s < len(plate):
            # Partial coverage: the polygon already delimits a substring.
            target = plate[s:e]
            mask_quad = poly
            glyph_text, glyph_quad = target, poly
            n_partial += 1
            branch = "partial"
        else:
            # Full coverage: split and sample, or there is no style context left.
            n = max(1, len([c for c in rec if c.isalnum()]))
            n = min(n, len(plate)) if n >= len(plate) else n
            if n < 2:
                n_reject += 1
                rows.append((stem, "too-short", rec, plate, ""))
                continue
            k = rng.randint(1, n - 1)
            st = rng.randint(0, n - k)
            mask_quad = span_quad(poly, n, st, st + k)
            lo = min(len(plate), int(round(st * len(plate) / n)))
            hi = min(len(plate), lo + k)
            target = plate[lo:hi] or plate[:k]
            # The glyph is the whole string across the whole region, then cut by
            # the mask, so untouched characters keep the width they have in the
            # photograph.
            glyph_text, glyph_quad = plate, poly
            n_full += 1
            branch = f"full(n={n},k={k},s={st})"

        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(a.images, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            n_reject += 1
            rows.append((stem, "image-missing", rec, plate, ""))
            continue
        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [mask_quad.astype(np.int32)], 255)
        cv2.imwrite(os.path.join(dirs["partial_masks"], stem + ".png"), mask)
        try:
            glyph = draw_glyph(font, glyph_text, glyph_quad, W, H)
        except SystemExit:
            n_reject += 1
            rows.append((stem, "glyph-failed", rec, plate, ""))
            continue
        cv2.imwrite(os.path.join(dirs["partial_glyphs"], stem + ".png"),
                    np.where(mask > 127, glyph, 0))
        coords = " ".join(str(int(round(v))) for pt in mask_quad for v in pt)
        with open(os.path.join(dirs["partial_labels_txt"], stem + ".txt"), "w",
                  encoding="utf-8") as f:
            f.write(f"{target} {coords}\n")
        rows.append((stem, branch, rec, plate, target))

    n_ok = n_partial + n_full
    print(f"{len(stems)} plates -> {n_ok} annotated "
          f"({n_partial} partial, {n_full} full-coverage), "
          f"{n_empty} with no detection, {n_reject} rejected")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write("stem\tbranch\trecognised\tplate\ttarget\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
        print(f"per-plate report: {a.report}")


if __name__ == "__main__":
    main()
