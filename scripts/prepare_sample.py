#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the four inference inputs from a plate photograph.

`infer.py` reads four parallel directories keyed by a shared stem, but nothing
in this repository produced them — the released dataset arrived with them
already built. This script closes that gap so the model can be run on your own
plates.

You supply the crop, the text it currently shows, and the four corners of the
text region. The script works out the sub-region for the characters you want to
replace and writes:

    filtered_plate/<stem>.jpg      the crop, unchanged
    partial_masks/<stem>.png       the region to repaint, white
    partial_glyphs/<stem>.png      the target text warped into that region
    partial_labels_txt/<stem>.txt  target text followed by the region's corners

Example — replace characters 3 to 5 of a seven-character plate:

    python scripts/prepare_sample.py \
        --image my_plate.jpg --text RBE8700 \
        --quad "16,21 8,68 172,101 180,52" \
        --span 3:6 --target 999 \
        --out data/mine --stem my_plate

The corners may be given in any order; they are sorted internally. Take them
from a text detector, or read them off the image once by hand.

**The target must have the same number of characters as the original plate
text.** The glyph condition is built the way the released data was: the whole
plate text is rendered across the full text region, then the mask cuts out the
part being replaced. Each character therefore lands where it genuinely belongs
and keeps the width of its untouched neighbours. Change the character count and
every character shifts, so the untouched ones would no longer line up with the
photograph underneath; the script refuses that rather than producing a plate
that looks subtly broken.

Fidelity against the released data, measured by regenerating an archived sample:
the corner coordinates come out identical, the mask at IoU 0.99, and the glyph
at IoU 0.69 strict — 0.86 once a two-pixel shift is allowed, with strokes 3%
heavier. Strokes are about eight pixels wide, so a one-pixel offset costs a lot
of IoU; the layout, font, and extent all match. Close enough to drive the model,
not a bit-exact reconstruction of the archive. The archive itself is published,
so there is no need to rebuild it.
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def order_points(pts):
    """Sort four corners into top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)
    xs = pts[np.argsort(pts[:, 0])]
    left, right = xs[:2], xs[2:]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    return np.float32([left[0], right[0], right[1], left[1]])


def span_quad(quad, n_chars, start, end):
    """The sub-region holding characters [start, end) of an n-character string.

    Characters are assumed evenly spaced along the plate, so the sub-region's
    edges are linear interpolations of the full region's top and bottom edges.
    Verified against the released annotations: a 7-character plate whose first
    6 characters were masked reproduces the archived corners exactly.
    """
    tl, tr, br, bl = quad
    t0, t1 = start / n_chars, end / n_chars
    top_a, top_b = tl + (tr - tl) * t0, tl + (tr - tl) * t1
    bot_a, bot_b = bl + (br - bl) * t0, bl + (br - bl) * t1
    return np.float32([top_a, top_b, bot_b, bot_a])


def best_font_size(font, text, w, h):
    size, best = 8, 8
    while size < 400:
        f = font.font_variant(size=size)
        box = f.getbbox(text)
        if box[2] - box[0] > w or box[3] - box[1] > h:
            break
        best, size = size, size + 4
    return best


def draw_glyph(font, text, quad, width, height):
    """Render text and warp it into the quad, matching the training conditions.

    The rendered text is cropped to its ink and resized to the quad's extent, so
    it fills the region exactly — which is also why the character count has to
    match.
    """
    p0, p1, p2, _ = quad
    w, h = int(np.linalg.norm(p0 - p1)), int(np.linalg.norm(p1 - p2))
    if w <= 1 or h <= 1:
        raise SystemExit(f"span region is degenerate: {w}x{h} px")
    f = font.font_variant(size=best_font_size(font, text, w, h))
    big = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(big).text((w, h), text, font=f, fill=255)
    arr = np.array(big)
    rows, cols = np.any(arr > 0, axis=1), np.any(arr > 0, axis=0)
    if not rows.any():
        raise SystemExit(f"the font has no glyphs for {text!r}")
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    resized = cv2.resize(arr[r0:r1 + 1, c0:c1 + 1], (w, h), interpolation=cv2.INTER_LINEAR)
    M = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad.astype(np.float32))
    return cv2.warpPerspective(resized, M, (width, height), borderValue=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="plate crop")
    ap.add_argument("--text", required=True, help="text the plate currently shows")
    ap.add_argument("--quad", required=True,
                    help='four corners of the whole text region, "x,y x,y x,y x,y"')
    ap.add_argument("--span", required=True, help='character range to replace, e.g. "3:6"')
    ap.add_argument("--target", default=None,
                    help="replacement text; omit to reconstruct the original span")
    ap.add_argument("--font", required=True, help="TrueType font for the plate face")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", default=None, help="output name (default: the image's)")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    H, W = img.shape[:2]

    pts = [tuple(float(v) for v in p.split(",")) for p in a.quad.split()]
    if len(pts) != 4:
        raise SystemExit(f"--quad needs four points, got {len(pts)}")
    quad = order_points(pts)

    start, end = (int(v) for v in a.span.split(":"))
    if not 0 <= start < end <= len(a.text):
        raise SystemExit(f"--span {a.span} is outside 0:{len(a.text)}")
    source = a.text[start:end]
    target = a.target if a.target is not None else source
    if len(target) != len(source):
        raise SystemExit(
            f"--target {target!r} has {len(target)} characters but the span "
            f"{source!r} has {len(source)}. The glyph is laid out across the "
            f"whole plate before the mask cuts it, so changing the count shifts "
            f"every character — including the ones still visible in the "
            f"photograph. Replace like for like.")

    sq = span_quad(quad, len(a.text), start, end)
    # Full text across the full region, then cut by the mask — the released
    # conditions were built this way, and re-rendering only the span instead
    # would stretch those characters to fill it and leave them a different
    # width from the ones still visible in the photograph.
    full_text = a.text[:start] + target + a.text[end:]
    stem = a.stem or os.path.splitext(os.path.basename(a.image))[0]
    dirs = {k: os.path.join(a.out, k) for k in
            ("filtered_plate", "partial_masks", "partial_glyphs", "partial_labels_txt")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    cv2.imwrite(os.path.join(dirs["filtered_plate"], stem + ".jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [sq.astype(np.int32)], 255)
    cv2.imwrite(os.path.join(dirs["partial_masks"], stem + ".png"), mask)
    font = ImageFont.truetype(a.font, 60)
    full_glyph = draw_glyph(font, full_text, quad, W, H)
    cv2.imwrite(os.path.join(dirs["partial_glyphs"], stem + ".png"),
                np.where(mask > 127, full_glyph, 0))
    # Truncate rather than round: this is what produced the released
    # annotations, and matching it keeps regenerated samples identical.
    corners = " ".join(f"{int(v)}" for v in sq.reshape(-1))
    with open(os.path.join(dirs["partial_labels_txt"], stem + ".txt"), "w") as f:
        f.write(f"{target} {corners}\n")

    kind = "reconstruct" if target == source else f"{source} -> {target}"
    print(f"{stem}: {kind}   span {start}:{end} of {len(a.text)}   "
          f"mask {int(mask.sum() / 255):,} px")
    print(f"wrote four files under {a.out}")


if __name__ == "__main__":
    main()
