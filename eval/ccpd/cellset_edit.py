#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCPD 任意格子集合編輯（新檔，不動既有檔案）
==========================================
解決兩個問題
------------
1. **分隔圓點消失**
   既有的 make_mask_quad(quad, s, k) 產生「從 CELLS[s][0] 到 CELLS[s+k-1][1]」
   的單一大矩形。CELLS 在 cell 1 (…126) 與 cell 2 (140…) 之間留了 126–140 的
   間隙，那裡是車牌的方形分隔點。片段一旦跨過該邊界，點就被遮掉，模型必須
   憑空重畫它 —— 而點在來源解析度下往往只有 6 px（實測 194×98 的車牌上是
   30 個像素，佔整個 glyph 的 1%），重畫幾乎必然失敗。

   這支改成 **逐格四邊形取聯集**：遮罩只蓋住字元格本身，間隙不動，
   圓點的原始像素直接保留，不需要生成。

2. **只能遮連續片段**
   聯集遮罩天然支援任意格子集合，因此可以隨機挑不連續的格子
   （例如 {0, 3, 5}），這是既有管線做不到的。

附帶好處：遮罩面積變小，保留更多真實像素 —— 與論文「preserves most real
pixels」的主張同向。

模式
----
  --cells 0,2,5      指定格子
  --pick contig      連續片段（等價於既有行為，但不含間隙）
  --pick random      隨機挑 n 個格子，可不連續
  --pick mixed       必含 cell 0（中文），其餘從英數格隨機挑 —— 中英數混合
  --keep_dot 0|1     萬一遮罩真的蓋到間隙時是否補畫方點（預設 1）

用法
----
  python cellset_edit.py --dry_run --limit 8 --pick mixed --n 3
  python cellset_edit.py --pick mixed --n 3 --target_mode edit --limit 200 \
      --config <cfg> --lora <ckpt> --flux_dir <flux> --out <dir>
"""
import argparse
import importlib.util
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))   # repository root, so `import src` works

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


CCE = _load("cce", os.path.join(HERE, "ccpd_cn_edit.py"))
GM = _load("glyph_mixed", os.path.join(HERE, "glyph_mixed.py"))

# 保存兩個會被 monkeypatch 掉的原函式 —— build_mask / build_glyph 內部必須用原版，
# 否則會自己呼叫到自己（無限遞迴）。
_CV_FILL = cv2.fillPoly
_ORIG_DRAW = CCE.draw_glyph2
CELLS, PAD_Y = CCE.CELLS, CCE.PAD_Y
CANON_W, CANON_H = 440, 140
DOT_X = (126, 140)

PROVINCE_POOL = CCE.PROVINCES[:31]
LETTER_POOL = [c for c in CCE.ALPHABETS if c not in ("I", "O")]
ALNUM_POOL = [c for c in CCE.ADS if c not in ("I", "O")]

PROMPTS_PLURAL = [
    "Fill the masked characters '{text}' using the same color, font, and style as the surrounding text.",
    "The missing characters '{text}' should match the style and color of neighboring glyphs.",
    "Generate '{text}' in the same font, size, and color as adjacent text.",
    "Complete the masked characters '{text}' with consistent color, font, and style of surrounding characters.",
    "Fill in '{text}' so it blends seamlessly with the nearby text in font and color.",
]


def pool_for(cell):
    return PROVINCE_POOL if cell == 0 else (LETTER_POOL if cell == 1 else ALNUM_POOL)


def canon_to_img(quad):
    """canonical 440x140 -> 原圖座標的 homography"""
    q = CCE.order_points(quad).astype(np.float32)
    src = np.float32([[0, 0], [CANON_W, 0], [CANON_W, CANON_H], [0, CANON_H]])
    return cv2.getPerspectiveTransform(src, q)


def cell_quad(quad, cell, M=None, inset=0):
    """inset > 0 時把該格的遮罩往內縮 inset 個 canonical 像素。

    為什麼需要：CELLS[0]=(12,69) 與 CELLS[1]=(69,126) 在 x=69 直接相接。
    兩格同時被遮時，交界兩側都是待生成區，沒有任何真實像素當錨點，
    筆畫就會跨界（實測交界處墨水密度 0.177 -> 0.332，+88%）。
    只遮單格時隔壁的真實像素形成硬邊界，完全沒有這個問題（-0.004）。
    留一條細帶不遮，等於把「用未遮像素當風格參考」用在字元邊界上。"""
    M = canon_to_img(quad) if M is None else M
    x0, x1 = CELLS[cell]
    # 只在 X 方向內縮：相鄰格是左右相接，垂直方向沒有鄰居，縮 Y 只會白白切掉字
    x0, x1 = x0 + inset, x1 - inset
    y0, y1 = PAD_Y, CANON_H - PAD_Y
    c = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return cv2.perspectiveTransform(c.reshape(-1, 1, 2), M).reshape(4, 2).astype(np.float32)


def build_mask(quad, cells, W, H, inset=0):
    """逐格四邊形的聯集，各格往內縮 inset —— 分隔點與字元邊界都留下真實像素"""
    M = canon_to_img(quad)
    m = np.zeros((H, W), np.uint8)
    for c in cells:
        _CV_FILL(m, [cell_quad(quad, c, M, inset).astype(np.int32)], 255)
    return m


def build_glyph(quad, cells, text, W, H, cjk_font, latin_font):
    """逐字型、逐格對位；每個字畫進它自己那一格"""
    M = canon_to_img(quad)
    out = np.zeros((H, W, 1), np.float32)
    for ch, c in zip(text, cells):
        f = cjk_font if CCE.is_cjk(ch) else latin_font
        # glyph 不內縮：字要畫滿整格，內縮的只有遮罩
        out = np.maximum(out, _ORIG_DRAW(f, ch, cell_quad(quad, c, M), W, H))
    return out


def square_dot(quad, W, H, frac=0.42):
    """真實車牌的分隔點是方形，不是圓形。只在遮罩真的蓋到間隙時才需要。"""
    M = canon_to_img(quad)
    cx = (DOT_X[0] + DOT_X[1]) / 2.0
    half = (DOT_X[1] - DOT_X[0]) * frac
    canon = np.zeros((CANON_H, CANON_W), np.uint8)
    cv2.rectangle(canon, (int(cx - half), int(CANON_H / 2 - half)),
                  (int(cx + half), int(CANON_H / 2 + half)), 255, -1)
    return cv2.warpPerspective(canon, M, (W, H), borderValue=0)[..., None].astype(np.float32) / 255.0


def pick_cells(mode, n, rng):
    if mode == "contig":
        k = n
        s = rng.randint(0, 7 - k)
        return list(range(s, s + k))
    if mode == "random":
        return sorted(rng.sample(range(7), n))
    if mode == "mixed":                     # 必含中文格，其餘隨機（可不連續）
        rest = sorted(rng.sample(range(1, 7), max(0, n - 1)))
        return [0] + rest
    if mode == "alnum":                     # 只在英數格，隨機可不連續
        return sorted(rng.sample(range(2, 7), n))
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pick", choices=["contig", "random", "mixed", "alnum"], default="mixed")
    ap.add_argument("--n", type=int, default=3, help="遮幾格")
    ap.add_argument("--n_min", type=int, default=0, help=">0 時每張隨機取 n_min..n 格")
    ap.add_argument("--cells", default=None, help="固定格子，如 0,2,5")
    ap.add_argument("--target_mode", choices=["gt", "edit"], default="gt")
    ap.add_argument("--keep_dot", type=int, default=1)
    ap.add_argument("--inset", type=int, default=2,
                    help="每格遮罩往內縮的 canonical 像素數；>0 會在相鄰格之間留下真實像素當錨點")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prov_plan", default=None,
                    help="每行「stem<TAB>目標省份」。指定後中文格改用排班的省份而非隨機抽 —— "
                         "隨機抽會讓稀有省份幾乎抽不到，31 省的目標覆蓋就不均勻")
    known, rest = ap.parse_known_args()

    plan = {}
    if known.prov_plan:
        for line in open(known.prov_plan):
            st, tp = line.rstrip("\n").split("\t")
            plan[st] = tp
        print(f"[cellset] 目標省份排班 {len(plan)} 筆")

    fixed = [int(x) for x in known.cells.split(",")] if known.cells else None
    state = {"cells": []}

    # labels.txt 的 s/k 欄無法表達不連續的格子集合，另寫一份 cells.txt 旁檔，
    # 否則事後無法逐格計分（先前混合實驗就是因此只能用寬鬆指標）。
    _cells_log = []
    import atexit

    def _dump_cells():
        out = None
        for i, a in enumerate(rest):
            if a == "--out" and i + 1 < len(rest):
                out = rest[i + 1]
        if out and _cells_log:
            with open(os.path.join(out, "cells.txt"), "w") as f:
                for stem, cs, txt in _cells_log:
                    f.write(f"{stem}\t{','.join(map(str, cs))}\t{txt}\n")
            print(f"逐格記錄 -> {out}/cells.txt（{len(_cells_log)} 筆）")

    atexit.register(_dump_cells)

    def pick_span(mode, rng):
        n = known.n if known.n_min <= 0 else rng.randint(known.n_min, known.n)
        cells = fixed if fixed else pick_cells(known.pick, n, rng)
        state["cells"] = cells
        # 回傳值只是為了讓 CCE.main 的 records 有東西寫；真正的遮罩由下面決定
        return cells[0], len(cells)

    def pick_target(target_mode, chars, s, k, rng):
        cells = state["cells"]
        orig = "".join(chars[c] for c in cells)
        if known.target_mode == "gt":
            tgt = orig
        else:
            want = plan.get(state.get("stem"))
            out = []
            for c in cells:
                # 中文格若有排班就照排班走，其餘格子仍隨機（且不得與原字相同）
                if c == 0 and want:
                    out.append(want)
                else:
                    out.append(rng.choice([x for x in pool_for(c) if x != chars[c]]))
            tgt = "".join(out)
        _cells_log.append((state.get("stem", "?"), list(cells), tgt))
        return tgt

    def make_mask_quad(quad, s, k):
        """CCE.main 用它產生遮罩多邊形。這裡回傳所有格子的頂點集合，
        fillPoly 會把它們當成一個多邊形 —— 所以改由下面的 fill 覆寫。"""
        return cell_quad(quad, state["cells"][0])

    # CCE.main 內部是 cv2.fillPoly(mask_np, [mask_quad], 255)；
    # 換成逐格聯集 + （必要時）方形分隔點。
    _fill = _CV_FILL

    def fill_poly(img, pts, color, *a, **kw):
        cs = state.get("cells")
        if cs is not None and len(img.shape) == 2 and color == 255 and state.get("quad") is not None:
            H, W = img.shape
            img[:] = np.maximum(img, build_mask(state["quad"], cs, W, H, known.inset))
            return img
        return _fill(img, pts, color, *a, **kw)

    _parse = CCE.parse_ccpd_filename

    def parse_ccpd_filename(stem):
        r = _parse(stem)
        state["quad"] = r[0]
        state["stem"] = stem
        return r

    _draw = CCE.draw_glyph2

    def draw_glyph2(font, render_text, polygon, width, height):
        g = build_glyph(state["quad"], state["cells"], render_text, width, height,
                        _fonts["cjk"], _fonts["latin"])
        # 聯集遮罩只會因透視取整而碰到分隔點約 9%，那不需要重畫 ——
        # 重畫反而會與保留下來的真實像素疊成雙重點。只有蓋掉一半以上才補。
        m = build_mask(state["quad"], state["cells"], width, height, known.inset)
        gap = square_dot(state["quad"], width, height)
        gp = gap[..., 0] > 0
        if known.keep_dot and gp.sum() and np.logical_and(gp, m > 0).sum() / gp.sum() > 0.5:
            g = np.maximum(g, gap)
        return g

    _run = CCE.run_single

    def run_single(pipe, config, scene, glyph, mask, target_text, prompt_rng, seed):
        saved = CCE.PROMPTS
        if len(target_text) > 1:
            CCE.PROMPTS = PROMPTS_PLURAL
        try:
            return _run(pipe, config, scene, glyph, mask, target_text, prompt_rng, seed)
        finally:
            CCE.PROMPTS = saved

    from PIL import ImageFont as _IF
    cjk_path, latin_path = CCE.DEF_CJK_FONT, CCE.DEF_LATIN_FONT
    for i, a in enumerate(rest):
        if a == "--cjk_font" and i + 1 < len(rest): cjk_path = rest[i + 1]
        if a == "--latin_font" and i + 1 < len(rest): latin_path = rest[i + 1]
    _fonts = {"cjk": _IF.truetype(cjk_path, 60), "latin": _IF.truetype(latin_path, 60)}

    CCE.pick_span = pick_span
    CCE.pick_target = pick_target
    CCE.make_mask_quad = make_mask_quad
    CCE.parse_ccpd_filename = parse_ccpd_filename
    CCE.draw_glyph2 = draw_glyph2
    CCE.run_single = run_single
    cv2.fillPoly = fill_poly

    sys.argv = [sys.argv[0]] + rest + ["--mode", "random", "--target_mode", "gt",
                                       "--seed", str(known.seed)]
    print(f"[cellset] pick={known.pick}  n={known.n if known.n_min<=0 else f'{known.n_min}..{known.n}'}"
          f"  cells={fixed or '隨機'}  target={known.target_mode}"
          f"  遮罩=逐格聯集，內縮 {known.inset}px（相鄰格之間保留真實像素）")
    CCE.main()


if __name__ == "__main__":
    main()
