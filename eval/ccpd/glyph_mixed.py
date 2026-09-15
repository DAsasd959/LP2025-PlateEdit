#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐字型 glyph condition 渲染器（新檔，不改動任何既有檔案）
==========================================================
為什麼需要這支
--------------
既有兩處都是「整串挑一個字型」：

    cn_preprocess.py:142     font = self.cjk_font if is_cjk(text) else self.latin_font
    ccpd_cn_edit.py (main)   font = cjk_font if is_cjk(target_text) else latin_font

只要片段同時含中文與英數（例如遮 cell 0..2 的 "皖A5"），is_cjk 為真，
於是 A 和 5 也會被正黑體畫出來 —— 那不是車牌的英數字型（TWGen7_V1）。
glyph condition 本身就餵錯風格，模型學到／看到的都是錯的。

這支改成兩件事：
    1. **逐字選字型**：中文用 CJK 字型，英數用車牌拉丁字型
    2. **逐格對位**：每個字畫進「它自己那一格」的四邊形，而不是把整串
       拉伸鋪滿整個片段。真實車牌就是逐格排字，而且 CELLS 在 cell 1 與
       cell 2 之間有 126..140 的間隙（圓點分隔符），整串拉伸會把字推歪。
    3. **補圓點**：片段若同時覆蓋 cell 1 與 cell 2，遮罩會蓋掉分隔圓點；
       glyph 裡不畫它的話模型沒有依據，圓點會消失。

只需要「片段四邊形」就能算出逐格四邊形 —— 片段四邊形本身就是 canonical
矩形 [CELLS[s][0], CELLS[s+k-1][1]] 的透視像，反推回去即可，不必另外傳入
整張車牌的 quad。這讓它能直接替換 draw_glyph2。
"""
import importlib.util
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("cce", os.path.join(HERE, "ccpd_cn_edit.py"))
CCE = importlib.util.module_from_spec(_s)
_s.loader.exec_module(CCE)

CELLS, PAD_Y = CCE.CELLS, CCE.PAD_Y
CANON_W, CANON_H = 440, 140
DOT_X = (126, 140)              # cell 1 與 cell 2 之間的圓點分隔符區間


def _canon_rect(x0, x1):
    return np.array([[x0, PAD_Y], [x1, PAD_Y], [x1, CANON_H - PAD_Y], [x0, CANON_H - PAD_Y]],
                    dtype=np.float32)


def span_to_canon(span_quad, s, k):
    """片段四邊形 -> canonical 座標的反向 homography"""
    src = _canon_rect(CELLS[s][0], CELLS[s + k - 1][1])
    return cv2.getPerspectiveTransform(src, np.asarray(span_quad, dtype=np.float32))


def cell_quads(span_quad, s, k):
    """片段內每一格在原圖座標的四邊形"""
    M = span_to_canon(span_quad, s, k)
    out = []
    for i in range(k):
        c = _canon_rect(*CELLS[s + i])
        out.append(cv2.perspectiveTransform(c.reshape(-1, 1, 2), M).reshape(4, 2).astype(np.float32))
    return out


def dot_glyph(span_quad, s, k, W, H, r_frac=0.42):
    """在 canonical 空間畫圓點再透視到原圖 —— 片段跨過 cell 1/2 邊界時才需要"""
    M = span_to_canon(span_quad, s, k)
    cx = (DOT_X[0] + DOT_X[1]) / 2.0
    cy = CANON_H / 2.0
    r = (DOT_X[1] - DOT_X[0]) * r_frac
    canon = np.zeros((CANON_H, CANON_W), dtype=np.uint8)
    cv2.circle(canon, (int(round(cx)), int(round(cy))), max(2, int(round(r))), 255, -1)
    warped = cv2.warpPerspective(canon, M, (W, H), borderValue=0)
    return warped[..., None].astype(np.float32) / 255.0


def spans_dot(s, k):
    """片段是否覆蓋了 cell 1 與 cell 2 之間的圓點"""
    return s <= 1 and (s + k - 1) >= 2


def draw_glyph_mixed(span_quad, s, k, text, W, H, cjk_font, latin_font, draw_dot=True):
    """逐字、逐格渲染 glyph condition。回傳 (H, W, 1) float32 [0,1]"""
    if not text:
        return np.zeros((H, W, 1), dtype=np.float32)
    if len(text) != k:
        raise ValueError(f"text 長度 {len(text)} 與片段長度 k={k} 不符：{text!r}")

    out = np.zeros((H, W, 1), dtype=np.float32)
    for ch, cq in zip(text, cell_quads(span_quad, s, k)):
        font = cjk_font if CCE.is_cjk(ch) else latin_font
        out = np.maximum(out, CCE.draw_glyph2(font, ch, cq, W, H))
    if draw_dot and spans_dot(s, k):
        out = np.maximum(out, dot_glyph(span_quad, s, k, W, H))
    return out


def coverage(glyph_np, mask_np):
    """glyph 落在遮罩內的比例 —— 拿來確認逐格對位沒有跑掉"""
    g = glyph_np[..., 0] > 0
    m = mask_np > 0
    if not g.any():
        return 0.0, 0.0
    return float(g.sum()) / max(1.0, float(m.sum())), float((g & m).sum()) / float(g.sum())
