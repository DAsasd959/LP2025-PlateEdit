#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCPD 中英數混合片段編輯／重建（新檔，只 monkeypatch，不改動既有檔案）
======================================================================
這是「方案 (a)」：一個**連續**片段跨過 cell 0 的邊界，因此同時含中文省份字
與英數字。例如遮 cell 0..2 -> "皖A5" 改成 "冀B7"。

為什麼是連續片段：make_mask_quad(quad, s, k) 本來就吃「起點 + 長度」，
所以 (a) 只要換取樣器。分離的多塊遮罩（方案 b）需要遮罩聯集，管線目前
一張圖只產生一個四邊形，那是另一件工程，這支不做。

三處與既有流程不同，全部靠 monkeypatch 疊在 ccpd_cn_edit 之上：

    1. pick_span    -> 可跨越 cell 0 邊界的變長片段
    2. pick_target  -> 逐位用「該位置合法的字集」挑替換字
                       cell 0 = 31 個省份簡稱；cell 1 = 字母；cell 2..6 = 英數
    3. draw_glyph2  -> 換成 glyph_mixed.draw_glyph_mixed（逐字型、逐格對位、補圓點）

另外把 prompt 從單數 character 換成複數 characters（片段長度 > 1 時），
既有 PROMPTS 全是單數，多字片段的語意會不對。

用法
----
先驗證遮罩與 glyph 對位（不吃 GPU）：
  python mixed_span_edit.py --dry_run --limit 8 --span_mode cross --max_k 3

重建（有 ground truth，可算 LPIPS/FID）：
  python mixed_span_edit.py --target_mode gt --limit 300 --span_mode cross --max_k 3 \
      --config <cfg> --lora <ckpt> --flux_dir <flux> --out <dir>

混合編輯（換字，只能算 ACC/NED）：
  python mixed_span_edit.py --target_mode edit --limit 200 --span_mode cross --max_k 3 ...
"""
import argparse
import importlib.util
import os
import sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))   # repository root, so `import src` works

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CCE = _load("cce", os.path.join(HERE, "ccpd_cn_edit.py"))
GM = _load("glyph_mixed", os.path.join(HERE, "glyph_mixed.py"))

# 各格的合法字集。CCPD 的 cell 0 是省份簡稱、cell 1 是字母、cell 2..6 是英數。
# I 與 O 在真實車牌不使用（容易與 1/0 混淆），所以從英數池排除。
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


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--target_mode", choices=["gt", "edit"], default="gt")
    ap.add_argument("--span_mode", choices=["cross", "any", "fixed"], default="cross",
                    help="cross=一定跨過 cell 0 邊界（s=0, k>=2）；any=一般變長；fixed=固定 k")
    ap.add_argument("--max_k", type=int, default=3, help="片段最大長度（<=7）")
    ap.add_argument("--fixed_k", type=int, default=3, help="span_mode=fixed 時用的長度")
    ap.add_argument("--no_dot", action="store_true", help="片段跨過分隔圓點時不補畫圓點")
    ap.add_argument("--seed", type=int, default=42)
    known, rest = ap.parse_known_args()

    if not 1 <= known.max_k <= 7:
        sys.exit("--max_k 必須在 1..7")
    state = {"s": 0, "k": 1}

    # ---------------- 1. 取樣器 ----------------
    def pick_span(mode, rng):
        if known.span_mode == "cross":
            # s=0 保證含省份字，k>=2 保證同時含英數
            k = rng.randint(2, known.max_k) if known.max_k >= 2 else 2
            s = 0
        elif known.span_mode == "fixed":
            k = min(known.fixed_k, 7)
            s = rng.randint(0, 7 - k)
        else:
            k = rng.randint(1, known.max_k)
            s = rng.randint(0, 7 - k)
        state["s"], state["k"] = s, k
        return s, k

    # ---------------- 2. 逐位挑替換字 ----------------
    def pick_target(target_mode, chars, s, k, rng):
        orig = "".join(chars[s:s + k])
        if known.target_mode == "gt":
            return orig
        out = []
        for i, ch in enumerate(chars[s:s + k]):
            cand = [c for c in pool_for(s + i) if c != ch]
            out.append(rng.choice(cand))
        return "".join(out)

    # ---------------- 3. 逐字型、逐格對位的 glyph ----------------
    def draw_glyph2(font, render_text, polygon, width, height):
        """簽章與 CCE.draw_glyph2 相同，但忽略傳進來的單一 font。
        片段的 s/k 由 pick_span 記在 state 裡（CCE.main 的呼叫順序保證先於此）。"""
        return GM.draw_glyph_mixed(
            polygon, state["s"], state["k"], render_text, width, height,
            cjk_font=_fonts["cjk"], latin_font=_fonts["latin"],
            draw_dot=not known.no_dot)

    # CCE.main 會自己建字型物件，這裡先用同樣的預設路徑建一份給渲染器用
    from PIL import ImageFont
    cjk_path = DEF = CCE.DEF_CJK_FONT
    latin_path = CCE.DEF_LATIN_FONT
    for i, a in enumerate(rest):
        if a == "--cjk_font" and i + 1 < len(rest):
            cjk_path = rest[i + 1]
        if a == "--latin_font" and i + 1 < len(rest):
            latin_path = rest[i + 1]
    _fonts = {"cjk": ImageFont.truetype(cjk_path, size=60),
              "latin": ImageFont.truetype(latin_path, size=60)}

    # ---------------- 4. 複數 prompt ----------------
    _run_single = CCE.run_single

    def run_single(pipe, config, scene_pil, glyph_pil, mask_pil, target_text, prompt_rng, seed):
        saved = CCE.PROMPTS
        if len(target_text) > 1:
            CCE.PROMPTS = PROMPTS_PLURAL
        try:
            return _run_single(pipe, config, scene_pil, glyph_pil, mask_pil,
                               target_text, prompt_rng, seed)
        finally:
            CCE.PROMPTS = saved

    CCE.pick_span = pick_span
    CCE.pick_target = pick_target
    CCE.draw_glyph2 = draw_glyph2
    CCE.run_single = run_single

    # CCE.main 只認得 --mode / --target_mode；片段由上面的 pick_span 決定，
    # 所以固定傳 random / gt，真正的模式從 known 讀。
    sys.argv = [sys.argv[0]] + rest + ["--mode", "random", "--target_mode", "gt",
                                       "--seed", str(known.seed)]
    kmin = 2 if known.span_mode == "cross" else 1
    print(f"[mixed] span_mode={known.span_mode}  k={kmin}..{known.max_k}  "
          f"target_mode={known.target_mode}  補圓點={'否' if known.no_dot else '是'}")
    print(f"[mixed] 字集：cell0 {len(PROVINCE_POOL)} 省 / cell1 {len(LETTER_POOL)} 字母 / "
          f"cell2-6 {len(ALNUM_POOL)} 英數")
    print(f"[mixed] 字型：CJK {os.path.basename(cjk_path)}  拉丁 {os.path.basename(latin_path)}")
    CCE.main()


if __name__ == "__main__":
    main()
