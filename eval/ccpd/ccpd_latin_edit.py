#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 Stage-1 權重能不能編輯「英數字」（新檔，不修改 ccpd_cn_edit.py）
=====================================================================
動機：現在有兩份權重（LP 一份、CCPD 一份），想知道能不能合成一份通吃。
      第一個要問的問題是 —— 7/26 那個在 CCPD 合成資料上訓練的權重，
      本來就有 84.2% 的更新是花在英數字上（cache 抽樣實測：省份格只佔 15.8%），
      所以它理論上應該同時會英數字。這支腳本就是驗證這件事。

做法：沿用 ccpd_cn_edit.py 的全部幾何與推論流程，只替換兩個函式
  pick_span   -> 只在第 2..6 格（純英數字區）取連續片段，不碰省份格
  pick_target -> 把該片段換成「不同的」英數字（CCPD 的 ADS 字集，已排除 I/O）

用法：
  python ccpd_latin_edit.py --k 1 --limit 40 \
      --config runs_word_fill_plate_512/20260726-184929/config.yaml \
      --lora   runs_word_fill_plate_512/20260726-184929/ckpt/3000/adapter_model.safetensors \
      --out /tmp/S1_latin
"""
import argparse
import importlib.util
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cce", os.path.join(HERE, "ccpd_cn_edit.py"))
CCE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CCE)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--k", type=int, default=1, help="要改幾個連續字元（1..3）")
    ap.add_argument("--charset", choices=["alnum", "alpha", "digit"], default="alnum",
                    help="限制目標字元類別：全部 / 只英文字母 / 只數字")
    ap.add_argument("--seed", type=int, default=42)
    known, rest = ap.parse_known_args()

    K = max(1, min(3, known.k))
    # 第 2..6 格是五個英數字；起點限制在 2..(7-K)，保證片段完全落在英數字區
    LO, HI = 2, 7 - K

    def pick_span(mode, rng):
        return rng.randint(LO, HI), K

    def pick_target(target_mode, chars, s, k, rng):
        """把 chars[s:s+k] 換成不同的英數字；逐位保證與原字元不同"""
        pool = [c for c in CCE.ADS if c not in ("I", "O")]
        if known.charset == "alpha":
            pool = [c for c in pool if c.isalpha()]
        elif known.charset == "digit":
            pool = [c for c in pool if c.isdigit()]
        out = []
        for i in range(s, s + k):
            cand = [c for c in pool if c != chars[i]]
            out.append(rng.choice(cand))
        return "".join(out)

    CCE.pick_span = pick_span
    CCE.pick_target = pick_target

    # 交回 ccpd_cn_edit 的 main()，mode/target_mode 已被上面兩個函式接管
    sys.argv = [sys.argv[0]] + rest + ["--mode", "random", "--target_mode", "gt",
                                       "--seed", str(known.seed)]
    _pool = [c for c in CCE.ADS if c not in ("I", "O")]
    if known.charset == "alpha":
        _pool = [c for c in _pool if c.isalpha()]
    elif known.charset == "digit":
        _pool = [c for c in _pool if c.isdigit()]
    print(f"[latin_edit] 只編輯第 {LO}..{HI+K-1} 格，片段長度 K={K}，"
          f"charset={known.charset}（{len(_pool)} 類）")
    CCE.main()


if __name__ == "__main__":
    main()
