# -*- coding: utf-8 -*-
"""
CCPD 中文車牌 Stage-1 合成資料產生器（新檔，不修改 datagen.py / cfg.py / data_cfg.py）
=====================================================================================

背景：data_cfg.py 已經指向 CCPD 設定
        font_dir      = 'datasets/fonts/ccpd_ttf'      (CCDP_TEXT0620.ttf, 省份字 31/31)
        text_filepath = 'data/ccpd_plate_list.txt'     (40000 筆, 31 省均衡)
      所以不需要改任何程式，只要用自訂的輸出目錄與樣本數把它跑起來。

為什麼要走 Stage-1 而不是貼字合成：
      Stage-1 是「先把文字渲染進車牌 -> 再對整張施加退化」，GT / glyph mask /
      字元 bbox 由同一次渲染同時產出，所以筆畫粗細、大小、模糊與周圍一致。
      先前用 cn_composite.py 把字貼到已退化的真實照片上，模型就學到了貼圖感。

用法：
  cd synth/cn
  python cn_datagen_run.py --out ../../data/ccpd/synth_probe --num 200
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="輸出目錄（會建立 i_s/ t_b/ mask_s/ i_s_bbox/）")
    ap.add_argument("--num", type=int, default=200, help="產生張數")
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"輸出目錄已存在且非空，為避免覆蓋請換一個：{args.out}")

    # datagen.main() 內部把 i_s_bbox 寫到 cfg.data_dir（而圖片寫到傳入的 data_dir），
    # 為了讓四個子目錄落在同一處，這裡在 runtime 覆寫 cfg.data_dir。
    # 這是記憶體內的屬性指派，不會改動 cfg.py 檔案本身。
    import cfg
    cfg.data_dir = args.out

    from Synthtext import data_cfg
    print(f"[cn_datagen] font_dir      = {data_cfg.font_dir}")
    print(f"[cn_datagen] text_filepath = {data_cfg.text_filepath}")
    print(f"[cn_datagen] 輸出            = {args.out}   張數 = {args.num}")

    from datagen import main as datagen_main
    datagen_main(data_dir=args.out, sample_num=args.num)

    for sub in ("i_s", "t_b", "mask_s", "i_s_bbox"):
        p = os.path.join(args.out, sub)
        print(f"  {sub:10s} = {len(os.listdir(p)) if os.path.isdir(p) else 0}")


if __name__ == "__main__":
    main()
