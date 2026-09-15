#!/usr/bin/env python3
"""
CCPD 中文省份字元編輯測試
========================

目的：現行 eval_ccpd.py 只遮罩後 5 碼英數字元，省份中文字永遠是可見 context，
      因此並沒有「生成中文」。本腳本把字元格從 5 格擴成 7 格，讓遮罩可以涵蓋
      省份中文字與城市字母，並改用含 CJK 字形的字型渲染 glyph condition。

CCPD canonical plate = 440 x 140，七格邊界（含字間空隙，中間圓點落在 126~140）：
      province  [ 12,  69]
      city      [ 69, 126]
      last5     [140,197] [197,254] [254,311] [311,368] [368,425]

模式：
  --mode province      只遮省份中文字（k=1, s=0）── 最乾淨的「能不能畫中文」測試
  --mode province_city 遮省份+城市字母（k=2, s=0）
  --mode random        七格內隨機連續片段（k=1..6）

用法（需先掛上放 FLUX.1-Fill-dev-nf4 的磁碟）：
  python ccpd_cn_edit.py --mode province --limit 30 --out CCPD_cn_province
"""
import argparse
import os
import random
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# 注意：torch / yaml / safetensors / src.* 都在 load_model() 與 run_single() 內才 import，
# 這樣 --dry_run 可以在沒有 FLUX 環境的機器上檢查幾何與字型覆蓋。

# ============================================================
# 預設路徑（皆可用 CLI 覆蓋）
# ============================================================
DEF_PLATE_DIR = "data/ccpd/test_1000/plates"
DEF_CONFIG = "test_original/20260603-161724/config.yaml"
DEF_LORA = "test_original/20260603-161724/11000/adapter_model.safetensors"
# 含 33/33 省份字的黑體，最接近真實中國車牌字型
DEF_CJK_FONT = os.environ.get("CJK_FONT", "font/正黑體.ttf")
DEF_LATIN_FONT = os.environ.get("LATIN_FONT", "font/TWGen7_V1.ttf")

TARGET_SIZE = 512

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
             "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
             "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R",
             "S", "T", "U", "V", "W", "X", "Y", "Z", "O"]
ADS = ALPHABETS[:-1] + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]

# 七格邊界 (start, end)，canonical 440x140
CELLS = [(12, 69), (69, 126), (140, 197), (197, 254), (254, 311), (311, 368), (368, 425)]
PAD_Y = 12

PROMPTS = [
    "Fill the masked character '{text}' using the same color, font, and style as the surrounding text.",
    "The missing character '{text}' should match the style and color of neighboring glyphs.",
    "Complete the masked letter '{text}' with consistent color, font, and style of surrounding characters.",
]


def is_cjk(s):
    return any("一" <= ch <= "鿿" for ch in s)


# ============================================================
# CCPD 檔名解析（與 eval_ccpd.py 一致）
# ============================================================
def parse_ccpd_filename(filename):
    parts = filename.split("-")
    quad = np.array([[int(v) for v in p.split("&")] for p in parts[3].split("_")], dtype=np.float32)
    quad[:, 0] -= np.min(quad[:, 0])
    quad[:, 1] -= np.min(quad[:, 1])

    idx = list(map(int, parts[4].split("_")))
    chars = [PROVINCES[idx[0]], ALPHABETS[idx[1]]] + [ADS[i] for i in idx[2:7]]
    return quad, "".join(chars), chars


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    xs = pts[np.argsort(pts[:, 0]), :]
    left, right = xs[:2, :], xs[2:, :]
    left = left[np.argsort(left[:, 1]), :]
    right = right[np.argsort(right[:, 1]), :]
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def make_mask_quad(quad, s, k):
    """把 canonical 七格中的 [s, s+k-1] 片段投影回原圖座標"""
    quad = order_points(quad)
    src = np.array([[0, 0], [440, 0], [440, 140], [0, 140]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))

    x0 = CELLS[s][0]
    x1 = CELLS[s + k - 1][1]
    canon = np.array([[x0, PAD_Y], [x1, PAD_Y], [x1, 140 - PAD_Y], [x0, 140 - PAD_Y]], dtype=np.float32)
    return cv2.perspectiveTransform(canon.reshape(-1, 1, 2), M).reshape(4, 2).astype(np.float32)


def pick_span(mode, rng):
    if mode == "province":
        return 0, 1
    if mode == "province_city":
        return 0, 2
    k = rng.randint(1, 6)
    return rng.randint(0, 7 - k), k


# 只用 31 個真實省份簡稱做替換目標（排除 警/学/O 這三個特殊符號）
REAL_PROVINCES = PROVINCES[:31]


def pick_target(target_mode, chars, s, k, rng):
    """
    決定 glyph condition 要給什麼字。
      gt             ── 給原字（重建）
      cross_province ── 省份格給「不同的」省份字，OCR 猜多數類就沒用了
      latin          ── 省份格給一個英文字母，用來區分「字系問題」與「位置問題」
    """
    orig = "".join(chars[s:s + k])
    if target_mode == "gt" or s != 0:
        return orig
    if target_mode == "cross_province":
        pool = [c for c in REAL_PROVINCES if c != chars[0]]
        return rng.choice(pool) + orig[1:]
    if target_mode == "latin":
        return rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ")) + orig[1:]
    return orig


# ============================================================
# Glyph 渲染（與 eval_ccpd.py 一致，只是字型可切換）
# ============================================================
def find_best_font_size(font, text, w, h):
    if not text:
        return 60
    canvas = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(canvas).text((w, h), text, font=font.font_variant(size=100), fill=255)
    rows = np.any(np.array(canvas) > 0, axis=1)
    if not np.any(rows):
        return 60
    return max(10, min(int(100 * (h * 0.9) / (np.sum(rows) + 1e-6)), 600))


def draw_glyph2(font, render_text, polygon, width, height):
    if not render_text:
        return np.zeros((height, width, 1), dtype=np.float32)
    p0, p1, p2, _ = polygon
    w = int(np.linalg.norm(p0 - p1))
    h = int(np.linalg.norm(p1 - p2))
    if w <= 1 or h <= 1:
        return np.zeros((height, width, 1), dtype=np.float32)

    new_font = font.font_variant(size=find_best_font_size(font, render_text, w, h))
    big = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(big).text((w, h), render_text, font=new_font, fill=255)
    arr = np.array(big)

    rows, cols = np.any(arr > 0, axis=1), np.any(arr > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        print(f"  [警告] glyph 渲染為空 -> 字型可能缺少 '{render_text}' 的字形")
        return np.zeros((height, width, 1), dtype=np.float32)

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    resized = cv2.resize(arr[r0:r1 + 1, c0:c1 + 1], (w, h), interpolation=cv2.INTER_LINEAR)
    M = cv2.getPerspectiveTransform(
        np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32), polygon.astype(np.float32)
    )
    warped = cv2.warpPerspective(resized, M, (width, height), borderValue=0)
    return warped[..., None].astype(np.float32) / 255.0


# ============================================================
# 模型
# ============================================================
def load_model(config_path, lora_path, flux_dir):
    import torch
    import yaml
    from safetensors.torch import load_file
    from src.train.model import OminiModelFIll
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = OminiModelFIll(
        flux_pipe_id=flux_dir,
        lora_config=config["train"]["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=config["train"]["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=True,
    )
    sd = load_file(lora_path)
    sd = {k.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default").replace("transformer.", ""): v
          for k, v in sd.items()}
    model.transformer.load_state_dict(sd, strict=False)
    pipe = model.flux_pipe
    pipe.to("cuda")
    pipe.text_encoder.to("cuda")
    return pipe, config


def run_single(pipe, config, scene_pil, glyph_pil, mask_pil, target_text, prompt_rng, seed):
    import torch
    from src.flux.condition import Condition
    from src.flux.generate_fill import generate_fill
    hint = np.array(glyph_pil.resize((TARGET_SIZE, TARGET_SIZE)).convert("RGB")) / 255.0
    origin = scene_pil.resize((TARGET_SIZE, TARGET_SIZE)).convert("RGB")
    mask = np.array(mask_pil.resize((TARGET_SIZE, TARGET_SIZE)).convert("L")) / 255.0
    mask_rgb = np.stack([mask] * 3, axis=-1)
    prompt = prompt_rng.choice(PROMPTS).format(text=target_text)

    cond = Condition(condition_type="word_fill", condition=[hint, mask_rgb, origin], position_delta=[0, 0])
    with torch.no_grad():
        res = generate_fill(
            pipe, prompt=prompt, conditions=[cond], height=TARGET_SIZE, width=TARGET_SIZE,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            model_config=config.get("model", {}), default_lora=True,
        )
    return res.images[0], prompt


def debug_grid(scene, mask, glyph, gen, caption, path, caption_font=None):
    vis = TARGET_SIZE // 2
    panels = [scene.resize((vis, vis)).convert("RGB"),
              mask.resize((vis, vis)).convert("RGB"),
              glyph.resize((vis, vis)).convert("RGB"),
              gen.resize((vis, vis)).convert("RGB")]
    canvas = Image.new("RGB", (vis * 4, vis + 26), "white")
    for i, p in enumerate(panels):
        canvas.paste(p, (vis * i, 0))
    # 字幕含中文，必須用 TrueType CJK 字型，預設點陣字型會 UnicodeEncodeError
    ImageDraw.Draw(canvas).text((6, vis + 6), caption, fill="black", font=caption_font)
    canvas.save(path)


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["province", "province_city", "random"], default="province")
    ap.add_argument("--target_mode", choices=["gt", "cross_province", "latin"], default="gt",
                    help="glyph condition 給什麼字：原字／不同省份字／英文字母")
    ap.add_argument("--plate_dir", default=DEF_PLATE_DIR)
    ap.add_argument("--config", default=DEF_CONFIG)
    ap.add_argument("--lora", default=DEF_LORA)
    ap.add_argument("--flux_dir", default="FLUX.1-Fill-dev-nf4",
                    help="FLUX.1-Fill-dev-nf4 權重目錄（本地路徑）")
    ap.add_argument("--cjk_font", default=DEF_CJK_FONT)
    ap.add_argument("--latin_font", default=DEF_LATIN_FONT)
    ap.add_argument("--cjk_font_all", action="store_true",
                    help="所有片段都用 CJK 字型渲染（預設只有含中文的片段才用）")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true",
                    help="不載入 FLUX，只輸出 mask/glyph 疊圖，用來檢查幾何切分與字型覆蓋")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for p in (args.cjk_font, args.latin_font):
        if not os.path.isfile(p):
            raise SystemExit(f"字型不存在: {p}")
    if not args.dry_run and not os.path.isdir(args.flux_dir):
        raise SystemExit(f"找不到 FLUX 權重目錄: {args.flux_dir}\n"
                         f"請掛上存放 FLUX.1-Fill-dev-nf4 的磁碟，或用 --flux_dir 指定。")

    os.makedirs(os.path.join(args.out, "plates"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "debug"), exist_ok=True)

    cjk_font = ImageFont.truetype(args.cjk_font, size=60)
    latin_font = ImageFont.truetype(args.latin_font, size=60)
    caption_font = ImageFont.truetype(args.cjk_font, size=15)

    paths = sorted(glob(os.path.join(args.plate_dir, "*.jpg")) +
                   glob(os.path.join(args.plate_dir, "*.png")))[: args.limit]
    print(f"模式={args.mode}  樣本數={len(paths)}  輸出={args.out}")

    pipe = config = None
    if not args.dry_run:
        pipe, config = load_model(args.config, args.lora, args.flux_dir)
    prompt_rng = random.Random(args.seed)

    records = []
    for i, path in enumerate(tqdm(paths)):
        stem = os.path.splitext(os.path.basename(path))[0]
        rng = random.Random(f"{args.seed}-{stem}")
        try:
            quad, full_text, chars = parse_ccpd_filename(stem)
        except Exception as e:
            print(f"  跳過 {stem}: 檔名解析失敗 {e}")
            continue

        scene = Image.open(path).convert("RGB")
        W, H = scene.size

        s, k = pick_span(args.mode, rng)
        target_text = pick_target(args.target_mode, chars, s, k, rng)
        mask_quad = make_mask_quad(quad, s, k)

        mask_np = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask_np, [mask_quad.astype(np.int32)], 255)
        mask_pil = Image.fromarray(mask_np)

        font = cjk_font if (args.cjk_font_all or is_cjk(target_text)) else latin_font
        glyph_np = draw_glyph2(font, target_text, mask_quad, W, H)
        glyph_pil = Image.fromarray((glyph_np[..., 0] * 255).astype(np.uint8))

        if args.dry_run:
            # 把 mask 邊框畫在原圖上，方便肉眼檢查七格切分是否對準字元
            gen = scene.copy()
            ImageDraw.Draw(gen).polygon([tuple(p) for p in mask_quad], outline=(255, 0, 0))
            cover = float((glyph_np[..., 0] > 0).sum()) / max(1.0, float((mask_np > 0).sum()))
            print(f"  {stem[:24]}  masked='{target_text}'  cjk={is_cjk(target_text)}  glyph 覆蓋率={cover:.3f}")
        else:
            gen, prompt = run_single(pipe, config, scene, glyph_pil, mask_pil,
                                     target_text, prompt_rng, args.seed + i)
        gen.save(os.path.join(args.out, "plates", stem + ".png"))
        debug_grid(scene, mask_pil, glyph_pil, gen,
                   f"GT={full_text}  masked[{s}:{s+k}]='{target_text}'  cjk={is_cjk(target_text)}",
                   os.path.join(args.out, "debug", stem + "_debug.jpg"),
                   caption_font=caption_font)
        records.append((stem, full_text, s, k, target_text))

    with open(os.path.join(args.out, "labels.txt"), "w") as f:
        for stem, full_text, s, k, tgt in records:
            f.write(f"{stem}.png\t{full_text}\t{s}\t{k}\t{tgt}\n")
    n_cjk = sum(1 for r in records if is_cjk(r[4]))
    print(f"\n完成 {len(records)} 張，其中 {n_cjk} 張的編輯片段含中文字元")
    print(f"輸出：{args.out}/plates, {args.out}/debug, {args.out}/labels.txt")


if __name__ == "__main__":
    main()
