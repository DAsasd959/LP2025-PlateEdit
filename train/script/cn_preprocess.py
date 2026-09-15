#!/usr/bin/env python3
"""
中文省份字元的 stage2 cache 產生器（探針實驗專用）
==================================================

與 src/data/data_ccpd.py + src/train/preprocess_partial.py 的差別只有兩點：
  1. 字元格從「後 5 碼」5 格擴成 7 格，遮罩會涵蓋省份中文字（cell 0）
  2. glyph 用含 CJK 字形的字型渲染（正黑體），原本的 TWGen7_V1.ttf 只有 254 個
     glyph、0 個中文，所以現有 cache 裡從未出現過中文監督訊號

輸出的 .pt 格式與 preprocess_partial.py 完全一致，可直接餵給 src/train/train.py。
本檔案不修改任何既有檔案，cache 也寫到新目錄。

Run from the repository root:
  python train/script/cn_preprocess.py --src data/ccpd/train_8000/plates \
      --out cache/ccpd_stage2_train --province_ratio 0.5 --limit 8000

Fonts default to font/ in this repository; override with --cjk_font / --latin_font
or the CJK_FONT / LATIN_FONT environment variables.
"""
import argparse
import gc
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.flux.pipeline_tools import prepare_text_input

from diffusers import FluxFillPipeline
import torchvision.transforms as transforms

# ============================================================
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
             "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
             "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R",
             "S", "T", "U", "V", "W", "X", "Y", "Z", "O"]
ADS = ALPHABETS[:-1] + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]

# 七格邊界，CCPD canonical 440x140（中間圓點落在 126~140 的自然間隙）
CELLS = [(12, 69), (69, 126), (140, 197), (197, 254), (254, 311), (311, 368), (368, 425)]
PAD_Y = 12

_REPO = Path(__file__).resolve().parents[2]
# The Latin face has 254 glyphs and no CJK at all, so the province cell needs a
# separate font. Both live in font/; override per-run if you have licensed copies.
CJK_FONT = os.environ.get("CJK_FONT", str(_REPO / "font" / "正黑體.ttf"))
LATIN_FONT = os.environ.get("LATIN_FONT", str(_REPO / "font" / "TWGen7_V1.ttf"))

PROMPT_TEMPLATES = [
    "Fill the masked character '{text}' using the same color, font, and style as the surrounding text.",
    "The missing character '{text}' should match the style and color of neighboring glyphs.",
    "Generate '{text}' in the same font, size, and color as adjacent text.",
    "Replace the placeholder with '{text}', preserving the appearance of nearby letters.",
    "Complete the masked letter '{text}' with consistent color, font, and style of surrounding characters.",
    "Fill in '{text}' so it blends seamlessly with the nearby text in font and color.",
    "Render '{text}' matching the style, size, and color of the context text around it.",
]

IMAGE_SIZE = 512


def is_cjk(s):
    return any("一" <= ch <= "鿿" for ch in s)


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    xs = pts[np.argsort(pts[:, 0]), :]
    left, right = xs[:2, :], xs[2:, :]
    left = left[np.argsort(left[:, 1]), :]
    right = right[np.argsort(right[:, 1]), :]
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


class CnPlateDataset(Dataset):
    """七格版 CCPD 部分遮蔽資料集；遮罩可涵蓋省份中文字"""

    def __init__(self, data_root, image_size=IMAGE_SIZE, seed=42,
                 province_ratio=1.0, max_k=2, limit=None, offset=0):
        self.root = Path(data_root)
        self.seed = seed
        self.province_ratio = province_ratio      # 遮罩涵蓋 cell 0 的機率
        self.max_k = max_k
        exts = {".jpg", ".jpeg", ".png"}
        files = sorted(f.name for f in self.root.iterdir() if f.suffix.lower() in exts)
        files = files[offset:]
        self.files = files[:limit] if limit else files

        self.cjk_font = ImageFont.truetype(CJK_FONT, size=60)
        self.latin_font = ImageFont.truetype(LATIN_FONT, size=60)

        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()])
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        print(f"[CnPlateDataset] {len(self.files)} 張  province_ratio={province_ratio}  max_k={max_k}")

    def __len__(self):
        return len(self.files)

    def parse(self, filename):
        parts = filename.split("-")
        quad = np.array([[int(v) for v in p.split("&")] for p in parts[3].split("_")],
                        dtype=np.float32)
        quad[:, 0] -= np.min(quad[:, 0])
        quad[:, 1] -= np.min(quad[:, 1])
        idx = list(map(int, parts[4].split("_")))
        chars = [PROVINCES[idx[0]], ALPHABETS[idx[1]]] + [ADS[i] for i in idx[2:7]]
        return quad, chars

    def mask_quad(self, quad, s, k):
        q = order_points(quad)
        src = np.array([[0, 0], [440, 0], [440, 140], [0, 140]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, q.astype(np.float32))
        x0, x1 = CELLS[s][0], CELLS[s + k - 1][1]
        canon = np.array([[x0, PAD_Y], [x1, PAD_Y], [x1, 140 - PAD_Y], [x0, 140 - PAD_Y]],
                         dtype=np.float32)
        return cv2.perspectiveTransform(canon.reshape(-1, 1, 2), M).reshape(4, 2).astype(np.float32)

    def find_best_font_size(self, font, text, w, h):
        if not text:
            return 60
        canvas = Image.new("L", (w * 4, h * 4), 0)
        ImageDraw.Draw(canvas).text((w, h), text, font=font.font_variant(size=100), fill=255)
        rows = np.any(np.array(canvas) > 0, axis=1)
        if not np.any(rows):
            return 60
        return max(10, min(int(100 * (h * 0.9) / (np.sum(rows) + 1e-6)), 600))

    def draw_glyph(self, text, polygon, width, height):
        if not text:
            return np.zeros((height, width, 1), dtype=np.float32)
        font = self.cjk_font if is_cjk(text) else self.latin_font
        p0, p1, p2, _ = polygon
        w = int(np.linalg.norm(p0 - p1))
        h = int(np.linalg.norm(p1 - p2))
        if w <= 1 or h <= 1:
            return np.zeros((height, width, 1), dtype=np.float32)
        nf = font.font_variant(size=self.find_best_font_size(font, text, w, h))
        big = Image.new("L", (w * 4, h * 4), 0)
        ImageDraw.Draw(big).text((w, h), text, font=nf, fill=255)
        arr = np.array(big)
        rows, cols = np.any(arr > 0, axis=1), np.any(arr > 0, axis=0)
        if not np.any(rows) or not np.any(cols):
            return np.zeros((height, width, 1), dtype=np.float32)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        resized = cv2.resize(arr[r0:r1 + 1, c0:c1 + 1], (w, h), interpolation=cv2.INTER_LINEAR)
        M = cv2.getPerspectiveTransform(
            np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32), polygon.astype(np.float32))
        return cv2.warpPerspective(resized, M, (width, height),
                                   borderValue=0)[..., None].astype(np.float32) / 255.0

    def __getitem__(self, i):
        rng = random.Random(self.seed + i)          # 固定 per-index seed，保證可重現
        fn = self.files[i]
        img = Image.open(self.root / fn).convert("RGB")
        W, H = img.size
        quad, chars = self.parse(fn)

        k = rng.randint(1, self.max_k)
        if rng.random() < self.province_ratio:
            s = 0                                    # 一定涵蓋省份中文字
        else:
            s = rng.randint(2, 7 - k)                # 只遮後 5 碼英數
        text = "".join(chars[s:s + k])
        mq = self.mask_quad(quad, s, k)

        mask_np = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask_np, [mq.astype(np.int32)], 255)
        glyph = self.draw_glyph(text, mq, W, H)

        return {
            "image": self.to_tensor(img),
            "hint": self.mask_tf(Image.fromarray(mask_np, mode="L")),
            "condition": self.mask_tf(Image.fromarray((glyph[..., 0] * 255).astype(np.uint8))),
            "description": text,
            "filename": Path(fn).stem,
        }


# ============================================================
# 以下三個函式與 src/train/preprocess_partial.py 完全一致（複製而非修改）
# ============================================================
def encode_prompt(pipe, prompts):
    with torch.no_grad():
        pe, ppe, tid = prepare_text_input(pipe, prompts)
    return pe.cpu(), ppe.cpu(), tid.cpu()


def encode_image_to_packed_tokens(pipe, images):
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    images = 2.0 * images - 1.0
    images = images.to(pipe.device, dtype=torch.bfloat16)
    with torch.no_grad():
        lat = pipe.vae.encode(images).latent_dist.sample()
        lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        packed = pipe._pack_latents(lat, lat.shape[0], lat.shape[1], lat.shape[2], lat.shape[3])
        img_ids = pipe._prepare_latent_image_ids(lat.shape[0], lat.shape[2], lat.shape[3],
                                                pipe.device, pipe.dtype)
        if packed.shape[1] != img_ids.shape[0]:
            img_ids = pipe._prepare_latent_image_ids(lat.shape[0], lat.shape[2] // 2,
                                                    lat.shape[3] // 2, pipe.device, pipe.dtype)
    return packed, img_ids


def prepare_mask_tokens_official(pipe, mask, height, width):
    B = mask.shape[0]
    mask = F.interpolate(mask, size=(height, width), mode="nearest")
    lh, lw = height // 8, width // 8
    m = mask.view(B, 1, lh, 8, lw, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64, lh, lw)
    return pipe._pack_latents(m, B, 64, lh, lw)


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CCPD plates 目錄")
    ap.add_argument("--out", required=True, help="新的 cache 輸出目錄（不會覆蓋既有 cache）")
    ap.add_argument("--flux_path", default="./weights/flux_base")
    ap.add_argument("--cjk_font", default=None, help="override the province-cell font")
    ap.add_argument("--latin_font", default=None, help="override the alphanumeric font")
    ap.add_argument("--limit", type=int, default=1000,
                    help="處理張數上限；要全部處理請給一個足夠大的值")
    ap.add_argument("--offset", type=int, default=0,
                    help="跳過前 N 張（用來續傳，避免重跑已 cache 的部分）")
    ap.add_argument("--province_ratio", type=float, default=1.0)
    ap.add_argument("--max_k", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    global CJK_FONT, LATIN_FONT
    if args.cjk_font:   CJK_FONT = args.cjk_font
    if args.latin_font: LATIN_FONT = args.latin_font
    for _f in (CJK_FONT, LATIN_FONT):
        if not os.path.isfile(_f):
            raise SystemExit(f"font not found: {_f}\n"
                             "Put it in font/, or pass --cjk_font / --latin_font.")

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"輸出目錄已存在且非空，為避免覆蓋請換一個：{args.out}")
    os.makedirs(args.out, exist_ok=True)

    print("載入 FLUX（只需要 VAE + 文字編碼器）...")
    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
    pipe.transformer = None                     # 前處理用不到 transformer，丟掉省顯存
    gc.collect(); torch.cuda.empty_cache()
    pipe.to("cuda")

    ds = CnPlateDataset(args.src, seed=args.seed, province_ratio=args.province_ratio,
                        max_k=args.max_k, limit=args.limit, offset=args.offset)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=4, shuffle=False)

    n_cjk = 0
    prompt_rng = random.Random(args.seed)
    for batch in tqdm(dl):
        prompts = [prompt_rng.choice(PROMPT_TEMPLATES).format(text=t) for t in batch["description"]]
        pe, ppe, tid = encode_prompt(pipe, prompts)

        gt = batch["image"].to("cuda")
        hint = batch["hint"].to("cuda")
        cond = batch["condition"].to("cuda")
        mw = hint[:, 0:1, :, :]
        masked = gt * (1 - mw)

        img_tokens, img_ids = encode_image_to_packed_tokens(pipe, gt)
        masked_ref_tokens, _ = encode_image_to_packed_tokens(pipe, masked)
        ref_tokens = masked_ref_tokens.clone()
        mask_tokens = prepare_mask_tokens_official(pipe, mw, IMAGE_SIZE, IMAGE_SIZE).to(
            pipe.device, dtype=torch.bfloat16)
        cond_tokens, cond_ids = encode_image_to_packed_tokens(pipe, cond)
        pixel_gt = batch["image"].clone()

        for i, fname in enumerate(batch["filename"]):
            n_cjk += int(is_cjk(batch["description"][i]))
            torch.save({
                "prompt_embeds": pe[i].cpu().clone(),
                "pooled_prompt_embeds": ppe[i].cpu().clone(),
                "text_ids": tid.cpu().clone(),
                "img_tokens": img_tokens[i].cpu().clone(),
                "img_ids": img_ids.cpu().clone(),
                "masked_ref_tokens": masked_ref_tokens[i].cpu().clone(),
                "ref_tokens": ref_tokens[i].cpu().clone(),
                "mask_tokens": mask_tokens[i].cpu().clone(),
                "cond_tokens": cond_tokens[i].cpu().clone(),
                "cond_ids": cond_ids.cpu().clone(),
                "mask_pixel": mw[i].cpu().clone(),
                "pixel_gt": pixel_gt[i].cpu().clone(),
                "description": prompts[i],
            }, os.path.join(args.out, f"{fname}.pt"))

    print(f"完成，寫出 {len(ds)} 筆到 {args.out}；其中遮罩片段含中文 = {n_cjk} 筆")


if __name__ == "__main__":
    main()
