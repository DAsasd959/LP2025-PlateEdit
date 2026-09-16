"""
CCPD Zero-shot Partial Reconstruction Inference Script
======================================================

對 CCPD 測試集中的每張車牌影像：

1. 從 CCPD 檔名解析車牌四個角點座標（Quadrilateral）以及對應的真實車牌文字（GT Text）。
2. 建立中國標準車牌的 Canonical Plane（440 × 140），並利用透視變換將真實車牌映射至標準座標系。
3. 根據預先定義的後五碼字元物理邊界，隨機選擇 1–5 個連續字元作為重建目標，計算其對應的透視四邊形遮罩區域（Mask Polygon）。
4. 生成 Binary Mask，並將目標字元以字形提示（Glyph Hint）的形式渲染至對應區域。
5. 將原始車牌影像、Mask 與 Glyph Hint 作為條件輸入 FLUX Fill 模型，完成局部文字重建（Partial Reconstruction）。

輸出：
  CCPD_test_reconstruction/
    plates/{filename}.png
    debug/{filename}_debug.jpg
    labels.txt
"""

import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from safetensors.torch import load_file

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # repository root, so `import src` works
from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from src.train.model import OminiModelFIll


# ============================================================
# 路徑設定 (請替換為你的實際路徑)
# ============================================================
PLATE_DIR = os.environ.get("PLATE_DIR", "data/ccpd/test_1000/plates")
OUTPUT_DIR = "./CCPD_test1000_clear_test"

CONFIG_PATH = os.environ.get("CONFIG_PATH", "train/config/ccpd/cn_v2_stage2_fix.yaml")
LORA_PATH = os.environ.get("LORA_PATH",
                           "weights/ccpd_stage2_ckpt_10000/adapter_model.safetensors")
FONT_PATH = os.environ.get("LATIN_FONT", "font/TWGen7_V1.ttf")

TARGET_SIZE = 512
SEED = 42

os.makedirs(os.path.join(OUTPUT_DIR, "plates"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "debug"), exist_ok=True)

# ============================================================
# Prompt templates
# ============================================================
prompt_templates = [
    "Fill the masked character '{text}' using the same color, font, and style as the surrounding text.",
    "The missing character '{text}' should match the style and color of neighboring glyphs.",
    "Generate '{text}' in the same font, size, and color as adjacent text.",
    "Replace the placeholder with '{text}', preserving the appearance of nearby letters.",
    "Complete the masked letter '{text}' with consistent color, font, and style of surrounding characters.",
    "Fill in '{text}' so it blends seamlessly with the nearby text in font and color.",
    "Render '{text}' matching the style, size, and color of the context text around it."
]

# ============================================================
# CCPD 字典對照表
# ============================================================
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "O"]
ADS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]

# ============================================================
# 解析 CCPD 檔名 (取得 Quad 與 真實文字)
# ============================================================
def parse_ccpd_filename(filename):
    parts = filename.split('-')

    # 解析 quad（原圖座標）
    quad_str = parts[3]
    pts = quad_str.split('_')
    quad = []
    for p in pts:
        x, y = map(int, p.split('&'))
        quad.append([x, y])
    quad = np.array(quad, dtype=np.float32)

    # ← 改這裡：用 min(x), min(y) 而不是 bbox x1, y1
    x_min = np.min(quad[:, 0])
    y_min = np.min(quad[:, 1])
    quad[:, 0] -= x_min
    quad[:, 1] -= y_min

    # 解析文字（不變）
    text_str = parts[4]
    indices = list(map(int, text_str.split('_')))
    province = PROVINCES[indices[0]]
    city = ALPHABETS[indices[1]]
    last_5 = "".join([ADS[idx] for idx in indices[2:7]])
    full_text = province + city + last_5

    return quad, full_text, last_5

# ============================================================
# 幾何轉換與 Mask 生成邏輯
# ============================================================
def order_points(pts):
    """抗傾斜頂點排序 (左上, 右上, 右下, 左下)"""
    pts = np.array(pts, dtype=np.float32)
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most[0], left_most[1]

    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most[0], right_most[1]

    return np.array([tl, tr, br, bl], dtype=np.float32)

def generate_dynamic_mask_and_text(
    quad,
    last_5_text,
    rng
):
    """
    使用 CCPD 標準車牌幾何

    Canonical plate:
        width  = 440
        height = 140

    後五碼:
        [140,197]
        [197,254]
        [254,311]
        [311,368]
        [368,425]

    return:
        mask_quad
        target_text
    """

    quad = order_points(quad)

    # --------------------------------------------------
    # CCPD canonical plane
    # --------------------------------------------------

    src_pts = np.array([
        [0, 0],
        [440, 0],
        [440, 140],
        [0, 140]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(
        src_pts,
        quad.astype(np.float32)
    )

    # --------------------------------------------------
    # Physical character boundaries
    # --------------------------------------------------

    char_boundaries = [
        140,
        197,
        254,
        311,
        368,
        425
    ]

    pad_y_top = 12
    pad_y_bottom = 12

    # --------------------------------------------------
    # Random mask range
    # --------------------------------------------------

    k = rng.randint(1, 5)

    s = rng.randint(
        0,
        5 - k
    )

    target_text = last_5_text[s:s+k]

    # --------------------------------------------------
    # Canonical mask box
    # --------------------------------------------------

    start_x = char_boundaries[s]
    end_x = char_boundaries[s + k]

    mask_canon = np.array([
        [start_x, pad_y_top],
        [end_x,   pad_y_top],
        [end_x,   140 - pad_y_bottom],
        [start_x, 140 - pad_y_bottom]
    ], dtype=np.float32)

    # --------------------------------------------------
    # Project back to plate image
    # --------------------------------------------------

    mask_quad = cv2.perspectiveTransform(
        mask_canon.reshape(-1, 1, 2),
        M
    ).reshape(4, 2)

    return (
        mask_quad.astype(np.float32),
        target_text
    )

# ============================================================
# Glyph rendering (保持不變)
# ============================================================
def find_best_font_size(font, text, w, h):
    if not text: return 60
    canvas = Image.new('L', (w * 4, h * 4), 0)
    ImageDraw.Draw(canvas).text((w, h), text, font=font.font_variant(size=100), fill=255)
    arr = np.array(canvas)
    rows = np.any(arr > 0, axis=1)
    if not np.any(rows): return 60
    return max(10, min(int(100 * (h * 0.9) / (np.sum(rows) + 1e-6)), 600))

def draw_glyph2(font, render_text, polygon, width, height):
    if not render_text: return np.zeros((height, width, 1), dtype=np.float32)
    p0, p1, p2, p3 = polygon
    w = int(np.linalg.norm(p0 - p1))
    h = int(np.linalg.norm(p1 - p2))
    if w <= 1 or h <= 1: return np.zeros((height, width, 1), dtype=np.float32)
    
    if w <= 1 or h <= 1:
        print("  → Early exit: w or h too small")
        return np.zeros((height, width, 1), dtype=np.float32)

    new_font = font.font_variant(size=find_best_font_size(font, render_text, w, h))
    big = Image.new('L', (w * 4, h * 4), 0)
    ImageDraw.Draw(big).text((w, h), render_text, font=new_font, fill=255)
    arr = np.array(big)

    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if not np.any(rows) or not np.any(cols): return np.zeros((height, width, 1), dtype=np.float32)

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    resized = cv2.resize(arr[r_min:r_max + 1, c_min:c_max + 1], (w, h), interpolation=cv2.INTER_LINEAR)
    M = cv2.getPerspectiveTransform(np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32), polygon.astype(np.float32))
    warped = cv2.warpPerspective(resized, M, (width, height), borderValue=0)
    
    return warped[..., None].astype(np.float32) / 255.0

# ============================================================
# 模型與 Inference (保持不變)
# ============================================================
def load_model():
    with open(CONFIG_PATH, "r") as f: config = yaml.safe_load(f)
    model = OminiModelFIll(
        flux_pipe_id=os.environ.get("FLUX_DIR", "weights/flux_base"),
        lora_config=config["train"]["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=config["train"]["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=True,
    )
    state_dict = load_file(LORA_PATH)
    state_dict_new = {x.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default").replace("transformer.", ""): v for x, v in state_dict.items()}
    model.transformer.load_state_dict(state_dict_new, strict=False)
    pipe = model.flux_pipe
    pipe.to("cuda")
    pipe.text_encoder.to("cuda")
    return pipe, config

@torch.no_grad()
def run_single(pipe, config, scene_pil, glyph_pil, mask_pil, masked_text, prompt_rng):
    hint_img = glyph_pil.resize((TARGET_SIZE, TARGET_SIZE)).convert("RGB")
    origin_img = scene_pil.resize((TARGET_SIZE, TARGET_SIZE)).convert("RGB")
    mask_img = mask_pil.resize((TARGET_SIZE, TARGET_SIZE)).convert("L")
    hint_np = np.array(hint_img) / 255.0
    mask_np = np.array(mask_img) / 255.0
    mask_rgb = np.stack([mask_np] * 3, axis=-1)
    prompt = prompt_rng.choice(prompt_templates).format(text=masked_text)

    condition = Condition(condition_type="word_fill", condition=[hint_np, mask_rgb, origin_img], position_delta=[0, 0])
    res = generate_fill(
        pipe, prompt=prompt, conditions=[condition], height=TARGET_SIZE, width=TARGET_SIZE,
        generator=torch.Generator(device="cuda"), model_config=config.get("model", {}), default_lora=True,
    )
    return res.images[0], prompt

def make_debug_image(scene_pil, mask_pil, glyph_pil, generated_pil, prompt, save_path):
    vis_size = 256
    scene_vis = scene_pil.resize((vis_size, vis_size)).convert("RGB")
    mask_vis = mask_pil.resize((vis_size, vis_size)).convert("RGB")
    glyph_vis = glyph_pil.resize((vis_size, vis_size)).convert("RGB")
    generated_vis = generated_pil.resize((vis_size, vis_size)).convert("RGB")
    canvas = Image.new("RGB", (vis_size * 4, vis_size + 80), (255, 255, 255))
    canvas.paste(scene_vis, (0, 0))
    canvas.paste(mask_vis, (vis_size, 0))
    canvas.paste(glyph_vis, (vis_size * 2, 0))
    canvas.paste(generated_vis, (vis_size * 3, 0))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype(FONT_PATH, 20)
    except: font = ImageFont.load_default()
    for i, label in enumerate(["Original", "Mask", "Glyph", "Generated (Recon)"]):
        draw.text((i * vis_size + 10, vis_size + 5), label, fill=(0, 0, 0), font=font)
    draw.text((10, vis_size + 35), f"Prompt: {prompt}", fill=(20, 20, 20), font=font)
    canvas.save(save_path)

# ============================================================
# Main Loop
# ============================================================
def main():
    print("Loading font...")
    font = ImageFont.truetype(FONT_PATH, size=60)

    print("Loading model...")
    pipe, config = load_model()

    sample_rng = random.Random(SEED)
    prompt_rng = random.Random(SEED + 1)

    plate_paths = sorted(glob(os.path.join(PLATE_DIR, "*.jpg")))
    print(f"Found {len(plate_paths)} plates for Zero-shot Reconstruction.")

    label_lines = []
    skipped = 0

    for plate_path in tqdm(plate_paths, desc="Reconstructing"):
        try:
            filename = os.path.basename(plate_path)
            stem = Path(plate_path).stem
            
            # 1. 解析 CCPD 檔名取得 Quad 與 文字
            quad, full_text, last_5_text = parse_ccpd_filename(filename)
            if quad is None or not full_text:
                skipped += 1
                continue

            # 2. 動態生成 Mask 區域與對應的目標字元
            mask_quad, target_text = generate_dynamic_mask_and_text(quad, last_5_text, sample_rng)
            if not target_text:
                skipped += 1
                continue

            scene_pil = Image.open(plate_path).convert("RGB")
            W, H = scene_pil.size

            # 3. 繪製 Binary Mask
            mask_np = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask_np, [mask_quad.astype(np.int32)], 255)
            mask_pil = Image.fromarray(mask_np, mode="L")

            # 4. 渲染 Glyph
            glyph_arr = draw_glyph2(font, target_text, mask_quad, W, H)
            glyph_pil = Image.fromarray((glyph_arr[..., 0] * 255).astype(np.uint8))

            # 5. Inference 進行局部重建
            generated, used_prompt = run_single(
                pipe, config, scene_pil, glyph_pil, mask_pil, target_text, prompt_rng
            )

            # 6. 存檔
            out_name = stem + ".png"
            generated.save(os.path.join(OUTPUT_DIR, "plates", out_name))
            
            make_debug_image(
                scene_pil, mask_pil, glyph_pil, generated, used_prompt,
                os.path.join(OUTPUT_DIR, "debug", stem + "_debug.jpg")
            )

            label_lines.append(f"{out_name} {full_text}\n")

        except Exception as e:
            tqdm.write(f"[skip] {plate_path}: {e}")
            skipped += 1
            continue

    with open(os.path.join(OUTPUT_DIR, "labels.txt"), 'w') as f:
        f.writelines(label_lines)

    print(f"\n✅ Done. Reconstructed {len(label_lines)} CCPD plates.")
    print(f"   Skipped: {skipped}")
    print(f"   Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()