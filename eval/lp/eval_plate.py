import os
import random
from glob import glob
import numpy as np
from PIL import Image
import torch
import yaml

from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from src.train.model import OminiModelFIll
from safetensors.torch import load_file
from tqdm import tqdm

# ----------------------- 路徑設定 -----------------------
glyph_dir = "/media/avlab/disk2/LP2024/test/partial_glyphs"
image_dir = "/media/avlab/disk2/LP2024/test/filtered_plate"
mask_dir = "/media/avlab/disk2/LP2024/test/partial_masks"
label_dir = "/media/avlab/disk2/LP2024/test/partial_labels_txt"
output_dir = "/media/avlab/76b02113-59b9-4d03-9078-45cf20e14592/0_paper/FluxText_plate/LP2024_test"

os.makedirs(output_dir, exist_ok=True)

for f in os.listdir(output_dir):
    f_path = os.path.join(output_dir, f)
    if os.path.isfile(f_path):
        os.remove(f_path)

# ----------------------- 讀取 config 與模型 -----------------------
config_path = "runs_word_fill_plate_512/20260419-180408/config.yaml"
lora_path = "runs_word_fill_plate_512/20260419-180408/ckpt/27606/adapter_model.safetensors"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

model = OminiModelFIll(
    flux_pipe_id="FLUX.1-Fill-dev-nf4",
    lora_config=config["train"]["lora_config"],
    device="cuda",
    dtype=getattr(torch, config["dtype"]),
    optimizer_config=config["train"]["optimizer"],
    model_config=config.get("model", {}),
    gradient_checkpointing=True,
    byt5_encoder_config=None,
)

state_dict = load_file(lora_path)
state_dict_new = {
    x.replace("lora_A", "lora_A.default")
     .replace("lora_B", "lora_B.default")
     .replace("transformer.", ""): v
    for x, v in state_dict.items()
}
model.transformer.load_state_dict(state_dict_new, strict=False)
pipe = model.flux_pipe
pipe.to("cuda")
pipe.text_encoder.to("cuda")


prompt_templates = [
    "Fill the masked character '{text}' using the same color, font, and style as the surrounding text.",
    "The missing character '{text}' should match the style and color of neighboring glyphs.",
    "Generate '{text}' in the same font, size, and color as adjacent text.",
    "Replace the placeholder with '{text}', preserving the appearance of nearby letters.",
    "Complete the masked letter '{text}' with consistent color, font, and style of surrounding characters.",
    "Fill in '{text}' so it blends seamlessly with the nearby text in font and color.",
    "Render '{text}' matching the style, size, and color of the context text around it."
]


glyph_paths = glob(os.path.join(glyph_dir, "*.png"))


for glyph_path in tqdm(glyph_paths, desc="Processing Images"):

    base_name = os.path.splitext(os.path.basename(glyph_path))[0]
    img_path = os.path.join(image_dir, base_name + ".jpg")
    mask_path = os.path.join(mask_dir, base_name + ".png")
    label_path = os.path.join(label_dir, base_name + ".txt")

    if not os.path.exists(img_path):
        tqdm.write(f"[缺 image] {img_path}")

    if not os.path.exists(mask_path):
        tqdm.write(f"[缺 mask] {mask_path}")

    if not os.path.exists(label_path):
        tqdm.write(f"[缺 label] {label_path}")

    # ===== 讀 label =====
    with open(label_path, "r") as f:
        line = f.readline().strip()
        if not line:
            tqdm.write(f"[跳過] 空 label：{base_name}")
            continue
        text = line.split()[0]

    # ===== 隨機 prompt =====
    template = random.choice(prompt_templates)
    prompt = template.format(text=text)

    # ===== 建 condition =====
    hint_img = Image.open(glyph_path).resize((512, 512)).convert("RGB")
    origin_img = Image.open(img_path).resize((512, 512)).convert("RGB")
    mask_img = Image.open(mask_path).resize((512, 512)).convert("L")

    # glyph
    hint_np = np.array(hint_img) 
    hint_np = hint_np / 255.0

    # mask：直接 normalize
    mask_np = np.array(mask_img) / 255.0
    mask_rgb = np.stack([mask_np] * 3, axis=-1)

    # 原圖
    origin_img = Image.open(img_path).resize((512, 512)).convert("RGB")

    # ⭐ 順序：[glyph(反色), mask, 原圖]
    condition = Condition(
        condition_type="word_fill",
        condition=[hint_np, mask_rgb, origin_img], 
        position_delta=[0, 0],
    )

    # ===== 推論 =====
    res = generate_fill(
        pipe,
        prompt=prompt,
        conditions=[condition],
        height=512,
        width=512,
        generator=torch.Generator(device="cuda"),
        model_config=config.get("model", {}),
        default_lora=True,
    )

    # 儲存結果圖
    out_img = res.images[0]
    out_path = os.path.join(output_dir, f"{base_name}.png")
    out_img.save(out_path)

    # # 拼接對照圖
    comparison = Image.new("RGB", (512 * 4, 512))
    comparison.paste(origin_img, (0, 0))
    comparison.paste(out_img, (512, 0))
    comparison.paste(hint_img, (1024, 0))
    comparison.paste(mask_img, (1536, 0))
    comparison_path = os.path.join(output_dir, f"{base_name}_compare.png")
    comparison.save(comparison_path)

print("🎉 所有圖片已完成！")
