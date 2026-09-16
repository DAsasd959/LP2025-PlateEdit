#!/usr/bin/env python3
"""
低顯存版本的中文車牌編輯推論（新檔，不修改任何既有檔案）
========================================================

目的：在訓練佔用大部分 GPU 的情況下，仍能跑推論驗證 checkpoint。

作法：
  - T5-XXL 文字編碼器與 CLIP 留在 CPU（bf16 約 9 GB RAM），只做一次編碼
  - Transformer 以 NF4 量化放 GPU（約 6.5 GB）
  - VAE 放 GPU（0.17 GB）
  -> GPU 需求約 7 GB，可與訓練共存（訓練佔 ~16.5 GB / 24 GB）

代價：與訓練搶 GPU 算力，兩者都會變慢；本進程若 CUDA OOM 只會殺掉自己，
      不影響已配置好記憶體的訓練進程。

用法：
  python cn_edit_lowmem.py --lora runs_cn_probe/20260809-012134/ckpt/2000/adapter_model.safetensors \
      --config runs_cn_probe/20260809-012134/config.yaml \
      --mode province --target_mode cross_province --limit 8 --out /tmp/test_out
"""
import argparse
import importlib.util
import os
import random

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from tqdm import tqdm
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))   # repository root, so `import src` works

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cce", os.path.join(HERE, "ccpd_cn_edit.py"))
CCE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CCE)      # 幾何 / glyph / 檔名解析全部沿用，確保與先前實驗可比

TARGET = 512


def build_pipe(flux_dir, config_path, lora_path):
    from diffusers import FluxFillPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print("載入 transformer（NF4 -> GPU）...")
    tr = FluxTransformer2DModel.from_pretrained(
        flux_dir, subfolder="transformer", quantization_config=nf4,
        torch_dtype=torch.bfloat16, local_files_only=True)
    tr.requires_grad_(False)

    # 掛上 PEFT adapter，key 名稱才會與 checkpoint 相符（同 model.py:105-107）
    tr = get_peft_model(tr, LoraConfig(**cfg["train"]["lora_config"]))
    sd = load_file(lora_path)
    sd = {k.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default")
           .replace("transformer.", ""): v for k, v in sd.items()}
    missing, unexpected = tr.load_state_dict(sd, strict=False)
    hit = len(sd) - len(unexpected)
    print(f"LoRA 載入：{hit}/{len(sd)} 個 key 命中   （unexpected={len(unexpected)}）")
    if hit == 0:
        raise SystemExit("LoRA 完全沒載入，key 名稱不符")

    print("載入 pipeline 其餘部分（文字編碼器留在 CPU）...")
    pipe = FluxFillPipeline.from_pretrained(
        flux_dir, transformer=tr, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.vae.to("cuda").eval()
    pipe.text_encoder.to("cpu").eval()
    pipe.text_encoder_2.to("cpu").eval()
    print(f"GPU 佔用 = {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return pipe, cfg


def encode_prompts_on_cpu(pipe, prompts):
    """先用 CPU 上的文字編碼器把所有 prompt 編碼完，之後就能把編碼器卸掉。

    這樣做的原因：pipe.device 回報的是第一個模組的裝置，文字編碼器留在 CPU 會讓
    generate_fill:205 把影像搬到 CPU，再餵給 GPU 上的 VAE 造成裝置衝突。
    """
    import gc
    out = []
    with torch.no_grad():
        for p in prompts:
            pe, ppe, _ = pipe.encode_prompt(
                prompt=p, prompt_2=None, prompt_embeds=None, pooled_prompt_embeds=None,
                device=torch.device("cpu"), num_images_per_prompt=1,
                max_sequence_length=512, lora_scale=None)
            out.append((pe.to("cuda", torch.bfloat16), ppe.to("cuda", torch.bfloat16)))
    # 卸掉文字編碼器：釋放 ~9 GB RAM，並讓 pipe.device 回報 cuda
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    gc.collect()
    print(f"prompt 編碼完成 ({len(out)} 筆)，文字編碼器已卸除")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--flux_dir", default=os.path.join(HERE, "FLUX.1-Fill-dev"))
    ap.add_argument("--plate_dir", default=CCE.DEF_PLATE_DIR)
    ap.add_argument("--mode", choices=["province", "province_city", "random"], default="province")
    ap.add_argument("--target_mode", choices=["gt", "cross_province", "latin"], default="cross_province")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "debug"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "plates"), exist_ok=True)

    cjk = ImageFont.truetype(CCE.DEF_CJK_FONT, 60)
    lat = ImageFont.truetype(CCE.DEF_LATIN_FONT, 60)
    cap = ImageFont.truetype(CCE.DEF_CJK_FONT, 15)

    pipe, cfg = build_pipe(args.flux_dir, args.config, args.lora)
    from src.flux.condition import Condition
    from src.flux.generate_fill import generate_fill

    from glob import glob
    import cv2
    paths = sorted(glob(os.path.join(args.plate_dir, "*.jpg")))[: args.limit]

    # ---- 第一輪：只算幾何與 prompt（CPU），不動 GPU ----
    prng = random.Random(args.seed)
    plan = []
    for i, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        rng = random.Random(f"{args.seed}-{stem}")
        quad, full, chars = CCE.parse_ccpd_filename(stem)
        s, k = CCE.pick_span(args.mode, rng)
        tgt = CCE.pick_target(args.target_mode, chars, s, k, rng)
        plan.append((stem, p, quad, full, chars, s, k, tgt,
                     prng.choice(CCE.PROMPTS).format(text=tgt)))

    embeds = encode_prompts_on_cpu(pipe, [x[8] for x in plan])

    recs = []
    for i, (stem, p, quad, full, chars, s, k, tgt, prompt) in enumerate(tqdm(plan)):
        scene = Image.open(p).convert("RGB")
        W, H = scene.size
        mq = CCE.make_mask_quad(quad, s, k)

        mnp = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mnp, [mq.astype(np.int32)], 255)
        mask_pil = Image.fromarray(mnp)
        font = cjk if CCE.is_cjk(tgt) else lat
        g = CCE.draw_glyph2(font, tgt, mq, W, H)
        glyph_pil = Image.fromarray((g[..., 0] * 255).astype(np.uint8))

        hint = np.array(glyph_pil.resize((TARGET, TARGET)).convert("RGB")) / 255.0
        origin = scene.resize((TARGET, TARGET)).convert("RGB")
        m = np.array(mask_pil.resize((TARGET, TARGET)).convert("L")) / 255.0
        cond = Condition(condition_type="word_fill",
                         condition=[hint, np.stack([m] * 3, -1), origin], position_delta=[0, 0])
        pe, ppe = embeds[i]
        with torch.no_grad():
            # 傳入預先算好的 embedding，prompt=None -> encode_prompt 直接回傳不需編碼器
            res = generate_fill(pipe, prompt=None, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                                conditions=[cond], height=TARGET, width=TARGET,
                                generator=torch.Generator(device="cuda").manual_seed(args.seed + i),
                                model_config=cfg.get("model", {}), default_lora=True,
                                num_inference_steps=args.steps)
        gen = res.images[0]
        gen.save(os.path.join(args.out, "plates", stem + ".png"))
        CCE.debug_grid(scene, mask_pil, glyph_pil, gen,
                       f"GT={full}  masked[{s}:{s+k}]='{tgt}'",
                       os.path.join(args.out, "debug", stem + "_debug.jpg"), caption_font=cap)
        recs.append((stem, full, s, k, tgt))

    with open(os.path.join(args.out, "labels.txt"), "w") as f:
        for stem, full, s, k, t in recs:
            f.write(f"{stem}.png\t{full}\t{s}\t{k}\t{t}\n")
    print(f"完成 {len(recs)} 張 -> {args.out}")


if __name__ == "__main__":
    main()
