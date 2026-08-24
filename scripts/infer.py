#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partial license-plate editing — inference.

A CLI port of the original eval_plate.py, whose paths were hard-coded. The
generation path itself is unchanged: the condition is the triple
[glyph, mask, source image] in that order, which is what the adapter was
trained on. Swapping that order silently produces garbage, so it is fixed here
rather than exposed as an option.

Expected input layout (one file per sample, matched by stem):

    <data_root>/
        filtered_plate/       <stem>.jpg   source plate crop
        partial_masks/        <stem>.png   region to repaint, white = edit
        partial_glyphs/       <stem>.png   rendered target glyph
        partial_labels_txt/   <stem>.txt   target text, first whitespace token

Example:

    python scripts/infer.py \
        --data_root data/test \
        --config configs/lp2025_train.yaml \
        --lora weights/lp2025_27606/adapter_model.safetensors \
        --flux_dir FLUX.1-Fill-dev-nf4 \
        --out outputs/lp2025_test
"""
import argparse
import os
import random
import sys
from glob import glob

import numpy as np
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.flux.condition import Condition                    # noqa: E402
from src.flux.generate_fill import generate_fill             # noqa: E402
from src.train.model import OminiModelFIll                   # noqa: E402

# The seven templates the adapter saw during training. One is drawn per sample;
# a single fixed template measurably narrows the output distribution.
PROMPTS = [
    "Fill the masked character '{text}' using the same color, font, and style as the surrounding text.",
    "The missing character '{text}' should match the style and color of neighboring glyphs.",
    "Generate '{text}' in the same font, size, and color as adjacent text.",
    "Replace the placeholder with '{text}', preserving the appearance of nearby letters.",
    "Complete the masked letter '{text}' with consistent color, font, and style of surrounding characters.",
    "Fill in '{text}' so it blends seamlessly with the nearby text in font and color.",
    "Render '{text}' matching the style, size, and color of the context text around it.",
]
SIZE = 512


def load_model(config_path, lora_path, flux_dir):
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
        byt5_encoder_config=None,
    )
    # PEFT stores the adapter with a "default" sub-module and a "transformer."
    # prefix that the bare transformer does not have; rename before loading.
    sd = load_file(lora_path)
    sd = {k.replace("lora_A", "lora_A.default")
           .replace("lora_B", "lora_B.default")
           .replace("transformer.", ""): v for k, v in sd.items()}
    missing = model.transformer.load_state_dict(sd, strict=False)
    n_loaded = len(sd) - len(getattr(missing, "unexpected_keys", []))
    print(f"[lora] loaded {n_loaded}/{len(sd)} tensors from {os.path.basename(lora_path)}")
    if getattr(missing, "unexpected_keys", []):
        print(f"[lora] WARNING {len(missing.unexpected_keys)} unexpected keys — "
              f"check that --config matches the checkpoint")
    pipe = model.flux_pipe
    pipe.to("cuda")
    pipe.text_encoder.to("cuda")
    return pipe, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--config", default="configs/lp2025_train.yaml")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--flux_dir", default="FLUX.1-Fill-dev-nf4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true",
                    help="also write a source|output|glyph|mask strip per sample")
    a = ap.parse_args()

    d = {k: os.path.join(a.data_root, k) for k in
         ("filtered_plate", "partial_masks", "partial_glyphs", "partial_labels_txt")}
    for k, p in d.items():
        if not os.path.isdir(p):
            raise SystemExit(f"missing input directory: {p}")

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob(os.path.join(d["partial_glyphs"], "*.png")))
    if a.limit:
        stems = stems[:a.limit]
    print(f"{len(stems)} samples from {a.data_root}")

    os.makedirs(a.out, exist_ok=True)
    pipe, config = load_model(a.config, a.lora, a.flux_dir)
    rng = random.Random(a.seed)
    skipped = []

    for stem in tqdm(stems, desc="generating"):
        img_p = os.path.join(d["filtered_plate"], stem + ".jpg")
        msk_p = os.path.join(d["partial_masks"], stem + ".png")
        gly_p = os.path.join(d["partial_glyphs"], stem + ".png")
        lbl_p = os.path.join(d["partial_labels_txt"], stem + ".txt")
        if not all(os.path.exists(p) for p in (img_p, msk_p, gly_p, lbl_p)):
            skipped.append((stem, "missing file"))
            continue
        with open(lbl_p) as f:
            line = f.readline().strip()
        if not line:
            skipped.append((stem, "empty label"))
            continue
        text = line.split()[0]

        glyph = Image.open(gly_p).resize((SIZE, SIZE)).convert("RGB")
        source = Image.open(img_p).resize((SIZE, SIZE)).convert("RGB")
        mask = Image.open(msk_p).resize((SIZE, SIZE)).convert("L")
        mask_rgb = np.stack([np.array(mask) / 255.0] * 3, axis=-1)

        cond = Condition(condition_type="word_fill",
                         condition=[np.array(glyph) / 255.0, mask_rgb, source],
                         position_delta=[0, 0])
        res = generate_fill(
            pipe, prompt=rng.choice(PROMPTS).format(text=text), conditions=[cond],
            height=SIZE, width=SIZE,
            generator=torch.Generator(device="cuda").manual_seed(a.seed),
            model_config=config.get("model", {}), default_lora=True)
        out = res.images[0]
        out.save(os.path.join(a.out, f"{stem}.png"))

        if a.compare:
            strip = Image.new("RGB", (SIZE * 4, SIZE))
            for i, im in enumerate([source, out, glyph, mask.convert("RGB")]):
                strip.paste(im, (SIZE * i, 0))
            strip.save(os.path.join(a.out, f"{stem}_compare.png"))

    print(f"wrote {len(stems) - len(skipped)} images to {a.out}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}{' ...' if len(skipped) > 5 else ''}")


if __name__ == "__main__":
    main()
