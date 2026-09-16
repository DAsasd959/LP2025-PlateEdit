#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edit the text on any licence plate photograph, in a browser.

    python app.py --lora weights/lp_stage2_ckpt_27606/adapter_model.safetensors

Upload a photograph, click the four corners of the plate's text, type what it says
and what you want it to say. Nothing about this is tied to LP-2025 or CCPD — the
corners are what locate the text, so any plate works.

The model is loaded on the first edit, not at startup, which takes a couple of
minutes and about 18 GB of VRAM. Everything before that runs on CPU, so the mask
and glyph preview costs nothing and is worth checking before you spend a
generation on it.

    --mode lp     any plate, arbitrary length. Characters are assumed evenly
                  spaced across the marked region, and the target must have the
                  same length as the span it replaces.
    --mode ccpd   Chinese plates: seven cells, one province character then six
                  alphanumerics. Pick cells rather than a character range.
"""
import argparse
import os
import sys

import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image, ImageFont
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.flux.condition import Condition            # noqa: E402
from src.flux.generate_fill import generate_fill    # noqa: E402
from src.train.model import OminiModelFIll          # noqa: E402
from eval.lp.prepare_sample import order_points, span_quad, draw_glyph  # noqa: E402

SIZE = 512
PROMPT = ("Fill the masked character '{text}' using the same color, font, "
          "and style as the surrounding text.")
_state = {"pipe": None, "config": None}


def load_model(config_path, lora_path, flux_dir):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = OminiModelFIll(
        flux_pipe_id=flux_dir, lora_config=config["train"]["lora_config"],
        device="cuda", dtype=getattr(torch, config["dtype"]),
        optimizer_config=config["train"]["optimizer"],
        model_config=config.get("model", {}), gradient_checkpointing=False)
    sd = load_file(lora_path)
    sd = {k.replace("lora_A", "lora_A.default")
           .replace("lora_B", "lora_B.default")
           .replace("transformer.", ""): v for k, v in sd.items()}
    model.transformer.load_state_dict(sd, strict=False)
    pipe = model.flux_pipe
    pipe.to("cuda")
    pipe.text_encoder.to("cuda")
    return pipe, config


def mark(img, corners):
    """Draw the corners picked so far, and the quadrilateral once there are four."""
    if img is None:
        return None
    out = np.array(img).copy()
    for i, (x, y) in enumerate(corners):
        cv2.circle(out, (int(x), int(y)), 6, (255, 40, 40), -1)
        cv2.putText(out, str(i + 1), (int(x) + 9, int(y) - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 40, 40), 2)
    if len(corners) == 4:
        q = order_points(np.float32(corners)).astype(np.int32)
        cv2.polylines(out, [q], True, (40, 220, 90), 2)
    return out


def on_click(img, corners, evt: gr.SelectData):
    corners = list(corners or [])
    if len(corners) >= 4:
        corners = []
    corners.append([evt.index[0], evt.index[1]])
    n = len(corners)
    msg = (f"corner {n} of 4" if n < 4 else
           "four corners set — click again to start over")
    return mark(img, corners), corners, msg


def clear(img):
    return mark(img, []), [], "click the four corners of the plate's text"


def preview(img, corners, text, span, target, mode, cells, font_path):
    """Mask and glyph, on CPU. Same geometry the generator will use."""
    if img is None:
        raise gr.Error("upload a photograph first")
    if len(corners or []) != 4:
        raise gr.Error("mark all four corners of the text region")
    text = (text or "").strip().upper()
    if not text:
        raise gr.Error("type what the plate currently says")
    quad = order_points(np.float32(corners))
    W, H = img.size
    font = ImageFont.truetype(font_path, 60)

    if mode == "ccpd":
        idx = sorted({int(c) for c in (cells or "").replace(" ", "").split(",") if c != ""})
        if not idx or min(idx) < 0 or max(idx) > 6:
            raise gr.Error("cells must be numbers 0-6, e.g. 0,5,6")
        if len(text) != 7:
            raise gr.Error(f"a Chinese plate has seven characters; got {len(text)}")
        start, end = min(idx), max(idx) + 1
        n_chars = 7
    else:
        try:
            start, end = (int(v) for v in (span or "").split(":"))
        except ValueError:
            raise gr.Error('span looks like "3:6" — the characters to replace')
        if not 0 <= start < end <= len(text):
            raise gr.Error(f"span {span} is outside 0:{len(text)}")
        n_chars = len(text)

    tgt = (target or "").strip().upper() or text[start:end]
    if len(tgt) != end - start:
        raise gr.Error(f"target {tgt!r} has {len(tgt)} characters, the span has "
                       f"{end - start}. Every character shifts otherwise, and the "
                       f"untouched ones stop lining up with the photograph.")

    sq = span_quad(quad, n_chars, start, end)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [sq.astype(np.int32)], 255)
    # The whole string is rendered across the whole region and then cut by the
    # mask, so each character keeps the width of its untouched neighbours.
    full = text[:start] + tgt + text[end:]
    glyph = np.where(mask > 127, draw_glyph(font, full, quad, W, H), 0)
    return (Image.fromarray(mask), Image.fromarray(glyph),
            f"replacing {text[start:end]!r} with {tgt!r}")


def run(img, corners, text, span, target, mode, cells, seed, font_path, args):
    mask, glyph, note = preview(img, corners, text, span, target, mode, cells, font_path)
    if _state["pipe"] is None:
        gr.Info("loading the model — a couple of minutes, once per session")
        _state["pipe"], _state["config"] = load_model(args.config, args.lora, args.flux_dir)
    pipe, config = _state["pipe"], _state["config"]

    src = img.convert("RGB").resize((SIZE, SIZE))
    g = glyph.convert("RGB").resize((SIZE, SIZE))
    m = mask.resize((SIZE, SIZE))
    cond = Condition(condition_type="word_fill",
                     condition=[np.array(g) / 255.0,
                                np.stack([np.array(m) / 255.0] * 3, axis=-1), src],
                     position_delta=[0, 0])
    shown = note.split("with ")[-1].strip("'\"")
    res = generate_fill(pipe, prompt=PROMPT.format(text=shown), conditions=[cond],
                        height=SIZE, width=SIZE,
                        generator=torch.Generator(device="cuda").manual_seed(int(seed)),
                        model_config=config.get("model", {}), default_lora=True)
    return res.images[0], mask, glyph, note


def build(args):
    with gr.Blocks(title="StylePlate", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# StylePlate\n"
            "Mark the four corners of the plate's text, say what it reads and what "
            "it should read. Any plate — the corners are what locate the text, so "
            "this is not tied to either dataset.")
        corners = gr.State([])
        with gr.Row():
            with gr.Column():
                src = gr.Image(label="photograph — click four corners", type="pil",
                               interactive=True)
                marked = gr.Image(label="corners", interactive=False)
                hint = gr.Markdown("upload a photograph, then click the four corners "
                                   "of the plate's text")
                reset = gr.Button("clear corners")
            with gr.Column():
                mode = gr.Radio(["lp", "ccpd"], value=args.mode, label="plate type",
                                info="lp: any plate, arbitrary length · "
                                     "ccpd: Chinese, seven cells")
                text = gr.Textbox(label="what the plate says now",
                                  placeholder="RBE8700   or   皖AMZ250")
                span = gr.Textbox(label="characters to replace (lp)", value="0:2",
                                  info='a range, "3:6" is the 4th to 6th character')
                cells = gr.Textbox(label="cells to replace (ccpd)", value="0,5,6",
                                   info="0 is the province, 1-6 the alphanumerics")
                target = gr.Textbox(label="replace them with",
                                    info="same number of characters as the span")
                seed = gr.Number(label="seed", value=0, precision=0)
                with gr.Row():
                    prev_btn = gr.Button("preview mask + glyph (cpu)")
                    go = gr.Button("edit", variant="primary")
        with gr.Row():
            out = gr.Image(label="result", interactive=False)
            mask_v = gr.Image(label="mask", interactive=False)
            glyph_v = gr.Image(label="glyph condition", interactive=False)
        note = gr.Markdown()

        src.select(on_click, [src, corners], [marked, corners, hint])
        src.change(clear, [src], [marked, corners, hint])
        reset.click(clear, [src], [marked, corners, hint])
        f = gr.State(args.font)
        prev_btn.click(preview, [src, corners, text, span, target, mode, cells, f],
                       [mask_v, glyph_v, note])
        go.click(lambda *a: run(*a, args), [src, corners, text, span, target, mode,
                                            cells, seed, f], [out, mask_v, glyph_v, note])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", default="weights/lp_stage2_ckpt_27606/adapter_model.safetensors")
    ap.add_argument("--config", default="train/config/lp/lp2025_train.yaml")
    ap.add_argument("--flux_dir", default="weights/flux_base")
    ap.add_argument("--font", default="font/TWGen7_V1.ttf")
    ap.add_argument("--mode", choices=["lp", "ccpd"], default="lp")
    ap.add_argument("--share", action="store_true", help="public gradio link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if a.mode == "ccpd" and a.font == "font/TWGen7_V1.ttf":
        a.font = "font/正黑體.ttf"
    for p in (a.font,):
        if not os.path.isfile(p):
            raise SystemExit(f"font not found: {p}")
    build(a).launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
