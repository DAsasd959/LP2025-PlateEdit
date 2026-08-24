#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text accuracy of the edited plates: ACC and NED via a TRBA recogniser.

Ground truth comes from the filename, as in the original evaluation:
    1004_1_AXS9956.png  ->  AXS9956
i.e. the last underscore-separated field of the stem.

Normalisation matches the original eval_ocr.py exactly. It matters: without
stripping the separator the recogniser's '·' output never equals a GT string
that has no separator, and ACC collapses to near zero.

    upper-case  ->  drop '-', '·', ' '  ->  keep only A-Z and 0-9

ACC = fraction of images whose full string matches.
NED = mean(1 - edit_distance / max(len(pred), len(gt))); 1.0 when both empty.

Requires the deep-text-recognition-benchmark code on PYTHONPATH:

    python scripts/eval_ocr.py \
        --image_folder outputs/lp2025_test \
        --saved_model weights/trba_lp2025/best_accuracy.pth \
        --dtr_root /path/to/deep-text-recognition-benchmark
"""
import argparse
import os
import re
import sys

import torch
import torch.utils.data

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
KEEP = re.compile(r"[^A-Z0-9]")


def norm(s):
    return KEEP.sub("", s.upper().replace("·", "").replace("-", "").replace(" ", ""))


def gt_from_name(stem):
    return norm(stem.split("_")[-1])


def edit(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1,
                                     prev + (a[i - 1] != b[j - 1]))
    return dp[n]


class OPT:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--saved_model", required=True)
    ap.add_argument("--dtr_root", required=True,
                    help="clone of deep-text-recognition-benchmark")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--imgH", type=int, default=128)
    ap.add_argument("--imgW", type=int, default=128)
    ap.add_argument("--out", default=None, help="optional per-image TSV")
    a = ap.parse_args()

    sys.path.insert(0, a.dtr_root)
    from dataset import AlignCollate, RawDataset
    from model import Model
    from utils import AttnLabelConverter

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    o = OPT()
    o.image_folder, o.saved_model = a.image_folder, a.saved_model
    o.workers, o.batch_size, o.batch_max_length = 4, a.batch_size, 25
    o.imgH, o.imgW, o.rgb, o.character = a.imgH, a.imgW, True, CHARSET
    o.PAD = True
    o.Transformation, o.FeatureExtraction = "TPS", "ResNet"
    o.SequenceModeling, o.Prediction = "BiLSTM", "Attn"
    o.num_fiducial, o.input_channel, o.output_channel, o.hidden_size = 20, 3, 512, 256
    conv = AttnLabelConverter(o.character)
    o.num_class = len(conv.character)

    model = torch.nn.DataParallel(Model(o)).to(dev)
    model.load_state_dict(torch.load(o.saved_model, map_location=dev, weights_only=False))
    model.eval()

    coll = AlignCollate(imgH=o.imgH, imgW=o.imgW, keep_ratio_with_pad=o.PAD)
    dl = torch.utils.data.DataLoader(
        RawDataset(root=o.image_folder, opt=o), batch_size=o.batch_size,
        shuffle=False, num_workers=o.workers, collate_fn=coll, pin_memory=True)

    rows = []
    with torch.no_grad():
        for imgs, paths in dl:
            imgs = imgs.to(dev)
            n = imgs.size(0)
            length = torch.IntTensor([o.batch_max_length] * n).to(dev)
            tfp = torch.LongTensor(n, o.batch_max_length + 1).fill_(0).to(dev)
            _, idx = model(imgs, tfp, is_train=False).max(2)
            for p, s in zip(paths, conv.decode(idx, length)):
                stem = os.path.splitext(os.path.basename(p))[0]
                if stem.endswith("_compare"):        # skip the visual strips
                    continue
                pred = norm(s[:s.find("[s]")] if "[s]" in s else s)
                rows.append((stem, gt_from_name(stem), pred))

    if not rows:
        raise SystemExit("no images scored — check --image_folder")
    acc = sum(p == g for _, g, p in rows) / len(rows)
    ned = sum(1.0 if not p and not g else 1 - edit(p, g) / max(len(p), len(g))
              for _, g, p in rows) / len(rows)
    print(f"n = {len(rows)}")
    print(f"ACC = {acc:.4f}")
    print(f"NED = {ned:.4f}")

    if a.out:
        with open(a.out, "w") as f:
            for stem, g, p in rows:
                f.write(f"{stem}\t{g}\t{p}\t{'OK' if p == g else 'X'}\n")
        print(f"per-image -> {a.out}")


if __name__ == "__main__":
    main()
