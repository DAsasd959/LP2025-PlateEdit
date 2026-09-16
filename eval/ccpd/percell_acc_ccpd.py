#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-cell accuracy for a CCPD run, split into the Chinese province cell and the
alphanumeric cells.

A CCPD plate is seven cells: one province character then six alphanumerics. An
edit masks some of them and asks for a target string. Whole-plate accuracy hides
which half the model got wrong, and the aggregate per-cell figure averages them.
This scores each edited cell against its target and reports the province cell and
the alphanumeric cells separately.

    python eval/ccpd/percell_acc_ccpd.py \
        --gen_dir <output with cells.txt and labels.txt> \
        --saved_model weights/trba_ccpd_final/best_accuracy.pth \
        --dtr_root <clone of deep-text-recognition-benchmark>

Only cells the run actually masked are scored; untouched cells carry the original
plate's characters and say nothing about the model.
"""
import argparse, os, sys, re
from pathlib import Path

CHARSET = ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学")
KEEP = re.compile(r"[^0-9A-Z一-鿿]")


def norm(s):
    return KEEP.sub("", s.upper())


def is_cjk(c):
    return "一" <= c <= "鿿"


def read_run(d):
    """stem -> (cells, target, original plate text)"""
    cells = {}
    for line in open(os.path.join(d, "cells.txt"), encoding="utf-8"):
        stem, cs, tgt = line.rstrip("\n").split("\t")
        cells[stem] = ([int(c) for c in cs.split(",")], tgt)
    orig = {}
    for line in open(os.path.join(d, "labels.txt"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            orig[os.path.splitext(p[0])[0]] = norm(p[1])
    return {s: (c, t, orig[s]) for s, (c, t) in cells.items() if s in orig}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--saved_model", required=True)
    ap.add_argument("--dtr_root", required=True,
                    help="clone of clovaai/deep-text-recognition-benchmark")
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()

    sys.path.insert(0, a.dtr_root)
    import torch
    from utils import AttnLabelConverter
    from dataset import RawDataset, AlignCollate
    from model import Model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class OPT:
        pass
    o = OPT()
    o.image_folder = os.path.join(a.gen_dir, "plates")
    o.saved_model = a.saved_model
    o.workers, o.batch_size, o.batch_max_length = 4, a.batch_size, 25
    o.imgH = o.imgW = 128
    o.rgb, o.character, o.sensitive, o.PAD = True, CHARSET, False, True
    o.Transformation, o.FeatureExtraction = "TPS", "ResNet"
    o.SequenceModeling, o.Prediction = "BiLSTM", "Attn"
    o.num_fiducial, o.input_channel, o.output_channel, o.hidden_size = 20, 3, 512, 256

    conv = AttnLabelConverter(o.character)
    o.num_class = len(conv.character)
    model = torch.nn.DataParallel(Model(o)).to(dev)
    model.load_state_dict(torch.load(o.saved_model, map_location=dev, weights_only=False))
    model.eval()

    loader = torch.utils.data.DataLoader(
        RawDataset(root=o.image_folder, opt=o), batch_size=o.batch_size, shuffle=False,
        num_workers=o.workers, pin_memory=True,
        collate_fn=AlignCollate(imgH=o.imgH, imgW=o.imgW, keep_ratio_with_pad=o.PAD))

    preds = {}
    with torch.no_grad():
        for imgs, paths in loader:
            n = imgs.size(0)
            p = model(imgs.to(dev),
                      torch.LongTensor(n, o.batch_max_length + 1).fill_(0).to(dev),
                      is_train=False)
            _, idx = p.max(2)
            for pth, pr in zip(paths, conv.decode(idx, torch.IntTensor([o.batch_max_length] * n).to(dev))):
                preds[os.path.splitext(os.path.basename(pth))[0]] = norm(pr[:pr.find("[s]")])

    run = read_run(a.gen_dir)
    cn_ok = cn_n = al_ok = al_n = 0
    skipped = 0
    for stem, (cells, target, orig) in run.items():
        pr = preds.get(stem)
        # The recogniser must return seven cells for positional comparison to mean
        # anything; a short or long read is counted as wrong on every masked cell.
        if pr is None or len(orig) != 7:
            skipped += 1
            continue
        for i, c in enumerate(cells):
            if i >= len(target):
                break
            want = target[i]
            got = pr[c] if len(pr) == 7 else None
            hit = (got == want)
            if c == 0 or is_cjk(want):
                cn_n += 1; cn_ok += hit
            else:
                al_n += 1; al_ok += hit

    name = os.path.basename(a.gen_dir.rstrip("/"))
    print(f"{name}   plates = {len(run):,}" + (f"   (skipped {skipped})" if skipped else ""))
    tot_ok, tot_n = cn_ok + al_ok, cn_n + al_n
    for label, ok, n in (("all edited cells ", tot_ok, tot_n),
                         ("province cell    ", cn_ok, cn_n),
                         ("alphanumeric     ", al_ok, al_n)):
        print(f"  per-cell ACC  {label} = {ok/n:.4f}   ({ok}/{n})" if n else
              f"  per-cell ACC  {label} = —")


if __name__ == "__main__":
    main()
