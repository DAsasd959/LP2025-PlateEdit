#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify a LP2025 checkout before spending GPU hours on it.

Every failure this catches is one that otherwise shows up late and quietly:
a split whose glyphs and sources disagree produces silently fewer samples, and
a missing weight file surfaces only after the base model has finished loading.

    python scripts/check_data.py --root /path/to/LP2025 --weights weights/

Exit status is non-zero if anything required is missing, so it can gate a run.
"""
import argparse
import os
import sys

NEED = ["filtered_plate", "partial_glyphs", "partial_masks", "partial_labels_txt"]
EXPECT = {"train": 2569, "val": 620, "test": 3258}
WEIGHTS = [
    ("lp2025_27606/adapter_model.safetensors", "diffusion adapter", True),
    ("trba_lp2025/best_accuracy.pth", "recogniser for eval_ocr", False),
    ("epoch_100.pt", "ODM loss encoder, training only", False),
]


def stems(d):
    return {os.path.splitext(f)[0] for f in os.listdir(d)} if os.path.isdir(d) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LP2025 archive root")
    ap.add_argument("--weights", default="weights")
    a = ap.parse_args()

    # The test split sits at the archive root; train and val sit under data/.
    splits = {"train": os.path.join(a.root, "data", "train"),
              "val": os.path.join(a.root, "data", "val"),
              "test": a.root}

    bad = []
    print(f"{'split':7}{'source':>9}{'glyph':>8}{'mask':>8}{'label':>8}"
          f"{'usable':>9}{'expected':>10}")
    for name, root in splits.items():
        got = {d: stems(os.path.join(root, d)) for d in NEED}
        absent = [d for d, v in got.items() if v is None]
        if absent:
            print(f"{name:7}  missing directory: {', '.join(absent)}   under {root}")
            bad.append(f"{name}: missing {absent}")
            continue
        usable = (got["partial_glyphs"] & got["partial_masks"]
                  & got["partial_labels_txt"] & got["filtered_plate"])
        exp = EXPECT[name]
        print(f"{name:7}{len(got['filtered_plate']):>9,}{len(got['partial_glyphs']):>8,}"
              f"{len(got['partial_masks']):>8,}{len(got['partial_labels_txt']):>8,}"
              f"{len(usable):>9,}{exp:>10,}")
        if len(usable) != exp:
            short = got["partial_glyphs"] - got["filtered_plate"]
            bad.append(f"{name}: {len(usable)} usable, expected {exp}"
                       + (f"; {len(short)} glyphs have no source image" if short else ""))

    print()
    for rel, what, required in WEIGHTS:
        p = os.path.join(a.weights, rel)
        if os.path.exists(p):
            print(f"  found    {rel}  ({os.path.getsize(p) / 1e6:.0f} MB) — {what}")
        elif required:
            print(f"  MISSING  {rel} — {what}")
            bad.append(f"weights: {rel} absent")
        else:
            print(f"  absent   {rel} — {what} (optional)")

    if bad:
        print("\nProblems:")
        for b in bad:
            print(f"  - {b}")
        sys.exit(1)
    print("\nEverything required is present.")


if __name__ == "__main__":
    main()
