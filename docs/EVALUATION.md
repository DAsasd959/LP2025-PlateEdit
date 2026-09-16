# Evaluation

## Commands

```bash
# image fidelity -- FID, full-frame LPIPS, masked-region LPIPS
python eval/eval_image.py --gen_dir outputs/lp_recon \
    --real_dir data/lp/test/filtered_plate --mask_dir data/lp/test/partial_masks

# text accuracy -- needs a clone of deep-text-recognition-benchmark
python eval/eval_ocr.py --image_folder outputs/lp_recon \
    --saved_model weights/trba_lp2025/best_accuracy.pth --dtr_root <clone>

# CCPD, per-cell accuracy and region LPIPS
python eval/ccpd/eval_ccpd.py
python eval/ccpd/region_lpips_ccpd.py --gen_dir outputs/ccpd_recon \
    --plate_dir data/ccpd/test_1000/plates
```

`eval/ccpd/LEGACY.txt` lists scripts kept only to document how the published
numbers were originally computed; their paths point at drives that no longer exist.

## The definitions the published numbers use

All three were recovered by measurement, reproducing the published values on the
same 3,258 images. `eval/eval_image.py` implements them; `--fid_resize_gt` restores
the wrong behaviour if you want to see the difference.

**FID does not touch the ground truth. LPIPS resizes it, bilinear.** The thesis is
explicit and the asymmetry is deliberate: FID compares distributions, so it uses
"the generated 512x512 images and the ground-truth images in their original
resolutions without extra preprocessing"; LPIPS compares pixel for pixel, so it
"involves explicitly resizing the ground-truth images to 512x512". Measured on the
released outputs:

| | published | measured |
|---|---:|---:|
| FID | 4.78 | **4.7751** |
| Full LPIPS | 0.081 | **0.0807** |
| Region LPIPS | 0.062 | **0.0617** |

One mechanical consequence: `pytorch_fid` cannot batch native-resolution plates,
since they differ in size, so FID runs at `batch_size=1`. That is a requirement,
not a tuning choice, and batching changes nothing numerically.

**Region LPIPS masks the images; it does not crop them.** Both images are
multiplied by the mask and LPIPS is taken over the whole 512x512 frame, so
everything outside the mask is identical black in both inputs. Cropping to the
mask's bounding box instead gives 0.22 on the same LP outputs, and differs by
9-10x on CCPD, because the crop removes the large identical region that otherwise
dominates. Two papers reporting "region LPIPS" can differ by 3x on identical
images. The mask keeps whatever soft edge its own resize produces. On the CCPD
outputs the two definitions differ by 2.8x to 9.9x: province reconstruction scores
0.0174 masked against 0.1724 cropped.

**FID is not comparable across sample counts.** The same CCPD model scores 7.30 at
n=200 and 3.85 at n=1,000. Only compare runs of equal size.

## Recogniser ceiling

Manual review of 409 CCPD items found the recogniser missed 129 correct edits and
wrongly credited none, so every reported ACC is a lower bound. On LP the TRBA
evaluator scores 0.9996 on the same real photographs used as sources, so headroom
there is negligible.

## Full results

### LP-2025 ablation, n = 3,258, native resolution

| Setting | FID ↓ | Full LPIPS ↓ | Region LPIPS ↓ | ACC ↑ | NED ↑ |
|---|---:|---:|---:|---:|---:|
| Ours (both stages) | 4.7751 | 0.0807 | 0.0617 | **0.8076** | **0.9548** |
| Real data only | 5.9240 | 0.0871 | 0.0663 | 0.6378 | 0.9163 |
| Synthetic only | 8.2500 | 0.1228 | 0.0997 | 0.8082 | 0.9344 |
| No ODM loss | 3.4757 | 0.0785 | 0.0630 | 0.6676 | 0.9160 |

Against the published 4.78 / 0.081 / 0.062 / 0.952 — every image metric reproduces.
ACC and NED are averaged per image rather than over a distribution, so they drift
slightly with each generation run; FID matching to two decimals is what shows the
regenerated outputs are equivalent to the published ones.

**The no-ODM row beats the full method on FID and Full LPIPS while losing 14 points
of ACC.** Perceptual and distributional metrics cannot see whether the characters
are the *right* characters, so on their own they would select the worse model. The
mechanism has not been checked against the images themselves — look at
`LP2024_no_odm` before drawing a conclusion from this row.

Both tables below mask the **Chinese province character**, which the published
CCPD table never does: it masks only the trailing five alphanumerics, and the
scripts behind it cannot express a province mask at all — their cell grid starts at
x=140 while the province occupies x=12..69. There is no published counterpart to
either, and they should not be read against it.

They are also not comparable to each other. The stage ablation masks **one cell,
the province character alone**, in all 1,000 samples. The larger evaluation masks
**two or three contiguous cells crossing the province boundary**, so every sample
changes the Chinese character together with one or two alphanumerics — `沪E`,
`鄂X2`, `晋KQ`. Two or three cells is a harder problem than one, which is most of
why its scores are worse.

The definitions are the same as the LP tables above, so all of these are consistent
with those.

### CCPD2019 stage ablation — province character only, 1 cell, n = 1,000

| Setting | ACC ↑ | NED ↑ | FID ↓ | Full LPIPS ↓ | Region LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Stage 1 only | 0.3480 | 0.9067 | 6.0627 | 0.0664 | 0.0386 |
| Both stages | 0.9920 | 0.9989 | 4.2468 | 0.0408 | 0.0174 |
| Real photographs (ceiling) | 0.9960 | 0.9994 | — | — | — |

### CCPD2019 mixed spans — province + alphanumerics, 2-3 cells, n = 1,000 each

| | FID ↓ | Full ↓ | Region ↓ | per-cell ACC ↑ | NED ↑ |
|---|---:|---:|---:|---:|---:|
| Reconstruction | 9.2547 | 0.0725 | 0.0345 | 0.8982 | 0.9607 |
| Replacement | 11.9581 | 0.1065 | 0.0609 | 0.6946 | 0.8772 |

Replacement targets every masked cell, and its province targets are scheduled
rather than sampled, so all 31 appear 32 or 33 times each. Sampling them at random
leaves the rare ones almost untouched — `--prov_plan` exists for that reason.

**Region LPIPS by part of the mask.** Masks here cover the province cell and one
or two alphanumerics together, and a single figure averages the two. Splitting the
same mask — cell 0 alone, then the alphanumeric cells alone — scores each part of
the same generated image separately (`eval/ccpd/region_split_ccpd.py`; only Region
LPIPS decomposes this way, FID and full-image LPIPS cannot):

| | whole mask | Chinese cell | alphanumeric cells |
|---|---:|---:|---:|
| Reconstruction | 0.0345 | **0.0151** | **0.0200** |
| Replacement | 0.0609 | **0.0214** | **0.0406** |

The Chinese character is the easier half in both, and replacement costs it far
less: 0.0151 → 0.0214 for Chinese against 0.0200 → 0.0406 for alphanumerics. The
province has 31 candidates whose shapes differ sharply; the alphanumeric pool has
34 and contains 8/B, 0/O, 5/S. The two halves also sum to the whole — 0.0151 +
0.0200 = 0.0351 against 0.0345 measured — because the sub-masks do not overlap and
everything outside them is zero in both images.

Ground truth for these two is cut from the CCPD2019 scene images by the bounding
box of the filename's quadrilateral — the same rule `test_1000/plates` follows,
verified 6/6 against it. The 2,000 plates are spread across `ccpd_base`,
`ccpd_challenge` and other subsets, so all nine have to be searched.

ACC and NED never involve the ground-truth *image* -- the recogniser reads the
generated plate and the target string comes from the filename -- so they are
unaffected by how the ground truth is prepared.

### Province balancing

Cross-province replacement on 200 samples scores 0.4400 when stage 2 trains on the
raw 8,000 CCPD images and **0.8000** on the balanced 1,777 — under a quarter of the
data, nearly double the accuracy. The raw split is 95.86% one province (皖); six
provinces never appear and sixteen appear fewer than ten times.

That comparison draws its target provinces at random, so they are not evenly
covered and the figures lean toward whichever provinces came up often. The
replacement row above is the better basis for anything about province coverage:
same question, 1,000 samples, all 31 provinces equally represented.
