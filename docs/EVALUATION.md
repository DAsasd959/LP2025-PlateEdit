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

| | published | as defined | resize GT for FID | BICUBIC for LPIPS |
|---|---:|---:|---:|---:|
| FID | 4.78 | **4.7751** | 4.2167 | — |
| Full LPIPS | 0.081 | **0.0807** | — | 0.0792 |
| Region LPIPS | 0.062 | **0.0617** | — | 0.0650 |

Resizing the ground truth for FID costs 0.56 because the interpolation blurs both
sides toward each other. Using BICUBIC where the original used bilinear moves LPIPS
in both directions at once, which is why it is easy to miss.

Two mechanical consequences worth knowing. `pytorch_fid` cannot batch
native-resolution plates -- they differ in size, so `batch_size` must be 1; the
original script's `batch_size=1` was a requirement, not a preference. And batching
itself changes nothing: 4.2167 at batch 50 against 4.2166 at batch 1.

**Region LPIPS masks the images; it does not crop them.** Both images are
multiplied by the mask and LPIPS is taken over the whole 512x512 frame, so
everything outside the mask is identical black in both inputs. Cropping to the
mask's bounding box instead gives 0.22 on the same LP outputs, and differs by
9-10x on CCPD, because the crop removes the large identical region that otherwise
dominates. Two papers reporting "region LPIPS" can differ by 3x on identical
images. The mask keeps whatever soft edge its own resize produces -- forcing it to
NEAREST gives 0.0587, not the published 0.062. On the CCPD outputs the same two
definitions differ by 2.8x to 9.9x: province reconstruction scores 0.0174 masked
against 0.1724 cropped.

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

CCPD here edits the **Chinese province character**, which the published CCPD table
does not: that one masks only the trailing five alphanumerics, and the scripts
behind it cannot express a province mask at all — their cell grid starts at x=140
while the province occupies x=12..69. These numbers therefore have no published
counterpart to reproduce; they measure a different task on the same dataset.

They do use the same definitions as the LP tables above, so the two are internally
consistent. Do not read them against the published CCPD table: masking one Chinese
character is an easier problem than masking one to five alphanumerics, and the gap
between the two is task, not method.

### CCPD2019 stage ablation, province reconstruction, n = 1,000

| Setting | ACC ↑ | NED ↑ | FID ↓ | Full LPIPS ↓ | Region LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Stage 1 only | 0.3480 | 0.9067 | 6.0627 | 0.0664 | 0.0386 |
| Both stages | 0.9920 | 0.9989 | 4.2468 | 0.0408 | 0.0174 |
| Real photographs (ceiling) | 0.9960 | 0.9994 | — | — | — |

### CCPD2019 larger evaluation, n = 1,000 each

Spans here cross the province boundary — one contiguous run covering the Chinese
character and one or more alphanumerics.

| | FID ↓ | Full ↓ | Region ↓ | per-cell ACC ↑ | NED ↑ |
|---|---:|---:|---:|---:|---:|
| Reconstruction | 9.2547 | 0.0725 | 0.0345 | 0.8982 | 0.9607 |
| Replacement | 11.9581 | 0.1065 | 0.0609 | 0.6946 | 0.8772 |

Ground truth for these two is cut from the CCPD2019 scene images by the bounding
box of the filename's quadrilateral — the same rule `test_1000/plates` follows,
verified 6/6 against it. The 2,000 plates are spread across `ccpd_base`,
`ccpd_challenge` and other subsets, so all nine have to be searched.

Recomputed under the definitions above; the previous figures resized the ground
truth for FID and used BICUBIC for LPIPS, which moved FID by 9-20%. Ranking is
unchanged throughout, and ACC/NED are unaffected by either.

### Province balancing

Cross-province replacement on 200 samples scores 0.4400 when stage 2 trains on the
raw 8,000 CCPD images and **0.8000** on the balanced 1,777 — under a quarter of the
data, nearly double the accuracy. The raw split is 95.86% one province (皖); six
provinces never appear and sixteen appear fewer than ten times.
