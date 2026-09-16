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
NEAREST gives 0.0587, not the published 0.062.

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

### CCPD2019 stage ablation, province reconstruction, n = 1,000

| Setting | ACC ↑ | NED ↑ | FID ↓ | Full LPIPS ↓ | Region LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Stage 1 only | 0.3480 | 0.9067 | 5.4295 | 0.0604 | 0.0389 |
| Both stages | 0.9920 | 0.9989 | 3.5421 | 0.0345 | 0.0183 |
| Real photographs (ceiling) | 0.9960 | 0.9994 | — | — | — |

### CCPD2019 larger evaluation, n = 1,000 each

| | FID ↓ | Full ↓ | Region ↓ | per-cell ACC ↑ | NED ↑ |
|---|---:|---:|---:|---:|---:|
| Reconstruction | 8.5290 | 0.0705 | 0.0365 | 0.8982 | 0.9607 |
| Replacement | 10.9169 | 0.1044 | 0.0628 | 0.6946 | 0.8772 |

### Province balancing

Cross-province replacement on 200 samples scores 0.4400 when stage 2 trains on the
raw 8,000 CCPD images and **0.8000** on the balanced 1,777 — under a quarter of the
data, nearly double the accuracy. The raw split is 95.86% one province (皖); six
provinces never appear and sixteen appear fewer than ten times.
