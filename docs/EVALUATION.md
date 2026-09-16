# Evaluation

## Commands

```bash
# image fidelity -- FID, full-frame LPIPS, masked-region LPIPS
python eval/lp/eval_image.py --gen_dir outputs/lp_recon \
    --real_dir data/lp/test/filtered_plate --mask_dir data/lp/test/partial_masks

# text accuracy -- needs a clone of deep-text-recognition-benchmark
python eval/lp/eval_ocr.py --image_folder outputs/lp_recon \
    --saved_model weights/trba_lp2025/best_accuracy.pth --dtr_root <clone>

# CCPD, per-cell accuracy and region LPIPS
python eval/ccpd/eval_ccpd.py
python eval/ccpd/region_lpips_ccpd.py --gen_dir outputs/ccpd_recon \
    --plate_dir data/ccpd/test_1000/plates
```

`eval/ccpd/LEGACY.txt` lists scripts kept only to document how the published
numbers were originally computed; their paths point at drives that no longer exist.

## Three things that change the numbers

**Region LPIPS definition.** The published numbers multiply both images by the
mask and score the whole 512x512 frame. Cropping to the mask's bounding box
instead gives 0.22 rather than 0.062 on LP, and differs by 9-10x on CCPD. Same
images, same model.

**FID is not comparable across sample counts.** The same CCPD model scores 7.30 at
n=200 and 3.85 at n=1,000. Only compare runs of equal size.

**512² versus native resolution.** LP plates have a median size of 326x216 and
75.8% are smaller than 512 on both sides. Resizing to 512² magnifies them and
squares the aspect ratio: native-resolution FID runs about 28% higher and LPIPS
2-4% lower. The ranking between methods does not change.

## Recogniser ceiling

Manual review of 409 CCPD items found the recogniser missed 129 correct edits and
wrongly credited none, so every reported ACC is a lower bound. On LP the TRBA
evaluator scores 0.9996 on the same real photographs used as sources, so headroom
there is negligible.

## Full results

### LP-2025 ablation, n = 3,258, native resolution

| Setting | FID ↓ | Full LPIPS ↓ | Region LPIPS ↓ | ACC ↑ | NED ↑ |
|---|---:|---:|---:|---:|---:|
| Ours (both stages) | 5.3931 | 0.0768 | 0.0625 | 0.8076 | 0.9548 |
| Real data only | 6.5637 | 0.0853 | 0.0683 | 0.6378 | 0.9163 |
| Synthetic only | 8.7490 | 0.1196 | 0.1052 | 0.8082 | 0.9344 |
| No ODM loss | — | — | — | 0.6676 | 0.9160 |

At 512² the full method measures FID 4.2167 / 0.0792 / 0.0650 against the
published 4.78 / 0.081 / 0.062.

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
