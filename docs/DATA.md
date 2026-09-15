# Data layout

Everything below is relative to the repository root. `data/` and `cache/` are
git-ignored; create them yourself.

```
data/
├── lp/
│   ├── train/  val/  test/
│   │   ├── filtered_plate/        plate crops
│   │   ├── partial_masks/         region to repaint, white on black
│   │   ├── partial_glyphs/        target text warped into that region
│   │   └── partial_labels_txt/    target text + the region's four corners
│   └── synth/{train,val}/         i_s/ mask_s/ i_s_bbox/ t_b/
└── ccpd/
    ├── ccpd_base/                 full-scene images from CCPD2019
    ├── crops/                     plate crops (train/script/crop_plates_from_base.py)
    ├── train_8000/ valid_3000/ test_1000/
    └── balanced_train/            province-balanced (train/script/cn_composite.py)

cache/
├── lp_stage1_{train,val}     20,000 + 1,000    ~185 GB
├── lp_stage2_{train,val}      2,569 +   620    ~29 GB
├── ccpd_stage1_{train,val}   20,000 +   500    ~181 GB
└── ccpd_stage2_{train,val}    1,777 +    90
```

## Counts

| Split | LP-2025 | CCPD2019 |
|---|---|---|
| Stage-1 train / val | 20,000 / 1,000 synthetic | 20,000 / 500 synthetic |
| Stage-2 train / val | 2,569 / 620 real | 1,777 / 90 real, province-balanced |
| Test | 3,258 real | 1,000 real |

## Why CCPD stage 2 is balanced

The raw 8,000-image split is 95.86% 皖 (Anhui). Six provinces never appear and
sixteen appear fewer than ten times. Training on that distribution means the model
sees essentially one Chinese character. `train/script/cn_composite.py` builds a balanced
set by erasing the province cell of real plates and compositing a different
province character in its place, matching foreground and background colour, blur
and noise from neighbouring characters on the same plate. Cross-province
replacement accuracy goes from 0.4400 (raw 8,000) to 0.8000 (balanced 1,777).

## Cache size

Each sample caches to ~9.45 MB: a 512x4096 bf16 T5 embedding (4.19 MB), fp32
ground-truth pixels (3.15 MB), and the packed latents. A 270 KB PNG becomes
9.45 MB of cache, a 36x expansion. This is why images are shipped and the cache is
rebuilt locally rather than transferred.

## Validation independence

The LP stage-1 validation set is a separate 1,000-sample generation, not a slice
of the 20,000. All 21,000 source PNGs were md5-compared: zero overlap, and no
duplicates within the training set either. Plate *strings* do overlap (62% of
validation strings also appear in training) because both draw from the same
25,353-entry corpus -- the validation set measures unseen renderings, not unseen
plate numbers.
