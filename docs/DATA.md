# Data

## LP2025

Real photographs of Taiwanese motorcycle and car plates, cropped to the plate.
No synthetic images. Plate strings use `A–Z 0–9` and a centre separator; there
is no Chinese text, which is what distinguishes this half of the work from the
CCPD extension.

### Layout consumed by the code

Four parallel directories keyed by a shared stem. `src/data/data_real.py` reads
exactly these names:

```
data/<split>/
    filtered_plate/       <stem>.jpg    source plate crop
    partial_labels_txt/   <stem>.txt    target text, first whitespace token
    partial_glyphs/       <stem>.png    rendered target glyph
    partial_masks/        <stem>.png    region to repaint, white = edit
```

The sample list is derived from `partial_glyphs/*.png`, so a stem missing from
that directory is silently absent from training.

The test split is the exception: its source crops sit in
`cropped_plates_png_512/<stem>.png`, not `filtered_plate/<stem>.jpg`.
`scripts/infer.py` looks for both names; `src/data/data_real.py`, which is only
used for caching the training and validation splits, expects `filtered_plate`.

**The test split as archived is incomplete on the source side.** All 3,258
samples have a mask, a glyph, and a label, but this copy holds only 2,542 source
images — 716 stems (22.0%) are absent. The complete set exists on the machine the
data was originally prepared on; this is an incomplete transfer, not a loss.
`docs/test_missing_sources.txt` lists exactly which stems to copy across, so the
gap can be closed without moving the whole split:

```bash
rsync -av --files-from=<(sed 's/$/.png/' docs/test_missing_sources.txt) \
      <source-host>:/path/to/test/filtered_plate/ \
      data/test/cropped_plates_png_512/
```

Until then `scripts/infer.py` reports the missing stems as skipped rather than
failing, so a run over the test split yields 2,542 images.
`scripts/eval_image.py` likewise scores only the pairs it can form and prints how
many it dropped.

### Split sizes

| Split | Samples | Note |
|---|---:|---|
| train | 2,569 | filtered from 13,638 raw crops |
| val | 620 | |
| test | 3,258 | |

`filtered_plate` is a quality-filtered subset of `cropped_plates`; the filter
removes crops too small, too skewed, or too dark for the target character to be
legible at 512×512.

## Recogniser data

The TRBA evaluator is trained separately, on plate crops with their transcription:

| Split | Images | Unique plate strings |
|---|---:|---:|
| train | 13,243 | 9,392 |
| val | 3,340 | 2,927 |
| test | 16,657 | 11,107 |

Format is one line per image, `filename<space>plate_string`. Note that
`create_lmdb_dataset.py` in deep-text-recognition-benchmark splits on **tab** in
some forks and on **space** in others — check yours before building the LMDB, as
a mismatch produces an LMDB whose labels are all empty and a model that trains
to 0% accuracy without erroring.

### Two data-quality issues worth knowing

**Plate strings repeat across splits.** 46.3% of validation images and 47.9% of
test images carry a plate string that also appears in training. The same vehicle
photographed more than once ends up on both sides of the split. The photographs
differ, so this is not pixel-level leakage, but a recogniser evaluated this way
is partly being asked about strings it has already memorised. If you re-split,
group by plate string rather than sampling images at random.

**A small fraction of labels contain characters outside the charset.** 55 of
13,243 training labels (0.42%), 18 of 3,340 validation labels (0.54%), and 96 of
16,657 test labels (0.58%) contain `- * ) ] _` or lower-case letters — for
example `7656-VN`, `BH*416`, `Z0-1995`. The recogniser's charset is `A–Z 0–9 ·`,
so these labels can never be matched and act as a fixed ~0.5-point deduction
from any accuracy figure. Either clean them or report accuracy on the clean
subset and say which you did.

Label lengths run 5–8 characters, with 6 the most common.

## Caching

Training reads pre-encoded `.pt` files, not images. `scripts/build_cache.sh`
produces them:

```
cache/lp_train/    2,569 files
cache/lp_val/        620 files
```

Each file holds the VAE latents and text embeddings for one sample, **with the
mask already baked into the tokens**. A cache built for one mask set therefore
cannot be reused for a different one — change the masks and you must rebuild.
