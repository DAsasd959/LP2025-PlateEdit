# Data

## Download

The dataset is published as a release alongside the code:

```bash
BASE=https://github.com/DAsasd959/LP2025-PlateEdit/releases/download/data-v1.0
mkdir -p data && cd data
curl -L -O $BASE/lp2025_train.tar.gz     #  20 MB
curl -L -O $BASE/lp2025_val.tar.gz       #  36 MB
curl -L -O $BASE/lp2025_test.tar.gz      # 185 MB
curl -L -O $BASE/data_checksums.txt
sha256sum -c data_checksums.txt

mkdir -p train val && tar -xzf lp2025_train.tar.gz -C train
tar -xzf lp2025_val.tar.gz -C val
cd .. && tar -xzf data/lp2025_test.tar.gz        # test extracts to the root
```

That last line is deliberate: the test split's four directories sit at the
archive root, while train and val live under `data/`. Extracting the test
tarball into `data/` instead would leave `check_data.py` reporting it missing.

The archives hold only what inference and training read — the source crops and
the three partial-editing conditions. The wider archive this was cut from also
carries unfiltered crops, detector labels, and visualisations, none of which any
script here opens.

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

**Where the test split lives.** The evaluated test set is the one at the top
level of the LP2025 archive, not the partial copy under `data/test/`:

```
LP2025/
    filtered_plate/        3,918 source crops (a superset of the test samples)
    partial_glyphs/        3,258
    partial_masks/         3,258
    partial_labels_txt/    3,258
```

All 3,258 samples have a source image there — the set is complete, and the 3,258
published outputs correspond exactly to it. The `data/test/` tree is an earlier
partial copy holding 2,542 of the sources under a different directory name; point
`--data_root` at the top-level archive rather than at `data/test/`.

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
