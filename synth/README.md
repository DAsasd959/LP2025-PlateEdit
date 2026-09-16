# Synthetic plate generation

Stage 1 of both datasets trains on synthetic plates. The generators are here; the
plates themselves are not published, because generating them is cheap and the
output is 13 GB per dataset.

Two generators, same engine. `synth/tw/Synthtext/` and `synth/cn/Synthtext/` are
identical except for `data_cfg.py`, which chooses the font and the plate-text
corpus.

| | `synth/tw` (LP-2025, Taiwan) | `synth/cn` (CCPD2019, China) |
|---|---|---|
| Entry point | `datagen.py` | `datagen.py`, `cn_datagen_run.py` |
| Settings | `cfg.py` — `sample_num`, `save_name` | same |
| Font | `datasets/fonts/LP2022_2/TWGen7_V1.ttf` | `datasets/fonts/ccpd_ttf/CCDP_TEXT0620.ttf` |
| Corpus | `Synthtext/data/plate_list_final2.txt` | `Synthtext/data/ccpd_plate_list.txt` |
| Colours | `Synthtext/data/colors_new.cp` | same |
| Backgrounds | `datasets/bg_data/bg_img/` — **download separately** | same |

`synth/cn` also carries `check_bbox.py`, `check_text.py` and `make_rect_mask.py`,
one-off inspection utilities that are not part of the pipeline.

## The one thing you have to download

Backgrounds are the [SynthText](https://github.com/ankush-me/SynthText) background
image set — the same `bg_img` collection that project distributes, about 9 GB of
scene photographs with filenames like `ant+hill_1.jpg`. They are not redistributed
here. Fetch `bg_img.tar.gz` from SynthText and extract it so that each generator
sees `datasets/bg_data/bg_img/`.

`datasets/imnames.cp` is tracked here and lists the 8,010 filenames the generator
draws from. It reads that list rather than scanning the directory, so a download
covering those names is sufficient — extra files in the directory are ignored.

Everything else the generators read — both fonts, both corpora, the colour model —
is in this repository.

## Generating

The generators run in **their own environment**, not the training one: they need
pygame and Augmentor, and they never touch a diffusion model.

```bash
conda create -n synthtext python=3.10
conda activate synthtext
pip install -r synth/requirements.txt

cd synth/tw                     # or synth/cn
# edit cfg.py: sample_num and save_name
python datagen.py
```

Writes the four parallel directories the cache builder expects:

```
<save_name>/
    i_s/       <idx>.png   the rendered plate in a scene
    mask_s/    <idx>.png   character mask
    t_b/       <idx>.png   background without text
    i_s_bbox/  <idx>.txt   per-character boxes
    i_s.txt                <idx>.png <plate text>
    angle_stat.txt
```

Then build the token cache:

```bash
# LP stage 1
STAGE1_DATA=<generated dir> bash train/script/build_cache_stage1.sh

# CCPD stage 1 -- PP_PROVINCE_PROB forces the province character under the mask
# often enough to train it. At the default of 0, only 15.8% of masked spans
# contain Chinese and each province character sees ~474 updates over the run; at
# 0.5 that rises to 59.6% and ~11,920 updates, about 385 per province.
PP_PROVINCE_PROB=0.5 STAGE1_DATA=<generated dir> bash train/script/build_cache_stage1.sh
```

## Reproducibility

Each run draws plate text, background, font size, perspective and degradation at
random, so two runs of the same `cfg.py` produce statistically equivalent but not
identical sets. The published checkpoints were trained on one particular draw,
which is not republished; see docs/DATA.md.
