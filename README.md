# FLUX-Text-Plate — Style-Preserving Partial License-Plate Editing

Partial text editing for license plates: repaint a chosen span of characters while
keeping the plate's own background, blur, illumination and stroke weight. Built on
[FLUX-Text](https://github.com/AMAP-ML/FluxText) and keeping its layout, so the
`src/`, `train/config/`, `train/script/` conventions carry over unchanged.

Two datasets are supported end to end — training, inference and evaluation:

| | LP-2025 (Taiwan) | CCPD2019 (China) |
|---|---|---|
| Plate layout | alphanumeric, `-0123456789A-Z` | 7 cells: 1 Chinese province character + 6 alphanumerics |
| Annotation source | text detector (see [Scope](#scope-and-known-gaps)) | **encoded in the filename** — no detector needed |
| Stage 1 | 20,000 synthetic + 1,000 held out | 20,000 synthetic + 500 held out |
| Stage 2 | 2,569 real + 620 held out | 1,777 province-balanced real + 90 held out |
| Test split | 3,258 real | 1,000 real |

CCPD editing is not split into "Chinese" and "alphanumeric" modes: a single
contiguous span may cross the province boundary, so `皖A5` → `冀B7` is one edit.

---

## 📦 Datasets

| Dataset | Download | Notes |
|---|---|---|
| CCPD2019 | https://github.com/detectRecog/CCPD | Full scene images; crop plates with `train/script/crop_plates_from_base.py`. |
| LP-2025 | https://github.com/AvLab-CV/LP2025 | Taiwan plates. |

Both ship raw images. The **conditions** each model consumes — `partial_masks/`,
`partial_glyphs/`, `partial_labels_txt/` — are derived:

* **CCPD** — build them yourself; the filename carries everything needed:
  ```bash
  python train/script/crop_plates_from_base.py --list <file-list> --out_root data/ccpd/crops
  python eval/ccpd/build_ccpd_conditions.py \
      --src data/ccpd/crops --out data/ccpd/test_1000 --province_ratio 0.5 --max_k 3
  ```
  **To reproduce the published CCPD numbers, download the test conditions instead
  of rebuilding them.** The cell geometry is exact — given the same span, the mask
  this produces matches the published one at IoU 1.0000 on all 1,000 test plates —
  but the span itself is drawn at random and the original seed was not recorded, so
  a rebuild evaluates a different set of edits. Comparable numbers, not the same
  numbers.
* **LP-2025** — the derived conditions for the 2,569 / 620 / 3,258 splits are
  published as a release asset, and for the same reason should be preferred over
  rebuilding when reproducing published numbers. To build them for *your own*
  plates, use `eval/lp/prepare_sample.py`, which takes a crop, the text it shows,
  and the four corners of the text region; to annotate a whole set from scratch,
  see [docs/ANNOTATION.md](docs/ANNOTATION.md).

---

## 🛠️ Installation

```bash
conda create -n flux_text python=3.10
conda activate flux_text
pip install -r requirements.txt
```

`diffusers` is pinned to **0.32.2**. Later releases dropped `USE_PEFT_BACKEND`,
which `src/flux/` relies on; upgrading breaks LoRA loading.

Then place the base model and checkpoints under `weights/` — see
[`weights/README.md`](weights/README.md) — and two fonts under `font/` — see
[`font/README.md`](font/README.md).

---

## 🤗 Checkpoints

| Name | Trained on | Use it for |
|---|---|---|
| `lp_stage2_ckpt_27606` | LP-2025 | **the paper's LP numbers** |
| `lp_stage1_ckpt_21250` | synthetic TW | warm-starting LP stage 2 |
| `ccpd_stage2_ckpt_10000` | CCPD, province-balanced | CCPD inference, Chinese + alphanumeric |
| `ccpd_stage1_ckpt_20000` | synthetic CN | warm-starting CCPD stage 2 |

All are rank-32 QLoRA adapters over an NF4-quantised FLUX.1-Fill-dev.

---

## 🔥 Quick Start

### LP — run the released checkpoint on the test split

```bash
python eval/lp/infer.py \
    --data_root data/lp/test --config train/config/lp/lp2025_train.yaml \
    --lora weights/lp_stage2_ckpt_27606/adapter_model.safetensors \
    --flux_dir weights/flux_base --out outputs/lp_recon --limit 3 --compare
```

`--prompt_mode sequence` reproduces the paper's prompt sampling; the default
`stem` derives the template from the filename so a subset matches a full run.

### LP — replace characters

```bash
python eval/lp/lp_edit.py --target_mode edit \
    --lp_root data/lp/test --lora weights/lp_stage2_ckpt_27606/adapter_model.safetensors \
    --flux_dir weights/flux_base --max_span 2 --limit 100 --out outputs/lp_edit
```

### LP — your own plate

```bash
python eval/lp/prepare_sample.py \
    --image my_plate.jpg --text RBE8700 --quad "16,21 8,68 172,101 180,52" \
    --span 3:6 --target 999 --font font/TWGen7_V1.ttf --out data/mine --stem my_plate
```
The target must have the same character count as the original text; the glyph
condition renders the whole plate and the mask cuts out the replaced part, so
changing the count shifts every character.

### CCPD — Chinese and alphanumeric in one edit

```bash
# preview masks and glyph alignment, no GPU
python eval/ccpd/mixed_span_edit.py --dry_run --limit 8 --span_mode cross --max_k 3

# reconstruction (has ground truth, so LPIPS/FID are meaningful)
python eval/ccpd/mixed_span_edit.py --target_mode gt --span_mode cross --max_k 3 \
    --lora weights/ccpd_stage2_ckpt_10000/adapter_model.safetensors \
    --flux_dir weights/flux_base --limit 300 --out outputs/ccpd_recon

# replacement (ACC/NED only)
python eval/ccpd/mixed_span_edit.py --target_mode edit --span_mode cross --max_k 3 \
    --lora weights/ccpd_stage2_ckpt_10000/adapter_model.safetensors \
    --flux_dir weights/flux_base --limit 200 --out outputs/ccpd_edit
```
`ccpd_cn_edit.py` (province only) and `ccpd_latin_edit.py` (alphanumeric only)
remain available for ablations.

---

## 💪🏻 Training

Both datasets use the same two-stage recipe: synthetic pretraining with Prodigy,
then real-data fine-tuning with AdamW. Prodigy becomes unstable when warm-started
from a converged checkpoint, which is why stage 2 switches optimiser.

`WANDB_API_KEY` is never stored in this repository. Export it, put it in
`~/.netrc` under `machine api.wandb.ai`, or leave it unset and comment out the
`wandb:` block in the config.

### 1. Build the token cache

Each sample caches to ~9.45 MB, mostly the T5 embedding. Check free space first.

```bash
# LP stage 1  (20,000 + 1,000 -> ~185 GB)
STAGE1_DATA=data/lp/synth bash train/script/build_cache_stage1.sh
# LP stage 2  (2,569 + 620 -> ~29 GB)
bash train/script/build_cache.sh

# CCPD, province cell included in the mask 50% of the time
python train/script/cn_preprocess.py --src data/ccpd/crops --out cache/ccpd_stage2_train \
    --province_ratio 0.5 --max_k 2
```

CCPD stage 1 needs `--province_ratio 0.5` to raise the share of masked spans
containing Chinese from 15.8% to 59.6%; otherwise each province character is
updated ~474 times over the whole run instead of ~11,920.

### 2. Train

```bash
bash train/script/train_lp_stage1.sh        # paper values, batch 8 — needs >24 GB
bash train/script/train_lp_stage1_3090.sh   # batch 1 x accum 64 — fits 24 GB
bash train/script/train_lp_stage2.sh
bash train/script/train_ccpd_stage1.sh
bash train/script/train_ccpd_stage2.sh
```

Set `reuse_lora_path` in the stage-2 config before running it.

**On GPU memory.** The paper's LP stage 1 ran on an H100 at batch 8; the thesis
notes the framework "can also be trained on a single RTX 4090 (24GB) with a longer
training time". Batch 8 does not fit in 24 GB — measured on an RTX 3090, it is
72 MiB short even with `expandable_segments`, and batch 4 dies later in the ODM
loss's VAE decode, which keeps a full autograd graph and is not checkpointed.
`train_lp_stage1_3090.sh` uses batch 1 × accum 64: the same effective batch of 64,
the same 2,656 optimizer steps, 4.167 s/batch measured, about 8.8 days total.

**Steps are counted in batches, not optimizer updates.** `src/train/callbacks.py`
increments once per batch, so a checkpoint named `ckpt/21250` is 21,250 batches —
at batch 8 that is 170,000 image-views. The same point at batch 1 is `ckpt/170000`.

### 3. Configs

`train/config/{lp,ccpd}/` holds runnable configs with repository-relative paths.
Alongside them, `stage1_ORIGINAL.yaml` / `stage2_ORIGINAL.yaml` (LP) and
`stage1_AS_RUN.yaml` / `stage2_AS_RUN.yaml` (CCPD) are the files Lightning wrote
at startup for the published runs. They keep their original absolute paths on
purpose — they are the record of what was run, not something to execute.

---

## 📊 Evaluation

```bash
# image fidelity — FID, full-frame LPIPS, region LPIPS
python eval/lp/eval_image.py --gen_dir outputs/lp_recon \
    --real_dir data/lp/test/filtered_plate --mask_dir data/lp/test/partial_masks

# text accuracy — needs a clone of deep-text-recognition-benchmark
python eval/lp/eval_ocr.py --image_folder outputs/lp_recon \
    --saved_model weights/trba_lp2025/best_accuracy.pth --dtr_root <clone>

# CCPD, per-cell accuracy
python eval/ccpd/eval_ccpd.py ...
python eval/ccpd/region_lpips_ccpd.py --gen_dir outputs/ccpd_recon --plate_dir data/ccpd/test_1000/plates
```

`eval/ccpd/LEGACY.txt` lists two scripts kept only for comparison with the
paper's original computation; their paths point at drives that no longer exist.

### Three things that change the numbers

1. **Region LPIPS definition.** The paper multiplies both images by the mask and
   scores the whole 512² frame. Cropping to the mask's bounding box instead gives
   0.22 rather than 0.062 on LP, and differs by 9–10× on CCPD. Same images.
2. **FID is not comparable across sample counts.** The same CCPD model scores 7.30
   at n=200 and 3.85 at n=1,000.
3. **512² versus native size.** LP plates have a median of 326×216 and 75.8% are
   smaller than 512 on both sides. Resizing to 512² magnifies and squares the
   aspect ratio: native-size FID runs ~28% higher, LPIPS 2–4% lower. Ranking is
   unchanged.

---

## 📈 Results

### LP-2025 — ablation, n = 3,258, native resolution

| Setting | FID ↓ | Full LPIPS ↓ | Region LPIPS ↓ | ACC ↑ | NED ↑ |
|---|---:|---:|---:|---:|---:|
| Ours (both stages) | 5.3931 | 0.0768 | 0.0625 | 0.8076 | 0.9548 |
| Real data only | 6.5637 | 0.0853 | 0.0683 | 0.6378 | 0.9163 |
| Synthetic only | 8.7490 | 0.1196 | 0.1052 | 0.8082 | 0.9344 |
| No ODM loss | — | — | — | 0.6676 | 0.9160 |

At 512² the full method measures FID 4.2167 / 0.0792 / 0.0650 against the paper's
reported 4.78 / 0.081 / 0.062.

### CCPD2019 — stage ablation, province reconstruction, n = 1,000

| Setting | ACC ↑ | NED ↑ | FID ↓ | Full LPIPS ↓ | Region LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Stage 1 only | 0.3480 | 0.9067 | 5.4295 | 0.0604 | 0.0389 |
| Both stages | 0.9920 | 0.9989 | 3.5421 | 0.0345 | 0.0183 |
| Real photographs (ceiling) | 0.9960 | 0.9994 | — | — | — |

Larger evaluation, n = 1,000 each:

| | FID ↓ | Full ↓ | Region ↓ | per-cell ACC ↑ | NED ↑ |
|---|---:|---:|---:|---:|---:|
| Reconstruction | 8.5290 | 0.0705 | 0.0365 | 0.8982 | 0.9607 |
| Replacement | 10.9169 | 0.1044 | 0.0628 | 0.6946 | 0.8772 |

**Province balancing matters more than volume.** Cross-province replacement on 200
samples scores 0.4400 when stage 2 trains on the raw 8,000 images and **0.8000**
on the balanced 1,777 — under a quarter of the data, nearly double the accuracy.
The raw split is 95.86% one province (皖); six provinces never appear and sixteen
appear fewer than ten times.

Manual review of 409 CCPD items found the recogniser missed 129 correct edits and
wrongly credited none, so every reported ACC is a lower bound.

---

## Scope and known gaps

* **LP-2025 annotation needs a text detector.** LP-2025 ships plate strings but no
  geometry, so the text quadrilateral is found with DeepSolo++ and turned into
  conditions by `train/script/annotate_lp.py`. Both stages are here and the
  detector weights are a release asset; see [docs/ANNOTATION.md](docs/ANNOTATION.md),
  including how faithfully stage 2 reproduces the released annotations. The
  conditions for all three splits are also published, so you can skip the whole
  pipeline unless you are annotating your own photographs. CCPD needs none of it:
  its annotation comes from the filename.
* **Synthetic generation assets are not shipped.** `synth/*/datasets/` holds 9.1 GB
  of background crops and 654 MB of fonts. The generators are here; the assets are
  not.
* Stage-1 synthetic data for LP was regenerated for indices 0000–1337 after a
  path error on 2026-09-13. Statistically equivalent, not bit-identical to the run
  behind the published checkpoint.

---

## 🌹 Acknowledgement

Built on [FLUX-Text](https://github.com/AMAP-ML/FluxText) and
[FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev).
Evaluation uses [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).
Datasets: [CCPD2019](https://github.com/detectRecog/CCPD), [LP-2025](https://github.com/AvLab-CV/LP2025).

## 📚 Citation

```bibtex
@mastersthesis{styleplate2026,
  title  = {Style-Preserving Partial License Plate Editing},
  school = {National Taiwan University},
  year   = {2026}
}
```
