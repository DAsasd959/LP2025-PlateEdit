# StylePlate — Style-Preserving Partial License-Plate Editing

Repaint a chosen span of characters on a licence plate while keeping the plate's
own background, blur, illumination and stroke weight. Two datasets are supported
end to end: **LP-2025** (Taiwan, alphanumeric) and **CCPD2019** (China, one Chinese
province character followed by six alphanumerics).

CCPD editing is not split into "Chinese" and "alphanumeric" modes — one contiguous
span may cross the province boundary, so `皖A·GG901 → 吉A·CG901` is a single edit.

![examples](assets/teaser.jpg)

StylePlate is a derivative of [FLUX-Text](https://github.com/AMAP-ML/FluxText), not
a fork and not affiliated with it. The layout is kept close to upstream so the
directories mean the same thing in both.

---

## Where things are

| | LP-2025 | CCPD2019 |
|---|---|---|
| Inference | `eval/lp/infer.py` · `lp_edit.py` | `eval/ccpd/mixed_span_edit.py` |
| Training | `train/script/train_lp_stage{1,2}.sh` | `train/script/train_ccpd_stage{1,2}.sh` |
| Configs | `train/config/lp/` | `train/config/ccpd/` |
| Cache | `train/script/build_cache_stage1.sh` · `build_cache.sh` | `build_cache_stage1.sh` · `cn_preprocess.py` |
| Prepare your own plates | `eval/lp/prepare_sample.py` · `train/script/annotate_lp.py` | `train/script/crop_plates_from_base.py` · `eval/ccpd/build_ccpd_conditions.py` |
| Synthetic data | `synth/tw/` | `synth/cn/` |
| Scoring | `eval/eval_image.py` · `eval/eval_ocr.py` | + `eval/ccpd/eval_ccpd.py` |

Docs: [downloads](docs/CHECKPOINTS.md) · [data layout](docs/DATA.md) ·
[evaluation](docs/EVALUATION.md) · [LP annotation](docs/ANNOTATION.md) ·
[recognisers](docs/RECOGNIZER.md) · [synthesis](synth/README.md)

## Getting started

**1. Environment.**

```bash
conda create -n flux_text python=3.10 && conda activate flux_text
pip install -r requirements.txt
```

`diffusers` is pinned to **0.32.2** — later releases dropped `USE_PEFT_BACKEND`,
which `src/flux/` needs.

**2. Check it before you download 32 GB.**

```bash
bash smoke_test.sh
```

Runs everything that needs no weights and no GPU. A `--` line means "not downloaded
yet" and names where to get it; only a `FAIL` is a broken environment.

**3. Base model** — 32 GB, from HuggingFace, not a release asset:

```bash
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev --local-dir weights/flux_base
```

Quantised to NF4 on load; a pre-quantised copy is not needed.

**4. Checkpoints and data.** `R=https://github.com/DAsasd959/LP2025-PlateEdit/releases/download`

```bash
mkdir -p weights data/lp data/ccpd

# LP: released checkpoint, perceptual loss, recogniser
curl -L $R/v1.0/adapter_model.safetensors -o weights/lp_stage2_ckpt_27606/adapter_model.safetensors --create-dirs
curl -L $R/v1.0/adapter_config.json       -o weights/lp_stage2_ckpt_27606/adapter_config.json
curl -L $R/v1.0/epoch_100.pt              -o weights/odm_epoch_100.pt
curl -L $R/v1.0/best_accuracy.pth         -o weights/trba_lp2025/best_accuracy.pth --create-dirs

# CCPD: both stages, recogniser, and the subset with its conditions
for f in ccpd_stage2_ckpt_10000 ccpd_stage1_ckpt_20000 lp_stage1_ckpt_21250 trba_ccpd_final; do
  curl -L $R/v2.0/$f.tar.gz | tar xz -C weights
done
curl -L $R/v2.0/ccpd_subset.tar.gz | tar xz -C data/ccpd --strip-components=1

# LP splits, each with filtered_plate + partial_{masks,glyphs,labels_txt}.
# These three unpack at the archive root, so give each its own directory.
mkdir -p data/lp/train data/lp/val data/lp/test
curl -L $R/data-v1.0/lp2025_train.tar.gz | tar xz -C data/lp/train
curl -L $R/data-v1.0/lp2025_val.tar.gz   | tar xz -C data/lp/val
curl -L $R/data-v1.0/lp2025_test.tar.gz  | tar xz -C data/lp/test

curl -L $R/v2.0/SHA256SUMS.txt -o weights/SHA256SUMS.txt   # sha256sum -c to verify
```

[docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) lists every asset, its size, and what
it is. Two things are deliberately not published: the 32 GB base model above, and
the synthetic stage-1 data — generate that with [`synth/`](synth/README.md), which
runs in a second conda environment of its own.

**5. Check again.** `bash smoke_test.sh` should now show no `--` lines except any
part you chose to skip.

## 📦 Datasets

| | Full dataset | What this project used |
|---|---|---|
| LP-2025 | [AvLab-CV/LP2025](https://github.com/AvLab-CV/LP2025) | 2,569 train · 620 val · 3,258 test |
| CCPD2019 | [detectRecog/CCPD](https://github.com/detectRecog/CCPD) | 1,777 train (province-balanced) · 90 val · 1,000 test |

Step 4 fetches the subsets, masks, glyphs and labels included. **Prefer them over
rebuilding** if you want to reproduce the published numbers: the geometry is exact,
but both condition builders draw the masked span at random and the original seeds
were not recorded, so a rebuild measures a different set of edits.

## 🤗 Checkpoints

| | Stage 1 | Stage 2 | Trained on |
|---|---|---|---|
| LP-2025 | `lp_stage1_ckpt_21250` | **`lp_stage2_ckpt_27606`** | 20,000 synth / 1,000 val → 2,569 real / 620 val |
| CCPD2019 | `ccpd_stage1_ckpt_20000` | **`ccpd_stage2_ckpt_10000`** | 20,000 synth / 500 val → 1,777 balanced / 90 val |

Bold entries are the released checkpoints the published numbers come from. All are
rank-32 QLoRA adapters over NF4-quantised FLUX.1-Fill-dev. Stage 1 is there so you
can warm-start stage 2 without retraining it.

## 🔥 Quick start

Reconstruction puts the original characters back; replacement writes different
ones. Both are the same script under `--target_mode`, and **the two scripts default
to opposite modes**, so name it explicitly.

```bash
LP=weights/lp_stage2_ckpt_27606/adapter_model.safetensors
CN=weights/ccpd_stage2_ckpt_10000/adapter_model.safetensors

# LP, reconstruction -- reads the published conditions, so this is the one that
# reproduces the published numbers
python eval/lp/infer.py --data_root data/lp/test --config train/config/lp/lp2025_train.yaml \
    --lora $LP --flux_dir weights/flux_base --out outputs/lp_recon --limit 3 --compare

# LP, replacement
python eval/lp/lp_edit.py --target_mode edit --lp_root data/lp/test \
    --lora $LP --flux_dir weights/flux_base --max_span 2 --limit 100 --out outputs/lp_edit

# CCPD, reconstruction -- province and alphanumerics in one span
python eval/ccpd/mixed_span_edit.py --target_mode gt --span_mode cross --max_k 3 \
    --lora $CN --flux_dir weights/flux_base --limit 200 --out outputs/ccpd_recon

# CCPD, replacement
python eval/ccpd/mixed_span_edit.py --target_mode edit --span_mode cross --max_k 3 \
    --lora $CN --flux_dir weights/flux_base --limit 200 --out outputs/ccpd_edit
```

`--dry_run` on the CCPD editor previews masks and glyph alignment without loading
a model — worth running once before committing a GPU to a long job.

`lp_edit.py --target_mode gt` also reconstructs, but by re-rendering the glyph from
the annotated corners rather than reading the published one; use `infer.py` when
the published numbers are the point. `ccpd_cn_edit.py` (province only) and
`ccpd_latin_edit.py` (alphanumeric only) remain available for ablations.

**Your own plate.** Supply the crop, the text it shows and the four corners of the
text region:

```bash
python eval/lp/prepare_sample.py --image my_plate.jpg --text RBE8700 \
    --quad "16,21 8,68 172,101 180,52" --span 3:6 --target 999 \
    --font font/TWGen7_V1.ttf --out data/mine --stem my_plate
```

The target must have the same character count as the span it replaces.

## 💪🏻 Training

Two stages for both datasets: synthetic pretraining with Prodigy, then real-data
fine-tuning with AdamW. Prodigy destabilises when warm-started from a converged
checkpoint, which is why stage 2 switches optimiser.

```bash
# 1. cache  (each sample is ~9.45 MB, mostly the T5 embedding -- check free space)
STAGE1_DATA=<generated dir> bash train/script/build_cache_stage1.sh lp
PP_PROVINCE_PROB=0.5 STAGE1_DATA=<generated dir> \
    bash train/script/build_cache_stage1.sh ccpd
bash train/script/build_cache.sh                                   # LP stage 2
python train/script/cn_preprocess.py --src data/ccpd/crops \
    --out cache/ccpd_stage2_train --province_ratio 0.5 --max_k 2   # CCPD stage 2

# 2. train
bash train/script/train_lp_stage1.sh           # 24 GB; _h100.sh is the published batch-8 recipe
bash train/script/train_lp_stage2.sh
bash train/script/train_ccpd_stage1.sh
bash train/script/train_ccpd_stage2.sh
```

Set `reuse_lora_path` in the stage-2 config first.

Runs log to **your own** wandb account: export `WANDB_API_KEY`, or put it in
`~/.netrc` under `machine api.wandb.ai`. `WANDB_PROJECT` and `WANDB_ENTITY`
override the project and the account or team without editing a config. Leave the
key unset and training runs exactly the same, with logging skipped.

**GPU memory.** LP stage 1 as published ran on an H100 at batch 8, which does not
fit 24 GB. `train_lp_stage1.sh` uses batch 1 × accum 64 — same effective batch
of 64, same 2,656 optimizer steps, measured 4.167 s/batch, about 8.8 days.

**Steps count batches, not optimizer updates.** `ckpt/21250` is 21,250 batches; at
batch 8 that is 170,000 image-views, and the same point at batch 1 is `ckpt/170000`.

## 📊 Evaluation

See [docs/EVALUATION.md](docs/EVALUATION.md) for commands, full result tables, and
three definitional traps that change the numbers by up to 10× on identical images.

Headline: LP reaches FID 4.78 / ACC 0.808 against 5.92 / 0.638 for real data alone.
CCPD province reconstruction reaches ACC 0.9920 against a real-photograph ceiling
of 0.9960. Province balancing lifts cross-province replacement from 0.4400 to
0.8000 using under a quarter of the data.

The CCPD figures cover two tasks the published CCPD table does not: editing the
Chinese province character on its own, and editing a contiguous span that crosses
the province boundary so one edit changes the Chinese character and one or two
alphanumerics together.

## Scope

* LP-2025 ships plate strings but no geometry, so annotation needs a text detector
  — both stages are here, see [docs/ANNOTATION.md](docs/ANNOTATION.md). CCPD needs
  none: its filename carries the four vertices.
* SynthText backgrounds (~9 GB) are downloaded, not redistributed — [synth/README.md](synth/README.md).
* Stage-1 LP synthetic indices 0000–1337 were regenerated after a path error;
  statistically equivalent, not bit-identical to the published run.

## 🌹 Acknowledgement

Built on [FLUX-Text](https://github.com/AMAP-ML/FluxText) and
[FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev).
Text detection uses [DeepSolo++](https://github.com/ViTAE-Transformer/DeepSolo);
recognition scoring uses [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark);
synthetic backgrounds come from [SynthText](https://github.com/ankush-me/SynthText).
Datasets: [CCPD2019](https://github.com/detectRecog/CCPD), [LP-2025](https://github.com/AvLab-CV/LP2025).

## 📚 Citation

```bibtex
@mastersthesis{styleplate2026,
  title  = {Style-Preserving Partial License Plate Editing},
  school = {National Taiwan University},
  year   = {2026}
}
```

## License

Original work in this repository is MIT — see [LICENSE](LICENSE), which also lists
the terms of the third-party code, weights, fonts and datasets redistributed or
linked here.

Released model weights are LoRA adapters over FLUX.1-Fill-dev and inherit its
**non-commercial** terms, which also cover images generated with them.
