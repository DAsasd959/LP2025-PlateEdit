# Reproducibility

The directory holding the original training run's configuration was deleted.
Its wandb record survived, which recovers most of the setup. This page separates
what is documented from what is reconstructed, so that anyone reproducing the
work knows which is which.

Source of truth: `wandb/run-20260419_180409-gloox8hm/` — `output.log`,
`wandb-summary.json`, `wandb-metadata.json`, `requirements.txt` — plus the
released `adapter_config.json` and model card.

## Attested

Recovered from the run's own record.

| Item | Value | Evidence |
|---|---|---|
| Training samples | 2,569 | `output.log`: cached dataset line |
| Validation samples | 620 | `output.log` |
| Batch size | 1 | epoch boundaries fall at steps 2568, 5137, 7706 … = 2,569 steps/epoch |
| Validation interval | 1,284 steps (half an epoch) | validation lines in `output.log` |
| Steps completed | 43,784 (epoch 17) | `wandb-summary.json` |
| Released step | 27,606 | referenced by the original inference script |
| Wall-clock | 88,657 s ≈ 24.6 h | summary `runtime` |
| Hardware | 1× RTX 4090 24 GB, 20 logical cores, 64 GB RAM, CUDA 12.2 | `wandb-metadata.json` |
| Condition type | `word_fill` | `output.log` |
| Loss terms | diffusion + ODM both active | `train/loss_sd`, `train/loss_odm` logged |
| Final losses | train 0.0121 · val_sd 0.3432 · val_odm 0.2366 | summary |
| Base model | FLUX.1-Fill-dev, NF4 | model card |
| Quantisation | nf4, double_quant false, compute dtype bfloat16, quant_storage uint8 | model card |
| LoRA | r=32, α=32, gaussian init, dropout 0, no DoRA/rsLoRA, full target-module regex | `adapter_config.json` |
| Library versions | torch 2.4.0 · diffusers 0.32.2 · peft 0.17.1 · transformers 4.57.1 · lightning 2.2.4 · bitsandbytes 0.48.1 · prodigyopt 1.1.2 | `requirements.txt` |

## Inferred

Not recorded for this run. Taken from the nearest surviving configuration of the
same training script and marked `[lineage]` in `configs/lp2025_train.yaml`.

| Item | Value used | Basis |
|---|---|---|
| Optimiser | Prodigy, lr = 1 | `prodigyopt` was installed in the recorded environment and every sibling run of this script used Prodigy at lr 1 |
| Optimiser extras | weight_decay 0.01, safeguard_warmup true, use_bias_correction true | sibling configuration |
| ODM loss weights | `w_loss_1..4 = 20`, `w_loss_f = 1` | sibling configuration; the run logged `loss_odm`, so the term was on, but its weights are not attested |
| Image size | 512 | sibling configuration; consistent with the 512×512 inference path |
| Gradient checkpointing | true | sibling configuration |
| Dataloader workers | 8 | sibling configuration |

If your reproduction diverges, these six rows are where to look first.

## Missing

| Item | Status |
|---|---|
| Random seed | not recorded — expect run-to-run variation |
| Learning-rate schedule | not recorded; Prodigy is adaptive, so probably none, but unconfirmed |
| Original training cache | only 890 of the 2,569 `.pt` files remain; rebuild from `filtered_plate` with `scripts/build_cache.sh` |

## Reproducing from scratch

1. Build the cache from `data/train` and `data/val`.
2. Train with `configs/lp2025_train.yaml`. Expect roughly 25 hours to 43,784
   steps on a 24 GB GPU; stopping near 28,000 steps loses nothing measurable.
3. Generate on `data/test` with the step-27,606 checkpoint, or with your own
   checkpoint at the best validation loss.
4. Evaluate. Report FID with its N, and say whether Region LPIPS is being used
   as a quality measure — it only is for reconstruction.

Because the seed is unrecorded, treat exact metric equality as out of reach and
compare against the reported figures within their stated intervals.
