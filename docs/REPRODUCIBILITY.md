# Reproducibility

Both training runs behind the released checkpoint are fully attested. Each
one's configuration is the file Lightning wrote at startup, recovered from
the run directory:

| Stage | Run directory | Config |
|---|---|---|
| 1 · synthetic pretraining | `20260415-150207` | `configs/lp2025_stage1.yaml` |
| 2 · real fine-tuning | `20260419-180408` | `configs/lp2025_train.yaml` |

Stage 2's directory timestamp matches the wandb record
`run-20260419_180409-gloox8hm` to the second, and its `ckpt/27606/adapter_model.safetensors`
is byte-identical to the released adapter (sha256 `61d52f45f3e220c3…`). Its
`reuse_lora_path` points at stage 1's `ckpt/21250`.

> An earlier revision of this page said the run directory had been deleted and
> that the optimiser was Prodigy at lr 1. Both were wrong. Four values in the
> stage-2 config had been reconstructed from a sibling run: the optimiser, the
> gradient accumulation, the save interval, and a `reuse_lora_path` that was
> missing entirely. Anything trained from that revision does not reproduce the
> paper.

## What the two stages actually ran

`src/train/callbacks.py` increments its step counter once per **batch**, not
per optimizer step. Read every step count below in those units.

| | Stage 1 | Stage 2 (released) |
|---|---:|---:|
| Data | 20,000 synthetic | 2,569 real |
| Validation | 1,000 synthetic | 620 real |
| Batch size | 8 | 1 |
| Gradient accumulation | 8 | 64 |
| Effective batch | 64 | 64 |
| Optimiser | Prodigy | AdamW |
| Learning rate | 1 | 1e-4 |
| Weight decay | 0.01 | 0.01 |
| Save interval | 1,250 | 642 |
| Checkpoint taken | 21,250 | 27,606 |
| Image exposures | 170,000 | 27,606 |
| Optimizer updates | 2,656 | 431 |
| Epochs | 8.5 | 10.7 |

Both checkpoints divide exactly by their save interval — 21,250 = 1,250 × 17,
27,606 = 642 × 43 — so neither is a hand-picked step.

Neither number is where its run stopped. Stage 1 continued to 28,750; stage 2
continued to 43,656 (epoch 17, 24.6 h wall-clock). Each is the checkpoint that
was carried forward, not an endpoint.

The optimiser differs between stages on purpose. Prodigy adapts its own rate
and suppresses it when started from warm weights, so stage 2 — which resumes
from stage 1's adapter — uses AdamW at a fixed 1e-4.

## Also attested

| Item | Value | Evidence |
|---|---|---|
| Wall-clock (stage 2) | 88,657 s ≈ 24.6 h | `wandb-summary.json` |
| Hardware (stage 2) | 1× RTX 4090 24 GB, 20 logical cores, 64 GB RAM, CUDA 12.2 | `wandb-metadata.json` |
| Hardware (stage 1) | a machine with room for batch 8 at 512² — the cache paths are under `/work/u1258075`, not this host | stage-1 config |
| Condition type | `word_fill` | both configs |
| Loss terms | diffusion + ODM, weights 20/20/20/20 and `w_loss_f` 1 | both configs |
| Final losses (stage 2) | train 0.0121 · val_sd 0.3432 · val_odm 0.2366 | summary |
| Base model | FLUX.1-Fill-dev, NF4 | both configs name `FLUX.1-Fill-dev-nf4` |
| LoRA | r=32, α=32, gaussian init, dropout 0, no DoRA/rsLoRA | `adapter_config.json` |
| Image size | 512 | both configs |
| Gradient checkpointing | true | both configs |
| Dataloader workers | 8 | both configs |
| Library versions | torch 2.4.0 · diffusers 0.32.2 · peft 0.17.1 · transformers 4.57.1 · lightning 2.2.4 · bitsandbytes 0.48.1 · prodigyopt 1.1.2 | `requirements.txt` |

## Where the shipped configs deviate

Three deliberate departures, none of which changes the weights:

| Field | Original | Shipped | Why |
|---|---|---|---|
| `max_steps` | `-1` | 21250 / 43784 | Both runs were stopped by hand. The shipped values stop at the checkpoint that was actually used, and at where stage 2 ended. |
| `gradient_checkpointing` | true | false in stage 1 | Numerically identical; it trades memory for speed, and a 96 GiB card does not need the trade. Set it back to true on a 24 GB card. |
| `limit_val_batches` | nested under `dataset` | at `train` level | `train.py` reads `training_config.get("limit_val_batches")`, so nested it was silently ignored and every validation walked the full split. Fixing the placement only changes how long validation takes. |

Stage 1's shipped data is one 20,000-sample generation split 19,500 / 500,
against the original's 20,000 train and a separately generated 1,000 for
validation — 2.5% less training data.

## Missing

| Item | Status |
|---|---|
| Random seed | not recorded — expect run-to-run variation |
| Learning-rate schedule | neither config sets one; stage 1's Prodigy is adaptive and stage 2's AdamW runs at a fixed 1e-4 |
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
