# Checkpoints

## Released adapter

| | |
|---|---|
| Step | 27,606 (epoch ~10–11) |
| Files | `adapter_model.safetensors` (222 MB), `adapter_config.json`, `README.md` |
| Base | FLUX.1-Fill-dev, NF4-quantised |
| LoRA | r=32, α=32, gaussian init, dropout 0, no DoRA, no rsLoRA |

Place it at `weights/lp2025_27606/` or pass any path via `--lora`.

The adapter is stored in PEFT layout: keys carry a `transformer.` prefix and
lack the `.default` sub-module that a live PEFT model expects.
`scripts/infer.py` renames them on load and prints how many tensors matched — if
that count is far below the total, the config and the checkpoint disagree.

## Why step 27,606

Validation loss over the run, sampled every 1,284 steps:

| Step | val/loss_sd |
|---:|---:|
| 2,568 | 0.3461 |
| 14,129 | 0.3358 |
| 26,974 | **0.3232 — lowest of the run** |
| 28,258 | 0.3432 |
| 29,543 | 0.3232 |
| 43,672 | 0.3432 |

The released step sits next to the run's best validation loss, and the following
16,000 steps never beat it. Validation loss is essentially flat from the first
epoch onward — it moves between 0.323 and 0.347 for the whole run — so the
checkpoint choice is not sharply determined by loss alone, and training past
roughly 28k steps buys nothing measurable.

## Recogniser checkpoint

The TRBA evaluator used for ACC/NED:

| | |
|---|---|
| Architecture | TPS-ResNet-BiLSTM-Attn |
| Charset | `A–Z 0–9 ·` (num_class 38) |
| Input | 128×128, PAD=True |
| Training | 300,000 iterations, batch 32, Adadelta lr=1 |
| Data | LP2025 train 13,243 / val 3,340 |

Read `docs/DATA.md` on the cross-split plate-string overlap before quoting its
validation accuracy as a held-out figure.
