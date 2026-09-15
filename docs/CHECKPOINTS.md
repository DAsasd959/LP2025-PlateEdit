# Checkpoints

All adapters are rank-32 QLoRA over an NF4-quantised FLUX.1-Fill-dev. Download
from Releases into `weights/`, keeping the directory names below, then
`sha256sum -c weights/checksums.txt`.

| Directory | Size | Dataset | Stage | Notes |
|---|---:|---|---|---|
| `lp_stage1_ckpt_21250/` | 222 MB | LP-2025 synthetic | 1 | 21,250 batches x 8 = 170,000 image-views |
| `lp_stage2_ckpt_27606/` | 222 MB | LP-2025 real | 2 | **the published LP checkpoint** |
| `ccpd_stage1_ckpt_20000/` | 222 MB | CCPD synthetic | 1 | built with `--province_ratio 0.5` |
| `ccpd_stage2_ckpt_10000/` | 222 MB | CCPD balanced real | 2 | **the published CCPD checkpoint** |
| `odm_epoch_100.pt` | 677 MB | — | — | ODM perceptual loss, used by every run |
| `trba_lp2025/` | 191 MB | — | — | TRBA recogniser for LP evaluation |
| `trba_ccpd_final/` | 191 MB | — | — | TRBA recogniser for CCPD evaluation, 98.199% on real plates |

`flux_base/` is not a release asset. Download FLUX.1-Fill-dev (32 GB) from
https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev and place it at
`weights/flux_base`. A pre-quantised NF4 copy is not needed: `src/train/model.py`
applies `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=bfloat16)` on load, which produces the same NF4 weights.

## Checkpoint naming

`src/train/callbacks.py` increments its counter once per **batch**, not per
optimizer update. A checkpoint named `ckpt/21250` is therefore 21,250 batches. At
the paper's batch size of 8 that is 170,000 image-views, or about 8.5 epochs over
the 20,000 synthetic plates. Training the same schedule at batch 1 x accum 64
produces the identical model at `ckpt/170000`.

## Stage 2 warm start

Stage 2 resumes from a stage-1 adapter through `reuse_lora_path`. Point it either
at your own stage-1 output or at the released stage-1 adapter. Do not warm-start
stage 1 itself: it uses Prodigy, whose D-adaptation suppresses the learned rate
when starting from converged weights. That is why stage 2 switches to AdamW.
