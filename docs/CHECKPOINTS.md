# Downloads

Everything the code expects but does not ship. Extract into `weights/` keeping the
directory names below — configs and scripts refer to them — then
`sha256sum -c weights/checksums.txt`.

## Weights

| Extract to `weights/` | Size | Release · asset |
|---|---:|---|
| `flux_base/` | 32 GB | [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), not a release asset |
| `lp_stage2_ckpt_27606/` | 222 MB | `v1.0` · `adapter_model.safetensors` + `adapter_config.json` |
| `lp_stage1_ckpt_21250/` | 222 MB | `v2.0` · `lp_stage1_ckpt_21250.tar.gz` |
| `ccpd_stage2_ckpt_10000/` | 222 MB | `v2.0` · `ccpd_stage2_ckpt_10000.tar.gz` |
| `ccpd_stage1_ckpt_20000/` | 222 MB | `v2.0` · `ccpd_stage1_ckpt_20000.tar.gz` |
| `odm_epoch_100.pt` | 677 MB | `v1.0` · `epoch_100.pt` |
| `trba_lp2025/` | 191 MB | `v1.0` · `best_accuracy.pth` |
| `trba_ccpd_final/` | 191 MB | `v2.0` · `trba_ccpd_final.tar.gz` |
| `deepsolo/r50_data3_multilingual_finetune.pth` | 173 MB | `v2.0` · `deepsolo_r50_multilingual.pth` — rename on extract; only for annotating new LP photographs |

Stage 2 of each dataset is the published checkpoint; stage 1 is there so you can
warm-start stage 2 without retraining it. Adapters are rank-32 QLoRA over
NF4-quantised FLUX.1-Fill-dev and inherit its non-commercial terms.
`odm_epoch_100.pt` comes from upstream FLUX-Text, not from this project.

## Data

| Extract to `data/` | Size | Release · asset |
|---|---:|---|
| `lp/{train,val,test}/` | 347 MB | `data-v1.0` · `lp2025_{train,val,test}.tar.gz` |
| `ccpd/` | 48 MB | `v2.0` · `ccpd_subset.tar.gz` — 1,860 balanced train, 93 val, 1,000 test with conditions |

The archive holds 1,860 training images; 1,777 of them cache successfully and are
what the published run trained on, which is the figure the tables report. Same for
validation: 93 images, 90 cached.
| CCPD2019 source | 177 MB | `dataset-v1.0` · `dataset_ccpd2019.tar.gz` |
| LP stage-1 synthetic | 3.2 GB | `stage1-v1.0` · `lp2025_stage1.tar.gz.part{0,1}` — `cat` them before extracting |

Prefer these over rebuilding conditions yourself. The geometry is exact — a CCPD
rebuild reproduces the published masks at IoU 1.0000 given the same span — but the
span is drawn at random and the original seeds were not recorded, so a rebuild
measures a different set of edits.

## Checkpoint naming

`src/train/callbacks.py` counts **batches**, not optimizer updates. `ckpt/21250` is
21,250 batches; at batch 8 that is 170,000 image-views, and the same point trained
at batch 1 lands at `ckpt/170000`.

Do not warm-start stage 1 — it uses Prodigy, whose D-adaptation suppresses the
learned rate from converged weights. That is why stage 2 switches to AdamW.
