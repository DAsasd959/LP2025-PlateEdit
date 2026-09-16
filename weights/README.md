# Weights

None are tracked in git. Download them from the repository's Releases page and
place them here, keeping these names — the configs and scripts refer to them.

| Path | Size | What it is |
|---|---|---|
| `flux_base/` | 32 GB | FLUX.1-Fill-dev, from HuggingFace. Quantised to NF4 on load; a pre-quantised copy is not needed. |
| `odm_epoch_100.pt` | 710 MB | ODM perceptual-loss model. Both stages of both datasets use it. |
| `lp_stage1_ckpt_21250/` | 231 MB | LP stage-1 adapter (the paper's run). |
| `lp_stage2_ckpt_27606/` | 231 MB | **LP released checkpoint** — the paper's numbers come from this one. |
| `ccpd_stage1_ckpt_20000/` | 231 MB | CCPD stage-1 adapter. |
| `ccpd_stage2_ckpt_10000/` | 231 MB | CCPD released checkpoint. |
| `trba_lp2025/` | 200 MB | TRBA recogniser for LP evaluation. |
| `trba_ccpd_final/` | 200 MB | TRBA recogniser for CCPD evaluation (98.199% on real plates). |

`flux_base` comes from https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
rather than from a release asset.

Verify what you downloaded with `sha256sum -c checksums.txt`.

## Conditions

Two more release assets are not weights but are needed to reproduce the published
numbers, because both condition builders draw their masked span at random and the
original seeds were not recorded:

| Asset | Size | Contents |
|---|---:|---|
| `lp2025_conditions.tar.gz` | ~56 MB | `partial_{masks,glyphs,labels_txt}` for the 2,569 / 620 / 3,258 LP splits |
| `ccpd_test1000_conditions.tar.gz` | ~14 MB | the same three directories for the 1,000 CCPD test plates |

Rebuilding them locally is supported and the geometry is exact — the CCPD builder
reproduces the published masks at IoU 1.0000 given the same span — but a rebuild
samples different spans, so it evaluates a different set of edits.
