# LP2025 Partial License-Plate Editing

Style-preserving partial editing of Taiwanese license plates with a QLoRA-tuned
FLUX.1-Fill model. Given a plate photograph, a mask over one or more character
cells, and a rendered target glyph, the model repaints only the masked cells and
leaves the rest of the plate — its lighting, blur, plate frame, and background —
untouched.

This repository is the LP2025 half of the work: data preparation, training,
inference, and the three evaluation metrics used in the paper.

<!-- Add a qualitative figure here: source | output | glyph | mask.
     `scripts/infer.py --compare` writes exactly that strip per sample. -->

## What is here

```
configs/lp2025_train.yaml    training configuration, with the provenance of every value
scripts/build_cache.sh       encode a split into VAE latents + text embeddings
scripts/train.sh             fine-tune the LoRA adapter
scripts/infer.py             generate edited plates
scripts/eval_ocr.py          ACC / NED through a TRBA recogniser
scripts/eval_image.py        FID, full-image LPIPS, masked-region LPIPS
src/                         model, data, loss, and training code
docs/DATA.md                 dataset layout and how each split is built
docs/REPRODUCIBILITY.md      what is attested, what is inferred, and what is missing
docs/CHECKPOINTS.md          released weights and how the step was chosen
```

## Requirements

One NVIDIA GPU with 24 GB is enough for both training and inference; the
original run used a single RTX 4090 under CUDA 12.2 with Python 3.10.

```bash
conda create -n lp2025 python=3.10 -y
conda activate lp2025
pip install -r requirements.txt
```

Versions in `requirements.txt` are pinned to the ones recorded in the original
training environment. `diffusers==0.32.2` in particular is not interchangeable —
later releases changed the FLUX transformer's forward signature that
`src/flux/` patches.

## Setup

**1. Base model.** Download FLUX.1-Fill-dev and quantise it to NF4, or use a
prepared NF4 copy. Point every script at that directory:

```bash
export FLUX_DIR=./FLUX.1-Fill-dev-nf4
```

The adapter was trained on the NF4-quantised base with `bnb_4bit_quant_type=nf4`,
`bnb_4bit_use_double_quant=false`, and `bnb_4bit_compute_dtype=bfloat16`. Loading
it onto an unquantised or double-quantised base changes the numerics it was
tuned against.

**2. ODM loss weights** (training only). The perceptual text loss needs its
pretrained encoder at `weights/epoch_100.pt` (~710 MB). Inference does not.

**3. Data.** See [docs/DATA.md](docs/DATA.md). The layout is four parallel
directories keyed by a shared stem:

```
data/<split>/
    filtered_plate/       <stem>.jpg    source plate crop
    partial_masks/        <stem>.png    region to repaint, white = edit
    partial_glyphs/       <stem>.png    rendered target glyph
    partial_labels_txt/   <stem>.txt    target text
```

| Split | Samples |
|---|---:|
| train | 2,569 |
| val | 620 |
| test | 3,258 |

## Train

```bash
bash scripts/build_cache.sh          # ~25 min for the training split on a 4090
bash scripts/train.sh configs/lp2025_train.yaml
```

Training reads the cache, not the images, and the mask is baked into the cached
tokens — so a cache built for one mask set cannot be reused for another.

Batch size is 1, so one epoch is 2,569 steps. The released checkpoint is step
27,606, about epoch 10–11. The original run continued to 43,784 steps without
improving validation loss; see [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) before
deciding to train longer.

## Generate

```bash
python scripts/infer.py \
    --data_root data/test \
    --config configs/lp2025_train.yaml \
    --lora weights/lp2025_27606/adapter_model.safetensors \
    --flux_dir $FLUX_DIR \
    --out outputs/lp2025_test \
    --compare
```

## Evaluate

```bash
# text accuracy — needs a clone of deep-text-recognition-benchmark
python scripts/eval_ocr.py \
    --image_folder outputs/lp2025_test \
    --saved_model weights/trba_lp2025/best_accuracy.pth \
    --dtr_root ../deep-text-recognition-benchmark

# image fidelity
python scripts/eval_image.py \
    --gen_dir outputs/lp2025_test \
    --real_dir data/test/filtered_plate \
    --mask_dir data/test/partial_masks
```

### Reading the metrics

Two of these numbers are routinely misread, so state them carefully:

**FID is not comparable across sample sizes.** It falls as N grows. On one fixed
image set we measured 7.28 at N=200 against 3.85 at N=1,000 — a factor of two
from sample size alone. Only compare FID values computed at the same N, and
always report N.

**Region LPIPS is a quality measure only for reconstruction.** When the target
text equals the original text there is a ground-truth image and lower is better.
When characters were deliberately replaced, the edited region is *supposed* to
differ from the source, so the value carries no quality information. Pass
`--edit` and the output says so.

**ACC is a lower bound, not an estimate.** The recogniser errs in one direction:
in a 409-sample human study on the CCPD extension of this work it never credited
a wrong character, but it missed 129 correct ones. Treat reported accuracy as a
floor on what the generator produced.

## Reproducibility

The original training run's directory was deleted; its wandb record survived.
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) marks every configuration
value as attested, inferred, or missing, so anyone reproducing this knows which
numbers are documented and which are reconstructed. Two blocks — the ODM loss
weights and the optimiser's secondary parameters — are taken from the same
script's nearest surviving configuration rather than from that run.

## Citation

<!-- Fill in once the paper reference is final. -->

## License

No license file is included yet. Add one before publishing: without it, others
have no permission to use the code. Note that FLUX.1-Fill-dev carries its own
non-commercial license, which governs the base weights independently of whatever
you choose here.
