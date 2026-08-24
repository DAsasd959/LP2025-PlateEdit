#!/usr/bin/env bash
# Pre-encode the dataset into VAE latents + text embeddings.
#
# Training reads these caches, not the images. The mask is baked into the
# cached tokens, so a cache built for one mask set cannot be reused for another.
#
#   PP_DATASET=real   selects src/data/data_real.py, the LP2025 layout
#   PP_DATA_ROOT      the split to encode
#   PP_OUTPUT_ROOT    where the .pt files land
#
# Runtime is roughly 25 min for 2,569 samples on an RTX 4090.
set -euo pipefail
cd "$(dirname "$0")/.."

FLUX="${FLUX_DIR:-./FLUX.1-Fill-dev-nf4}"

for split in train val; do
  echo "=== caching $split ==="
  PP_DATASET=real \
  PP_DATA_ROOT="data/$split" \
  PP_OUTPUT_ROOT="cache/lp_$split" \
  PP_FLUX_PATH="$FLUX" \
  python -m src.train.preprocess_partial
done

echo "train: $(ls cache/lp_train | wc -l) files   (expected 2569)"
echo "val:   $(ls cache/lp_val   | wc -l) files   (expected 620)"
