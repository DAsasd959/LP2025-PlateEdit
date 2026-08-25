#!/usr/bin/env bash
# Pre-encode the synthetic stage-1 data into VAE latents + text embeddings.
#
#   PP_DATASET=plate  selects src/data/data_plate_partial.py, which reads
#                     <root>/{i_s,mask_s,i_s_bbox} — the Synthplate layout.
#
# SIZE WARNING. Each sample caches to ~9.45 MB, most of it the T5 embedding
# (512x4096 bf16 = 4.19 MB) and the fp32 ground-truth pixels (3.15 MB). A
# 270 KB PNG therefore becomes 9.45 MB of cache, a 36x expansion:
#
#     19,500 train  ->  ~173 GB
#        500 val    ->    ~4.7 GB
#
# This is why the images are shipped and the cache is rebuilt on the training
# machine rather than transferred. Check free space before starting.
#
# ~68 min for 20,000 samples on an RTX 3090.
set -euo pipefail
cd "$(dirname "$0")/.."

FLUX="${FLUX_DIR:-./FLUX.1-Fill-dev}"
DATA="${STAGE1_DATA:-data/stage1}"

if [ ! -d "$FLUX" ]; then
  echo "base model not found at $FLUX — set FLUX_DIR" >&2; exit 1
fi

NEED=190
FREE=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "$FREE" -lt "$NEED" ]; then
  echo "only ${FREE} GB free here; the stage-1 cache needs about ${NEED} GB." >&2
  echo "Point PP_OUTPUT_ROOT at a larger volume, or cache fewer samples." >&2
  exit 1
fi

for split in train val; do
  src="$DATA/$split"
  [ -d "$src/i_s" ] || { echo "missing $src/i_s" >&2; exit 1; }
  echo "=== caching $split ($(ls "$src/i_s" | wc -l) samples) ==="
  PP_DATASET=plate \
  PP_DATA_ROOT="$src" \
  PP_OUTPUT_ROOT="cache/stage1_$split" \
  PP_FLUX_PATH="$FLUX" \
  FLUX_QUANTIZE="${FLUX_QUANTIZE:-nf4}" \
  python -m src.train.preprocess_partial
done

echo "train: $(ls cache/stage1_train | wc -l) files"
echo "val:   $(ls cache/stage1_val   | wc -l) files"
