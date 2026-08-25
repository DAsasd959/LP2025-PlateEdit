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
# CACHE_ROOT can point at a different volume. On a pod whose home is a small
# NFS share, the cache belongs on the large local disk instead; symlink
# ./cache there, or set this.
CACHE="${CACHE_ROOT:-cache}"

if [ ! -d "$FLUX" ]; then
  echo "base model not found at $FLUX — set FLUX_DIR" >&2; exit 1
fi

# Check the volume the cache actually lands on, following symlinks — the
# working directory is often on a different, much smaller filesystem.
mkdir -p "$CACHE"
NEED=$(( ( $(ls "$DATA/train/i_s" 2>/dev/null | wc -l) + $(ls "$DATA/val/i_s" 2>/dev/null | wc -l) ) * 10 / 1024 + 1 ))
FREE=$(df -BG --output=avail "$(readlink -f "$CACHE")" | tail -1 | tr -dc '0-9')
echo "cache target: $(readlink -f "$CACHE")  (${FREE} GB free, needs ~${NEED} GB)"
if [ "$FREE" -lt "$NEED" ]; then
  echo "not enough room. Point CACHE_ROOT at a larger volume." >&2
  exit 1
fi

for split in train val; do
  src="$DATA/$split"
  [ -d "$src/i_s" ] || { echo "missing $src/i_s" >&2; exit 1; }
  echo "=== caching $split ($(ls "$src/i_s" | wc -l) samples) ==="
  PP_DATASET=plate \
  PP_DATA_ROOT="$src" \
  PP_OUTPUT_ROOT="$CACHE/stage1_$split" \
  PP_FLUX_PATH="$FLUX" \
  FLUX_QUANTIZE="${FLUX_QUANTIZE:-nf4}" \
  python -m src.train.preprocess_partial
done

echo "train: $(ls "$CACHE/stage1_train" | wc -l) files"
echo "val:   $(ls "$CACHE/stage1_val"   | wc -l) files"
