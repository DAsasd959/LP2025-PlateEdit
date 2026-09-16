#!/usr/bin/env bash
# Pre-encode the real LP-2025 splits for stage 2.
#
#   bash train/script/build_cache.sh
#
# Training reads these caches, not the images. The mask is baked into the cached
# tokens, so a cache built for one mask set cannot be reused for another.
#
# Output matches the cache_root in train/config/lp/lp2025_train.yaml:
#   cache/lp_stage2_train   cache/lp_stage2_val
#
# CCPD stage 2 uses train/script/cn_preprocess.py instead -- it needs the
# seven-cell layout so the Chinese province character can fall under the mask.
#
# About 25 minutes for 2,569 samples on an RTX 4090; ~29 GB of cache.
set -euo pipefail
cd "$(dirname "$0")/../.."

FLUX="${FLUX_DIR:-weights/flux_base}"
DATA="${LP_DATA:-data/lp}"
CACHE="${CACHE_ROOT:-cache}"

[ -d "$FLUX" ] || { echo "base model not found at $FLUX — set FLUX_DIR" >&2; exit 1; }

for split in train val; do
  [ -d "$DATA/$split/partial_glyphs" ] || { echo "missing $DATA/$split/partial_glyphs" >&2; exit 1; }
  echo "=== caching $split ($(ls "$DATA/$split/partial_glyphs" | wc -l) samples) ==="
  PP_DATASET=real \
  PP_DATA_ROOT="$DATA/$split" \
  PP_OUTPUT_ROOT="$CACHE/lp_stage2_$split" \
  PP_FLUX_PATH="$FLUX" \
  python -m src.train.preprocess_partial
done

echo "train: $(ls "$CACHE/lp_stage2_train" | wc -l) files   (expected 2569)"
echo "val:   $(ls "$CACHE/lp_stage2_val"   | wc -l) files   (expected 620)"
