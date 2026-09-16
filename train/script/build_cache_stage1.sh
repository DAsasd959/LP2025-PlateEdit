#!/usr/bin/env bash
# Pre-encode synthetic stage-1 plates into VAE latents + text embeddings.
#
#   bash train/script/build_cache_stage1.sh [lp|ccpd]
#
# Both datasets use the same Synthplate layout -- <root>/{i_s,mask_s,i_s_bbox}
# with train/ and val/ underneath -- so one script serves both. The argument only
# decides where the cache lands, matching the cache_root in each config:
#
#   lp    -> cache/stage1_train       cache/stage1_val
#   ccpd  -> cache/ccpd_stage1_train  cache/ccpd_stage1_val
#
# For CCPD, set PP_PROVINCE_PROB=0.5. It is read in src/data/data_plate_partial.py
# and forces the Chinese province cell under the mask that often. At the default
# of 0 only 15.8% of masked spans contain Chinese and each province character sees
# about 474 updates across the run; at 0.5 that becomes 59.6% and ~11,920 updates.
#
#   PP_PROVINCE_PROB=0.5 STAGE1_DATA=<generated dir> \
#       bash train/script/build_cache_stage1.sh ccpd
#
# SIZE. Each sample caches to ~9.45 MB, most of it the T5 embedding (512x4096 bf16
# = 4.19 MB) and the fp32 ground-truth pixels (3.15 MB) -- a 270 KB PNG becomes
# 9.45 MB, a 36x expansion. The published runs used:
#
#   LP     20,000 train -> ~177 GB    1,000 val -> ~8.9 GB
#   CCPD   20,000 train -> ~177 GB      500 val -> ~4.5 GB
#
# This is why the images are generated locally rather than the cache transferred.
# About 68 minutes for 20,000 samples on an RTX 3090.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASET="${1:-lp}"
case "$DATASET" in
  lp)   PREFIX=stage1 ;;
  ccpd) PREFIX=ccpd_stage1 ;;
  *)    echo "usage: $0 [lp|ccpd]" >&2; exit 1 ;;
esac

FLUX="${FLUX_DIR:-weights/flux_base}"
DATA="${STAGE1_DATA:-data/${DATASET}/synth}"
# CACHE_ROOT can point at another volume; symlink ./cache there, or set it.
CACHE="${CACHE_ROOT:-cache}"

[ -d "$FLUX" ] || { echo "base model not found at $FLUX — set FLUX_DIR" >&2; exit 1; }

mkdir -p "$CACHE"
NEED=$(( ( $(ls "$DATA/train/i_s" 2>/dev/null | wc -l) + $(ls "$DATA/val/i_s" 2>/dev/null | wc -l) ) * 10 / 1024 + 1 ))
FREE=$(df -BG --output=avail "$(readlink -f "$CACHE")" | tail -1 | tr -dc '0-9')
echo "cache target: $(readlink -f "$CACHE")  (${FREE} GB free, needs ~${NEED} GB)"
[ "$FREE" -ge "$NEED" ] || { echo "not enough room. Point CACHE_ROOT at a larger volume." >&2; exit 1; }

for split in train val; do
  src="$DATA/$split"
  [ -d "$src/i_s" ] || { echo "missing $src/i_s" >&2; exit 1; }
  echo "=== caching $split ($(ls "$src/i_s" | wc -l) samples, province_prob=${PP_PROVINCE_PROB:-0}) ==="
  PP_DATASET=plate \
  PP_DATA_ROOT="$src" \
  PP_OUTPUT_ROOT="$CACHE/${PREFIX}_$split" \
  PP_FLUX_PATH="$FLUX" \
  FLUX_QUANTIZE="${FLUX_QUANTIZE:-nf4}" \
  python -m src.train.preprocess_partial
done

echo "train: $(ls "$CACHE/${PREFIX}_train" | wc -l) files"
echo "val:   $(ls "$CACHE/${PREFIX}_val"   | wc -l) files"
