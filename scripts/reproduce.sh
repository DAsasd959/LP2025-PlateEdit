#!/usr/bin/env bash
# Reproduce the LP2025 test-set numbers from the released checkpoint.
#
#   bash scripts/reproduce.sh /path/to/LP2025 /path/to/deep-text-recognition-benchmark
#
# Expected on the full test split (n = 3258): ACC 0.8109, NED 0.9549.
# Generation dominates the runtime — about 17 s per image on an RTX 4090, so the
# full split is roughly 15 hours. Set LIMIT to sample a subset first.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${1:?usage: reproduce.sh <LP2025-root> <deep-text-recognition-benchmark>}"
DTR="${2:?usage: reproduce.sh <LP2025-root> <deep-text-recognition-benchmark>}"
FLUX="${FLUX_DIR:-./FLUX.1-Fill-dev}"
OUT="${OUT_DIR:-outputs/lp2025_test}"
LIMIT="${LIMIT:-0}"

echo "=== checking the checkout ==="
python scripts/check_data.py --root "$ROOT" --weights weights

echo "=== generating -> $OUT ==="
python scripts/infer.py \
    --data_root "$ROOT" \
    --config configs/lp2025_train.yaml \
    --lora weights/lp2025_27606/adapter_model.safetensors \
    --flux_dir "$FLUX" \
    --out "$OUT" \
    --limit "$LIMIT"

echo "=== text accuracy ==="
python scripts/eval_ocr.py \
    --image_folder "$OUT" \
    --saved_model weights/trba_lp2025/best_accuracy.pth \
    --dtr_root "$DTR"

echo "=== image fidelity ==="
python scripts/eval_image.py \
    --gen_dir "$OUT" \
    --real_dir "$ROOT/filtered_plate" \
    --mask_dir "$ROOT/partial_masks"
