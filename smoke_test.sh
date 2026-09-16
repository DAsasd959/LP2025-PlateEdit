#!/usr/bin/env bash
# What works here, before you download 32 GB to find out.
#
#   bash smoke_test.sh
#
# Runs every check that needs no model weights and no GPU: imports, fonts, the
# condition builders, the synthetic generators, and a mask/glyph preview of the
# CCPD editor. Anything whose data is not present is skipped and said so -- a skip
# is not a failure, it tells you what to fetch.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
pass=0; fail=0; skip=0
ok(){   printf '  \033[32mok\033[0m    %s\n' "$1"; pass=$((pass+1)); }
no(){   printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); }
sk(){   printf '  --    %s  (%s)\n' "$1" "$2"; skip=$((skip+1)); }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "environment"
$PY - <<'P' && ok "python imports" || no "python imports"
import sys
import torch, diffusers, transformers, peft, bitsandbytes, lightning, lpips, cv2
from PIL import Image
print(f"    python {sys.version.split()[0]}  torch {torch.__version__}  "
      f"diffusers {diffusers.__version__}  cuda {torch.version.cuda}")
print(f"    gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none visible'}")
P
[ "$(  $PY -c 'import diffusers;print(diffusers.__version__)' 2>/dev/null)" = "0.32.2" ] \
  && ok "diffusers pinned at 0.32.2" || no "diffusers is not 0.32.2 -- LoRA loading will break"

echo "repository modules"
$PY - <<'P' && ok "src/ imports" || no "src/ imports"
import sys; sys.path.insert(0, ".")
for m in ("src.flux.generate_fill", "src.train.model", "src.train.train",
          "src.data.data_ccpd", "src.data.data_real", "src.loss.ocr_loss.odm_loss"):
    __import__(m)
P

echo "fonts"
for f in font/TWGen7_V1.ttf "font/正黑體.ttf"; do
  [ -f "$f" ] && ok "$f" || no "$f missing -- see font/README.md"
done

echo "downloads"
[ -d weights/flux_base ] && ok "weights/flux_base" || sk "weights/flux_base" "docs/CHECKPOINTS.md"
for d in data/lp/test data/ccpd; do
  [ -d "$d" ] && ok "$d" || sk "$d" "docs/CHECKPOINTS.md"
done
for s in tw cn; do
  n=$(ls "synth/$s/datasets/bg_data/bg_img" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && ok "synth/$s backgrounds ($n)" || sk "synth/$s backgrounds" "synth/README.md"
done

echo "CCPD condition builder (cpu)"
SRC=${CCPD_PLATES:-data/ccpd/test_1000/plates}
if [ -d "$SRC" ]; then
  $PY eval/ccpd/build_ccpd_conditions.py --src "$SRC" --out "$TMP/cond" \
      --province_ratio 0.5 --max_k 3 --limit 8 >/dev/null 2>&1 \
    && [ "$(ls "$TMP/cond/partial_masks" 2>/dev/null | wc -l)" -eq 8 ] \
    && ok "build_ccpd_conditions.py" || no "build_ccpd_conditions.py"
else
  sk "build_ccpd_conditions.py" "no CCPD plates"
fi

echo "LP sample preparation (cpu)"
LP=${LP_PLATES:-data/lp/test/filtered_plate}
img=$(ls "$LP"/*.jpg 2>/dev/null | head -1)
if [ -n "${img:-}" ]; then
  stem=$(basename "$img" .jpg); txt=${stem##*_}
  $PY eval/lp/prepare_sample.py --image "$img" --text "$txt" \
      --quad "10,10 10,60 200,70 200,15" --span 0:2 --font font/TWGen7_V1.ttf \
      --out "$TMP/lp" --stem probe >/dev/null 2>&1 \
    && [ -f "$TMP/lp/partial_masks/probe.png" ] \
    && ok "prepare_sample.py" || no "prepare_sample.py"
else
  sk "prepare_sample.py" "no LP plates"
fi

echo "CCPD editor geometry (cpu, --dry_run)"
if [ -d "$SRC" ]; then
  $PY eval/ccpd/mixed_span_edit.py --dry_run --limit 4 --span_mode cross --max_k 3 \
      --plate_dir "$SRC" --out "$TMP/dry" >/dev/null 2>&1 \
    && [ -s "$TMP/dry/labels.txt" ] && ok "mixed_span_edit.py --dry_run" \
    || no "mixed_span_edit.py --dry_run"
else
  sk "mixed_span_edit.py --dry_run" "no CCPD plates"
fi

echo "synthetic generators (cpu, second environment)"
# SYNTH_PYTHON points at the environment built from synth/requirements.txt.
SP_="${SYNTH_PYTHON:-$PY}"
$SP_ -c "import pygame, Augmentor" 2>/dev/null || {
  sk "synth generators" "pygame/Augmentor not here -- set SYNTH_PYTHON, see synth/README.md"
  SP_=""
}
for s in tw cn; do
  if [ -z "$SP_" ]; then :
  elif [ "$(ls "synth/$s/datasets/bg_data/bg_img" 2>/dev/null | wc -l)" -gt 0 ]; then
    ( cd "synth/$s" && $SP_ - <<'P' >/dev/null 2>&1
import cfg, sys
sys.argv = [sys.argv[0]]
from Synthtext.gen import datagen
g = datagen()
g.gen_srnet_data_with_background()
P
    ) && ok "synth/$s generator" || no "synth/$s generator"
  else
    sk "synth/$s generator" "backgrounds not downloaded"
  fi
done

echo
printf '%d ok, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || echo "A failure here is a broken environment, not a missing download."
exit $(( fail > 0 ))
