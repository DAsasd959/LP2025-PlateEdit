#!/usr/bin/env bash
# Pack synthetic stage-1 data for transfer to the training machine.
#
#   bash scripts/pack_stage1.sh <synth_root> <out_dir> [n_train] [n_val]
#
# Takes the Synthplate output directory and writes one tarball holding
#
#     stage1/train/{i_s,mask_s,i_s_bbox}
#     stage1/val/{i_s,mask_s,i_s_bbox}
#
# t_b (the clean background) is deliberately left out: the training path never
# opens it, and it is as large as i_s.
#
# The 'h' in czhf dereferences symlinks. Without it a tarball unpacked on
# another machine contains dangling links that fail at Image.open() — the bug
# that shipped in the first data release.
set -euo pipefail

SRC="${1:?usage: pack_stage1.sh <synth_root> <out_dir> [n_train] [n_val]}"
OUT="${2:?missing out_dir}"
NTRAIN="${3:-19500}"
NVAL="${4:-500}"

for d in i_s mask_s i_s_bbox; do
  [ -d "$SRC/$d" ] || { echo "missing $SRC/$d" >&2; exit 1; }
done

TOTAL=$(ls "$SRC/i_s" | wc -l)
if [ "$TOTAL" -lt $((NTRAIN + NVAL)) ]; then
  echo "$SRC holds $TOTAL samples; asked for $NTRAIN + $NVAL." >&2; exit 1
fi

mkdir -p "$OUT"
STAGE=$(mktemp -d "$OUT/.pack.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT

# Validation is taken from the tail so that shrinking n_train leaves the same
# held-out samples — two runs at different lengths stay comparable.
mapfile -t STEMS < <(ls "$SRC/i_s" | sed 's/\.png$//' | sort)
for split in train val; do
  if [ "$split" = train ]; then sel=("${STEMS[@]:0:$NTRAIN}")
  else sel=("${STEMS[@]: -$NVAL}"); fi
  for d in i_s mask_s i_s_bbox; do mkdir -p "$STAGE/stage1/$split/$d"; done
  for s in "${sel[@]}"; do
    cp "$SRC/i_s/$s.png"        "$STAGE/stage1/$split/i_s/"
    cp "$SRC/mask_s/$s.png"     "$STAGE/stage1/$split/mask_s/"
    cp "$SRC/i_s_bbox/$s.txt"   "$STAGE/stage1/$split/i_s_bbox/"
  done
  echo "$split: ${#sel[@]} samples"
done

TAR="$OUT/lp2025_stage1.tar.gz"
tar -czhf "$TAR" -C "$STAGE" stage1
echo "symlinks in tarball: $(tar -tvzf "$TAR" | grep -c '^l' || true)   (must be 0)"
sha256sum "$TAR" | tee "$OUT/stage1_checksum.txt"
du -h "$TAR"
