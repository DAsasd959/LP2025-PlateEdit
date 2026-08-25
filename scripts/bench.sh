#!/usr/bin/env bash
# Measure real seconds-per-step on this machine before committing to a long run.
#
#   bash scripts/bench.sh [steps] [config]
#
# Runs a short training job with checkpointing, sampling and validation pushed
# out of range, then reports the throughput the progress bar settled on. Model
# load takes a few minutes and is excluded from the per-step figure.
#
# Reference, measured: RTX 3090, NF4, gradient checkpointing on -> 4.46 s/step
# (runs_cn_v2_stage1: 20,000 steps in 24 h 46 m).
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-200}"
BASE="${2:-configs/lp2025_stage1.yaml}"
OUT="${BENCH_OUT:-runs_bench}"
CFG="$OUT/bench.yaml"
LOG="$OUT/bench.log"
mkdir -p "$OUT"

python - "$BASE" "$CFG" "$STEPS" "$OUT" <<'PY'
import sys, yaml
base, out_cfg, steps, out_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
c = yaml.safe_load(open(base))
t = c["train"]
t["max_steps"] = steps
t["save_interval"] = steps * 10          # never fires
t["sample_interval"] = steps * 10
t["dataset"]["val_check_interval"] = steps * 10
t["save_path"] = out_dir
t.pop("wandb", None)                     # keep the probe out of the project history
yaml.safe_dump(c, open(out_cfg, "w"), sort_keys=False, allow_unicode=True)
print(f"benchmark config: {out_cfg}  ({steps} steps)")
PY

export XFL_CONFIG="$CFG"
export TOKENIZERS_PARALLELISM=false
echo "quantisation: ${FLUX_QUANTIZE:-nf4}   (set FLUX_QUANTIZE=none on ROCm)"

START=$(date +%s)
python -m src.train.train 2>&1 | tee "$LOG"
WALL=$(( $(date +%s) - START ))

echo
echo "=================== benchmark ==================="
printf 'wall clock      %d s (includes model load)\n' "$WALL"

# The progress bar reports either it/s or s/it; normalise to s/step.
RATE=$(grep -oE '[0-9.]+(it/s|s/it)' "$LOG" | tail -1)
if [ -n "$RATE" ]; then
  SPS=$(python - "$RATE" <<'PY'
import sys, re
m = re.match(r'([0-9.]+)(it/s|s/it)', sys.argv[1])
v = float(m.group(1))
print(f"{v if m.group(2)=='s/it' else 1.0/v:.3f}")
PY
)
  echo "steady state    $SPS s/step   (from '$RATE')"
  python - "$SPS" <<'PY'
import sys
s = float(sys.argv[1])
print("\nprojected stage-1 training time:")
for n in (3000, 8000, 19500):
    h = n * s / 3600
    print(f"  {n:>6,} steps   {h:6.1f} h" + ("   <- config default" if n == 19500 else ""))
PY
else
  echo "could not parse a rate from the progress bar; see $LOG"
fi
echo "================================================="
