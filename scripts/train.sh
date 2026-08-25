#!/usr/bin/env bash
# Fine-tune the LoRA adapter. The entry point takes its configuration from the
# XFL_CONFIG environment variable, not from argv.
#
# The released checkpoint is step 27606. The original run continued to 43784
# (epoch 17) without improving validation loss, so there is no reason to train
# longer than about 28k steps — see docs/REPRODUCIBILITY.md.
#
# ~24.6 h for the full 43,784 steps on one RTX 4090 (24 GB).
set -euo pipefail
cd "$(dirname "$0")/.."

export XFL_CONFIG="${1:-configs/lp2025_train.yaml}"
export TOKENIZERS_PARALLELISM=false

# Training takes the base model from the config, not from FLUX_DIR — the other
# scripts read that variable, so check the configured path before the run
# spends minutes loading only to fail on a missing directory.
FLUX_CFG=$(python - "$XFL_CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["flux_path"])
PY
)
for key in flux_path train_cache_root valid_cache_root odm_loss; do :; done
if [ ! -d "$FLUX_CFG" ]; then
  echo "flux_path in $XFL_CONFIG points at $FLUX_CFG, which does not exist." >&2
  echo "Edit that field, or place the base model there." >&2
  exit 1
fi

echo "config: $XFL_CONFIG"
echo "base model: $FLUX_CFG"
python -m src.train.train
