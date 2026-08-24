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
echo "config: $XFL_CONFIG"
python -m src.train.train
