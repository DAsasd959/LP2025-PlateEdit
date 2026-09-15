#!/usr/bin/env bash
# LP stage 2 -- real LP-2025 fine-tune. Set reuse_lora_path first.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
run_train "${1:-train/config/lp/lp2025_train.yaml}"
