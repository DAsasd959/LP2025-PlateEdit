#!/usr/bin/env bash
# LP stage 1 -- paper values (batch 8). Needs more than 24 GB; see the 3090 variant.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
run_train "${1:-train/config/lp/lp2025_stage1.yaml}"
