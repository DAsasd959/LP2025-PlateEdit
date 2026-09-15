#!/usr/bin/env bash
# LP stage 1 on a 24 GB card (batch 1 x accum 64). Same effective batch, same 2656 optimizer steps.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
run_train "${1:-train/config/lp/lp2025_stage1_3090_bs1.yaml}"
