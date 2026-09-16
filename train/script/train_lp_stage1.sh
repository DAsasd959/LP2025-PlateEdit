#!/usr/bin/env bash
# LP stage 1. Batch 1 x accum 64 -- fits 24 GB, same effective batch of 64 and the
# same 2,656 optimizer steps as the published H100 run. See train_lp_stage1_h100.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
run_train "${1:-train/config/lp/lp2025_stage1.yaml}"
