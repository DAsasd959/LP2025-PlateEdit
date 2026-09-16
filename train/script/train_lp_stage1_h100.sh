#!/usr/bin/env bash
# LP stage 1 as published: batch 8 on an H100. Does NOT fit 24 GB -- use train_lp_stage1.sh there.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
run_train "${1:-train/config/lp/lp2025_stage1_h100.yaml}"
