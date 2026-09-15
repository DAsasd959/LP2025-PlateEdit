#!/usr/bin/env bash
# CCPD stage 2 -- province-balanced real fine-tune.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
run_train "${1:-train/config/ccpd/cn_v2_stage2_fix.yaml}"
