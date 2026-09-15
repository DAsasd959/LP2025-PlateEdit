#!/usr/bin/env bash
# CCPD stage 1 -- synthetic pretraining.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
run_train "${1:-train/config/ccpd/cn_v2_stage1.yaml}"
