#!/usr/bin/env bash
# Helper launcher for Muon optimizer experiments in TorchTitan.
# Mirrors run_train.sh but defaults to the Muon test config.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

NGPU="${NGPU:-1}"
LOG_RANK="${LOG_RANK:-0}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/torchtitan/experiments/muon/muon_test.toml}"
TRAIN_MODULE="${TRAIN_MODULE:-torchtitan.train}"
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"

PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter="${LOG_RANK}" \
    --role rank \
    --tee 3 \
    -m "${TRAIN_MODULE}" \
    --job.config_file "${CONFIG_FILE}" \
    "$@"
