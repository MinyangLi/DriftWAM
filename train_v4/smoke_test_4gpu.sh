#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NGPU=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2}"
if [[ ! "${MAX_TRAIN_STEPS}" =~ ^[123]$ ]]; then
    echo "The history stress test supports only 1-3 optimizer updates." >&2
    exit 2
fi
export SAVE_INTERVAL=1000000000
export SKIP_FINAL_CHECKPOINT=1
export PRECOMPUTE_TEACHER_SIGNATURES=0
export OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-fs/wam/runs/driftwam_v4_history_stress_$(date -u +%Y%m%dT%H%M%SZ)}"

# Fail before loading data/models when the four training devices are absent.
python - <<'PYGPU'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit("History stress test requires exactly four visible CUDA GPUs")
PYGPU

exec bash "${SCRIPT_DIR}/run_flashwam500_clean50.sh" --history-stress-test "$@"
