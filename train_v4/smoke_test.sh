#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STUDENT_INIT_SOURCE="flash_wam"
export STUDENT_INIT_PATH="/root/autodl-fs/wam/models/FlashWAM-RoboTwin"
export TEACHER_PATH="/root/autodl-fs/wam/models/lingbot-va-posttrain-robotwin"
export DATASET_PATH="/root/autodl-tmp/robotwin-lerobot/lerobot_robotwin_eef_clean_50"
export TEACHER_VIDEO_BANK_PATH="/root/autodl-tmp/robotwin-lerobot/teacher_video_bank_clean_50"
export TEACHER_SIGNATURE_CACHE_PATH="${TEACHER_SIGNATURE_CACHE_PATH:-/root/autodl-tmp/robotwin-lerobot/teacher_signature_cache_clean_50_seed42_v3}"
export EMPTY_EMB_PATH="/root/autodl-tmp/robotwin-lerobot/empty_emb.pt"

export NGPU="1"
export EXPECTED_DATASET_COUNT="50"
export EXPECTED_SOURCE_COUNT="2492"
export EXPECTED_WORLD_SIZE="1"
export EXPECTED_GLOBAL_BATCH_SIZE="1"
export BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="1"
export MAX_TRAIN_STEPS="1"
export WARMUP_STEPS="0"
export SAVE_INTERVAL="1000000000"
export NUM_WORKERS="0"
export DISABLE_WANDB="1"
export EXECUTION_RESPONSE_MODE="${EXECUTION_RESPONSE_MODE:-recompute_selected}"
export EXECUTION_LOSS_WEIGHT="${EXECUTION_LOSS_WEIGHT:-0.1}"
export SKIP_FINAL_CHECKPOINT="1"
export EXPERIMENT_NAME="driftwam_v4_action_response_smoke"
export ACTION_CONSISTENCY_LOSS_WEIGHT="${ACTION_CONSISTENCY_LOSS_WEIGHT:-1.0}"
export ACTION_FLOW_MATCHING_LOSS_WEIGHT="${ACTION_FLOW_MATCHING_LOSS_WEIGHT:-0.01}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-fs/wam/runs/driftwam_v4_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! -f "${TEACHER_SIGNATURE_CACHE_PATH}/CACHE_COMPLETE.json" ]]; then
    echo "Complete the teacher-signature cache before the smoke test:" >&2
    echo "  bash ${SCRIPT_DIR}/precompute_teacher_signatures.sh" >&2
    exit 2
fi

exec bash "${SCRIPT_DIR}/run.sh" "$@"
