#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

NGPU="${NGPU:-4}"
if [[ ! "${NGPU}" =~ ^[0-9]+$ ]] || (( NGPU < 1 || 32 % NGPU != 0 )); then
    echo "NGPU must be a positive divisor of 32 so global batch remains 32." >&2
    exit 2
fi

export STUDENT_INIT_SOURCE="flash_wam"
export STUDENT_INIT_PATH="/root/autodl-fs/wam/models/FlashWAM-RoboTwin"
export TEACHER_PATH="/root/autodl-fs/wam/models/lingbot-va-posttrain-robotwin"
export DATASET_PATH="/root/autodl-tmp/robotwin-lerobot/lerobot_robotwin_eef_clean_50"
export TEACHER_VIDEO_BANK_PATH="/root/autodl-tmp/robotwin-lerobot/teacher_video_bank_clean_50"
export TEACHER_SIGNATURE_CACHE_PATH="${TEACHER_SIGNATURE_CACHE_PATH:-/root/autodl-tmp/robotwin-lerobot/teacher_signature_cache_clean_50_seed42_v3}"
export EMPTY_EMB_PATH="/root/autodl-tmp/robotwin-lerobot/empty_emb.pt"

export EXPERIMENT_NAME="driftwam_v4_action_response_clean50_seed42"
export EXPECTED_DATASET_COUNT="50"
export EXPECTED_SOURCE_COUNT="2492"
export EXPECTED_WORLD_SIZE="${NGPU}"
export EXPECTED_GLOBAL_BATCH_SIZE="32"
export BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="$((32 / NGPU))"
export MAX_UNSYNCED_HISTORY_FRAMES="${MAX_UNSYNCED_HISTORY_FRAMES:-24}"
export WARMUP_STEPS="100"
export SEED="42"
export TEACHER_SIGNATURE_NOISE_SEED="${TEACHER_SIGNATURE_NOISE_SEED:-42}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
export DIAGNOSTICS_INTERVAL="${DIAGNOSTICS_INTERVAL:-10}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-250}"
export NUM_WORKERS="0"
if [[ "${DISABLE_WANDB:-1}" != "1" ]]; then
    echo "Formal v4 profiles require DISABLE_WANDB=1; diagnostics are written locally." >&2
    exit 2
fi
export DISABLE_WANDB="1"
export EXECUTION_RESPONSE_MODE="${EXECUTION_RESPONSE_MODE:-recompute_selected}"
export EXECUTION_LOSS_WEIGHT="${EXECUTION_LOSS_WEIGHT:-0.1}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-fs/wam/runs/driftwam_v4_action_response_clean50_seed42}"
export ACTION_CONSISTENCY_LOSS_WEIGHT="${ACTION_CONSISTENCY_LOSS_WEIGHT:-1.0}"
export ACTION_FLOW_MATCHING_LOSS_WEIGHT="${ACTION_FLOW_MATCHING_LOSS_WEIGHT:-0.01}"

if [[ -n "${RESUME_FROM_PATH:-}" || -n "${RESUME_FROM_STEP:-}" ]]; then
    if [[ "${ALLOW_RESUME:-0}" != "1" ]]; then
        echo "Resume variables are set; pass ALLOW_RESUME=1 to resume intentionally." >&2
        exit 2
    fi
elif [[ "${ALLOW_RESUME:-0}" == "1" ]]; then
    echo "ALLOW_RESUME=1 requires RESUME_FROM_PATH or RESUME_FROM_STEP." >&2
    exit 2
fi

if [[ ! -f "${TEACHER_VIDEO_BANK_PATH}/BANK_COMPLETE.json" ]]; then
    cd "${PROJECT_ROOT}"
    python -m train_v4.validate_teacher_bank \
        --dataset-path "${DATASET_PATH}" \
        --teacher-video-bank-path "${TEACHER_VIDEO_BANK_PATH}" \
        --teacher-model-path "${TEACHER_PATH}" \
        --expected-dataset-count "${EXPECTED_DATASET_COUNT}" \
        --expected-source-count "${EXPECTED_SOURCE_COUNT}" \
        --write-marker
fi

if [[ ! -f "${TEACHER_SIGNATURE_CACHE_PATH}/CACHE_COMPLETE.json" ]]; then
    if [[ "${PRECOMPUTE_TEACHER_SIGNATURES:-1}" != "1" ]]; then
        echo "Teacher-signature cache is incomplete: ${TEACHER_SIGNATURE_CACHE_PATH}" >&2
        echo "Set PRECOMPUTE_TEACHER_SIGNATURES=1 or run precompute_teacher_signatures.sh first." >&2
        exit 2
    fi
    CACHE_NGPU="${CACHE_NGPU:-${NGPU}}" \
        bash "${SCRIPT_DIR}/precompute_teacher_signatures.sh"
fi

exec bash "${SCRIPT_DIR}/run.sh" "$@"
