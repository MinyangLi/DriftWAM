#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ -z "${STUDENT_INIT_SOURCE:-}" ]]; then
    echo "STUDENT_INIT_SOURCE must be explicitly set to flash_wam or teacher" >&2
    exit 2
fi

ARGS=(--student-init-source "${STUDENT_INIT_SOURCE}")
[[ -n "${DATASET_PATH:-}" ]] && ARGS+=(--dataset-path "${DATASET_PATH}")
[[ -n "${EMPTY_EMB_PATH:-}" ]] && ARGS+=(--empty-emb-path "${EMPTY_EMB_PATH}")
[[ -n "${TEACHER_VIDEO_BANK_PATH:-}" ]] && ARGS+=(--teacher-video-bank-path "${TEACHER_VIDEO_BANK_PATH}")
[[ -n "${TEACHER_SIGNATURE_CACHE_PATH:-}" ]] && ARGS+=(--teacher-signature-cache-path "${TEACHER_SIGNATURE_CACHE_PATH}")
[[ -n "${TEACHER_PATH:-}" ]] && ARGS+=(--teacher-model-path "${TEACHER_PATH}")
[[ -n "${STUDENT_INIT_PATH:-}" ]] && ARGS+=(--student-init-path "${STUDENT_INIT_PATH}")
[[ -n "${OUTPUT_DIR:-}" ]] && ARGS+=(--output-dir "${OUTPUT_DIR}")
[[ -n "${RESUME_FROM_PATH:-}" ]] && ARGS+=(--resume-from-path "${RESUME_FROM_PATH}")
[[ -n "${RESUME_FROM_STEP:-}" ]] && ARGS+=(--resume-from-step "${RESUME_FROM_STEP}")
[[ -n "${EXPERIMENT_NAME:-}" ]] && ARGS+=(--experiment-name "${EXPERIMENT_NAME}")
[[ -n "${EXPECTED_DATASET_COUNT:-}" ]] && ARGS+=(--expected-dataset-count "${EXPECTED_DATASET_COUNT}")
[[ -n "${EXPECTED_SOURCE_COUNT:-}" ]] && ARGS+=(--expected-source-count "${EXPECTED_SOURCE_COUNT}")
[[ -n "${EXPECTED_WORLD_SIZE:-}" ]] && ARGS+=(--expected-world-size "${EXPECTED_WORLD_SIZE}")
[[ -n "${EXPECTED_GLOBAL_BATCH_SIZE:-}" ]] && ARGS+=(--expected-global-batch-size "${EXPECTED_GLOBAL_BATCH_SIZE}")
[[ -n "${MAX_TRAIN_STEPS:-}" ]] && ARGS+=(--max-train-steps "${MAX_TRAIN_STEPS}")
[[ -n "${BATCH_SIZE:-}" ]] && ARGS+=(--batch-size "${BATCH_SIZE}")
[[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]] && ARGS+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")
[[ -n "${WARMUP_STEPS:-}" ]] && ARGS+=(--warmup-steps "${WARMUP_STEPS}")
[[ -n "${SEED:-}" ]] && ARGS+=(--seed "${SEED}")
[[ -n "${TEACHER_SIGNATURE_NOISE_SEED:-}" ]] && ARGS+=(--teacher-signature-noise-seed "${TEACHER_SIGNATURE_NOISE_SEED}")
[[ -n "${SAVE_INTERVAL:-}" ]] && ARGS+=(--save-interval "${SAVE_INTERVAL}")
[[ -n "${DIAGNOSTICS_INTERVAL:-}" ]] && ARGS+=(--diagnostics-interval "${DIAGNOSTICS_INTERVAL}")
[[ -n "${NUM_WORKERS:-}" ]] && ARGS+=(--num-workers "${NUM_WORKERS}")
[[ -n "${EXECUTION_RESPONSE_MODE:-}" ]] && ARGS+=(--execution-response-mode "${EXECUTION_RESPONSE_MODE}")
[[ -n "${EXECUTION_LOSS_WEIGHT:-}" ]] && ARGS+=(--execution-loss-weight "${EXECUTION_LOSS_WEIGHT}")
[[ -n "${ACTION_CONSISTENCY_LOSS_WEIGHT:-}" ]] && ARGS+=(--action-consistency-loss-weight "${ACTION_CONSISTENCY_LOSS_WEIGHT}")
[[ -n "${ACTION_FLOW_MATCHING_LOSS_WEIGHT:-}" ]] && ARGS+=(--action-flow-matching-loss-weight "${ACTION_FLOW_MATCHING_LOSS_WEIGHT}")
if [[ -n "${VIDEO_BANDWIDTH:-}" || -n "${ACTION_BANDWIDTH:-}" ]]; then
    if [[ -z "${VIDEO_BANDWIDTH:-}" || -z "${ACTION_BANDWIDTH:-}" ]]; then
        echo "VIDEO_BANDWIDTH and ACTION_BANDWIDTH must be set together" >&2
        exit 2
    fi
    ARGS+=(--video-bandwidth "${VIDEO_BANDWIDTH}")
    ARGS+=(--action-bandwidth "${ACTION_BANDWIDTH}")
fi

[[ "${DISABLE_OPTIMIZER_CHECKPOINT:-0}" == "1" ]] && ARGS+=(--disable-optimizer-checkpoint)
[[ "${DISABLE_OPTIMIZER_STATE_OFFLOAD:-0}" == "1" ]] && ARGS+=(--disable-optimizer-state-offload)
[[ "${SKIP_FINAL_CHECKPOINT:-0}" == "1" ]] && ARGS+=(--skip-final-checkpoint)
[[ "${DISABLE_WANDB:-0}" == "1" ]] && ARGS+=(--disable-wandb)
ARGS+=("$@")

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-1}"
# PyTorch 2.9 in this environment still reads the legacy variable.
# Otherwise allocator diagnostics report expandable_segments=False.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${PROJECT_ROOT}"

torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    --module train_v4.train \
    "${ARGS[@]}"
