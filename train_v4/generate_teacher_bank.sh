#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29502}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-1}"
export PYTHONUNBUFFERED=1

ARGS=()
[[ -n "${DATASET_PATH:-}" ]] && ARGS+=(--dataset-path "${DATASET_PATH}")
[[ -n "${EMPTY_EMB_PATH:-}" ]] && ARGS+=(--empty-emb-path "${EMPTY_EMB_PATH}")
[[ -n "${TEACHER_VIDEO_BANK_PATH:-}" ]] && ARGS+=(--teacher-video-bank-path "${TEACHER_VIDEO_BANK_PATH}")
[[ -n "${TEACHER_PATH:-}" ]] && ARGS+=(--teacher-model-path "${TEACHER_PATH}")
[[ -n "${LINGBOT_VA_ROOT:-}" ]] && ARGS+=(--lingbot-va-root "${LINGBOT_VA_ROOT}")
[[ -n "${TEACHER_VIDEO_NUM_INFERENCE_STEPS:-}" ]] && ARGS+=(--num-inference-steps "${TEACHER_VIDEO_NUM_INFERENCE_STEPS}")
[[ -n "${TEACHER_VIDEO_GUIDANCE_SCALE:-}" ]] && ARGS+=(--guidance-scale "${TEACHER_VIDEO_GUIDANCE_SCALE}")
[[ -n "${TEACHER_BANK_CANDIDATE_BATCH_SIZE:-}" ]] && ARGS+=(--candidate-batch-size "${TEACHER_BANK_CANDIDATE_BATCH_SIZE}")
[[ -n "${TEACHER_BANK_SOURCE_BATCH_SIZE:-}" ]] && ARGS+=(--source-batch-size "${TEACHER_BANK_SOURCE_BATCH_SIZE}")
[[ -n "${TEACHER_BANK_SEED:-}" ]] && ARGS+=(--seed "${TEACHER_BANK_SEED}")
[[ -n "${START_INDEX:-}" ]] && ARGS+=(--start-index "${START_INDEX}")
[[ -n "${END_INDEX:-}" ]] && ARGS+=(--end-index "${END_INDEX}")
[[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max-samples "${MAX_SAMPLES}")
[[ "${OVERWRITE:-0}" == "1" ]] && ARGS+=(--overwrite)

cd "${PROJECT_ROOT}"

torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    --module train_v4.generate_teacher_bank \
    "${ARGS[@]}"
