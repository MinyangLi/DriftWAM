#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

CACHE_NGPU="${CACHE_NGPU:-1}"
CACHE_MASTER_PORT="${CACHE_MASTER_PORT:-29503}"

ARGS=()
[[ -n "${DATASET_PATH:-}" ]] && ARGS+=(--dataset-path "${DATASET_PATH}")
[[ -n "${EMPTY_EMB_PATH:-}" ]] && ARGS+=(--empty-emb-path "${EMPTY_EMB_PATH}")
[[ -n "${TEACHER_VIDEO_BANK_PATH:-}" ]] && ARGS+=(--teacher-video-bank-path "${TEACHER_VIDEO_BANK_PATH}")
[[ -n "${TEACHER_SIGNATURE_CACHE_PATH:-}" ]] && ARGS+=(--teacher-signature-cache-path "${TEACHER_SIGNATURE_CACHE_PATH}")
[[ -n "${TEACHER_PATH:-}" ]] && ARGS+=(--teacher-model-path "${TEACHER_PATH}")
[[ -n "${LINGBOT_VA_ROOT:-}" ]] && ARGS+=(--lingbot-va-root "${LINGBOT_VA_ROOT}")
[[ -n "${TEACHER_SIGNATURE_NOISE_SEED:-}" ]] && ARGS+=(--teacher-signature-noise-seed "${TEACHER_SIGNATURE_NOISE_SEED}")
[[ -n "${EXPECTED_DATASET_COUNT:-}" ]] && ARGS+=(--expected-dataset-count "${EXPECTED_DATASET_COUNT}")
[[ -n "${EXPECTED_SOURCE_COUNT:-}" ]] && ARGS+=(--expected-source-count "${EXPECTED_SOURCE_COUNT}")
[[ -n "${CACHE_START_INDEX:-}" ]] && ARGS+=(--start-index "${CACHE_START_INDEX}")
[[ -n "${CACHE_END_INDEX:-}" ]] && ARGS+=(--end-index "${CACHE_END_INDEX}")
[[ -n "${CACHE_MAX_SOURCES:-}" ]] && ARGS+=(--max-sources "${CACHE_MAX_SOURCES}")
ARGS+=("$@")

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
cd "${PROJECT_ROOT}"

torchrun \
    --nproc_per_node="${CACHE_NGPU}" \
    --master_port="${CACHE_MASTER_PORT}" \
    --module train_v4.precompute_teacher_signatures \
    "${ARGS[@]}"
