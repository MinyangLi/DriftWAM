"""Precompute all paired teacher-video action signatures for v4 training."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from .engine.bank_validation import verify_teacher_bank_marker
from .engine.teacher_signature_cache import (
    CACHE_COMPLETE_FILENAME,
    TeacherSignatureCache,
    prepare_teacher_signature_cache,
    write_teacher_signature_cache_complete,
)
from .engine.training_data import (
    DriftingTrainingDataset,
    collate_training_samples,
)
from .forwards.action_signature import ActionSignatureGenerator
from .model_setup import configure_frozen_teacher
from .training_config import TrainingConfig


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute all 50-step teacher actions required by train_v4"
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--empty-emb-path", type=Path)
    parser.add_argument("--teacher-video-bank-path", type=Path)
    parser.add_argument("--teacher-signature-cache-path", type=Path)
    parser.add_argument("--teacher-model-path", type=Path)
    parser.add_argument("--lingbot-va-root", type=Path)
    parser.add_argument("--teacher-signature-noise-seed", type=int)
    parser.add_argument("--expected-dataset-count", type=int)
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--max-sources", type=int)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    base = TrainingConfig()
    overrides = {
        "dataset_path": args.dataset_path,
        "empty_emb_path": args.empty_emb_path,
        "teacher_video_bank_path": args.teacher_video_bank_path,
        "teacher_signature_cache_path": args.teacher_signature_cache_path,
        "teacher_model_path": args.teacher_model_path,
        "lingbot_va_root": args.lingbot_va_root,
        "teacher_signature_noise_seed": args.teacher_signature_noise_seed,
        "expected_dataset_count": args.expected_dataset_count,
        "expected_source_count": args.expected_source_count,
    }
    return replace(
        base,
        **{name: value for name, value in overrides.items() if value is not None},
    )


def initialize_distributed(config: TrainingConfig) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("teacher-signature precomputation requires CUDA")
    torch.cuda.set_device(config.local_rank)
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=config.rank,
            world_size=config.world_size,
        )
    return torch.device("cuda", config.local_rank)


def selected_source_indices(
    source_count: int,
    *,
    start_index: int,
    end_index: int | None,
    max_sources: int | None,
) -> range:
    """Resolve a valid half-open source range for resumable cache filling."""

    if start_index < 0 or start_index > source_count:
        raise ValueError("--start-index is outside the dataset")
    stop = source_count if end_index is None else end_index
    if stop < start_index or stop > source_count:
        raise ValueError("--end-index is outside the dataset")
    if max_sources is not None:
        if max_sources < 1:
            raise ValueError("--max-sources must be positive")
        stop = min(stop, start_index + max_sources)
    return range(start_index, stop)


def run(config: TrainingConfig, args: argparse.Namespace) -> None:
    device = initialize_distributed(config)
    dtype = getattr(torch, config.param_dtype)

    preparation_error = None
    if config.rank == 0:
        try:
            verify_teacher_bank_marker(config)
            prepare_teacher_signature_cache(config)
        except Exception as exc:
            preparation_error = f"{type(exc).__name__}: {exc}"
    errors = [preparation_error]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(errors, src=0)
    if errors[0] is not None:
        raise RuntimeError(f"cache preflight failed: {errors[0]}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    dataset = DriftingTrainingDataset(config)
    selected = selected_source_indices(
        len(dataset),
        start_index=args.start_index,
        end_index=args.end_index,
        max_sources=args.max_sources,
    )
    selected_list = list(selected)
    if not selected_list:
        raise ValueError("selected source range is empty")
    complete_range = (
        args.start_index == 0
        and (args.end_index is None or args.end_index == len(dataset))
        and args.max_sources is None
    )
    if complete_range and config.rank == 0:
        # A full pass republishes this marker only after every entry has been
        # read or generated successfully. A partial diagnostic pass must not
        # invalidate an already complete shared cache.
        (
            config.teacher_signature_cache_path / CACHE_COMPLETE_FILENAME
        ).unlink(missing_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    expected_entries = sum(
        len(dataset.frame_starts_for_index(source_index))
        for source_index in selected_list
    )
    local_sources = selected_list[config.rank :: config.world_size]
    local_entries = sum(
        len(dataset.frame_starts_for_index(source_index))
        for source_index in local_sources
    )

    teacher = configure_frozen_teacher(config, device=device)
    generator = ActionSignatureGenerator(teacher, config)
    cache = TeacherSignatureCache(
        config,
        generator,
        allow_misses=True,
    )

    hits = 0
    misses = 0
    progress = tqdm(
        total=local_entries,
        desc=f"teacher actions rank {config.rank}",
        disable=config.rank != 0,
        dynamic_ncols=True,
    )
    try:
        for source_index in local_sources:
            for frame_start in dataset.frame_starts_for_index(source_index):
                sample = dataset[(source_index, frame_start)]
                batch = collate_training_samples([sample]).to(
                    device,
                    dtype=dtype,
                )
                result = cache.get_or_compute(
                    batch.sample_ids,
                    batch.teacher_videos,
                    batch.condition(),
                )
                hits += result.hits
                misses += result.misses
                progress.update(1)
                del result, batch, sample
    finally:
        progress.close()

    totals = torch.tensor(
        [local_entries, hits, misses],
        device=device,
        dtype=torch.long,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.barrier()
    processed, global_hits, global_misses = map(int, totals.tolist())
    if processed != expected_entries:
        raise RuntimeError(
            f"processed {processed} entries but selected range requires "
            f"{expected_entries}"
        )

    if config.rank == 0:
        logger.info(
            "Teacher-signature cache pass complete: entries=%d hits=%d misses=%d",
            processed,
            global_hits,
            global_misses,
        )
        if complete_range:
            write_teacher_signature_cache_complete(
                config,
                expected_entries=expected_entries,
            )
            logger.info("Published CACHE_COMPLETE.json")
        else:
            logger.info(
                "Partial source range completed; no completion marker was written"
            )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s | rank={config.rank} | "
            "%(levelname)s | %(name)s | %(message)s"
        ),
    )
    try:
        run(config, args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
