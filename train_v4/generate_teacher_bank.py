"""CLI for building the condition-matched offline teacher-video bank."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor
from tqdm import tqdm

from .engine.teacher_bank_data import (
    TeacherBankSource,
    TeacherBankSourceDataset,
    source_shape,
)
from .engine.bank_format import FRAME_STARTS_KEY, TEACHER_VIDEOS_KEY
from .engine.bank_validation import (
    remove_bank_complete_marker,
    validate_teacher_bank_and_write,
)
from .forwards.action_signature import ActionSignatureCondition
from .forwards.teacher_video import TeacherVideoGenerator
from .model_setup import configure_lingbot_video_teacher
from .training_config import TrainingConfig


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate four frozen-teacher video candidates for every aligned "
            "latent chunk"
        )
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--empty-emb-path", type=Path)
    parser.add_argument("--teacher-video-bank-path", type=Path)
    parser.add_argument("--teacher-model-path", type=Path)
    parser.add_argument("--lingbot-va-root", type=Path)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--candidate-batch-size", type=int)
    parser.add_argument("--source-batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig()
    overrides = {
        "dataset_path": args.dataset_path,
        "empty_emb_path": args.empty_emb_path,
        "teacher_video_bank_path": args.teacher_video_bank_path,
        "teacher_model_path": args.teacher_model_path,
        "teacher_video_num_inference_steps": args.num_inference_steps,
        "lingbot_va_root": args.lingbot_va_root,
        "teacher_video_guidance_scale": args.guidance_scale,
        "teacher_bank_candidate_batch_size": args.candidate_batch_size,
        "teacher_bank_source_batch_size": args.source_batch_size,
        "teacher_bank_seed": args.seed,
    }
    selected = {
        name: value
        for name, value in overrides.items()
        if value is not None
    }
    for name in (
        "dataset_path",
        "empty_emb_path",
        "teacher_video_bank_path",
        "teacher_model_path",
        "lingbot_va_root",
    ):
        if name in selected:
            selected[name] = selected[name].expanduser().resolve()
    return replace(config, **selected)


def initialize_distributed(config: TrainingConfig) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("teacher-video generation requires CUDA")
    torch.cuda.set_device(config.local_rank)
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=config.rank,
            world_size=config.world_size,
        )
    return torch.device("cuda", config.local_rank)


def stable_candidate_seed(
    base_seed: int,
    sample_id: str,
    frame_start: int,
    candidate_index: int,
) -> int:
    """Derive a seed independent of rank count and traversal order."""

    payload = (
        f"{base_seed}\\0{sample_id}\\0{frame_start}\\0{candidate_index}"
    ).encode("utf-8")
    value = int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        byteorder="little",
        signed=False,
    )
    return value & ((1 << 63) - 1)


def make_video_noise(
    source: TeacherBankSource,
    *,
    frame_start: int,
    candidate_start: int,
    candidate_count: int,
    config: TrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create independently seeded noise with shape [1,K,C,2,H,W]."""

    shape = source_shape(source, config.frame_chunk_size)
    candidates = []
    for candidate_index in range(
        candidate_start,
        candidate_start + candidate_count,
    ):
        generator = torch.Generator(device=device)
        generator.manual_seed(
            stable_candidate_seed(
                config.teacher_bank_seed,
                source.sample_id,
                frame_start,
                candidate_index,
            )
        )
        candidates.append(
            torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
    return torch.stack(candidates, dim=0).unsqueeze(0)


def make_batched_video_noise(
    sources: Sequence[TeacherBankSource],
    *,
    frame_start: int,
    candidate_start: int,
    candidate_count: int,
    config: TrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create stable per-source noise with shape [B,K,C,2,H,W]."""

    return torch.cat(
        [
            make_video_noise(
                source,
                frame_start=frame_start,
                candidate_start=candidate_start,
                candidate_count=candidate_count,
                config=config,
                device=device,
                dtype=dtype,
            )
            for source in sources
        ],
        dim=0,
    )


def batch_condition(
    sources: Sequence[TeacherBankSource],
    frame_start: int,
) -> ActionSignatureCondition:
    """Stack same-position causal conditions from independent sources."""

    history_video = None
    history_action = None
    if frame_start > 0:
        history_video = torch.stack(
            [source.latents[:, :frame_start] for source in sources],
            dim=0,
        )
        history_action = torch.stack(
            [source.actions[:, :frame_start] for source in sources],
            dim=0,
        )
    return ActionSignatureCondition(
        text_emb=torch.stack([source.text_emb for source in sources], dim=0),
        frame_start=frame_start,
        history_video=history_video,
        history_action=history_action,
        history_frame_start=0,
    )


def atomic_save_bank(entry: dict[str, Tensor], destination: Path) -> None:
    """Publish a complete bank file without exposing partial writes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-rank0-{os.getpid()}"
    )
    try:
        torch.save(entry, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def broadcast_rank_zero_bool(value: bool, device: torch.device) -> bool:
    if not dist.is_initialized():
        return value
    control = torch.tensor(
        [int(value) if dist.get_rank() == 0 else 0],
        dtype=torch.int64,
        device=device,
    )
    dist.broadcast(control, src=0)
    return bool(control.item())


def check_dataset_length(length: int, device: torch.device) -> None:
    """Prevent mismatched FSDP work loops across ranks."""

    if not dist.is_initialized():
        return
    minimum = torch.tensor([length], dtype=torch.int64, device=device)
    maximum = minimum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if minimum.item() != maximum.item():
        raise RuntimeError("dataset length differs across distributed ranks")


def selected_indices(
    dataset_length: int,
    *,
    start_index: int,
    end_index: int | None,
    max_samples: int | None,
) -> range:
    if start_index < 0 or start_index > dataset_length:
        raise ValueError("--start-index is outside the dataset")
    stop = dataset_length if end_index is None else end_index
    if stop < start_index or stop > dataset_length:
        raise ValueError("--end-index is outside the selected dataset range")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("--max-samples must be positive")
        stop = min(stop, start_index + max_samples)
    return range(start_index, stop)


def generate_sources(
    sources: Sequence[TeacherBankSource],
    generator: TeacherVideoGenerator,
    config: TrainingConfig,
    device: torch.device,
) -> list[dict[str, Tensor] | None]:
    """Generate every valid chunk for a small batch of source items."""

    if not sources:
        return []

    dtype = generator.dtype
    stored_chunks: list[list[Tensor]] = [[] for _ in sources]
    max_chunk_count = max(len(source.frame_starts) for source in sources)

    for chunk_index in range(max_chunk_count):
        active_indices = [
            source_index
            for source_index, source in enumerate(sources)
            if chunk_index < len(source.frame_starts)
        ]
        active_sources = [sources[index] for index in active_indices]
        frame_start = int(
            active_sources[0].frame_starts[chunk_index].item()
        )
        if any(
            int(source.frame_starts[chunk_index].item()) != frame_start
            for source in active_sources[1:]
        ):
            raise ValueError(
                "batched sources must use the same frame start per chunk"
            )

        candidate_parts: list[list[Tensor]] = [
            [] for _ in active_sources
        ]
        for candidate_start in range(
            0,
            config.teacher_candidate_count,
            config.teacher_bank_candidate_batch_size,
        ):
            candidate_count = min(
                config.teacher_bank_candidate_batch_size,
                config.teacher_candidate_count - candidate_start,
            )
            noise = make_batched_video_noise(
                active_sources,
                frame_start=frame_start,
                candidate_start=candidate_start,
                candidate_count=candidate_count,
                config=config,
                device=device,
                dtype=dtype,
            )
            videos = generator(
                noise,
                batch_condition(active_sources, frame_start),
                initial_frame=torch.stack(
                    [source.initial_frame for source in active_sources],
                    dim=0,
                ),
            )
            if config.rank == 0:
                for active_index, source_videos in enumerate(videos):
                    candidate_parts[active_index].append(
                        source_videos.to("cpu")
                    )
            del videos, noise

        if config.rank == 0:
            for active_index, source_index in enumerate(active_indices):
                stored_chunks[source_index].append(
                    torch.cat(candidate_parts[active_index], dim=0)
                    .to(torch.bfloat16)
                    .contiguous()
                )

    if config.rank != 0:
        return [None for _ in sources]
    return [
        {
            TEACHER_VIDEOS_KEY: torch.stack(source_chunks, dim=0),
            FRAME_STARTS_KEY: source.frame_starts.clone().to(dtype=torch.long),
        }
        for source, source_chunks in zip(sources, stored_chunks, strict=True)
    ]


def generate_source(
    source: TeacherBankSource,
    generator: TeacherVideoGenerator,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, Tensor] | None:
    """Compatibility wrapper for generating one source item."""

    return generate_sources((source,), generator, config, device)[0]


def run(config: TrainingConfig, args: argparse.Namespace) -> None:
    device = initialize_distributed(config)
    torch.manual_seed(config.teacher_bank_seed)
    torch.cuda.manual_seed_all(config.teacher_bank_seed)
    torch.set_float32_matmul_precision("high")

    if not config.dataset_path.is_dir():
        raise FileNotFoundError(f"dataset not found: {config.dataset_path}")
    if not config.empty_emb_path.is_file():
        raise FileNotFoundError(
            f"empty text embedding not found: {config.empty_emb_path}"
        )
    if not config.teacher_model_path.is_dir():
        raise FileNotFoundError(
            f"teacher checkpoint not found: {config.teacher_model_path}"
        )

    dataset = TeacherBankSourceDataset(config)
    check_dataset_length(len(dataset), device)
    indices = selected_indices(
        len(dataset),
        start_index=args.start_index,
        end_index=args.end_index,
        max_samples=args.max_samples,
    )

    negative_text_emb = torch.load(
        config.empty_emb_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(negative_text_emb, Tensor):
        raise TypeError("empty embedding file must contain one tensor")

    if config.rank == 0:
        config.teacher_video_bank_path.mkdir(parents=True, exist_ok=True)
        remove_bank_complete_marker(config.teacher_video_bank_path)
        logger.info("Dataset: %s (%d source items)", config.dataset_path, len(dataset))
        logger.info("Teacher: %s", config.teacher_model_path)
        logger.info("Teacher bank: %s", config.teacher_video_bank_path)
        logger.info(
            "Selected indices [%d, %d); candidates=%d, candidate_batch=%d, "
            "source_batch=%d, video_steps=%d, guidance=%.3g",
            indices.start,
            indices.stop,
            config.teacher_candidate_count,
            config.teacher_bank_candidate_batch_size,
            config.teacher_bank_source_batch_size,
            config.teacher_video_num_inference_steps,
            config.teacher_video_guidance_scale,
        )
    if dist.is_initialized():
        dist.barrier()

    logger.info("Loading frozen teacher")
    teacher = configure_lingbot_video_teacher(config, device=device)
    video_generator = TeacherVideoGenerator(
        teacher,
        config,
        negative_text_emb,
    )

    progress = tqdm(
        total=len(indices),
        desc="Teacher bank",
        disable=config.rank != 0,
        dynamic_ncols=True,
    )
    saved = 0
    skipped = 0
    pending_sources: list[TeacherBankSource] = []
    pending_destinations: list[Path] = []

    def flush_pending() -> None:
        nonlocal saved
        if not pending_sources:
            return
        entries = generate_sources(
            pending_sources,
            video_generator,
            config,
            device,
        )
        if config.rank == 0:
            for source, entry, destination in zip(
                pending_sources,
                entries,
                pending_destinations,
                strict=True,
            ):
                if entry is None:
                    raise RuntimeError("rank zero did not receive a bank entry")
                atomic_save_bank(entry, destination)
                saved += 1
            progress.set_postfix(
                saved=saved,
                skipped=skipped,
                source_batch=len(pending_sources),
                chunks="+".join(
                    str(len(source.frame_starts))
                    for source in pending_sources
                ),
            )
        progress.update(len(pending_sources))
        pending_sources.clear()
        pending_destinations.clear()

    try:
        for source_index in indices:
            destination = dataset.bank_path(source_index)
            should_generate = broadcast_rank_zero_bool(
                args.overwrite or not destination.is_file(),
                device,
            )
            if not should_generate:
                skipped += 1
                progress.update(1)
                continue

            pending_sources.append(dataset[source_index])
            pending_destinations.append(destination)
            if (
                len(pending_sources)
                >= config.teacher_bank_source_batch_size
            ):
                flush_pending()
        flush_pending()
    finally:
        progress.close()

    if config.rank == 0:
        logger.info(
            "Teacher bank generation complete: saved=%d skipped=%d",
            saved,
            skipped,
        )
        if indices.start == 0 and indices.stop == len(dataset):
            logger.info("Validating the complete teacher bank")
            manifest = validate_teacher_bank_and_write(config)
            logger.info(
                "Published teacher bank completion marker for %d sources",
                manifest["source_count"],
            )
        else:
            logger.info(
                "Partial generation does not publish a completion marker"
            )


def main() -> None:
    if dist.is_initialized():
        dist.barrier()
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
