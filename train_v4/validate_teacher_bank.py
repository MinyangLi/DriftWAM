"""Validate an existing teacher-video bank and publish its completion marker."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .engine.bank_validation import (
    bank_complete_path,
    validate_teacher_bank_and_write,
    verify_teacher_bank_marker,
)
from .training_config import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--teacher-video-bank-path", type=Path)
    parser.add_argument("--teacher-model-path", type=Path)
    parser.add_argument("--expected-dataset-count", type=int)
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument("--write-marker", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    base = TrainingConfig()
    overrides = {
        "dataset_path": args.dataset_path,
        "teacher_video_bank_path": args.teacher_video_bank_path,
        "teacher_model_path": args.teacher_model_path,
        "expected_dataset_count": args.expected_dataset_count,
        "expected_source_count": args.expected_source_count,
    }
    return replace(
        base,
        **{name: value for name, value in overrides.items() if value is not None},
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    if args.write_marker:
        manifest = validate_teacher_bank_and_write(config)
        print(
            f"wrote {bank_complete_path(config.teacher_video_bank_path)} "
            f"for {manifest['source_count']} sources"
        )
    else:
        manifest = verify_teacher_bank_marker(config)
        print(
            f"teacher bank is ready: {manifest['validated_file_count']} entries"
        )


if __name__ == "__main__":
    main()
