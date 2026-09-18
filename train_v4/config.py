"""Configuration for action-response-guided video drifting."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_LINGBOT_VA_ROOT = Path(__file__).resolve().parents[1] / "lingbot-va"
EXECUTION_RESPONSE_MODES = frozenset(
    {"off", "reuse_all", "recompute_selected"}
)


def _optional_path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


@dataclass(slots=True)
class DistillationConfig:
    """Settled Sections 3.1--4 settings and model locations."""

    lingbot_va_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("LINGBOT_VA_ROOT", _DEFAULT_LINGBOT_VA_ROOT)
        ).expanduser().resolve()
    )
    teacher_model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "TEACHER_PATH",
                _DEFAULT_LINGBOT_VA_ROOT
                / "checkpoints"
                / "lingbot-va-posttrain-robotwin",
            )
        ).expanduser().resolve()
    )
    student_init_source: str = field(
        default_factory=lambda: os.environ.get(
            "STUDENT_INIT_SOURCE",
            "teacher",
        ).strip().lower()
    )
    # Used when student_init_source="flash_wam". It may point at either the
    # checkpoint root or its Transformer directory.
    student_init_path: Path | None = field(
        default_factory=lambda: _optional_path(os.environ.get("STUDENT_INIT_PATH"))
    )

    # Forward/input dtype; online masters, Adam moments, and EMA stay FP32.
    param_dtype: str = "bfloat16"
    signature_horizon: int = 1
    frame_chunk_size: int = 2
    video_num_train_timesteps: int = 1000
    video_student_num_inference_steps: int = 1
    video_snr_shift: float = 5.0
    action_per_frame: int = 16
    action_dim: int = 30
    action_teacher_num_inference_steps: int = 50
    action_num_train_timesteps: int = 1000
    action_snr_shift: float = 1.0
    action_num_ddim_timesteps: int = 2
    action_huber_c: float = 0.001
    # Fixed main-experiment data source; serialized for provenance/resume.
    action_supervision_source: str = field(default="ground_truth", init=False)
    action_consistency_loss_weight: float = 1.0
    action_flow_matching_loss_weight: float = 0.01
    ema_decay: float = 0.995
    attn_window: int = 72
    teacher_candidate_count: int = 4
    student_candidate_count: int = 4
    beta_rep: float = 1.0
    video_drifting_loss_weight: float = 1.0
    execution_response_mode: str = field(
        default_factory=lambda: os.environ.get(
            "EXECUTION_RESPONSE_MODE",
            "off",
        ).strip().lower()
    )
    execution_loss_weight: float = field(
        default_factory=lambda: float(
            os.environ.get("EXECUTION_LOSS_WEIGHT", "0.1")
        )
    )
    learning_rate: float = 2e-6
    video_bandwidth: float | None = None
    action_bandwidth: float | None = None

    def __post_init__(self) -> None:
        self.lingbot_va_root = Path(
            self.lingbot_va_root
        ).expanduser().resolve()
        self.teacher_model_path = Path(
            self.teacher_model_path
        ).expanduser().resolve()
        if self.student_init_path is not None:
            self.student_init_path = Path(
                self.student_init_path
            ).expanduser().resolve()
        self.student_init_source = self.student_init_source.strip().lower()
        if self.student_init_source not in {"flash_wam", "teacher"}:
            raise ValueError(
                "student_init_source must be 'flash_wam' or 'teacher'"
            )

        if self.signature_horizon < 1:
            raise ValueError("signature_horizon must be at least 1")
        if self.frame_chunk_size != 2:
            raise ValueError("frame_chunk_size must be 2")
        if self.video_num_train_timesteps != 1000:
            raise ValueError(
                "video_num_train_timesteps must be 1000 for v4"
            )
        if self.video_student_num_inference_steps != 1:
            raise ValueError(
                "video_student_num_inference_steps must be 1 for v4"
            )
        if not math.isfinite(self.video_snr_shift) or self.video_snr_shift <= 0:
            raise ValueError("video_snr_shift must be finite and positive")
        if self.action_per_frame != 16:
            raise ValueError("action_per_frame must be 16")
        if self.action_dim != 30:
            raise ValueError("action_dim must be 30")
        if self.action_teacher_num_inference_steps < 1:
            raise ValueError(
                "action_teacher_num_inference_steps must be positive"
            )
        if self.action_num_train_timesteps != 1000:
            raise ValueError(
                "action_num_train_timesteps must be 1000 for v4"
            )
        if self.action_num_ddim_timesteps != 2:
            raise ValueError("action_num_ddim_timesteps must be 2 for v4")
        if (
            self.action_num_train_timesteps
            % self.action_num_ddim_timesteps
            != 0
        ):
            raise ValueError(
                "action_num_train_timesteps must be divisible by "
                "action_num_ddim_timesteps"
            )
        if not math.isfinite(self.action_huber_c) or self.action_huber_c <= 0:
            raise ValueError("action_huber_c must be finite and positive")
        if (
            not math.isfinite(self.action_consistency_loss_weight)
            or self.action_consistency_loss_weight <= 0
        ):
            raise ValueError(
                "action_consistency_loss_weight must be finite and positive"
            )
        if (
            not math.isfinite(self.action_flow_matching_loss_weight)
            or self.action_flow_matching_loss_weight < 0
        ):
            raise ValueError(
                "action_flow_matching_loss_weight must be finite and "
                "non-negative"
            )
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be finite and lie in [0, 1)")
        if self.teacher_candidate_count != 4:
            raise ValueError("teacher_candidate_count must be 4 for v4")
        if self.student_candidate_count != 4:
            raise ValueError("student_candidate_count must be 4 for v4")
        if not math.isfinite(self.beta_rep) or self.beta_rep < 0:
            raise ValueError("beta_rep must be finite and non-negative")
        if (
            not math.isfinite(self.video_drifting_loss_weight)
            or self.video_drifting_loss_weight < 0
        ):
            raise ValueError(
                "video_drifting_loss_weight must be finite and non-negative"
            )
        self.execution_response_mode = (
            self.execution_response_mode.strip().lower()
        )
        if self.execution_response_mode not in EXECUTION_RESPONSE_MODES:
            choices = ", ".join(sorted(EXECUTION_RESPONSE_MODES))
            raise ValueError(
                f"execution_response_mode must be one of: {choices}"
            )
        if (
            not math.isfinite(self.execution_loss_weight)
            or self.execution_loss_weight < 0
        ):
            raise ValueError(
                "execution_loss_weight must be finite and non-negative"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (self.video_bandwidth is None) != (self.action_bandwidth is None):
            raise ValueError(
                "video_bandwidth and action_bandwidth must be set together"
            )
        if self.video_bandwidth is not None:
            self.set_kernel_bandwidths(
                video=self.video_bandwidth,
                action=self.action_bandwidth,
            )

    def set_kernel_bandwidths(self, *, video: float, action: float) -> None:
        """Store one calibrated pair of fixed kernel bandwidths."""

        video = float(video)
        action = float(action)
        if not math.isfinite(video) or video <= 0:
            raise ValueError("video bandwidth must be finite and positive")
        if not math.isfinite(action) or action <= 0:
            raise ValueError("action bandwidth must be finite and positive")
        if self.video_bandwidth is not None:
            current_video = float(self.video_bandwidth)
            current_action = float(self.action_bandwidth)
            if current_video != video or current_action != action:
                raise RuntimeError("kernel bandwidths are fixed once set")
        self.video_bandwidth = video
        self.action_bandwidth = action

    @property
    def action_consistency_stride(self) -> int:
        return self.action_num_train_timesteps // self.action_num_ddim_timesteps

    @property
    def selected_student_init_path(self) -> Path:
        """Return the checkpoint selected for online and EMA students."""

        if self.student_init_source == "teacher":
            return self.teacher_model_path
        if self.student_init_path is None:
            raise ValueError(
                "student_init_path is required when "
                "student_init_source='flash_wam'"
            )
        return self.student_init_path
