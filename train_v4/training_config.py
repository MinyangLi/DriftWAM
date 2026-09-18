"""Runtime configuration for action-response-guided video drifting."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import DistillationConfig


_DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/robotwin-lerobot")
_DEFAULT_DATASET_PATH = (
    _DEFAULT_DATA_ROOT / "lerobot_robotwin_eef_clean_50"
)
_DEFAULT_EMPTY_EMB_PATH = _DEFAULT_DATA_ROOT / "empty_emb.pt"
_DEFAULT_TEACHER_VIDEO_BANK = (
    _DEFAULT_DATA_ROOT / "teacher_video_bank_clean_50"
)
_DEFAULT_TEACHER_SIGNATURE_CACHE = (
    _DEFAULT_DATA_ROOT / "teacher_signature_cache_clean_50_seed42_v3"
)
_DEFAULT_MODEL_ROOT = Path("/root/autodl-fs/wam/models")
_DEFAULT_TEACHER_PATH = (
    _DEFAULT_MODEL_ROOT / "lingbot-va-posttrain-robotwin"
)
_DEFAULT_FLASH_WAM_PATH = _DEFAULT_MODEL_ROOT / "FlashWAM-RoboTwin"
_DEFAULT_OUTPUT_DIR = Path("/root/autodl-fs/wam/runs/train_v4")


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _optional_path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _official_norm_stat() -> dict[str, tuple[float, ...]]:
    q01 = (
        -0.06172713458538055,
        -3.6716461181640625e-05,
        -0.08783501386642456,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -0.3547105032205582,
        -1.3113021850585938e-06,
        -0.11975435614585876,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ) + (0.0,) * 16
    q99 = (
        0.3462600058317184,
        0.39966784834861746,
        0.14745532035827624,
        1.0,
        1.0,
        1.0,
        1.0,
        0.034201726913452024,
        0.39142737388610793,
        0.1792279863357542,
        1.0,
        1.0,
        1.0,
        1.0,
    ) + (0.0,) * 14 + (1.0, 1.0)
    return {"q01": q01, "q99": q99}


@dataclass(slots=True)
class TrainingConfig(DistillationConfig):
    """Experiment, data, optimizer, logging, and distributed settings."""

    # Model locations. ``student_init_source`` is inherited and selects
    # either ``student_init_path`` or ``teacher_model_path``.
    teacher_model_path: Path = field(
        default_factory=lambda: _path_from_env(
            "TEACHER_PATH",
            _DEFAULT_TEACHER_PATH,
        )
    )
    student_init_path: Path | None = field(
        default_factory=lambda: _path_from_env(
            "STUDENT_INIT_PATH",
            _DEFAULT_FLASH_WAM_PATH,
        )
    )

    # The loader recursively discovers task repositories below dataset_path.
    # For the official full training set, point DATASET_PATH at the sibling
    # directory named ``lerobot_robotwin_eef_aug_500`` instead.
    dataset_path: Path = field(
        default_factory=lambda: _path_from_env(
            "DATASET_PATH",
            _DEFAULT_DATASET_PATH,
        )
    )
    empty_emb_path: Path = field(
        default_factory=lambda: _path_from_env(
            "EMPTY_EMB_PATH",
            _DEFAULT_EMPTY_EMB_PATH,
        )
    )
    teacher_video_bank_path: Path = field(
        default_factory=lambda: _path_from_env(
            "TEACHER_VIDEO_BANK_PATH",
            _DEFAULT_TEACHER_VIDEO_BANK,
        )
    )
    teacher_signature_cache_path: Path = field(
        default_factory=lambda: _path_from_env(
            "TEACHER_SIGNATURE_CACHE_PATH",
            _DEFAULT_TEACHER_SIGNATURE_CACHE,
        )
    )
    output_dir: Path = field(
        default_factory=lambda: _path_from_env(
            "OUTPUT_DIR",
            _DEFAULT_OUTPUT_DIR,
        )
    )
    resume_from_path: Path | None = field(
        default_factory=lambda: _optional_path_from_env("RESUME_FROM_PATH")
    )
    resume_from_step: int | None = None
    experiment_name: str = "driftwam"
    expected_dataset_count: int | None = None
    expected_source_count: int | None = None
    expected_world_size: int | None = None
    expected_global_batch_size: int | None = None
    save_optimizer_state: bool = True
    # AdamW moments are unused during forward/backward. Keep them on CPU
    # between optimizer updates so long-history microbatches retain headroom.
    offload_optimizer_state: bool = True
    save_final_checkpoint: bool = True
    # Delete obsolete full checkpoints BEFORE saving; reserve one slot for new.
    retain_checkpoint_count: int = 1

    # Native RoboTwin dataset semantics.
    env_type: str = "robotwin_tshape"
    obs_cam_keys: tuple[str, ...] = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    action_norm_method: str = "quantiles"
    used_action_channel_ids: tuple[int, ...] = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        28,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        29,
    )
    norm_stat: dict[str, tuple[float, ...]] = field(
        default_factory=_official_norm_stat
    )
    cfg_prob: float = 0.0

    # Offline teacher-video bank generation. Candidate and source batching
    # share one teacher forward; classifier-free guidance doubles the final
    # model batch once more.
    teacher_video_num_inference_steps: int = 25
    teacher_video_guidance_scale: float = 5.0
    teacher_bank_candidate_batch_size: int = 4
    teacher_bank_source_batch_size: int = 2
    teacher_bank_seed: int = 42
    teacher_signature_noise_seed: int = 42

    # Optimization and run length. The algorithm learning rate remains the
    # inherited 2e-6 settled in DECISIONS.md.
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_unsynced_history_frames: int = field(
        default_factory=lambda: int(
            os.environ.get("MAX_UNSYNCED_HISTORY_FRAMES", "24")
        )
    )
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    optimizer_epsilon: float = 1e-8
    max_grad_norm: float = 2.0
    warmup_steps: int = 100
    max_train_steps: int = 10_000
    # Explicit, bounded memory test; the formal sampler is unchanged.
    history_stress_test: bool = False
    # Data loading, reporting, and checkpoint cadence.
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 42
    log_interval: int = 10
    diagnostics_interval: int = 50
    save_interval: int = 50
    gc_interval: int = 50
    enable_wandb: bool = True
    wandb_project: str = "driftwam_train_v4"
    wandb_entity: str | None = None

    # torchrun supplies these environment variables.
    rank: int = field(
        default_factory=lambda: int(os.environ.get("RANK", "0"))
    )
    local_rank: int = field(
        default_factory=lambda: int(os.environ.get("LOCAL_RANK", "0"))
    )
    world_size: int = field(
        default_factory=lambda: int(os.environ.get("WORLD_SIZE", "1"))
    )

    def __post_init__(self) -> None:
        super(TrainingConfig, self).__post_init__()
        self.dataset_path = Path(self.dataset_path).expanduser().resolve()
        self.empty_emb_path = Path(self.empty_emb_path).expanduser().resolve()
        self.teacher_video_bank_path = Path(
            self.teacher_video_bank_path
        ).expanduser().resolve()
        self.teacher_signature_cache_path = Path(
            self.teacher_signature_cache_path
        ).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.resume_from_path is not None:
            self.resume_from_path = Path(
                self.resume_from_path
            ).expanduser().resolve()

        if self.action_norm_method != "quantiles":
            raise ValueError("action_norm_method must be 'quantiles'")
        if len(self.used_action_channel_ids) != 16:
            raise ValueError("used_action_channel_ids must contain 16 entries")
        if len(set(self.used_action_channel_ids)) != 16:
            raise ValueError("used_action_channel_ids must be unique")
        if any(
            index < 0 or index >= self.action_dim
            for index in self.used_action_channel_ids
        ):
            raise ValueError("used_action_channel_ids contains an invalid index")
        if any(len(values) != self.action_dim for values in self.norm_stat.values()):
            raise ValueError("every norm_stat vector must have action_dim entries")
        if not 0.0 <= self.cfg_prob <= 1.0:
            raise ValueError("cfg_prob must lie in [0, 1]")
        if self.teacher_video_num_inference_steps < 1:
            raise ValueError(
                "teacher_video_num_inference_steps must be positive"
            )
        if (
            not math.isfinite(self.teacher_video_guidance_scale)
            or self.teacher_video_guidance_scale <= 0
        ):
            raise ValueError(
                "teacher_video_guidance_scale must be finite and positive"
            )
        if not (
            1
            <= self.teacher_bank_candidate_batch_size
            <= self.teacher_candidate_count
        ):
            raise ValueError(
                "teacher_bank_candidate_batch_size must lie between 1 and "
                "teacher_candidate_count"
            )
        if self.teacher_bank_seed < 0:
            raise ValueError("teacher_bank_seed must be non-negative")
        if self.teacher_signature_noise_seed < 0:
            raise ValueError(
                "teacher_signature_noise_seed must be non-negative"
            )

        positive_ints = {
            "teacher_bank_source_batch_size": self.teacher_bank_source_batch_size,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_train_steps": self.max_train_steps,
            "log_interval": self.log_interval,
            "diagnostics_interval": self.diagnostics_interval,
            "save_interval": self.save_interval,
            "gc_interval": self.gc_interval,
            "world_size": self.world_size,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.max_unsynced_history_frames < 0:
            raise ValueError("max_unsynced_history_frames must be non-negative")
        if self.max_unsynced_history_frames % self.frame_chunk_size != 0:
            raise ValueError(
                "max_unsynced_history_frames must align to frame_chunk_size"
            )
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.resume_from_step is not None and self.resume_from_step < 0:
            raise ValueError("resume_from_step must be non-negative")
        optional_positive_ints = {
            "expected_dataset_count": self.expected_dataset_count,
            "expected_source_count": self.expected_source_count,
            "expected_world_size": self.expected_world_size,
            "expected_global_batch_size": self.expected_global_batch_size,
            "retain_checkpoint_count": self.retain_checkpoint_count,
        }
        for name, value in optional_positive_ints.items():
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when configured")
        if self.expected_world_size not in (None, self.world_size):
            raise ValueError(
                f"expected world size {self.expected_world_size}, got "
                f"{self.world_size}"
            )
        if not 0.0 < self.beta1 < 1.0 or not 0.0 < self.beta2 < 1.0:
            raise ValueError("optimizer betas must lie in (0, 1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.optimizer_epsilon) or self.optimizer_epsilon <= 0:
            raise ValueError("optimizer_epsilon must be finite and positive")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")

    @property
    def inverse_used_action_channel_ids(self) -> tuple[int, ...]:
        """Map native 30-channel indices into the physical 16-channel order."""

        inverse = [len(self.used_action_channel_ids)] * self.action_dim
        for physical_index, native_index in enumerate(
            self.used_action_channel_ids
        ):
            inverse[native_index] = physical_index
        return tuple(inverse)

    @property
    def effective_batch_size_per_rank(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def effective_global_batch_size(self) -> int:
        return self.effective_batch_size_per_rank * self.world_size

    def validate_runtime_contract(self) -> None:
        if self.history_stress_test:
            if not 1 <= self.max_train_steps <= 3:
                raise ValueError("history stress test requires 1-3 optimizer updates")
            if self.resume_from_path is not None or self.resume_from_step is not None:
                raise ValueError("history stress test must start fresh")
            if self.save_final_checkpoint or self.save_interval <= self.max_train_steps:
                raise ValueError("history stress test must disable checkpoint saves")
        if self.expected_global_batch_size not in (
            None,
            self.effective_global_batch_size,
        ):
            raise ValueError(
                f"expected global batch size {self.expected_global_batch_size}, "
                f"got {self.effective_global_batch_size}"
            )
