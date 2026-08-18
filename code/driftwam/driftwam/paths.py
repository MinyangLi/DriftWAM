"""Centralised filesystem locations, all overridable by environment variable.

Nothing here imports torch or Flash-WAM, so this module is safe to read from a
CPU-only shell to find out where things live.
"""

from __future__ import annotations

import os
from pathlib import Path

_E = os.environ.get


def _p(env: str, default: str) -> Path:
    return Path(_E(env, default)).expanduser()


# Flash-WAM checkout that we borrow the model/dataset/scheduler code from. We
# never write inside it.
FLASH_WAM = _p("FLASH_WAM_DIR", "/root/autodl-tmp/wam/code/Flash-WAM")

DATASET_ROOT = _p("DATASET_ROOT", "/root/autodl-fs/wam/datasets/robotwin-lerobot")
DATASET_PATH = _p("DATASET_PATH", str(DATASET_ROOT / "lerobot_robotwin_eef_clean_50"))

# empty_emb.pt ships one level above the dataset variant, but Flash-WAM's config
# derives it as <dataset_path>/empty_emb.pt. bootstrap.py reconciles the two.
EMPTY_EMB_PATH = _p("EMPTY_EMB_PATH", str(DATASET_ROOT / "empty_emb.pt"))

TEACHER_PATH = _p("TEACHER_PATH", "/root/autodl-fs/wam/models/lingbot-va-posttrain-robotwin")
STUDENT_PATH = _p("STUDENT_PATH", "/root/autodl-fs/wam/models/FlashWAM-RoboTwin")
DINO_PATH = _p("DINO_PATH", "/root/autodl-fs/wam/models/dinov3-vitb16-pretrain-lvd1689m")

# Experiment outputs. On autodl, /root/autodl-fs survives instance teardown
# while /root/autodl-tmp does not, so results go to -fs.
EXP_ROOT = _p("EXP_ROOT", "/root/autodl-fs/wam/exp")
EXP01_DIR = _p("EXP01_DIR", str(EXP_ROOT / "exp01"))


def transformer_dir(model_root: Path | str) -> Path:
    return Path(model_root) / "transformer"


def vae_dir(model_root: Path | str) -> Path:
    return Path(model_root) / "vae"


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}")
    return path


def describe() -> str:
    rows = [
        ("Flash-WAM code", FLASH_WAM),
        ("dataset variant", DATASET_PATH),
        ("empty_emb.pt", EMPTY_EMB_PATH),
        ("teacher", TEACHER_PATH),
        ("student (Flash-WAM)", STUDENT_PATH),
        ("DINOv3", DINO_PATH),
        ("exp01 output", EXP01_DIR),
    ]
    w = max(len(k) for k, _ in rows)
    return "\n".join(
        f"  {k:<{w}}  {'ok  ' if Path(v).exists() else 'MISS'}  {v}" for k, v in rows
    )


if __name__ == "__main__":
    print(describe())
