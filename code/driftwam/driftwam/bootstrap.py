"""Make Flash-WAM's `wan_va` / `distillation` packages importable, then hand back
a patched copy of its config.

Flash-WAM is not installable and its modules assume they are run from inside the
repo with `sys.path` already containing `wan_va/` and `distillation/`. Both dirs
expose very generic top-level names (`config`, `utils`, `dataset`, `modules`), so
everything here is kept out of the way inside the `driftwam` package and the
import is done exactly once, in the one order that works:

    OMP env fix  ->  env vars for config  ->  sys.path  ->  flash-attn stub  ->  config

The stub has to be installed before `modules.model` is imported, because that
module does a top-level `import flash_attn_interface`.

Typical use::

    from driftwam import bootstrap
    cfg = bootstrap.setup()
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

from . import paths

# libgomp aborts on OMP_NUM_THREADS=0, which is what autodl exports for a 0.5-core
# container. Must be repaired before torch/numpy pull in libgomp.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    try:
        if int(os.environ.get(_var, "1")) < 1:
            os.environ[_var] = "1"
    except ValueError:
        os.environ[_var] = "1"

_STATE: dict[str, object] = {}


def _add_sys_path() -> None:
    fw = paths.require(paths.FLASH_WAM, "Flash-WAM checkout")
    for sub in ("wan_va", "distillation"):
        d = str(paths.require(fw / sub, f"Flash-WAM {sub}/"))
        if d not in sys.path:
            sys.path.append(d)


def _check_provenance(mod: ModuleType, expected_parent: Path) -> None:
    got = Path(getattr(mod, "__file__", "") or "").resolve()
    if expected_parent.resolve() not in got.parents:
        raise ImportError(
            f"module {mod.__name__!r} resolved to {got}, not inside {expected_parent}. "
            "Another package on sys.path is shadowing Flash-WAM's generic module names."
        )


def setup(dataset_path: str | Path | None = None, quiet: bool = True):
    """Idempotently prepare the interpreter and return Flash-WAM's `cfg`.

    `cfg` is the module-level EasyDict from `distillation/config.py`, mutated in
    place so that `empty_emb_path` points at the file as actually shipped. Callers
    get the same object every time, matching how Flash-WAM itself treats it.
    """
    if "cfg" in _STATE and dataset_path is None:
        return _STATE["cfg"]

    ds = Path(dataset_path) if dataset_path is not None else paths.DATASET_PATH
    # distillation/config.py reads these at import time.
    os.environ["DATASET_PATH"] = str(ds)
    os.environ.setdefault("TEACHER_PATH", str(paths.TEACHER_PATH))

    _add_sys_path()

    from patches import install_flash_attn_stub  # noqa: PLC0415  (needs sys.path)

    if quiet:
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()):
            install_flash_attn_stub()
    else:
        install_flash_attn_stub()

    import config as fw_config  # noqa: PLC0415

    _check_provenance(fw_config, paths.FLASH_WAM)
    cfg = fw_config.cfg

    cfg.dataset_path = str(ds)
    cfg.empty_emb_path = str(paths.EMPTY_EMB_PATH)
    cfg.teacher_model_path = str(paths.TEACHER_PATH)

    _STATE["cfg"] = cfg
    return cfg


def load_empty_emb(cfg=None, device="cpu", dtype=None):
    """The unconditional text embedding used for classifier-free guidance.

    Stored as a 2-D (512, 4096) tensor; `Tensor.expand(B, -1, -1)` grows a new
    leading batch dim, which is how Flash-WAM's training step consumes it.
    """
    import torch  # noqa: PLC0415

    cfg = cfg or setup()
    p = paths.require(Path(cfg.empty_emb_path), "empty_emb.pt")
    emb = torch.load(p, map_location="cpu", weights_only=False)
    if emb.ndim != 2:
        raise ValueError(f"expected 2-D empty_emb, got {tuple(emb.shape)}")
    return emb.to(device=device, dtype=dtype or emb.dtype)


def find_sub_datasets(cfg=None) -> list[str]:
    """Absolute repo_ids of every task-level LeRobot dataset under the variant."""
    cfg = cfg or setup()
    from dataset.lerobot_latent_dataset import recursive_find_file  # noqa: PLC0415

    found = recursive_find_file(cfg.dataset_path, "info.json")
    return sorted(str(Path(f).parent.parent) for f in found)


def open_sub_dataset(repo_id: str, cfg=None):
    cfg = cfg or setup()
    from dataset.lerobot_latent_dataset import LatentLeRobotDataset  # noqa: PLC0415

    return LatentLeRobotDataset(repo_id=str(repo_id), config=cfg)
