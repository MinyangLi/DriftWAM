"""Loading the teacher and the Flash-WAM student, with the attention mode forced.

Why this module exists at all: both published checkpoints carry
`"attn_mode": "torch"` in `transformer/config.json`, and `attn_mode` is a
`register_to_config` argument, so `from_pretrained` faithfully restores it. In
`WanAttention.__init__` that selects

    def custom_sdpa(q, k, v):
        return F.scaled_dot_product_attention(q.transpose(1, 2), ...)

which takes no mask and no `is_causal`. Only `attn_mode="flex"` routes through
`FlexAttnFunc`, the sole consumer of the BlockMask that `init_mask` builds.

For the autoregressive `forward` path that is harmless: causality comes from the
KV cache holding only past keys. For `forward_train` it is not, because the entire
block-causal structure lives in that mask. Loaded as shipped, `forward_train`
runs fully bidirectional attention, and then

* noisy tokens attend to the clean ground truth of their own chunk, i.e. straight
  to the answer, so any "conditional distribution" measured from it is fiction;
* chunks stop being independent, so one forward no longer yields several
  conditions;
* batch slots leak, so K noise draws are not independent samples.

`tests/test_forward_tiny.py` fails four invariants under "torch" and passes all of
them under "flex", which is the check that pinned this down.
"""

from __future__ import annotations

from pathlib import Path

import torch

from . import paths


def load_transformer(
    model_root: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_mode: str = "flex",
    verbose: bool = True,
):
    """Load a `transformer/` directory, overriding `attn_mode`.

    `forward_train` hard-casts its inputs to bf16, so bf16 weights are the only
    consistent choice; fp32 weights raise a dtype mismatch in the first Linear.
    That also means the bf16 noise floor cannot be dodged by promoting the model.
    """
    from modules.model import WanTransformer3DModel  # noqa: PLC0415

    root = paths.require(Path(model_root), "model root")
    tdir = paths.require(paths.transformer_dir(root), f"{root.name}/transformer")

    model = WanTransformer3DModel.from_pretrained(
        tdir, torch_dtype=dtype, attn_mode=attn_mode
    )
    model = model.to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)

    op = type(model.blocks[0].attn1.attn_op).__name__
    if attn_mode == "flex" and op != "FlexAttnFunc":
        raise RuntimeError(
            f"attn_mode override did not take effect: attention op is {op}. "
            "Without FlexAttnFunc the block-causal mask is never applied."
        )
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"loaded {root.name}/transformer: {n / 1e9:.2f}B params, "
              f"{dtype}, attn_op={op}")
    return model


def load_teacher(**kw):
    return load_transformer(paths.TEACHER_PATH, **kw)


def load_student(**kw):
    return load_transformer(paths.STUDENT_PATH, **kw)


def assert_same_architecture(a, b) -> None:
    """The distilled student must be architecturally identical to the teacher."""
    ka, kb = a.config, b.config
    diff = {
        k: (ka.get(k), kb.get(k))
        for k in set(ka) | set(kb)
        if k not in ("_name_or_path", "attn_mode") and ka.get(k) != kb.get(k)
    }
    if diff:
        raise RuntimeError(f"teacher/student architecture differs: {diff}")
