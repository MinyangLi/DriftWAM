"""Persistent FP32 training state, independently of forward compute dtype."""

import torch

TRAINING_NUMERICS = {
    "master_param_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "ema_dtype": "float32",
    "training_attention": "explicit_causal_eager_mask_v2",
    "action_objective": "gt_fp32_full_sequence_unclipped_v3",
}


def assert_fp32_model(model, name="model"):
    for key, parameter in model.named_parameters():
        if parameter.dtype != torch.float32:
            raise ValueError(f"{name}.{key} must retain FP32 master parameters")


def assert_fp32_optimizer(optimizer):
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if value is not None and value.dtype != torch.float32:
                raise ValueError(f"AdamW {key} must be FP32")
