"""Localise the apparent cross-stream coupling: structural or numerical?

Calls forward_train directly, four ways, for a range of batch sizes and both
compute dtypes:

  ref    : baseline
  same   : identical inputs again          -> must be bitwise equal
  act    : noisy action perturbed          -> video must be bitwise equal
  slots  : other batch slots perturbed     -> slot 0 must be bitwise equal

If the deviations vanish in fp32 they are bf16 rounding, and the invariants hold
structurally. If they survive fp32, something really does couple the streams.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

from driftwam import sampling as SP  # noqa: E402
from driftwam import shapes as S  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_forward_tiny import synthetic_item, tiny_model  # noqa: E402


def run(dtype, k, f=8, dev="cuda"):
    model = tiny_model(dev, dtype=dtype)
    spec = SP.SamplerSpec(video_steps=2, action_steps=4, cfg_scale=1.0)
    sampler = SP.Sampler(
        model, spec, patch_size=tuple(CFG.patch_size),
        chunk_size=CFG.frame_chunk_size, window_size=CFG.attn_window,
    )
    cond = SP.make_condition(synthetic_item(f), k=k, device=dev,
                             patch_size=tuple(CFG.patch_size), n_frames=f)
    g = torch.Generator(device=dev).manual_seed(7)
    v0, a0 = SP.initial_noise(cond, g)

    # warm up so every measured call is in the same compilation state
    sampler.velocity(cond, v0, 1.0, a0, 1.0)

    v_ref, a_ref = sampler.velocity(cond, v0, 1.0, a0, 1.0)
    v_same, a_same = sampler.velocity(cond, v0, 1.0, a0, 1.0)

    a_pert = torch.empty_like(a0).normal_(generator=g) * cond.actions_mask.float()
    v_act, _ = sampler.velocity(cond, v0, 1.0, a_pert, 1.0)

    v_pert = torch.empty_like(v0).normal_(generator=g)
    _, a_vid = sampler.velocity(cond, v_pert, 1.0, a0, 1.0)

    res = {
        "same_video": (v_same - v_ref).abs().max().item(),
        "same_action": (a_same - a_ref).abs().max().item(),
        "video_vs_action_pert": (v_act - v_ref).abs().max().item(),
        "action_vs_video_pert": (a_vid - a_ref).abs().max().item(),
        "video_scale": v_ref.abs().mean().item(),
    }

    if k > 1:
        v_slots = v0.clone()
        v_slots[1:] = torch.empty_like(v_slots[1:]).normal_(generator=g)
        v_s, _ = sampler.velocity(cond, v_slots, 1.0, a0, 1.0)
        res["slot0_vs_other_slots"] = (v_s[0] - v_ref[0]).abs().max().item()
        # and the perturbed slots should of course change
        res["slot1_changed"] = (v_s[1] - v_ref[1]).abs().max().item()
    return res


def main() -> int:
    for dtype in (torch.bfloat16, torch.float32):
        print(f"\n===== params/compute dtype = {dtype} =====")
        for k in (1, 2, 4):
            r = run(dtype, k)
            print(f"  K={k}  video scale {r['video_scale']:.4f}")
            for key in ("same_video", "same_action", "video_vs_action_pert",
                        "action_vs_video_pert", "slot0_vs_other_slots", "slot1_changed"):
                if key in r:
                    flag = "" if r[key] == 0.0 else "   <-- nonzero"
                    print(f"      {key:24s} {r[key]:.6e}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
