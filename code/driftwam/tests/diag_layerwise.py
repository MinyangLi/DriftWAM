"""Find the first tensor that diverges when only the noisy action stream changes.

Hooks every transformer block and compares the *video* token positions between a
baseline run and one where only the noisy action values were replaced. If the mask
does what the dense check says, the video positions must stay bitwise identical all
the way to the output. Anything else localises where information (or rounding
noise) crosses over.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

from driftwam import sampling as SP  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_forward_tiny import synthetic_item, tiny_model  # noqa: E402

K, F, DEV = 1, 8, "cuda"
LN = F * 12 * 10  # video tokens per sample
AN = F * 16  # action tokens per sample


def main() -> int:
    model = tiny_model(DEV, dtype=torch.bfloat16)
    spec = SP.SamplerSpec(video_steps=2, action_steps=4, cfg_scale=1.0)
    sampler = SP.Sampler(model, spec, patch_size=tuple(CFG.patch_size),
                         chunk_size=CFG.frame_chunk_size, window_size=CFG.attn_window)
    cond = SP.make_condition(synthetic_item(F), k=K, device=DEV,
                             patch_size=tuple(CFG.patch_size), n_frames=F)
    g = torch.Generator(device=DEV).manual_seed(7)
    v0, a0 = SP.initial_noise(cond, g)
    a_pert = torch.empty_like(a0).normal_(generator=g) * cond.actions_mask.float()

    captured: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        captured.append(out.detach().clone())

    handles = [b.register_forward_hook(hook) for b in model.blocks]

    sampler.velocity(cond, v0, 1.0, a0, 1.0)  # warm up
    captured.clear()
    v_ref, a_ref = sampler.velocity(cond, v0, 1.0, a0, 1.0)
    ref_acts = list(captured)

    captured.clear()
    v_new, _ = sampler.velocity(cond, v0, 1.0, a_pert, 1.0)
    new_acts = list(captured)
    for h in handles:
        h.remove()

    # segment layout inside the packed sequence
    seg = {
        "video_noisy": slice(0, K * LN),
        "video_clean": slice(K * LN, 2 * K * LN),
        "action_noisy": slice(2 * K * LN, 2 * K * LN + K * AN),
        "action_clean": slice(2 * K * LN + K * AN, 2 * K * LN + 2 * K * AN),
    }

    print(f"K={K} F={F}  blocks captured: {len(ref_acts)}")
    print(f"{'block':>6}  " + "  ".join(f"{n:>18}" for n in seg))
    for i, (r, n) in enumerate(zip(ref_acts, new_acts)):
        row = []
        for s in seg.values():
            d = (n[:, s] - r[:, s]).abs().max().item()
            row.append(f"{d:18.6e}")
        print(f"{i:>6}  " + "  ".join(row))

    dv = (v_new - v_ref).abs().max().item()
    print(f"\nfinal video velocity max diff: {dv:.6e}")
    n_diff = (v_new != v_ref).sum().item()
    print(f"elements differing: {n_diff} of {v_ref.numel()} "
          f"({100.0 * n_diff / v_ref.numel():.4f}%)")
    if n_diff:
        idx = (v_new != v_ref).nonzero()[0].tolist()
        print(f"first differing index {idx}: ref={v_ref[tuple(idx)].item():.8f} "
              f"new={v_new[tuple(idx)].item():.8f}")
        big = v_ref.abs().max().item()
        print(f"max |video velocity| = {big:.4f}  (1 bf16 ulp there = "
              f"{2 ** (torch.tensor(big).log2().floor().item() - 8):.6e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
