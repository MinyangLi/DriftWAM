"""Does flex_attention respect the mask at element level, and does the
BlockMask BLOCK_SIZE vs kernel BLOCK_M/N mismatch matter?

The decisive experiment: pick queries in the noisy-video segment, then perturb
*only* keys the mask forbids them from seeing. Any change in their output is a
mask violation. Run in fp32 so bf16 rounding cannot mask or mimic the effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as Fn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_mask import build_ids, dense_reference, segment_slices  # noqa: E402

B, F, CH, DEV = 1, 8, 2, "cuda"
LN, AN = F * 12 * 10, F * 16
KOPTS = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 64,
         "BLOCK_M2": 64, "BLOCK_N2": 32}


def make_mask_mod(seq_ids, frame_ids, noise_ids, window):
    def mod(b, h, q, kv):
        same = (seq_ids[q] == seq_ids[kv]) & (seq_ids[q] >= 0) & (seq_ids[kv] >= 0)
        c2c = (noise_ids[q] == 1) & (noise_ids[kv] == 1) & (frame_ids[kv] <= frame_ids[q])
        n2c = (noise_ids[q] == 0) & (noise_ids[kv] == 1) & (frame_ids[kv] < frame_ids[q])
        n2n = (noise_ids[q] == 0) & (noise_ids[kv] == 0) & (frame_ids[kv] == frame_ids[q])
        win = (frame_ids[q] - frame_ids[kv]).abs() <= window
        return same & (c2c | n2c | n2n) & win
    return mod


def main() -> int:
    torch.manual_seed(0)
    seq_ids, frame_ids, noise_ids = build_ids(b=B, f=F, chunk=CH)
    n = len(seq_ids)
    ref = dense_reference(seq_ids, frame_ids, noise_ids, window=CFG.attn_window).to(DEV)
    seg = segment_slices(b=B)
    print(f"B={B} F={F} total seq {n}")

    mod = make_mask_mod(
        seq_ids.long().to(DEV), frame_ids.long().to(DEV), noise_ids.long().to(DEV),
        CFG.attn_window,
    )
    masks = {
        "BLOCK_SIZE=128": create_block_mask(mod, 1, 1, n, n, device=DEV, BLOCK_SIZE=128),
        "BLOCK_SIZE=64": create_block_mask(mod, 1, 1, n, n, device=DEV, BLOCK_SIZE=64),
    }

    nh, hd = 2, 32
    q = torch.randn(1, nh, n, hd, device=DEV, dtype=torch.float32)
    k = torch.randn(1, nh, n, hd, device=DEV, dtype=torch.float32)
    v = torch.randn(1, nh, n, hd, device=DEV, dtype=torch.float32)
    out_ref = Fn.scaled_dot_product_attention(q, k, v, attn_mask=ref[None, None])

    # keys the noisy-video queries must not see: the noisy-action segment
    k_pert = k.clone()
    k_pert[:, :, seg["action_noisy"]] = torch.randn_like(k_pert[:, :, seg["action_noisy"]])
    v_pert = v.clone()
    v_pert[:, :, seg["action_noisy"]] = torch.randn_like(v_pert[:, :, seg["action_noisy"]])

    qs = seg["video_noisy"]
    print(f"\nqueries examined: noisy video [{qs.start}:{qs.stop}]")
    print(f"forbidden keys perturbed: noisy action "
          f"[{seg['action_noisy'].start}:{seg['action_noisy'].stop}]")
    print(f"{'variant':>26}  {'err vs SDPA':>14}  {'leak on perturb':>16}")

    fails = []
    for name, bm in masks.items():
        for kopts, tag in ((KOPTS, "kernel 64"), (None, "kernel default")):
            call = lambda K, V: flex_attention(  # noqa: E731
                q, K, V, block_mask=bm, **({"kernel_options": kopts} if kopts else {})
            )
            o1 = call(k, v)
            o2 = call(k_pert, v_pert)
            err = (o1 - out_ref).abs().max().item()
            leak = (o2[:, :, qs] - o1[:, :, qs]).abs().max().item()
            label = f"{name} / {tag}"
            print(f"{label:>26}  {err:14.3e}  {leak:16.3e}")
            if leak != 0.0:
                fails.append(label)

    print("\nSDPA reference control (same perturbation, dense mask):")
    o_ref2 = Fn.scaled_dot_product_attention(q, k_pert, v_pert, attn_mask=ref[None, None])
    print(f"  dense-masked SDPA leak: "
          f"{(o_ref2[:, :, qs] - out_ref[:, :, qs]).abs().max().item():.3e}")

    print(f"\n{'NO LEAK' if not fails else 'MASK VIOLATED in: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
