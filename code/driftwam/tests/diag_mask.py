"""Pinpoint whether the block-causal mask is built correctly and honoured.

Rebuilds the id vectors exactly as `FlexAttnFunc.init_mask` does, evaluates the
mask rules densely as ground truth, compares that against the BlockMask actually
produced, and finally compares flex_attention's output against a dense-masked
SDPA reference on the same q/k/v.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

B, F, CH = 4, 8, 2
GH, GW, PH, PW = 24, 20, 2, 2
APF = 16
LN = F * (GH // PH) * (GW // PW)  # 960 latent tokens per sample
AN = F * APF  # 128 action tokens per sample


def build_ids(b=B, f=F, chunk=CH):
    lat_seq = torch.arange(b)[:, None, None, None].expand(-1, f, GH // PH, GW // PW).flatten()
    act_seq = torch.arange(b)[:, None, None, None].expand(-1, f, APF, 1).flatten()
    seq_ids = torch.cat([lat_seq] * 2 + [act_seq] * 2)

    lat_fid = torch.arange(f)[None, :, None, None].expand(b, -1, GH // PH, GW // PW)[None].flatten()
    act_fid = torch.arange(f)[None, :, None, None].expand(b, -1, APF, 1)[None].flatten()
    frame_ids = torch.cat([lat_fid // chunk * 2] * 2 + [act_fid // chunk * 2 + 1] * 2)

    noise_ids = torch.cat([
        torch.zeros_like(lat_fid), torch.ones_like(lat_fid),
        torch.zeros_like(act_fid), torch.ones_like(act_fid),
    ])
    return seq_ids, frame_ids, noise_ids


def dense_reference(seq_ids, frame_ids, noise_ids, window=72):
    """The three rules, or'd, and'd with sequence and window masks."""
    s_q, s_k = seq_ids[:, None], seq_ids[None, :]
    f_q, f_k = frame_ids[:, None], frame_ids[None, :]
    n_q, n_k = noise_ids[:, None], noise_ids[None, :]

    same_seq = (s_q == s_k) & (s_q >= 0) & (s_k >= 0)
    clean2clean = (n_q == 1) & (n_k == 1) & (f_k <= f_q)
    noise2clean = (n_q == 0) & (n_k == 1) & (f_k < f_q)
    noise2noise = (n_q == 0) & (n_k == 0) & (f_k == f_q)
    within_window = (f_q - f_k).abs() <= window
    return same_seq & (clean2clean | noise2clean | noise2noise) & within_window


def segment_slices(b=B):
    n = b * LN
    a = b * AN
    return {
        "video_noisy": slice(0, n),
        "video_clean": slice(n, 2 * n),
        "action_noisy": slice(2 * n, 2 * n + a),
        "action_clean": slice(2 * n + a, 2 * n + 2 * a),
    }


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    seq_ids, frame_ids, noise_ids = build_ids()
    total = len(seq_ids)
    print(f"B={B} F={F}  tokens/sample: {LN} video + {AN} action  total seq {total}")

    ref = dense_reference(seq_ids, frame_ids, noise_ids, window=CFG.attn_window)
    seg = segment_slices()
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    print("\n[A] dense reference encodes the intended structure")
    check("noisy video cannot see noisy action",
          not ref[seg["video_noisy"], seg["action_noisy"]].any().item())
    check("noisy action cannot see noisy video",
          not ref[seg["action_noisy"], seg["video_noisy"]].any().item())
    check("noisy action CAN see clean video of its own chunk",
          ref[seg["action_noisy"], seg["video_clean"]].any().item())
    # cross-sample leakage
    q0 = slice(0, LN)  # sample 0 noisy video
    k1 = slice(LN, 2 * LN)  # sample 1 noisy video
    check("sample 0 cannot see sample 1", not ref[q0, k1].any().item())

    print("\n[B] BlockMask from init_mask matches the dense reference")
    from modules.model import FlexAttnFunc

    FlexAttnFunc.init_mask(
        (B, 48, F, GH, GW), (B, 30, F, APF, 1), 0, CH,
        window_size=CFG.attn_window, patch_size=(1, PH, PW), device=dev,
    )
    bm = FlexAttnFunc.attention_mask
    print(f"  BlockMask shape {bm.shape}  BLOCK_SIZE {bm.BLOCK_SIZE}")
    got = bm.to_dense()[0, 0].bool().cpu()
    if got.shape != ref.shape:
        check("dense shapes match", False, f"{tuple(got.shape)} vs {tuple(ref.shape)}")
    else:
        diff = (got != ref)
        check("BlockMask == dense reference", not diff.any().item(),
              f"{int(diff.sum())} differing entries of {diff.numel()}")
        if diff.any():
            extra = (got & ~ref).sum().item()
            missing = (~got & ref).sum().item()
            print(f"        allows {extra} it should not, forbids {missing} it should allow")

    print("\n[C] flex_attention honours the mask (vs dense-masked SDPA)")
    nh, hd = 2, 32
    q = torch.randn(1, nh, total, hd, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, nh, total, hd, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, nh, total, hd, device=dev, dtype=torch.bfloat16)

    out_flex = FlexAttnFunc.flex_attn(
        q, k, v, block_mask=bm,
        kernel_options={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32,
                        "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32},
    )
    out_sdpa = Fn.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=ref.to(dev)[None, None]
    )
    err = (out_flex.float() - out_sdpa).abs().max().item()
    scale = out_sdpa.abs().mean().item()
    check("flex_attention ~= masked SDPA", err < 0.05 * max(scale, 1e-6),
          f"max abs err {err:.3e}, output scale {scale:.3e}")

    print("\n[D] does perturbing one sample's keys change another sample's output?")
    k2 = k.clone()
    k2[:, :, LN:2 * LN] = torch.randn_like(k2[:, :, LN:2 * LN])  # sample 1 noisy video
    out2 = FlexAttnFunc.flex_attn(
        q, k2, v, block_mask=bm,
        kernel_options={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32,
                        "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32},
    )
    d0 = (out2[:, :, :LN] - out_flex[:, :, :LN]).abs().max().item()
    check("sample 0 output unchanged", d0 == 0.0, f"max diff {d0:.3e}")

    print(f"\n{'ALL PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
