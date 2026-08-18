"""End-to-end plumbing test for the sampler, on a tiny randomly-initialised model.

The invariants the sampling protocol leans on -- video/action decoupling, chunk
independence, batch isolation -- are properties of the attention mask, not of the
weights, so a 2-layer model with random weights tests them exactly as well as the
real 5B checkpoint and costs nothing. What this cannot check is anything about the
teacher's actual distribution.

Run:  python tests/test_forward_tiny.py [--device cuda] [--frames 8] [--k 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

from driftwam import sampling as SP  # noqa: E402
from driftwam import shapes as S  # noqa: E402


def tiny_model(device, dtype=torch.bfloat16, attn_mode="flex"):
    """A 2-layer stand-in with the real architecture.

    `attn_mode` defaults to "flex" because that is the only mode that applies the
    block-causal mask; "torch" maps to `custom_sdpa`, which ignores masks entirely.
    """
    from modules.model import WanTransformer3DModel

    torch.manual_seed(0)
    m = WanTransformer3DModel(
        patch_size=[1, 2, 2],
        num_attention_heads=2,
        attention_head_dim=32,
        in_channels=S.LATENT_CHANNELS,
        out_channels=S.LATENT_CHANNELS,
        action_dim=S.ACTION_DIM,
        text_dim=4096,
        ffn_dim=128,
        num_layers=2,
        attn_mode=attn_mode,
    )
    return m.to(device=device, dtype=dtype).eval()


def synthetic_item(n_frames: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(S.ACTION_DIM, n_frames, S.ACTION_PER_FRAME, 1, dtype=torch.bool)
    mask[S.VALID_ACTION_CHANNELS] = True
    return {
        "latents": torch.randn(
            S.LATENT_CHANNELS, n_frames, S.GRID_H, S.GRID_W, generator=g
        ).to(torch.bfloat16),
        "actions": torch.randn(
            S.ACTION_DIM, n_frames, S.ACTION_PER_FRAME, 1, generator=g
        ).clamp(-1, 1) * mask,
        "actions_mask": mask,
        "text_emb": torch.randn(512, 4096, generator=g).to(torch.bfloat16),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--attn-mode", default="flex", choices=["flex", "torch"])
    args = ap.parse_args()
    dev, f, k = args.device, args.frames, args.k

    print(f"device={dev}  F={f}  K={k}  attn_mode={args.attn_mode}")
    sl = S.sequence_length(f)
    print(f"tokens/sample: video {sl['video']}x2 + action {sl['action']}x2 = {sl['total']}"
          f"  -> batch total {k * sl['total']}, pad to 128 -> "
          f"{(128 - k * sl['total'] % 128) % 128}")

    model = tiny_model(dev, attn_mode=args.attn_mode)
    n_params = sum(p.numel() for p in model.parameters())
    op = type(model.blocks[0].attn1.attn_op).__name__
    print(f"tiny model: {n_params / 1e6:.2f} M params, attention op = {op}")

    empty_emb = torch.randn(512, 4096, dtype=torch.bfloat16, device=dev)
    spec = SP.SamplerSpec(video_steps=2, action_steps=4, cfg_scale=5.0)
    sampler = SP.Sampler(
        model, spec,
        patch_size=tuple(CFG.patch_size),
        chunk_size=CFG.frame_chunk_size,
        window_size=CFG.attn_window,
        empty_emb=empty_emb,
    )

    item = synthetic_item(f)
    cond = SP.make_condition(item, k=k, device=dev, patch_size=tuple(CFG.patch_size), n_frames=f)
    print(f"condition: F={cond.n_frames} chunks={cond.n_chunks} K={cond.k}")

    fails = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    print("\n[1] sigma grids")
    vs, vn = SP.sigma_grid(25, 5.0)
    as_, an = SP.sigma_grid(50, 1.0)
    check("video grid starts at 1.0", abs(vs[0].item() - 1.0) < 1e-9, f"{vs[0].item():.6f}")
    check("video grid ends above 0", vs[-1].item() > 0.15, f"last sigma {vs[-1].item():.4f}")
    check("video final step lands on 0", vn[-1].item() == 0.0)
    check("action grid ends above 0", as_[-1].item() > 0.0, f"last sigma {as_[-1].item():.4f}")
    check("action final step lands on 0", an[-1].item() == 0.0)
    check("grids strictly decreasing", bool((vs[1:] < vs[:-1]).all() and (as_[1:] < as_[:-1]).all()))

    print("\n[2] one forward, output shapes")
    g = torch.Generator(device=dev).manual_seed(1234)
    v0, a0 = SP.initial_noise(cond, g)
    check("noise shapes", v0.shape == cond.clean_latents.shape and a0.shape == cond.clean_actions.shape)
    check("action noise respects mask",
          bool((a0[~cond.actions_mask] == 0).all()))
    v_v, v_a = sampler.velocity(cond, v0, 1.0, a0, 1.0)
    check("video velocity shape", v_v.shape == (k, S.LATENT_CHANNELS, f, S.GRID_H, S.GRID_W),
          str(tuple(v_v.shape)))
    check("action velocity shape", v_a.shape == (k, S.ACTION_DIM, f, S.ACTION_PER_FRAME, 1),
          str(tuple(v_a.shape)))
    check("velocities finite", bool(torch.isfinite(v_v).all() and torch.isfinite(v_a).all()))

    print("\n[3] V2 determinism")
    det = SP.check_determinism(sampler, cond, v0, a0)
    check("video bitwise identical", det["video_identical"], f"max diff {det['video_max_diff']:.3e}")
    check("action bitwise identical", det["action_identical"], f"max diff {det['action_max_diff']:.3e}")

    print("\n[4] V4 modality decoupling (mask property, weight independent)")
    dec = SP.check_modality_decoupling(sampler, cond, g)
    check("video blind to noisy action",
          dec["video_response_to_action_perturbation"] == 0.0,
          f"response {dec['video_response_to_action_perturbation']:.3e} "
          f"vs scale {dec['video_velocity_scale']:.3e}")
    check("action blind to noisy video",
          dec["action_response_to_video_perturbation"] == 0.0,
          f"response {dec['action_response_to_video_perturbation']:.3e} "
          f"vs scale {dec['action_velocity_scale']:.3e}")

    print("\n[5] chunk independence (licenses many conditions per forward)")
    ci = SP.check_chunk_independence(sampler, cond, g)
    check("earlier chunks unaffected by last chunk noise", ci.get("independent", False),
          f"earlier {ci.get('earlier_chunk_response', float('nan')):.3e} / "
          f"perturbed {ci.get('perturbed_chunk_response', float('nan')):.3e}")

    print("\n[6] batch isolation (K samples must not leak into each other)")
    v_alt = v0.clone()
    v_alt[1:] = torch.empty_like(v_alt[1:]).normal_(generator=g)
    v_new, _ = sampler.velocity(cond, v_alt, 1.0, a0, 1.0)
    check("slot 0 unchanged when other slots change",
          bool(torch.equal(v_new[0], v_v[0])),
          f"max diff {(v_new[0] - v_v[0]).abs().max().item():.3e}")

    print("\n[7] identical epsilon in different batch slots -> identical output")
    v_rep, a_rep = SP.initial_noise(cond, g, n_replicate=k)
    v_r, a_r = sampler.velocity(cond, v_rep, 1.0, a_rep, 1.0)
    spread_v = (v_r - v_r[0:1]).abs().max().item()
    spread_a = (a_r - a_r[0:1]).abs().max().item()
    print(f"        video spread across slots {spread_v:.3e}  (scale {v_r.abs().mean().item():.3e})")
    print(f"        action spread across slots {spread_a:.3e}  (scale {a_r.abs().mean().item():.3e})")
    print("        ^ this is the V3 noise floor; nonzero is expected in bf16")

    print("\n[8] teacher ODE loop")
    out = sampler.teacher_ode(cond, v0, a0, record_x0=True)
    check("video output shape", out["video"].shape == cond.clean_latents.shape)
    check("action output shape", out["action"].shape == cond.clean_actions.shape)
    check("outputs finite", bool(torch.isfinite(out["video"]).all() and torch.isfinite(out["action"]).all()))
    check("action output respects mask", bool((out["action"][~cond.actions_mask] == 0).all()))
    traj = out["trajectory"]
    check("recorded video steps == video_steps", len(traj["x0_video"]) == spec.video_steps,
          f"{len(traj['x0_video'])}")
    check("recorded action steps == action_steps", len(traj["x0_action"]) == spec.action_steps,
          f"{len(traj['x0_action'])}")

    print("\n[9] one-step student path")
    st = sampler.student_one_step(cond, v0, a0)
    check("c_skip/c_out at sigma=1", abs(st["c_skip"] - 0.2) < 1e-9 and abs(st["c_out"] - 0.4472) < 1e-4,
          f"c_skip={st['c_skip']:.4f} c_out={st['c_out']:.4f}")
    check("student video shape", st["video"].shape == cond.clean_latents.shape)
    check("student action respects mask", bool((st["action"][~cond.actions_mask] == 0).all()))

    print("\n[10] exactness of the interleaved schedule vs separate loops")
    # With CFG off, running the two branches in one interleaved loop must equal
    # running each branch on its own -- the whole justification for sharing forwards.
    spec_nocfg = SP.SamplerSpec(video_steps=2, action_steps=4, cfg_scale=1.0)
    s2 = SP.Sampler(model, spec_nocfg, patch_size=tuple(CFG.patch_size),
                    chunk_size=CFG.frame_chunk_size, window_size=CFG.attn_window,
                    empty_emb=empty_emb)  # noqa: E501
    inter = s2.teacher_ode(cond, v0, a0)
    v_only = _video_only(s2, cond, v0, a0)
    a_only = _action_only(s2, cond, v0, a0)
    check("interleaved video == video-only loop", bool(torch.equal(inter["video"], v_only)),
          f"max diff {(inter['video'] - v_only).abs().max().item():.3e}")
    check("interleaved action == action-only loop", bool(torch.equal(inter["action"], a_only)),
          f"max diff {(inter['action'] - a_only).abs().max().item():.3e}")

    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


@torch.no_grad()
def _video_only(sampler, cond, video_x, action_x):
    """Advance only video, holding the action input pinned at its initial state."""
    sig, nxt = sampler._v_sigmas.tolist(), sampler._v_next.tolist()
    a_sig = sampler._a_sigmas.tolist()
    for i, s in enumerate(sig):
        v, _ = sampler.velocity(cond, video_x, s, action_x, a_sig[i * sampler.spec.video_every])
        video_x = video_x + v * (nxt[i] - s)
    return video_x


@torch.no_grad()
def _action_only(sampler, cond, video_x, action_x):
    """Advance only action, holding the video input pinned at its initial state."""
    sig, nxt = sampler._a_sigmas.tolist(), sampler._a_next.tolist()
    v_sig = sampler._v_sigmas.tolist()
    mask = cond.actions_mask.float()
    for i, s in enumerate(sig):
        vi = min(i // sampler.spec.video_every, sampler.spec.video_steps - 1)
        _, a = sampler.velocity(cond, video_x, v_sig[vi], action_x, s)
        action_x = (action_x + a * (nxt[i] - s)) * mask
    return action_x


if __name__ == "__main__":
    raise SystemExit(main())
