"""Smoke test on the real teacher: invariants, noise floor, throughput, fidelity.

Everything here must pass before any variance number is worth reading. Order is
cheapest-first so a failure costs the least time.

  python scripts/smoke_teacher.py --frames 8 --k 4 [--full-ode] [--cfg 5.0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

from driftwam import metrics as M  # noqa: E402
from driftwam import models  # noqa: E402
from driftwam import paths  # noqa: E402
from driftwam import sampling as SP  # noqa: E402
from driftwam import shapes as S  # noqa: E402

RESULTS: dict = {}
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def pick_item(min_frames: int):
    """First sub-dataset segment with at least `min_frames` latent frames."""
    for repo in bootstrap.find_sub_datasets(CFG):
        ds = bootstrap.open_sub_dataset(repo, CFG)
        for i in range(len(ds)):
            it = ds[i]
            if it["latents"].shape[1] >= min_frames:
                return it, Path(repo).name, i
    raise RuntimeError(f"no segment with F >= {min_frames}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--full-ode", action="store_true",
                    help="run the complete 75-forward ODE for V1 fidelity")
    ap.add_argument("--out", default=str(Path(paths.EXP01_DIR) / "smoke.json"))
    args = ap.parse_args()
    dev, f, k = "cuda", args.frames, args.k

    print("=" * 72)
    print(f"device {torch.cuda.get_device_name(0)}  F={f}  K={k}  CFG={args.cfg}")
    sl = S.sequence_length(f)
    print(f"tokens/sample {sl['total']}  batch total {k * sl['total']}  "
          f"pad {(128 - k * sl['total'] % 128) % 128}")

    print("\n[0] load")
    t0 = time.time()
    teacher = models.load_teacher(device=dev, dtype=torch.bfloat16, attn_mode="flex")
    print(f"  teacher loaded in {time.time() - t0:.1f}s")
    check("attention op is FlexAttnFunc",
          type(teacher.blocks[0].attn1.attn_op).__name__ == "FlexAttnFunc")
    # teacher.config reports the override, so read what is actually on disk
    shipped = json.loads(
        (paths.transformer_dir(paths.TEACHER_PATH) / "config.json").read_text()
    ).get("attn_mode")
    print(f"  attn_mode on disk: {shipped!r} -> overridden to 'flex' at load")
    check("override was needed", shipped != "flex",
          "checkpoint would otherwise run unmasked attention in forward_train")
    empty_emb = bootstrap.load_empty_emb(CFG, device=dev)
    check("empty_emb is (512, 4096)", tuple(empty_emb.shape) == (512, 4096),
          f"{tuple(empty_emb.shape)} {empty_emb.dtype}")

    item, task, idx = pick_item(f)
    print(f"  item: {task} seg {idx}, native F={item['latents'].shape[1]} -> cropped to {f}")

    spec = SP.SamplerSpec(cfg_scale=args.cfg)
    sampler = SP.Sampler(teacher, spec, patch_size=tuple(CFG.patch_size),
                         chunk_size=CFG.frame_chunk_size, window_size=CFG.attn_window,
                         empty_emb=empty_emb)
    cond = SP.make_condition(item, k=k, device=dev, patch_size=tuple(CFG.patch_size),
                             n_frames=f, task=task)
    print(f"  condition: F={cond.n_frames} chunks={cond.n_chunks} K={cond.k}")

    g = torch.Generator(device=dev).manual_seed(1234)
    v0, a0 = SP.initial_noise(cond, g)

    print("\n[1] warm up (compiles create_block_mask + flex_attention)")
    t0 = time.time()
    sampler.velocity(cond, v0, 1.0, a0, 1.0)
    torch.cuda.synchronize()
    print(f"  first forward (incl. compile): {time.time() - t0:.1f}s")

    print("\n[2] V2 determinism")
    det = SP.check_determinism(sampler, cond, v0, a0)
    check("video bitwise stable", det["video_identical"], f"{det['video_max_diff']:.3e}")
    check("action bitwise stable", det["action_identical"], f"{det['action_max_diff']:.3e}")
    RESULTS["V2"] = det

    print("\n[3] V4 modality decoupling (licenses the interleaved 75-forward schedule)")
    dec = SP.check_modality_decoupling(sampler, cond, g)
    check("video blind to noisy action", dec["video_response_to_action_perturbation"] == 0.0,
          f"{dec['video_response_to_action_perturbation']:.3e} vs scale "
          f"{dec['video_velocity_scale']:.3e}")
    check("action blind to noisy video", dec["action_response_to_video_perturbation"] == 0.0,
          f"{dec['action_response_to_video_perturbation']:.3e} vs scale "
          f"{dec['action_velocity_scale']:.3e}")
    RESULTS["V4"] = dec

    print("\n[4] chunk independence (licenses ceil(F/2) conditions per forward)")
    ci = SP.check_chunk_independence(sampler, cond, g)
    check("earlier chunks unaffected", ci.get("independent", False),
          f"earlier {ci.get('earlier_chunk_response', float('nan')):.3e}, "
          f"perturbed {ci.get('perturbed_chunk_response', float('nan')):.3e}")
    RESULTS["chunk_independence"] = ci

    print("\n[5] batch isolation")
    v_ref, _ = sampler.velocity(cond, v0, 1.0, a0, 1.0)
    v_alt = v0.clone()
    v_alt[1:] = torch.empty_like(v_alt[1:]).normal_(generator=g)
    v_new, _ = sampler.velocity(cond, v_alt, 1.0, a0, 1.0)
    d0 = (v_new[0] - v_ref[0]).abs().max().item()
    check("slot 0 exactly unchanged", d0 == 0.0, f"{d0:.3e}")
    RESULTS["batch_isolation_slot0_diff"] = d0

    print("\n[6] V3 bf16 noise floor: same epsilon, different batch positions")
    v_rep, a_rep = SP.initial_noise(cond, g, n_replicate=k)
    v_r, a_r = sampler.velocity(cond, v_rep, 1.0, a_rep, 1.0)
    floor_v = M.noise_floor(v_r)
    floor_a = M.noise_floor(a_r * cond.actions_mask.float())
    print(f"  video  v_floor {floor_v['v_floor']:.4e}  rho_floor {floor_v['rho_floor']:.4e}  "
          f"max dev {floor_v['max_abs_dev']:.3e}")
    print(f"  action v_floor {floor_a['v_floor']:.4e}  rho_floor {floor_a['rho_floor']:.4e}  "
          f"max dev {floor_a['max_abs_dev']:.3e}")
    print("  NOTE this floor is layout-induced: identical epsilon at different batch")
    print("  positions takes a different block-sparse reduction order. It inflates")
    print("  measured variance, so the real runs keep each item at a fixed batch")
    print("  position and vary epsilon across outer iterations instead.")
    RESULTS["V3_floor_batched"] = {"video": floor_v, "action": floor_a}

    print("\n[7] same epsilon at a FIXED batch position across separate calls")
    v_one = v0[:1].repeat(k, 1, 1, 1, 1)
    a_one = a0[:1].repeat(k, 1, 1, 1, 1)
    outs = []
    for _ in range(3):
        vv, _ = sampler.velocity(cond, v_one, 1.0, a_one, 1.0)
        outs.append(vv[0:1].clone())
    spread = max((o - outs[0]).abs().max().item() for o in outs)
    check("fixed position is reproducible", spread == 0.0, f"{spread:.3e}")
    RESULTS["fixed_position_spread"] = spread

    print("\n[8] throughput")
    torch.cuda.synchronize()
    n_rep = 5
    t0 = time.time()
    for _ in range(n_rep):
        sampler.velocity(cond, v0, 1.0, a0, 1.0)
    torch.cuda.synchronize()
    per_fwd = (time.time() - t0) / n_rep
    tokens = k * sl["total"]
    tflop = 2 * 5.0e9 * tokens / 1e12
    print(f"  {per_fwd * 1000:.1f} ms per forward at B={k} ({tokens} tokens)")
    print(f"  ~{tflop / per_fwd:.0f} TFLOP/s effective (weights-only estimate)")
    print(f"  peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    fwd_per_cond_sample = spec.n_forwards / (k * cond.n_chunks)
    print(f"  {spec.n_forwards} forwards per ODE -> "
          f"{per_fwd * spec.n_forwards:.1f}s per {k} samples x {cond.n_chunks} chunks")
    RESULTS["throughput"] = {
        "sec_per_forward": per_fwd, "batch": k, "tokens": tokens,
        "tflops_effective": tflop / per_fwd,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "forwards_per_ode": spec.n_forwards,
        "forwards_per_condition_sample": fwd_per_cond_sample,
    }

    if args.full_ode:
        print(f"\n[9] full ODE ({spec.n_forwards} forwards) + V1 fidelity")
        t0 = time.time()
        out = sampler.teacher_ode(cond, v0, a0)
        torch.cuda.synchronize()
        print(f"  elapsed {time.time() - t0:.1f}s")
        gen_v, gen_a = out["video"], out["action"]
        true_v, true_a = cond.clean_latents, cond.clean_actions
        # V1 on chunks that have visual history; chunk 0 is text-only conditioned
        later = [fr for c in range(1, cond.n_chunks)
                 for fr in S.video_frames_of_chunk(c, cond.n_frames)]
        rel_v = M.sampler_fidelity(gen_v[:, :, later].mean(0), true_v[0, :, later])
        act_frames = S.action_frame_slice(cond.n_frames)
        rel_a = M.sampler_fidelity(
            gen_a[:, S.VALID_ACTION_CHANNELS][:, :, act_frames].mean(0),
            true_a[0, S.VALID_ACTION_CHANNELS][:, act_frames],
        )
        print(f"  video relative L2 (mean vs truth, chunks>=1): {rel_v:.4f}")
        print(f"  action relative L2 (valid channels, frames>=1): {rel_a:.4f}")
        check("video fidelity < 0.6", rel_v < 0.6, f"{rel_v:.4f}")
        vs = M.condition_stats(gen_v[:, :, later])
        as_ = M.condition_stats(gen_a[:, S.VALID_ACTION_CHANNELS][:, :, act_frames])
        print(f"  video  rho {vs.rho:.4e}  PR {vs.participation_ratio:.2f}  "
              f"min_pair {vs.min_pair_ratio:.3f}")
        print(f"  action rho {as_.rho:.4e}  PR {as_.participation_ratio:.2f}  "
              f"min_pair {as_.min_pair_ratio:.3f}")
        print("  (single condition set, K small: indicative only, not a verdict)")
        RESULTS["full_ode"] = {
            "video_rel_l2": rel_v, "action_rel_l2": rel_a,
            "video": vs.as_row(), "action": as_.as_row(),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    RESULTS["fails"] = FAILS
    RESULTS["config"] = {"frames": f, "k": k, "cfg": args.cfg, "task": task}
    Path(args.out).write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\nwrote {args.out}")
    print("ALL PASSED" if not FAILS else f"FAILED: {', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
