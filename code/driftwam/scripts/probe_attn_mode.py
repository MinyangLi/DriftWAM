"""Was the released Flash-WAM student trained with the block-causal mask or without it?

Nothing in the Flash-WAM repo ever sets `attn_mode`, so `distillation/trainer.py`
inherits `"torch"` from config.json and runs `forward_train` with unmasked
attention. Either the released training code differs from what produced the
checkpoint, or the student really was trained unmasked. The checkpoint itself can
say which.

The two modes give opposite signatures:

* trained **masked**  -> flex reconstructs well; torch feeds it tokens it never
  learned to attend to, so accuracy degrades.
* trained **unmasked** -> torch reconstructs almost perfectly, because a noisy
  token can read the clean ground truth of its own chunk and copy the answer,
  while flex removes that shortcut and accuracy collapses.

Also settles how *I* must evaluate the student for H7: faithfully, in whichever
mode it was trained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

from driftwam import metrics as M  # noqa: E402
from driftwam import models, paths  # noqa: E402
from driftwam import sampling as SP  # noqa: E402
from driftwam import shapes as S  # noqa: E402


def set_attn_mode(model, mode: str) -> str:
    """Swap the attention op in place, so the 5B weights load only once.

    `FlexAttnFunc` is an nn.Module and registers under `_modules`, while
    `custom_sdpa` is a bare function; nn.Module.__setattr__ refuses to overwrite a
    registered child with a non-module, so clear both slots first.
    """
    from modules.model import FlexAttnFunc, custom_sdpa  # noqa: PLC0415

    for b in model.blocks:
        for attn, is_cross in ((b.attn1, False), (b.attn2, True)):
            attn._modules.pop("attn_op", None)
            attn.__dict__.pop("attn_op", None)
            if mode == "flex":
                attn.attn_op = FlexAttnFunc(is_cross)
            else:
                object.__setattr__(attn, "attn_op", custom_sdpa)
    return type(model.blocks[0].attn1.attn_op).__name__


def fidelity(pred_v, pred_a, cond) -> tuple[float, float]:
    """Relative L2 against ground truth, on the parts that carry information."""
    later = [fr for c in range(1, cond.n_chunks)
             for fr in S.video_frames_of_chunk(c, cond.n_frames)]
    af = S.action_frame_slice(cond.n_frames)
    ch = S.VALID_ACTION_CHANNELS
    rel_v = M.sampler_fidelity(pred_v[0, :, later], cond.clean_latents[0, :, later])
    rel_a = M.sampler_fidelity(pred_a[0][ch][:, af], cond.clean_actions[0][ch][:, af])
    return rel_v, rel_a


def flow_matching_residual(sampler, cond, sigma, generator) -> tuple[float, float]:
    """Relative error on the actual pretraining objective, at one sigma.

    With `x_t = (1-s) x_0 + s eps`, flow matching regresses `v = eps - x_0`. This is
    what the teacher was literally trained to minimise, so it is the fair way to ask
    which attention mode a checkpoint expects, and unlike a consistency one-step it
    stays meaningful for a non-distilled model.
    """
    eps_v, eps_a = SP.initial_noise(cond, generator)
    x0_v, x0_a = cond.clean_latents, cond.clean_actions
    xt_v = (1 - sigma) * x0_v + sigma * eps_v
    xt_a = ((1 - sigma) * x0_a + sigma * eps_a) * cond.actions_mask.float()
    pv, pa = sampler.velocity(cond, xt_v, sigma, xt_a, sigma)

    later = [fr for c in range(1, cond.n_chunks)
             for fr in S.video_frames_of_chunk(c, cond.n_frames)]
    af = S.action_frame_slice(cond.n_frames)
    ch = S.VALID_ACTION_CHANNELS
    rel_v = M.sampler_fidelity(pv[0, :, later], (eps_v - x0_v)[0, :, later])
    rel_a = M.sampler_fidelity(pa[0][ch][:, af], (eps_a - x0_a)[0][ch][:, af])
    return rel_v, rel_a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--items", type=int, default=4)
    ap.add_argument("--out", default=str(Path(paths.EXP01_DIR) / "attn_mode_probe.json"))
    args = ap.parse_args()
    dev, f = "cuda", args.frames

    print("=" * 74)
    for name, p in (("teacher", paths.TEACHER_PATH), ("student", paths.STUDENT_PATH)):
        shipped = json.loads(
            (paths.transformer_dir(p) / "config.json").read_text()
        ).get("attn_mode")
        print(f"{name:>8} config.json attn_mode = {shipped!r}")

    # gather a few items
    items = []
    for repo in bootstrap.find_sub_datasets(CFG):
        ds = bootstrap.open_sub_dataset(repo, CFG)
        for i in range(len(ds)):
            it = ds[i]
            if it["latents"].shape[1] >= f:
                items.append((it, Path(repo).name, i))
                break
        if len(items) >= args.items:
            break
    print(f"\n{len(items)} items at F={f}: " + ", ".join(t for _, t, _ in items))

    results: dict = {}
    for who, loader in (("student", models.load_student), ("teacher", models.load_teacher)):
        print(f"\n{'=' * 74}\n{who}")
        model = loader(device=dev, dtype=torch.bfloat16, attn_mode="flex", verbose=False)
        empty = bootstrap.load_empty_emb(CFG, device=dev)
        spec = SP.SamplerSpec(cfg_scale=1.0)
        results[who] = {}

        for mode in ("flex", "torch"):
            op = set_attn_mode(model, mode)
            sampler = SP.Sampler(model, spec, patch_size=tuple(CFG.patch_size),
                                 chunk_size=CFG.frame_chunk_size,
                                 window_size=CFG.attn_window, empty_emb=empty)
            acc: dict[str, list[float]] = {k: [] for k in
                                           ("fm_v", "fm_a", "os_v", "os_a")}
            for it, task, idx in items:
                cond = SP.make_condition(it, k=1, device=dev,
                                         patch_size=tuple(CFG.patch_size), n_frames=f,
                                         task=task)
                g = torch.Generator(device=dev).manual_seed(99)
                a, b = flow_matching_residual(sampler, cond, 0.5, g)
                acc["fm_v"].append(a)
                acc["fm_a"].append(b)

                g = torch.Generator(device=dev).manual_seed(99)
                v0, a0 = SP.initial_noise(cond, g)
                out = sampler.student_one_step(cond, v0, a0)
                a, b = fidelity(out["video"], out["action"], cond)
                acc["os_v"].append(a)
                acc["os_a"].append(b)

            mean = {k: sum(v) / len(v) for k, v in acc.items()}
            results[who][mode] = {"attn_op": op, "mean": mean, "per_item": acc}
            print(f"  {mode:>5} ({op:>12})  "
                  f"flow-matching residual  video {mean['fm_v']:.4f}  "
                  f"action {mean['fm_a']:.4f}")
            print(f"  {'':>5} {'':>14}  "
                  f"one-step vs truth       video {mean['os_v']:.4f}  "
                  f"action {mean['os_a']:.4f}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 74}\nverdict (on the flow-matching residual)")
    for who in results:
        for key, what in (("fm_v", "video"), ("fm_a", "action")):
            fx = results[who]["flex"]["mean"][key]
            tc = results[who]["torch"]["mean"][key]
            better = "masked (flex)" if fx < tc else "UNMASKED (torch)"
            print(f"  {who:>8} {what:>6}: prefers {better:>16}  "
                  f"flex {fx:.4f} vs torch {tc:.4f}  ({tc / max(fx, 1e-9):.2f}x)")
    print("\n  Student preferring masked attention about as strongly as the teacher")
    print("  means it was trained with the mask, and the released distillation code")
    print("  merely lost the attn_mode override somewhere in refactoring.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
