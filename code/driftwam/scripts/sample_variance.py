"""Draw K conditional samples per item and write them to disk.

Orchestration (see `sampling.stack_items` for why): the batch dimension holds B
*different* items, and epsilon varies across an outer loop over k, so every item
keeps one fixed batch slot for all of its K draws. Batching the K draws of one
item instead would place each draw at a different offset, and the block-sparse
reduction order depends on the offset, manufacturing ~0.35% of spurious spread.

Cost is `ceil(n_items/B) * K * 75` forwards for the teacher and
`ceil(n_items/B) * K * 1` for the student.

  python scripts/sample_variance.py --subject teacher --items 20 --k 8 --cfg 5
  python scripts/sample_variance.py --subject student --items 20 --k 8
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

from driftwam import models, paths  # noqa: E402
from driftwam import sampling as SP  # noqa: E402
from driftwam import shapes as S  # noqa: E402


def select_items(n_items: int, n_tasks: int, min_frames: int, seed: int = 0) -> list[dict]:
    """Spread items over tasks, taking each from a different episode.

    Reads the F scan rather than opening every dataset, so selection is instant
    and reproducible. Tasks are sorted and picked at an even stride so the subset
    is not biased toward alphabetically early tasks.
    """
    scan_path = Path(paths.EXP01_DIR) / "f_scan.json"
    if not scan_path.exists():
        raise FileNotFoundError(
            f"{scan_path} missing; run scripts/scan_frames.py first"
        )
    scan = json.loads(scan_path.read_text())

    eligible = {
        task: [i for i, f in enumerate(info["F"]) if f >= min_frames]
        for task, info in sorted(scan.items())
    }
    eligible = {t: segs for t, segs in eligible.items() if segs}
    if not eligible:
        raise RuntimeError(f"no task has any segment with F >= {min_frames}")

    tasks = sorted(eligible)
    n_tasks = min(n_tasks, len(tasks))
    stride = len(tasks) / n_tasks
    chosen_tasks = [tasks[int(i * stride)] for i in range(n_tasks)]

    per_task = -(-n_items // n_tasks)  # ceil
    rng = torch.Generator().manual_seed(seed)
    picked: list[dict] = []
    for task in chosen_tasks:
        segs = eligible[task]
        # evenly spaced across the episode list, deterministic given the seed
        order = torch.randperm(len(segs), generator=rng).tolist()
        for j in order[:per_task]:
            picked.append({"task": task, "segment": segs[j],
                           "F_native": scan[task]["F"][segs[j]]})
            if len(picked) == n_items:
                return picked
    return picked[:n_items]


def load_items(manifest: list[dict]) -> list[dict]:
    """Open each selected segment once, grouped by task to avoid re-opening."""
    by_task: dict[str, list[dict]] = {}
    for m in manifest:
        by_task.setdefault(m["task"], []).append(m)

    roots = {Path(r).name: r for r in bootstrap.find_sub_datasets(CFG)}
    out: list[dict] = []
    for task, entries in by_task.items():
        if task not in roots:
            raise RuntimeError(f"task {task} not found on disk")
        ds = bootstrap.open_sub_dataset(roots[task], CFG)
        for e in entries:
            item = ds[e["segment"]]
            out.append({**e, "item": item})
    # restore the manifest order so slot assignment is reproducible
    order = {(m["task"], m["segment"]): i for i, m in enumerate(manifest)}
    out.sort(key=lambda r: order[(r["task"], r["segment"])])
    return out


def save_trajectories(traj_store, sigmas, records, args, out_dir) -> None:
    """Reorganise recorded x0 predictions into (T, K, ...) per item.

    `teacher_ode` yields, per draw, a list over ODE steps of (B, ...) tensors.
    M8 needs the dispersion across draws at a fixed sigma, so the k and step axes
    have to be transposed and split by item. Kept in fp16: only the relative
    dispersion per step is used, and its smallest value is four orders of
    magnitude above fp16 resolution.
    """
    tdir = out_dir / "traj"
    tdir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(records):
        blob = {"sigmas_video": sigmas["video"], "sigmas_action": sigmas["action"],
                "task": r["task"], "segment": r["segment"],
                "F": args.frames, "K": args.k, "cfg": args.cfg}
        for key in ("x0_video", "x0_action"):
            # traj_store[key][k] is a list over steps of (B, ...) tensors
            per_step = [
                torch.stack([traj_store[key][k][t][i] for k in range(args.k)])
                for t in range(len(traj_store[key][0]))
            ]
            blob[key] = torch.stack(per_step)  # (T, K, ...)
        torch.save(blob, tdir / f"{r['task']}__seg{r['segment']:04d}.pt")


@torch.no_grad()
def run_batch(sampler, records, args, dev, out_dir) -> dict:
    """K draws for one micro-batch of items, each item pinned to its own slot."""
    meta = [{"task": r["task"], "segment": r["segment"]} for r in records]
    cond = SP.stack_items(
        [r["item"] for r in records], device=dev,
        patch_size=tuple(CFG.patch_size), n_frames=args.frames, meta=meta,
    )
    b = cond.k
    keys = ("video", "action") if args.subject == "teacher" else (
        "video", "action", "video_x0")
    store = {i: {k: [] for k in keys} for i in range(b)}

    # Samples are kept in fp32. fp16 would add a quantization variance of about
    # (ULP^2)/12 ~ 1e-7, only ~30x below the v_intra a Dirac-like branch would
    # show, and it is exactly the small-variance regime the experiment has to
    # resolve. fp32 costs 6 MB per item, so there is nothing to trade off.
    eps0 = None
    # one untimed forward so create_block_mask / flex_attention compilation (~35s)
    # does not land inside the throughput number
    g_warm = torch.Generator(device=dev).manual_seed(0)
    vw, aw = SP.initial_noise(cond, g_warm)
    sampler.velocity(cond, vw, 1.0, aw, 1.0)
    torch.cuda.synchronize()

    traj_store: dict[str, list] = {"x0_video": [], "x0_action": []}
    sigmas: dict[str, list] = {}

    t_start = time.time()
    for k in range(args.k):
        g = torch.Generator(device=dev).manual_seed(args.seed * 100003 + k)
        v0, a0 = SP.initial_noise(cond, g)
        if k == 0:
            eps0 = (v0.clone(), a0.clone())
        if args.subject == "teacher":
            out = sampler.teacher_ode(cond, v0, a0, record_x0=args.record_x0)
            if args.record_x0:
                tr = out.pop("trajectory")
                traj_store["x0_video"].append(tr["x0_video"])
                traj_store["x0_action"].append(tr["x0_action"])
                sigmas = {"video": tr["sigmas_video"], "action": tr["sigmas_action"]}
        else:
            out = sampler.student_one_step(cond, v0, a0)
        for i in range(b):
            for key in keys:
                store[i][key].append(out[key][i].float().cpu())
        done = k + 1
        el = time.time() - t_start
        print(f"    k={done}/{args.k}  {el:.0f}s elapsed, "
              f"{el / done:.1f}s per draw", flush=True)

    # V3 for this orchestration: replay k=0's epsilon; a fixed slot must reproduce
    # bitwise, so the layout-induced floor is zero by construction rather than by
    # assumption.
    floor = None
    if args.verify_floor:
        v0, a0 = eps0
        if args.subject == "teacher":
            rep = sampler.teacher_ode(cond, v0, a0)
        else:
            rep = sampler.student_one_step(cond, v0, a0)
        dv, da = [], []
        for i in range(b):
            ref_v = store[i]["video"][0].to(dev)
            ref_a = store[i]["action"][0].to(dev)
            dv.append((rep["video"][i] - ref_v).abs().max().item())
            da.append((rep["action"][i] - ref_a).abs().max().item())
        floor = {"video_max_dev": max(dv), "action_max_dev": max(da),
                 "bitwise": max(max(dv), max(da)) == 0.0}
        print(f"    V3 replay: video dev {max(dv):.3e}, action dev {max(da):.3e}, "
              f"bitwise={floor['bitwise']}")

    for i, r in enumerate(records):
        path = out_dir / f"{r['task']}__seg{r['segment']:04d}.pt"
        blob = {k: torch.stack(store[i][k]) for k in keys}  # (K, ...) fp32
        blob.update({
            "truth_video": cond.clean_latents[i].cpu(),
            "truth_action": cond.clean_actions[i].cpu(),
            "actions_mask": cond.actions_mask[i].cpu(),
            "task": r["task"], "segment": r["segment"],
            "F": args.frames, "F_native": r["F_native"],
            "K": args.k, "cfg": args.cfg, "subject": args.subject,
        })
        torch.save(blob, path)

    if args.record_x0 and args.subject == "teacher":
        save_trajectories(traj_store, sigmas, records, args, out_dir)
        print(f"    wrote {len(records)} trajectories "
              f"({len(sigmas['video'])} video steps, {len(sigmas['action'])} action)")
    return {"floor": floor, "sec": time.time() - t_start, "n_items": b}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", choices=["teacher", "student"], default="teacher")
    ap.add_argument("--items", type=int, default=20)
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--batch", type=int, default=10,
                    help="items per micro-batch; 0 = all at once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-floor", action="store_true", default=True)
    ap.add_argument("--no-verify-floor", dest="verify_floor", action="store_false")
    ap.add_argument("--record-x0", action="store_true",
                    help="M8/H6: keep the predicted x0 at every ODE step "
                         "(~25x the storage, so use few items)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda"

    tag = f"{args.subject}_cfg{args.cfg:g}_F{args.frames}_K{args.k}"
    if args.record_x0:
        tag += "_traj"
    out_dir = Path(args.out) if args.out else Path(paths.EXP01_DIR) / "samples" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"subject={args.subject}  items={args.items} over {args.tasks} tasks  "
          f"K={args.k}  CFG={args.cfg}  F={args.frames}")
    manifest = select_items(args.items, args.tasks, args.frames, args.seed)
    print(f"selected {len(manifest)} items from "
          f"{len({m['task'] for m in manifest})} tasks")

    spec = SP.SamplerSpec(cfg_scale=args.cfg)
    n_fwd_per_draw = spec.n_forwards if args.subject == "teacher" else 1
    batch = args.batch if args.batch > 0 else len(manifest)
    n_batches = -(-len(manifest) // batch)
    total_fwd = n_batches * (args.k + (1 if args.verify_floor else 0)) * n_fwd_per_draw
    print(f"micro-batch {batch} -> {n_batches} batches, "
          f"{n_fwd_per_draw} forwards per draw, {total_fwd} forwards total")

    print("\nloading items ...")
    records = load_items(manifest)
    print(f"  {len(records)} items loaded")

    print("loading model ...")
    load = models.load_teacher if args.subject == "teacher" else models.load_student
    model = load(device=dev, dtype=torch.bfloat16, attn_mode="flex")
    empty = bootstrap.load_empty_emb(CFG, device=dev)
    sampler = SP.Sampler(model, spec, patch_size=tuple(CFG.patch_size),
                         chunk_size=CFG.frame_chunk_size,
                         window_size=CFG.attn_window, empty_emb=empty)

    t0 = time.time()
    floors, done, sec_sampling = [], 0, 0.0
    for bi in range(n_batches):
        chunk = records[bi * batch:(bi + 1) * batch]
        print(f"\n[batch {bi + 1}/{n_batches}] {len(chunk)} items: "
              + ", ".join(f"{r['task'].split('-')[0]}#{r['segment']}" for r in chunk))
        res = run_batch(sampler, chunk, args, dev, out_dir)
        if res["floor"]:
            floors.append(res["floor"])
        done += res["n_items"]
        sec_sampling += res["sec"]
        el = time.time() - t0
        print(f"  batch done in {res['sec']:.0f}s | total {el / 60:.1f} min | "
              f"{done}/{len(records)} items | peak mem "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    elapsed = time.time() - t0
    run_meta = {
        "subject": args.subject, "items": len(records), "tasks": args.tasks,
        "K": args.k, "cfg": args.cfg, "frames": args.frames, "batch": batch,
        "seed": args.seed, "manifest": manifest,
        "forwards_total": total_fwd, "elapsed_sec": elapsed,
        # excludes the per-batch compile warm-up, which is a fixed ~35s cost
        "sampling_sec": sec_sampling,
        "sec_per_forward": sec_sampling / max(total_fwd, 1),
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "floor_checks": floors,
        "floor_bitwise_all": all(f["bitwise"] for f in floors) if floors else None,
        "n_chunks_per_item": S.n_chunks(args.frames),
        "conditions_total": len(records) * S.n_chunks(args.frames),
    }
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=1))
    print(f"\n{'=' * 74}")
    print(f"wrote {done} items to {out_dir}")
    print(f"elapsed {elapsed / 60:.1f} min ({sec_sampling / 60:.1f} min sampling), "
          f"{sec_sampling / max(total_fwd, 1) * 1000:.0f} ms per forward at B={batch}")
    if floors:
        print(f"V3 replay bitwise identical in every batch: "
              f"{run_meta['floor_bitwise_all']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
