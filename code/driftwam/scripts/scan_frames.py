"""Scan every sub-dataset for its per-segment F, and pin down the shape assumptions.

Writes two files:

  f_scan.json   task -> {n_seg, F: [...], F_dist, F_min, F_max}, used by
                sample_variance.py to pick items without opening any dataset
  shapes.json   the layout facts every other script relies on, each one checked
                against real data rather than assumed

The checks matter because two of these facts were wrong in the first draft of the
plan: F was assumed to be 16-20 when it is often 5, and action latent frame 0 was
assumed to be data when it is constant padding that `actions_mask` marks valid.

  python scripts/scan_frames.py [--full] [--items-per-task 1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

from driftwam import paths  # noqa: E402
from driftwam import shapes as S  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


def check_item(item: dict, task: str) -> dict:
    """Assert the layout of one real item and return what was observed."""
    lat, act = item["latents"], item["actions"]
    msk, txt = item["actions_mask"], item["text_emb"]
    f = lat.shape[1]
    obs = {
        "task": task,
        "F": f,
        "latents": list(lat.shape), "latents_dtype": str(lat.dtype),
        "actions": list(act.shape), "actions_dtype": str(act.dtype),
        "actions_mask": list(msk.shape), "text_emb": list(txt.shape),
    }

    check(f"[{task}] latents (48, F, 24, 20)",
          tuple(lat.shape) == (S.LATENT_CHANNELS, f, S.GRID_H, S.GRID_W),
          str(tuple(lat.shape)))
    check(f"[{task}] actions (30, F, 16, 1)",
          tuple(act.shape) == (S.ACTION_DIM, f, S.ACTION_PER_FRAME, 1),
          str(tuple(act.shape)))
    check(f"[{task}] text_emb (512, 4096)", tuple(txt.shape) == (512, 4096),
          str(tuple(txt.shape)))

    # the mask must select exactly the 16 channels shapes.py claims
    flagged = sorted(torch.nonzero(msk.any(dim=(1, 2, 3))).flatten().tolist())
    check(f"[{task}] mask selects VALID_ACTION_CHANNELS",
          flagged == S.VALID_ACTION_CHANNELS,
          f"{len(flagged)} channels" if flagged == S.VALID_ACTION_CHANNELS
          else f"got {flagged}")
    obs["flagged_channels"] = flagged

    # action frame 0 is constant padding, and is flagged valid
    a0 = act[S.VALID_ACTION_CHANNELS, 0].to(torch.float64)
    std0 = a0.reshape(len(S.VALID_ACTION_CHANNELS), -1).std(dim=-1)
    check(f"[{task}] action frame 0 is constant per channel",
          bool((std0 == 0).all()), f"max std {std0.max().item():.3e}")
    check(f"[{task}] frame 0 is nonetheless flagged valid",
          bool(msk[S.VALID_ACTION_CHANNELS, 0].all()))
    obs["frame0_values"] = [round(v, 6) for v in a0[:, 0, 0].tolist()]

    later = act[S.VALID_ACTION_CHANNELS][:, S.action_frame_slice(f)].to(torch.float64)
    std_later = later.reshape(len(S.VALID_ACTION_CHANNELS), -1).std(dim=-1)
    obs["active_channels"] = S.active_action_channels(act)
    obs["n_active_channels"] = len(obs["active_channels"])
    print(f"        frames>=1 per-channel std: min {std_later.min().item():.3e} "
          f"max {std_later.max().item():.3e}; "
          f"{len(obs['active_channels'])}/16 channels actually move")

    # invalid channels must be exactly zero, or masking is not doing its job
    invalid = [c for c in range(S.ACTION_DIM) if c not in S.VALID_ACTION_CHANNELS]
    check(f"[{task}] invalid channels are exactly zero",
          bool((act[invalid] == 0).all()))

    # camera tiling covers the grid with no overlap and no gap
    cover = torch.zeros(S.GRID_H, S.GRID_W, dtype=torch.int32)
    for cam in S.CAMERAS:
        cover[cam.rows, cam.cols] += 1
        check(f"[{task}] {cam.name} pixels/16 == latent grid",
              cam.pixel_h // S.DINO_PATCH == cam.latent_h
              and cam.pixel_w // S.DINO_PATCH == cam.latent_w,
              f"{cam.pixel_h}x{cam.pixel_w} -> {cam.latent_h}x{cam.latent_w}")
    check(f"[{task}] cameras tile 24x20 exactly once", bool((cover == 1).all()))
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-per-task", type=int, default=1,
                    help="items to shape-check per task")
    ap.add_argument("--check-tasks", type=int, default=3,
                    help="how many tasks to shape-check (scan of F covers all)")
    ap.add_argument("--reuse-scan", action="store_true",
                    help="reuse an existing f_scan.json instead of re-reading "
                         "every segment")
    args = ap.parse_args()

    out_dir = Path(paths.EXP01_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    repos = bootstrap.find_sub_datasets(CFG)
    print(f"found {len(repos)} sub-datasets\n")

    print("=" * 74)
    print("shape assumptions, checked against real items")
    print("=" * 74)
    observed = []
    for repo in repos[:args.check_tasks]:
        ds = bootstrap.open_sub_dataset(repo, CFG)
        for i in range(min(args.items_per_task, len(ds))):
            observed.append(check_item(ds[i], Path(repo).name))

    print("\n" + "=" * 74)
    print("F scan over every segment")
    print("=" * 74)
    scan: dict[str, dict] = {}
    t0 = time.time()
    total_seg = 0
    cached = out_dir / "f_scan.json"
    if args.reuse_scan and cached.exists():
        scan = json.loads(cached.read_text())
        total_seg = sum(v["n_seg"] for v in scan.values())
        print(f"reused {cached} ({len(scan)} tasks, {total_seg} segments)")
    else:
        for n, repo in enumerate(repos, 1):
            name = Path(repo).name
            ds = bootstrap.open_sub_dataset(repo, CFG)
            fs = [ds[i]["latents"].shape[1] for i in range(len(ds))]
            scan[name] = {
                "n_seg": len(fs), "F": fs,
                "F_dist": {str(k): v for k, v in sorted(Counter(fs).items())},
                "F_min": min(fs), "F_max": max(fs),
            }
            total_seg += len(fs)
            print(f"[{n:>2}/{len(repos)}] {name:<52} n={len(fs):>3} "
                  f"F {min(fs)}-{max(fs)}", flush=True)

    all_f = [f for v in scan.values() for f in v["F"]]
    dist = dict(sorted(Counter(all_f).items()))
    print(f"\nloaded {len(scan)}/{len(repos)} sub-datasets, {total_seg} segments "
          f"in {time.time() - t0:.0f}s")
    print(f"global F dist: {dist}")
    for thr in (8, 16):
        n_ok = sum(1 for f in all_f if f >= thr)
        print(f"  F >= {thr:>2}: {n_ok} segments ({100 * n_ok / len(all_f):.1f}%), "
              f"{sum(1 for v in scan.values() if v['F_max'] >= thr)} tasks")
    print(f"total conditions at native F: {sum(S.n_chunks(f) for f in all_f)}")

    (out_dir / "f_scan.json").write_text(json.dumps(scan, indent=1))

    shapes_doc = {
        "latent_channels": S.LATENT_CHANNELS,
        "grid": [S.GRID_H, S.GRID_W],
        "action_dim": S.ACTION_DIM,
        "action_per_frame": S.ACTION_PER_FRAME,
        "frame_chunk_size": S.FRAME_CHUNK_SIZE,
        "vae_temporal_ratio": S.VAE_TEMPORAL_RATIO,
        "action_pad_frames": S.ACTION_PAD_FRAMES,
        "valid_action_channels": S.VALID_ACTION_CHANNELS,
        "action_groups": S.ACTION_GROUPS,
        "cameras": [
            {"name": c.name, "rows": [c.rows.start, c.rows.stop],
             "cols": [c.cols.start, c.cols.stop],
             "pixels": [c.pixel_h, c.pixel_w],
             "latent": [c.latent_h, c.latent_w]} for c in S.CAMERAS
        ],
        "sequence_length_F8": S.sequence_length(8),
        "sequence_length_F16": S.sequence_length(16),
        "F_distribution": {str(k): v for k, v in dist.items()},
        "n_segments": total_seg,
        "n_tasks": len(scan),
        "observed_items": observed,
        "checks_failed": FAILS,
    }
    (out_dir / "shapes.json").write_text(json.dumps(shapes_doc, indent=1))
    print(f"\nwrote {out_dir / 'f_scan.json'} and {out_dir / 'shapes.json'}")
    print("ALL SHAPE CHECKS PASSED" if not FAILS else f"FAILED: {', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
