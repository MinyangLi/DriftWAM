"""Re-measure the teacher's conditional dispersion in DINOv3 feature space.

Latent space put the video branch at R = 0.028 (conditional std 16.9% of
marginal), which is only marginally usable for an explicit drifting loss, and the
CFG sweep showed CFG is a bias knob rather than a variance knob. The one remaining
lever is the space the loss is measured in: two frames that differ by little in VAE
latents may differ by a lot in a semantic space, in which case a drifting loss
placed there sees more to work with. This script produces the same M1-M6 statistics
as `compute_stats.py`, on DINOv3 features, from the same saved samples, so the two
numbers are directly comparable.

Each condition contributes one row per DINOv3 block (2, 5, 8 -- DriftWorld's
choice) and per view. Features are reduced to one vector per *latent* frame by
averaging the four pixel frames that frame expands to, which keeps D aligned with
the latent-space measurement's temporal resolution and keeps it equal across
chunks.

  python scripts/extract_dino.py --items 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

from driftwam import decode as D  # noqa: E402
from driftwam import features as FT  # noqa: E402
from driftwam import metrics as M  # noqa: E402
from driftwam import paths  # noqa: E402
from driftwam import shapes as S  # noqa: E402

CAM_ORDER = [c.name for c in S.CAMERAS]


def pool_to_latent_frames(feats: torch.Tensor, groups: list[int]) -> torch.Tensor:
    """(t, gh, gw, C) pixel-frame features -> (n_latent, gh, gw, C).

    `groups` gives how many consecutive pixel frames each latent frame owns (1 for
    latent frame 0, 4 otherwise).

    Averaging matches the latent-space measurement's temporal resolution, which
    makes the two R values comparable dimension-for-dimension. It also suppresses
    whatever the draws disagree about *within* a latent frame -- differences in
    motion timing, for instance -- so `--pool none` exists to check that the
    reduction is not what limits the measured dispersion. That unpooled form is
    also closer to a per-frame feature loss.
    """
    out, i = [], 0
    for g in groups:
        out.append(feats[i:i + g].mean(dim=0))
        i += g
    if i != feats.shape[0]:
        raise ValueError(f"groups cover {i} frames, got {feats.shape[0]}")
    return torch.stack(out)


def chunk_groups(chunk: int, n_latent: int) -> list[int]:
    return [len(D.pixel_frames_of_latent_frame(f))
            for f in S.video_frames_of_chunk(chunk, n_latent)]


@torch.no_grad()
def features_for_item(vae, dino, rec: dict, chunks: list[int], blocks: tuple[int, ...],
                      batch: int, dev: str, pool_frames: bool = True,
                      field: str = "video") -> dict:
    """block -> camera -> chunk -> (K+1, n_latent, gh, gw, C), truth at index K."""
    sample = rec[field].float().to(dev)
    truth = rec["truth_video"].float().to(dev)
    k, n_lat = sample.shape[0], sample.shape[2]

    store: dict = {b: {cam: {} for cam in CAM_ORDER} for b in blocks}
    for cam in S.CAMERAS:
        grid = FT.expected_patch_grid(cam)
        if grid != FT.latent_grid_of(cam):
            raise RuntimeError(f"{cam.name}: patch grid {grid} != latent grid "
                               f"{FT.latent_grid_of(cam)}")
        for b in blocks:
            for c in chunks:
                store[b][cam.name][c] = [None] * (k + 1)

        for c, draw, pix in D.decode_swap_batch(vae, sample, truth, cam, chunks,
                                               batch=batch):
            frames = pix.permute(1, 0, 2, 3)          # (t, 3, H, W)
            feats = FT.patch_features(dino, frames, blocks=blocks, batch=32,
                                      grid=grid)
            groups = chunk_groups(c, n_lat)
            slot = k if draw < 0 else draw
            for b in blocks:
                store[b][cam.name][c][slot] = (
                    pool_to_latent_frames(feats[b], groups) if pool_frames
                    else feats[b]
                )

    for b in blocks:
        for cam in CAM_ORDER:
            for c in chunks:
                rows = store[b][cam][c]
                if any(r is None for r in rows):
                    raise RuntimeError(f"missing rows for block {b} {cam} chunk {c}")
                store[b][cam][c] = torch.stack(rows)
    return store


def condition_rows(store: dict, chunks: list[int], blocks: tuple[int, ...],
                   k: int) -> tuple[dict, dict]:
    """Per-view ConditionStats, plus per-channel within-variance for the check."""
    stats: dict = {}
    chan: dict = {}
    for b in blocks:
        for view in CAM_ORDER + ["combined"]:
            for c in chunks:
                # (K+1, positions, C): positions are (latent frame, patch), and the
                # combined view concatenates the cameras in a fixed order so that
                # it is the analogue of the composite latent tensor.
                cams = CAM_ORDER if view == "combined" else [view]
                ch = torch.cat(
                    [store[b][cam][c].reshape(k + 1, -1, store[b][cam][c].shape[-1])
                     for cam in cams], dim=1
                )
                x = ch.flatten(1)
                stats.setdefault(view, {}).setdefault(b, {})[c] = M.condition_stats(
                    x[:k], truth=x[k]
                )
                # (K, P, C) -> per-channel variance across draws, averaged over
                # positions. Lets the pooled R be recomputed after dividing each
                # channel by its global scale, which is the check that R is not
                # being set by a handful of high-norm DINO channels.
                s = ch[:k].to(torch.float64)
                chan.setdefault(view, {}).setdefault(b, {})[c] = {
                    "within_var": s.var(dim=0, unbiased=True).mean(dim=0).cpu(),
                    "sum": s.sum(dim=(0, 1)).cpu(),
                    "sumsq": s.pow(2).sum(dim=(0, 1)).cpu(),
                    "n": s.shape[0] * s.shape[1],
                }
    # Condition means are what M2 needs later; nothing else is worth the disk.
    for view in stats:
        for b in stats[view]:
            for c in stats[view][b]:
                st = stats[view][b][c]
                st.mean = st.mean.cpu()
    return stats, chan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=None, help="sample dir to re-measure")
    ap.add_argument("--items", type=int, default=0, help="0 = all")
    ap.add_argument("--blocks", default="2,5,8")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--field", default="video",
                    help="which stored prediction to decode. For the student use "
                         "video_x0: its `video` is the consistency output "
                         "c_skip * x + c_out * f, whose spread is mostly the "
                         "analytic skip term and which is underscaled by ~2x, so "
                         "decoding it measures injected noise rather than what the "
                         "student learned")
    ap.add_argument("--pool", choices=("latent", "none"), default="latent",
                    help="average the pixel frames of each latent frame, or keep "
                         "every pixel frame as its own dimension")
    ap.add_argument("--skip-chunk0", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    blocks = tuple(int(x) for x in args.blocks.split(","))

    sdir = Path(args.samples) if args.samples else (
        Path(paths.EXP01_DIR) / "samples" / "teacher_cfg5_F8_K8")
    files = sorted(sdir.glob("*.pt"))
    if args.items:
        files = files[:args.items]
    out_dir = Path(args.out) if args.out else (
        Path(paths.EXP01_DIR) / "dino" / sdir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} items from {sdir}\nblocks {blocks} -> {out_dir}")

    vae = D.load_wan_vae(paths.vae_dir(paths.TEACHER_PATH), device=dev,
                         dtype=torch.bfloat16)
    dino = FT.load_dino(paths.DINO_PATH, device=dev, dtype=torch.float32)

    t_start = time.time()
    done = 0
    for i, f in enumerate(files):
        rec = torch.load(f, map_location="cpu")
        tag = f"{rec['task'].split('-')[0]}#{rec['segment']}"
        dest = out_dir / f"{f.stem}.pt"
        if dest.exists() and not args.overwrite:
            print(f"[{i + 1}/{len(files)}] {tag}: exists, skipped")
            continue

        n_lat, k = rec["video"].shape[2], rec["video"].shape[0]
        chunks = [c for c in range(S.n_chunks(n_lat))
                  if len(S.video_frames_of_chunk(c, n_lat)) == S.FRAME_CHUNK_SIZE]
        if args.skip_chunk0:
            chunks = [c for c in chunks if c != 0]

        t0 = time.time()
        store = features_for_item(vae, dino, rec, chunks, blocks, args.batch, dev,
                                  pool_frames=args.pool == "latent",
                                  field=args.field)
        stats, chan = condition_rows(store, chunks, blocks, k)
        del store
        torch.cuda.empty_cache()

        torch.save({
            "task": rec["task"], "segment": rec["segment"], "F": rec["F"],
            "K": k, "cfg": rec.get("cfg"), "subject": rec.get("subject"),
            "blocks": list(blocks), "chunks": chunks, "views": CAM_ORDER + ["combined"],
            "pool": args.pool, "field": args.field, "stats": stats, "chan": chan,
        }, dest)
        done += 1
        el = time.time() - t_start
        print(f"[{i + 1}/{len(files)}] {tag}: {time.time() - t0:.1f}s "
              f"(elapsed {el / 60:.1f} min, eta "
              f"{el / done * (len(files) - i - 1) / 60:.1f} min)", flush=True)

    print(f"\nwrote {done} items to {out_dir} in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
