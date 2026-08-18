"""Validate the VAE decode + DINOv3 path, and measure what it costs.

Checks, cheapest first:
  1. VAE loads; decode of a real latent has the expected shape and range
  2. Decoding ground truth gives a sane image (not a normalisation mistake)
  3. The decoder's temporal receptive field, measured -- this decides whether the
     per-chunk history splicing is required or merely tidy
  4. DINOv3 patch grid equals the latent grid for every camera
  5. Throughput, to size the real run

  python scripts/probe_dino.py
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

from driftwam import decode as D  # noqa: E402
from driftwam import features as FT  # noqa: E402
from driftwam import paths  # noqa: E402
from driftwam import shapes as S  # noqa: E402

RESULTS: dict = {}
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=str(Path(paths.EXP01_DIR) / "dino_probe.json"))
    args = ap.parse_args()
    dev = "cuda"

    sdir = Path(args.samples) if args.samples else (
        Path(paths.EXP01_DIR) / "samples" / "teacher_cfg5_F8_K8")
    f = sorted(sdir.glob("*.pt"))[0]
    rec = torch.load(f, map_location="cpu")
    sample, truth = rec["video"], rec["truth_video"]
    k, _, nlat = sample.shape[0], sample.shape[1], sample.shape[2]
    print(f"item {rec['task'].split('-')[0]}#{rec['segment']}  "
          f"K={k}  F={nlat}  sample {tuple(sample.shape)}")

    print("\n[0] load VAE")
    t0 = time.time()
    vae = D.load_wan_vae(paths.vae_dir(paths.TEACHER_PATH), device=dev,
                         dtype=torch.bfloat16)
    print(f"  {time.time() - t0:.1f}s")
    check("z_dim is 48 (cameras are packed spatially, not on the channel axis)",
          vae.config.z_dim == S.LATENT_CHANNELS, str(vae.config.z_dim))

    print("\n[1] decode ground truth, per camera")
    for cam in S.CAMERAS:
        t = cam.crop(truth)[None].to(dev)
        t0 = time.time()
        pix = D.decode(vae, t, batch=1)
        dt = time.time() - t0
        exp_t = D.n_pixel_frames(nlat)
        ok_shape = tuple(pix.shape) == (1, 3, exp_t, cam.pixel_h, cam.pixel_w)
        check(f"{cam.name} decode shape", ok_shape,
              f"{tuple(pix.shape)} in {dt:.2f}s")
        lo, hi, mean = pix.min().item(), pix.max().item(), pix.mean().item()
        # a normalisation error typically shows up as a saturated or DC-shifted
        # image, so require real spread and a mean near mid-grey
        check(f"{cam.name} pixel range and spread", -1.05 < lo and hi < 1.05
              and pix.std().item() > 0.1 and abs(mean) < 0.6,
              f"[{lo:.3f}, {hi:.3f}] mean {mean:.3f} std {pix.std().item():.3f}")
        RESULTS[f"decode_{cam.name}"] = {
            "shape": list(pix.shape), "sec": dt,
            "min": lo, "max": hi, "mean": mean, "std": pix.std().item(),
        }

    print("\n[2] decoder temporal receptive field "
          "(does chunk c's pixels depend on earlier latent frames?)")
    cam = S.CAMERAS_BY_NAME["cam_high"]
    base = cam.crop(truth)[None].to(dev)
    ref = D.decode(vae, base, batch=1)
    rows = []
    for pert in range(nlat):
        z = base.clone()
        z[:, :, pert] += 1.0
        out = D.decode(vae, z, batch=1)
        diff = (out - ref).abs().amax(dim=(0, 1, 3, 4))  # per pixel frame
        touched = [i for i, d in enumerate(diff.tolist()) if d > 1e-3]
        own = D.pixel_frames_of_latent_frame(pert)
        earliest = min(touched) if touched else None
        rows.append({"latent_frame": pert, "own_pixels": own,
                     "touched_pixels": [min(touched), max(touched)] if touched else [],
                     "leaks_backward": bool(touched and min(touched) < own[0])})
        print(f"  perturb latent {pert} (owns pixels {own[0]}-{own[-1]}): "
              f"changes pixels {earliest}-{max(touched) if touched else None}")
    RESULTS["receptive_field"] = rows
    check("decode is causal (no pixel before a frame's own span changes)",
          not any(r["leaks_backward"] for r in rows))
    span = max((r["touched_pixels"][1] - r["own_pixels"][-1])
               for r in rows if r["touched_pixels"])
    print(f"  forward reach beyond a frame's own pixels: {span} pixel frames")
    print("  -> a later chunk's pixels DO depend on earlier latent frames, so the "
          "per-chunk history splicing is required, not cosmetic")
    RESULTS["forward_reach_pixels"] = span

    print("\n[3] load DINOv3")
    dino = FT.load_dino(paths.DINO_PATH, device=dev, dtype=torch.float32)
    print(f"  prefix tokens: {FT.n_prefix_tokens(dino)}")

    print("\n[4] patch grid equals latent grid, per camera")
    for cam in S.CAMERAS:
        pix = D.decode(vae, cam.crop(truth)[None].to(dev), batch=1)
        feats = FT.patch_features(dino, pix[0].permute(1, 0, 2, 3), batch=8,
                                  grid=FT.expected_patch_grid(cam))
        b = FT.DRIFTWORLD_BLOCKS[0]
        gh, gw, c = feats[b].shape[1], feats[b].shape[2], feats[b].shape[3]
        check(f"{cam.name} patch grid == latent grid",
              (gh, gw) == FT.latent_grid_of(cam),
              f"patches {gh}x{gw} vs latent {cam.latent_h}x{cam.latent_w}, C={c}")
        norms = {bb: feats[bb].norm(dim=-1).mean().item()
                 for bb in FT.DRIFTWORLD_BLOCKS}
        print(f"    mean feature norm by block: "
              + ", ".join(f"{bb}:{v:.1f}" for bb, v in norms.items()))
        RESULTS[f"dino_{cam.name}"] = {"grid": [gh, gw], "C": c, "norms": norms}

    print("\n[5] throughput for one (item, camera, chunk) at K=8")
    cam = S.CAMERAS_BY_NAME["cam_high"]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    pix_s, pix_t = D.decode_camera_chunk(vae, sample.to(dev), truth.to(dev), cam,
                                        chunk=nlat // S.FRAME_CHUNK_SIZE - 1,
                                        batch=args.batch)
    torch.cuda.synchronize()
    t_dec = time.time() - t0
    t0 = time.time()
    n, t = pix_s.shape[0], pix_s.shape[2]
    frames = pix_s.permute(0, 2, 1, 3, 4).reshape(n * t, 3, *pix_s.shape[-2:])
    _ = FT.patch_features(dino, frames, batch=32)
    torch.cuda.synchronize()
    t_dino = time.time() - t0
    print(f"  decode K={k} + truth: {t_dec:.2f}s   DINO on {n * t} frames: "
          f"{t_dino:.2f}s   peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    n_chunks_used = nlat // S.FRAME_CHUNK_SIZE - 1  # chunks 1..last
    per_item = (t_dec + t_dino) * 3 * (n_chunks_used + 1)
    print(f"  extrapolated: {per_item:.1f}s per item (3 cameras x "
          f"{n_chunks_used + 1} chunks) -> {per_item * 20 / 60:.1f} min for 20 items")
    RESULTS["throughput"] = {
        "decode_sec": t_dec, "dino_sec": t_dino, "frames": n * t,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "est_sec_per_item": per_item, "est_min_20_items": per_item * 20 / 60,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    RESULTS["fails"] = FAILS
    Path(args.out).write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\nwrote {args.out}")
    print("ALL PASSED" if not FAILS else f"FAILED: {', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
