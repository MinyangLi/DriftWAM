"""Check the decode against the original video, not against my expectations.

The dataset keeps the source mp4s alongside the pre-encoded latents, and each
latent file records the exact original frame indices it covers. So the whole chain
-- T-shape crop, per-channel denormalisation, camera-to-crop assignment, causal
temporal mapping -- can be checked end to end by decoding the ground-truth latent
and comparing it to the frames it was made from. A wrong crop or a skipped
denormalisation cannot survive this; it is a much stronger check than asking
whether the pixel histogram looks reasonable.

  python scripts/validate_decode.py --task adjust_bottle --segment 39
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

from driftwam import decode as D  # noqa: E402
from driftwam import paths  # noqa: E402
from driftwam import shapes as S  # noqa: E402


def read_frames(mp4: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    """Decode sequentially and keep the wanted indices; episodes are short."""
    import av  # noqa: PLC0415

    want = set(wanted)
    out: dict[int, np.ndarray] = {}
    with av.open(str(mp4)) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = frame.to_ndarray(format="rgb24")
            if len(out) == len(want):
                break
    return out


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Both in [0, 1]."""
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="adjust_bottle")
    ap.add_argument("--segment", type=int, default=39)
    ap.add_argument("--out", default=str(Path(paths.EXP01_DIR) / "decode_check"))
    args = ap.parse_args()
    dev = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = next(s for s in bootstrap.find_sub_datasets(CFG)
                if Path(s).name.startswith(args.task))
    ds = bootstrap.open_sub_dataset(root, CFG)
    meta = ds.new_metas[args.segment]
    ep, f0, f1 = meta["episode_index"], meta["start_frame"], meta["end_frame"]
    print(f"{Path(root).name}  segment {args.segment}  episode {ep}  "
          f"local frames {f0}..{f1}")

    chunk = ds.meta.get_episode_chunk(ep)
    latent_root = Path(ds.latent_path) / f"chunk-{chunk:03d}"
    video_root = Path(root) / "videos" / f"chunk-{chunk:03d}"

    item = ds[args.segment]
    truth = item["latents"]
    print(f"composite latent {tuple(truth.shape)}")

    vae = D.load_wan_vae(paths.vae_dir(paths.TEACHER_PATH), device=dev,
                         dtype=torch.bfloat16)

    results = {}
    panels = []
    for cam in S.CAMERAS:
        key = f"observation.images.{cam.name}"
        lat = torch.load(latent_root / key /
                         f"episode_{ep:06d}_{f0}_{f1}.pth", weights_only=False)
        frame_ids = [int(x) for x in lat["frame_ids"]]
        stride = frame_ids[1] - frame_ids[0] if len(frame_ids) > 1 else 1

        z = cam.crop(truth)[None].to(dev)
        pix = D.decode(vae, z, batch=1)[0]           # (3, T, H, W) in [-1, 1]
        got = (pix.permute(1, 0, 2, 3).clamp(-1, 1) + 1) / 2   # (T, 3, H, W)
        n = got.shape[0]

        want = frame_ids[:n]
        raw = read_frames(video_root / key / f"episode_{ep:06d}.mp4", want)
        missing = [w for w in want if w not in raw]
        ref = torch.stack([torch.from_numpy(raw[w]) for w in want if w in raw])
        ref = ref.permute(0, 3, 1, 2).float().to(dev) / 255.0

        note = ""
        if ref.shape[-2:] != got.shape[-2:]:
            note = f"  (mp4 {tuple(ref.shape[-2:])} -> resized)"
            ref = torch.nn.functional.interpolate(
                ref, size=got.shape[-2:], mode="bilinear", antialias=True,
                align_corners=False)

        m = min(ref.shape[0], got.shape[0])
        p_all = psnr(got[:m].cpu(), ref[:m].cpu())
        # Per-frame, because a temporal off-by-one shows up as one bad frame
        # rather than a uniformly mediocre average.
        per = [psnr(got[i].cpu(), ref[i].cpu()) for i in range(m)]
        # A shifted-by-one comparison is the control: if the alignment were
        # wrong, the shifted PSNR would beat the aligned one.
        p_shift = psnr(got[:m - 1].cpu(), ref[1:m].cpu()) if m > 1 else float("nan")

        print(f"\n{cam.name}: latent {tuple(cam.crop(truth).shape)} -> "
              f"pixels {tuple(pix.shape)}{note}")
        print(f"  original frames {want[0]}..{want[-1]} stride {stride}, "
              f"{len(want)} of them{', MISSING ' + str(missing) if missing else ''}")
        print(f"  PSNR aligned {p_all:.2f} dB   shifted-by-1 {p_shift:.2f} dB   "
              f"per-frame min {min(per):.2f} max {max(per):.2f}")
        results[cam.name] = {
            "psnr": p_all, "psnr_shifted": p_shift,
            "psnr_per_frame": per, "frame_ids": want, "stride": stride,
            "decoded_shape": list(pix.shape),
        }
        panels.append((cam.name, got[0].cpu(), ref[0].cpu(),
                       got[m // 2].cpu(), ref[m // 2].cpu()))

    try:
        from PIL import Image  # noqa: PLC0415

        for name, g0, r0, gm, rm in panels:
            rows = []
            for g, r in ((g0, r0), (gm, rm)):
                pair = torch.cat([g, r], dim=2)
                rows.append((pair.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img = np.concatenate(rows, axis=0)
            Image.fromarray(img).save(out_dir / f"{name}.png")
        print(f"\nwrote side-by-side images (decoded | original) to {out_dir}")
    except ImportError:
        print("\nPIL missing, skipped image dump")

    ok = all(v["psnr"] > 20.0 and v["psnr"] > v["psnr_shifted"]
             for v in results.values())
    (out_dir / "psnr.json").write_text(json.dumps(results, indent=1))
    print("DECODE VALID" if ok else "DECODE SUSPECT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
