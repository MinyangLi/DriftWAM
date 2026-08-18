"""Decode VAE latents back to pixels, one camera at a time.

Two things here are easy to get wrong and both would silently corrupt a variance
measurement.

**Cameras must be decoded separately.** The three views are packed *spatially*
into one 24x20 grid (the VAE's own z_dim is 48; it is not three 16-channel
encodings stacked on the channel axis). Decoding the composite grid would run the
decoder's spatial convolutions across the seam between cam_high and the wrists,
producing pixels that belong to no camera.

**The decode must be given the same history the model was given.** Sampling is
teacher-forced: chunk c was generated conditioned on the *true* latents of chunks
< c, so the eight generated frames are not a coherent rollout. The VAE decoder is
causal in time, so decoding all eight frames of a sample together makes chunk c's
pixels depend on the *generated* earlier chunks, which vary across draws and would
leak their dispersion into chunk c. `splice_history` rebuilds, per chunk, the
sequence the protocol actually implies: ground truth before the chunk, the sample
inside it.
"""

from __future__ import annotations

import torch

from . import shapes

Tensor = torch.Tensor

VAE_TEMPORAL_RATIO = shapes.VAE_TEMPORAL_RATIO


def load_wan_vae(vae_dir, device="cuda", dtype=torch.bfloat16, verbose: bool = True):
    """`AutoencoderKLWan` from the checkpoint's `vae/` folder."""
    from modules.utils import load_vae  # noqa: PLC0415  (needs bootstrap)

    vae = load_vae(str(vae_dir), torch_dtype=dtype, torch_device=device)
    vae.eval()
    vae.requires_grad_(False)
    if verbose:
        n = sum(p.numel() for p in vae.parameters())
        print(f"loaded VAE: {n / 1e6:.0f}M params, z_dim={vae.config.z_dim}, "
              f"spatial x{vae.config.scale_factor_spatial}, "
              f"temporal x{vae.config.scale_factor_temporal}, {dtype}")
    return vae


def denormalise(latents: Tensor, vae) -> Tensor:
    """Undo the dataset's per-channel latent normalisation.

    The encoder side stores `z = (mu - mean) / std`, so `decode` has to be handed
    `mu = z * std + mean`. Skipping this produces a plausible-looking but wrong
    image, which is exactly the kind of error a variance ratio would survive.
    """
    c = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device=latents.device,
                        dtype=torch.float32).view(1, c, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=latents.device,
                       dtype=torch.float32).view(1, c, 1, 1, 1)
    return latents.float() * std + mean


def pixel_frames_of_latent_frame(f: int) -> list[int]:
    """Causal 4x temporal compression: frame 0 stands alone, the rest cover 4."""
    if f == 0:
        return [0]
    r = VAE_TEMPORAL_RATIO
    return list(range(r * f - (r - 1), r * f + 1))


def n_pixel_frames(n_latent: int) -> int:
    return 1 + VAE_TEMPORAL_RATIO * (n_latent - 1)


def pixel_frames_of_chunk(chunk: int, n_latent: int,
                          chunk_size: int = shapes.FRAME_CHUNK_SIZE) -> list[int]:
    out: list[int] = []
    for f in shapes.video_frames_of_chunk(chunk, n_latent, chunk_size):
        out.extend(pixel_frames_of_latent_frame(f))
    return out


def splice_history(sample: Tensor, truth: Tensor, chunk: int,
                   chunk_size: int = shapes.FRAME_CHUNK_SIZE) -> Tensor:
    """Ground truth everywhere except `chunk`, which comes from the sample.

    `sample` is (K, C, F, h, w) and `truth` is (C, F, h, w); the result is
    (K, C, F, h, w). Frames after the chunk are left as ground truth rather than
    cropped away: the decode is causal (verified in `probe_dino.py`), so they
    cannot affect the chunk's pixels, and keeping the full length lets every chunk
    share one batch shape.
    """
    frames = shapes.video_frames_of_chunk(chunk, sample.shape[2], chunk_size)
    lo, hi = frames[0], frames[-1] + 1
    out = truth[None].expand(sample.shape[0], -1, -1, -1, -1).clone()
    out[:, :, lo:hi] = sample[:, :, lo:hi]
    return out


def build_swap_batch(sample: Tensor, truth: Tensor, chunks: list[int],
                     chunk_size: int = shapes.FRAME_CHUNK_SIZE
                     ) -> tuple[Tensor, list[tuple[int, int]]]:
    """One row per (chunk, draw), plus a final all-truth row.

    Each row is ground truth with a single chunk swapped in from one draw, so the
    dispersion visible in that chunk's pixels comes from that chunk alone. The
    all-truth row serves every chunk at once, again by causality.

    Returns the (n_rows, C, F, h, w) batch and the row labels, where the truth row
    is labelled `(-1, -1)`.
    """
    rows = [splice_history(sample, truth, c, chunk_size) for c in chunks]
    labels = [(c, k) for c in chunks for k in range(sample.shape[0])]
    batch = torch.cat(rows + [truth[None]])
    labels.append((-1, -1))
    return batch, labels


@torch.no_grad()
def decode(vae, latents: Tensor, batch: int = 4) -> Tensor:
    """(B, 48, F, h, w) normalised latents -> (B, 3, T, H, W) pixels in [-1, 1].

    Chunked over the batch because the decoder's activations at full resolution
    dominate memory, and `use_slicing` would serialise to batch 1.
    """
    outs = []
    for i in range(0, latents.shape[0], batch):
        z = denormalise(latents[i:i + batch], vae).to(vae.dtype)
        outs.append(vae.decode(z, return_dict=False)[0].float())
    return torch.cat(outs)


@torch.no_grad()
def decode_swap_batch(vae, sample: Tensor, truth: Tensor, camera,
                      chunks: list[int], batch: int = 8):
    """Yield `(chunk, draw, pixels)` for every row of the swap batch.

    `pixels` is (3, t, H, W), already narrowed to the pixel frames that belong to
    the row's chunk; the truth row is yielded once per chunk with draw `-1`. Rows
    are decoded in groups and released immediately, since holding every row's full
    29-frame decode would cost far more memory than the statistics need.
    """
    s = camera.crop(sample)
    t = camera.crop(truth)
    rows, labels = build_swap_batch(s, t, chunks)
    n_lat = rows.shape[2]
    keep = {c: [p for p in pixel_frames_of_chunk(c, n_lat)
                if p < n_pixel_frames(n_lat)] for c in chunks}

    for i in range(0, rows.shape[0], batch):
        pix = decode(vae, rows[i:i + batch], batch=batch)
        for j, (c, k) in enumerate(labels[i:i + batch]):
            if c >= 0:
                yield c, k, pix[j][:, keep[c]]
            else:
                for cc in chunks:
                    yield cc, -1, pix[j][:, keep[cc]]
        del pix
