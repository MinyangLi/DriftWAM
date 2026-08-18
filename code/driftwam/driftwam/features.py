"""DINOv3 patch features, in the configuration DriftWorld reports using.

DriftWorld takes "features from the output of the 2nd, 5th, and 8th blocks in the
DINOv3 ViT-B/16" and adds a drifting loss in that space to the one in VAE latent
space. The blocks are taken separately here so each can be judged on its own,
since an early block is close to raw colour statistics while a middle block is
where semantics appear, and the question is precisely whether the *semantic* space
sees dispersion that the latent space does not.

Patch16 is what makes the comparison clean: cam_high's 256x320 gives a 16x20 patch
grid and the wrists' 128x160 give 8x10, each identical to that camera's latent
grid, so a per-position statistic maps 1:1 between the two spaces with no
resampling. DINOv2's patch14 would not divide.

Prefix tokens (1 CLS + 4 register) are dropped: they are global, not spatial, and
DINOv3's register tokens are known to carry very large norms that would dominate a
plain per-element variance.
"""

from __future__ import annotations

import torch

from . import shapes

Tensor = torch.Tensor

# DriftWorld's choice, as indices into `hidden_states` (entry 0 is the embedding
# output, so entry i is the output of block i).
DRIFTWORLD_BLOCKS = (2, 5, 8)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dino(model_dir, device="cuda", dtype=torch.float32, verbose: bool = True):
    """DINOv3 ViT-B/16. Needs transformers >= 4.56 for the `dinov3_vit` type."""
    from transformers import AutoModel  # noqa: PLC0415

    model = AutoModel.from_pretrained(str(model_dir), torch_dtype=dtype)
    model.eval().requires_grad_(False)
    model.to(device)
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        cfg = model.config
        print(f"loaded DINOv3: {n / 1e6:.0f}M params, hidden={cfg.hidden_size}, "
              f"layers={cfg.num_hidden_layers}, patch={cfg.patch_size}, {dtype}")
    return model


def n_prefix_tokens(model) -> int:
    """CLS plus however many register tokens this checkpoint uses."""
    return 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)


def to_dino_input(pixels: Tensor) -> Tensor:
    """VAE output in [-1, 1] -> ImageNet-normalised, (N, 3, H, W)."""
    x = (pixels.clamp(-1, 1) + 1.0) / 2.0
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def patch_features(
    model,
    pixels: Tensor,
    blocks: tuple[int, ...] = DRIFTWORLD_BLOCKS,
    batch: int = 32,
    grid: tuple[int, int] | None = None,
) -> dict[int, Tensor]:
    """Per-block patch features for a stack of frames.

    `pixels` is (N, 3, H, W) in [-1, 1]. Returns block -> (N, gh, gw, C).
    """
    if pixels.ndim != 4:
        raise ValueError(f"expected (N, 3, H, W), got {tuple(pixels.shape)}")
    p = int(model.config.patch_size)
    h, w = pixels.shape[-2:]
    if h % p or w % p:
        raise ValueError(f"{h}x{w} is not divisible by patch size {p}")
    gh, gw = h // p, w // p
    if grid is not None and (gh, gw) != tuple(grid):
        raise ValueError(f"patch grid {(gh, gw)} != expected {tuple(grid)}")
    skip = n_prefix_tokens(model)

    out: dict[int, list[Tensor]] = {b: [] for b in blocks}
    for i in range(0, pixels.shape[0], batch):
        x = to_dino_input(pixels[i:i + batch].to(model.dtype))
        hs = model(pixel_values=x, output_hidden_states=True).hidden_states
        for b in blocks:
            f = hs[b][:, skip:]
            if f.shape[1] != gh * gw:
                raise RuntimeError(
                    f"block {b}: {f.shape[1]} patch tokens for a {gh}x{gw} grid; "
                    f"prefix count {skip} is probably wrong"
                )
            out[b].append(f.reshape(f.shape[0], gh, gw, f.shape[-1]).float())
    return {b: torch.cat(v) for b, v in out.items()}


def latent_grid_of(camera) -> tuple[int, int]:
    return camera.latent_h, camera.latent_w


def expected_patch_grid(camera, patch: int = shapes.DINO_PATCH) -> tuple[int, int]:
    """The whole point of patch16: this must equal the camera's latent grid."""
    return camera.pixel_h // patch, camera.pixel_w // patch


def standardise_channels(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Divide each feature channel by its std over the whole stack.

    A plain per-element variance in DINO space is dominated by a handful of
    high-norm channels, and that is also what an unweighted L2 drifting loss would
    see, so the raw number is the operationally relevant one. This gives the
    robustness check: if the variance ratio survives per-channel whitening, it is
    not an artefact of a few outlier dimensions.
    """
    flat = x.reshape(-1, x.shape[-1])
    return x / flat.std(dim=0).clamp_min(eps)
