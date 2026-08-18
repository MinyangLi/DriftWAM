"""Layout of a LingBot-VA RoboTwin training item, and how to take it apart.

A dataset item is::

    latents       (48, F, 24, 20)  bf16   composite "T-shape" of three cameras
    text_emb      (512, 4096)      bf16
    actions       (30, F, 16, 1)   f32    quantile-normalised to [-1, 1]
    actions_mask  (30, F, 16, 1)   bool

Two properties of this layout are load-bearing for any variance measurement and
are easy to get wrong, so they are spelled out here and asserted in
`scripts/inspect_data.py`:

1.  The three camera latents are packed into one 24x20 grid: the two 8x10 wrist
    views sit side by side on top, and the 16x20 overhead view below. Pixel
    dimensions divide by 16 into exactly these grids, which is why DINOv3/16
    patch tokens align 1:1 with latent cells.

2.  Action latent frame 0 is *not* data. `_action_post_process` front-pads the
    raw action stream by one full latent frame (`frame_stride * 4 == 16` steps)
    of zeros before normalising, so frame 0 carries a per-channel constant
    (normalise(0)) that is identical in every episode, yet is flagged valid by
    `actions_mask`. Averaging it into a variance statistic silently drags the
    result toward zero. Use `action_frame_slice()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

LATENT_CHANNELS = 48
GRID_H, GRID_W = 24, 20
ACTION_DIM = 30
ACTION_PER_FRAME = 16
FRAME_CHUNK_SIZE = 2
VAE_TEMPORAL_RATIO = 4
DINO_PATCH = 16

# Number of leading action latent frames that are constant zero-padding.
ACTION_PAD_FRAMES = 1


@dataclass(frozen=True)
class CameraView:
    name: str
    rows: slice
    cols: slice
    pixel_h: int
    pixel_w: int

    @property
    def latent_h(self) -> int:
        return self.rows.stop - self.rows.start

    @property
    def latent_w(self) -> int:
        return self.cols.stop - self.cols.start

    @property
    def n_cells(self) -> int:
        return self.latent_h * self.latent_w

    def crop(self, latents: torch.Tensor) -> torch.Tensor:
        """Slice this camera out of a composite tensor whose last two dims are H, W."""
        return latents[..., self.rows, self.cols]


# Order matches `_cat_video_latents`: wrists concatenated along W, then stacked
# above cam_high along H.
CAMERAS: tuple[CameraView, ...] = (
    CameraView("cam_left_wrist", slice(0, 8), slice(0, 10), 128, 160),
    CameraView("cam_right_wrist", slice(0, 8), slice(10, 20), 128, 160),
    CameraView("cam_high", slice(8, 24), slice(0, 20), 256, 320),
)
CAMERAS_BY_NAME = {c.name: c for c in CAMERAS}

# Aligned action channel groups (post-permutation indices, see module docstring
# of driftwam.shapes and `inverse_used_action_channel_ids`).
ACTION_GROUPS: dict[str, list[int]] = {
    "left_arm_trans": [0, 1, 2],
    "left_arm_rot": [3, 4, 5, 6],
    "right_arm_trans": [7, 8, 9],
    "right_arm_rot": [10, 11, 12, 13],
    "left_gripper": [28],
    "right_gripper": [29],
}
VALID_ACTION_CHANNELS: list[int] = sorted(c for g in ACTION_GROUPS.values() for c in g)
CHANNEL_GROUP = {c: g for g, cs in ACTION_GROUPS.items() for c in cs}


def n_chunks(n_frames: int, chunk_size: int = FRAME_CHUNK_SIZE) -> int:
    """Chunks spanned by `n_frames` latent frames; the last one may be partial."""
    return (n_frames + chunk_size - 1) // chunk_size


def chunk_of_frame(frame: int, chunk_size: int = FRAME_CHUNK_SIZE) -> int:
    return frame // chunk_size


def frame_ids(n_frames: int, action: bool, chunk_size: int = FRAME_CHUNK_SIZE) -> list[int]:
    """The per-frame ordering key the model's block-causal mask is built from.

    Video frames in a chunk share an even id; the action frames of the same chunk
    take the next odd id, which is what makes an action attend to the clean video
    of its own chunk while video cannot see that action.
    """
    return [f // chunk_size * 2 + (1 if action else 0) for f in range(n_frames)]


def video_frames_of_chunk(chunk: int, n_frames: int, chunk_size: int = FRAME_CHUNK_SIZE) -> list[int]:
    return [f for f in range(n_frames) if f // chunk_size == chunk]


def action_frame_slice(n_frames: int) -> list[int]:
    """Action latent frames that carry real data (drops the constant pad frame)."""
    return list(range(ACTION_PAD_FRAMES, n_frames))


def expected_latent_frames(n_video_frames: int) -> int:
    """VAE temporal compression: a causal 4x ratio keeps the first frame whole."""
    return 1 + (n_video_frames - 1) // VAE_TEMPORAL_RATIO


def video_tokens_per_frame(patch_h: int = 2, patch_w: int = 2) -> int:
    return (GRID_H // patch_h) * (GRID_W // patch_w)


def sequence_length(n_frames: int, patch_h: int = 2, patch_w: int = 2) -> dict[str, int]:
    """Token budget of one `forward_train` call (noisy + clean copies of both streams)."""
    v = n_frames * video_tokens_per_frame(patch_h, patch_w)
    a = n_frames * ACTION_PER_FRAME
    return {"video": v, "action": a, "total": 2 * (v + a)}


class ActionNorm:
    """Quantile normalisation used by the dataset, and its inverse.

    `norm = (raw - q01) / (q99 - q01 + eps) * 2 - 1`, applied *after* channels are
    permuted into aligned order, so `q01`/`q99` are indexed by aligned channel.

    Variance in normalised units is not comparable across channels: one unit is
    ~0.20 m of arm translation but a full 0.5 of gripper travel. Use `scale` to
    put per-channel statistics back into physical units before comparing them.
    """

    EPS = 1e-6

    def __init__(self, norm_stat: dict, dtype=torch.float64):
        self.q01 = torch.as_tensor(norm_stat["q01"], dtype=dtype)
        self.q99 = torch.as_tensor(norm_stat["q99"], dtype=dtype)
        if self.q01.shape != (ACTION_DIM,) or self.q99.shape != (ACTION_DIM,):
            raise ValueError(f"norm_stat must have {ACTION_DIM} entries per bound")

    @property
    def span(self) -> torch.Tensor:
        return self.q99 - self.q01 + self.EPS

    @property
    def scale(self) -> torch.Tensor:
        """Physical units per unit of normalised action, per aligned channel."""
        return self.span / 2.0

    def denormalise(self, norm: torch.Tensor, channel_dim: int = 0) -> torch.Tensor:
        shape = [1] * norm.ndim
        shape[channel_dim] = ACTION_DIM
        q01 = self.q01.to(norm.device, norm.dtype).view(shape)
        span = self.span.to(norm.device, norm.dtype).view(shape)
        return (norm + 1.0) / 2.0 * span + q01

    def to_physical_std(self, std_norm: torch.Tensor, channels: list[int]) -> torch.Tensor:
        """Rescale a per-channel std from normalised to physical units."""
        s = self.scale.to(std_norm.device, std_norm.dtype)[channels]
        return std_norm * s


def split_cameras(latents: torch.Tensor) -> dict[str, torch.Tensor]:
    """Composite (..., H, W) -> per-camera crops."""
    if latents.shape[-2:] != (GRID_H, GRID_W):
        raise ValueError(f"expected (..., {GRID_H}, {GRID_W}), got {tuple(latents.shape)}")
    return {c.name: c.crop(latents) for c in CAMERAS}


def active_action_channels(
    actions: torch.Tensor, channels: list[int] | None = None, tol: float = 1e-8
) -> list[int]:
    """Channels that actually move in the given batch of items.

    RoboTwin has plenty of single-arm tasks in which the idle arm's seven pose
    channels are byte-identical across the episode. They are flagged valid by the
    mask, so pooling over `VALID_ACTION_CHANNELS` would average a genuine spread
    together with structural zeros.

    `actions` is (N, 30, F, 16, 1) or (30, F, 16, 1).
    """
    if actions.ndim == 4:
        actions = actions.unsqueeze(0)
    channels = channels if channels is not None else VALID_ACTION_CHANNELS
    frames = action_frame_slice(actions.shape[2])
    a = actions[:, channels][:, :, frames].to(torch.float64)
    spread = a.reshape(a.shape[0], a.shape[1], -1).std(dim=-1).amax(dim=0)
    return [c for c, s in zip(channels, spread.tolist()) if s > tol]
