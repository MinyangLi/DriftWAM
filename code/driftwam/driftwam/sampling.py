"""Teacher-forced reverse-ODE sampling and one-step student sampling.

Flash-WAM ships no inference path, so this drives `forward_train` directly. That
is also the honest choice: a drifting loss, if adopted, would be computed inside
`_train_step`, so the conditional distribution that matters is the one
`forward_train` exposes.

Protocol. The clean halves of both streams are pinned to ground truth and only
the noisy halves are integrated. Three properties of the block-causal mask make
this cheap and exact:

* Every chunk denoises independently against its own true history, so one forward
  evaluates all `ceil(F/2)` conditions of an item at once.
* Noisy video and noisy action tokens cannot see each other, so both branches can
  sit at unrelated points on their own sigma grids within a single forward. This
  is what lets 25 video steps and 50 action steps share 50 conditional forwards
  rather than needing 75. `check_modality_decoupling` verifies it against real
  weights; if it ever fails, the interleaving is invalid.
* `seq_ids` isolates batch elements in self- *and* cross-attention, so the K
  independent noise draws go in the batch dimension.

Cost per condition set: 50 conditional + 25 unconditional forwards = 75.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange

from . import shapes

Tensor = torch.Tensor


@dataclass(frozen=True)
class SamplerSpec:
    """Sampling grid, defaults taken from `va_robotwin_cfg`."""

    video_steps: int = 25
    action_steps: int = 50
    video_shift: float = 5.0
    action_shift: float = 1.0
    cfg_scale: float = 5.0
    num_train_timesteps: int = 1000
    sigma_data: float = 0.5

    def __post_init__(self) -> None:
        if self.action_steps % self.video_steps != 0:
            raise ValueError(
                f"action_steps ({self.action_steps}) must be a multiple of "
                f"video_steps ({self.video_steps}) for the interleaved schedule"
            )

    @property
    def video_every(self) -> int:
        return self.action_steps // self.video_steps

    @property
    def uses_cfg(self) -> bool:
        return abs(self.cfg_scale - 1.0) > 1e-9

    @property
    def n_forwards(self) -> int:
        return self.action_steps + (self.video_steps if self.uses_cfg else 0)


def sigma_grid(steps: int, shift: float) -> tuple[Tensor, Tensor]:
    """Descending sigmas, and the sigma each step lands on.

    Mirrors `FlowMatchScheduler(shift=..., sigma_min=0.0, extra_one_step=True)`.
    The grid stops short of zero (25 video steps end at sigma=0.172) and
    `FlowMatchScheduler.step` compensates by treating the last step as a jump to
    zero. Pairing `sigmas[i]` with `sigmas[i+1]` and stopping would leave the
    sample visibly noisy, so `next_sigmas` carries that explicit zero.
    """
    from utils import FlowMatchScheduler  # noqa: PLC0415  (needs bootstrap)

    sched = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(steps)
    sigmas = sched.sigmas.to(torch.float64)
    if len(sigmas) != steps:
        raise RuntimeError(f"expected {steps} sigmas, got {len(sigmas)}")
    return sigmas, torch.cat([sigmas[1:], torch.zeros(1, dtype=sigmas.dtype)])


@dataclass
class Condition:
    """A batch ready for `forward_train`.

    The batch dimension carries either K noise draws of one item
    (`make_condition`) or one draw of each of B different items (`stack_items`).
    Nothing downstream depends on which, because batch slots are provably
    isolated; `slots` records what each position means.
    """

    clean_latents: Tensor  # (B, 48, F, 24, 20)
    clean_actions: Tensor  # (B, 30, F, 16, 1)
    actions_mask: Tensor  # (B, 30, F, 16, 1)
    text_emb: Tensor  # (B, 512, 4096)
    video_grid_id: Tensor  # (B, 4, F*12*10)
    action_grid_id: Tensor  # (B, 4, F*16)
    item_index: int = -1
    episode_index: int = -1
    task: str = ""
    slots: list[dict] | None = None  # per-position metadata for stacked batches

    @property
    def k(self) -> int:
        return self.clean_latents.shape[0]

    @property
    def n_frames(self) -> int:
        return self.clean_latents.shape[2]

    @property
    def n_chunks(self) -> int:
        return shapes.n_chunks(self.n_frames)


def make_condition(
    item: dict,
    k: int,
    device: torch.device | str,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    n_frames: int | None = None,
    item_index: int = -1,
    episode_index: int = -1,
    task: str = "",
) -> Condition:
    """Replicate a dataset item K times and precompute its RoPE grids.

    `n_frames` crops the item. Episode length runs from 5 to 31 latent frames
    across RoboTwin tasks and `create_block_mask` recompiles whenever (B, F)
    changes, so callers group items of equal F.
    """
    from utils import get_mesh_id  # noqa: PLC0415

    lat, act = item["latents"], item["actions"]
    mask, text = item["actions_mask"], item["text_emb"]
    f = n_frames if n_frames is not None else lat.shape[1]
    if f > lat.shape[1]:
        raise ValueError(f"asked for F={f} but item has only {lat.shape[1]} frames")
    lat, act, mask = lat[:, :f], act[:, :f], mask[:, :f]

    def rep(t: Tensor) -> Tensor:
        return t.unsqueeze(0).expand(k, *t.shape).contiguous().to(device)

    p_t, p_h, p_w = patch_size
    v_grid = get_mesh_id(
        f // p_t, shapes.GRID_H // p_h, shapes.GRID_W // p_w,
        t=0, f_w=1, f_shift=0, action=False,
    )
    a_grid = get_mesh_id(f, shapes.ACTION_PER_FRAME, 1, t=1, f_w=1, f_shift=0, action=True)

    mask_dev = rep(mask)
    return Condition(
        clean_latents=rep(lat).float(),
        # the dataset already zeroes invalid channels, but `_add_noise` re-applies
        # the mask and the ODE state must stay masked at every step
        clean_actions=rep(act).float() * mask_dev.float(),
        actions_mask=mask_dev,
        text_emb=rep(text),
        video_grid_id=v_grid[None].expand(k, -1, -1).contiguous().to(device),
        action_grid_id=a_grid[None].expand(k, -1, -1).contiguous().to(device),
        item_index=item_index,
        episode_index=episode_index,
        task=task,
    )


def stack_items(
    items: list[dict],
    device: torch.device | str,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    n_frames: int = 8,
    meta: list[dict] | None = None,
) -> Condition:
    """Put B *different* items in the batch dimension, one noise draw each.

    This is the orchestration the measurement actually uses. Batching the K draws
    of a single item would put each draw at a different batch offset, and the
    block-sparse reduction order depends on that offset, so identical epsilon at
    different positions disagrees by ~0.35% in bf16 -- an error that inflates
    measured variance, the one direction that could make a degenerate branch look
    healthy. Pinning each item to a fixed slot and varying epsilon across outer
    iterations removes it, at identical total FLOPs, because slot isolation is
    exact.

    Every item must be croppable to `n_frames`; equal F keeps one compiled mask.
    """
    from utils import get_mesh_id  # noqa: PLC0415

    if not items:
        raise ValueError("need at least one item")
    b = len(items)
    lat, act, msk, txt = [], [], [], []
    for it in items:
        f_native = it["latents"].shape[1]
        if f_native < n_frames:
            raise ValueError(f"item has F={f_native}, need >= {n_frames}")
        lat.append(it["latents"][:, :n_frames])
        act.append(it["actions"][:, :n_frames])
        msk.append(it["actions_mask"][:, :n_frames])
        txt.append(it["text_emb"])

    mask = torch.stack(msk).to(device)
    p_t, p_h, p_w = patch_size
    v_grid = get_mesh_id(
        n_frames // p_t, shapes.GRID_H // p_h, shapes.GRID_W // p_w,
        t=0, f_w=1, f_shift=0, action=False,
    )
    a_grid = get_mesh_id(n_frames, shapes.ACTION_PER_FRAME, 1, t=1, f_w=1, f_shift=0,
                         action=True)
    return Condition(
        clean_latents=torch.stack(lat).to(device).float(),
        clean_actions=torch.stack(act).to(device).float() * mask.float(),
        actions_mask=mask,
        text_emb=torch.stack(txt).to(device),
        video_grid_id=v_grid[None].expand(b, -1, -1).contiguous().to(device),
        action_grid_id=a_grid[None].expand(b, -1, -1).contiguous().to(device),
        slots=meta,
    )


def initial_noise(
    cond: Condition, generator: torch.Generator, n_replicate: int = 0
) -> tuple[Tensor, Tensor]:
    """Independent epsilon per batch slot, optionally duplicating the first few.

    `n_replicate > 1` makes that many leading slots share one epsilon. Their
    outputs should be identical, so their spread is the bf16 / block-sparse
    reduction noise floor (V3), measured under exactly the kernels and batch shape
    used for the real numbers.
    """
    v = torch.empty_like(cond.clean_latents).normal_(generator=generator)
    a = torch.empty_like(cond.clean_actions).normal_(generator=generator)
    if n_replicate > 1:
        v[:n_replicate] = v[0]
        a[:n_replicate] = a[0]
    return v, a * cond.actions_mask.float()


class Sampler:
    """Wraps one transformer and drives it over the reverse ODE."""

    def __init__(
        self,
        model,
        spec: SamplerSpec,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        chunk_size: int = shapes.FRAME_CHUNK_SIZE,
        window_size: int = 72,
        empty_emb: Tensor | None = None,
    ):
        self.model = model
        self.spec = spec
        self.patch_size = tuple(patch_size)
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.empty_emb = empty_emb
        self._v_sigmas, self._v_next = sigma_grid(spec.video_steps, spec.video_shift)
        self._a_sigmas, self._a_next = sigma_grid(spec.action_steps, spec.action_shift)

    def _input_dict(
        self,
        cond: Condition,
        video_x: Tensor,
        video_sigma: float,
        action_x: Tensor,
        action_sigma: float,
        text_emb: Tensor,
    ) -> dict:
        k, f = cond.k, cond.n_frames
        ts, dev = self.spec.num_train_timesteps, video_x.device

        def per_frame(sigma: float) -> Tensor:
            return torch.full((k, f), float(sigma) * ts, dtype=torch.float32, device=dev)

        zeros = torch.zeros((k, f), dtype=torch.float32, device=dev)
        return {
            "latent_dict": {
                "noisy_latents": video_x,
                "latent": cond.clean_latents,
                "timesteps": per_frame(video_sigma),
                "cond_timesteps": zeros,
                "grid_id": cond.video_grid_id,
                "text_emb": text_emb,
            },
            "action_dict": {
                "noisy_latents": action_x,
                "latent": cond.clean_actions,
                "timesteps": per_frame(action_sigma),
                "cond_timesteps": zeros,
                "grid_id": cond.action_grid_id,
                "text_emb": text_emb,
                "actions_mask": cond.actions_mask,
            },
            "chunk_size": self.chunk_size,
            "window_size": self.window_size,
        }

    @torch.no_grad()
    def velocity(
        self,
        cond: Condition,
        video_x: Tensor,
        video_sigma: float,
        action_x: Tensor,
        action_sigma: float,
        text_emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """One forward, returning (video v, action v) in data layout.

        `forward_train` casts its inputs to bf16 and writes them back into the
        dict it was handed, so the dict is rebuilt per call and the caller's fp32
        ODE state is never clobbered.
        """
        from utils import data_seq_to_patch  # noqa: PLC0415

        d = self._input_dict(
            cond, video_x, video_sigma, action_x, action_sigma,
            cond.text_emb if text_emb is None else text_emb,
        )
        v_seq, a_seq = self.model(d, train_mode=True)
        v = data_seq_to_patch(
            self.patch_size, v_seq, cond.n_frames,
            shapes.GRID_H, shapes.GRID_W, batch_size=cond.k,
        )
        a = rearrange(a_seq, "b (f n) c -> b c f n 1", f=cond.n_frames)
        return v.float(), a.float()

    def _uncond_text(self, cond: Condition) -> Tensor:
        if self.empty_emb is None:
            raise RuntimeError("CFG requested but no empty_emb was provided")
        return self.empty_emb.to(cond.text_emb.dtype).expand(cond.k, -1, -1)

    @torch.no_grad()
    def teacher_ode(
        self,
        cond: Condition,
        video_x: Tensor,
        action_x: Tensor,
        record_x0: bool = False,
        progress=None,
    ) -> dict:
        """Integrate both branches to sigma=0 and return the generated sample.

        Both grids advance inside one loop over action steps, video stepping every
        `video_every`-th iteration. Legal only because the noisy halves of the two
        streams are mutually invisible.
        """
        spec = self.spec
        v_sig, v_next = self._v_sigmas.tolist(), self._v_next.tolist()
        a_sig, a_next = self._a_sigmas.tolist(), self._a_next.tolist()
        uncond_text = self._uncond_text(cond) if spec.uses_cfg else None
        mask = cond.actions_mask.float()

        traj: dict[str, list] = {
            "sigmas_video": [], "x0_video": [], "sigmas_action": [], "x0_action": [],
        }
        iv = 0
        for ia in range(spec.action_steps):
            advance_video = (ia % spec.video_every == 0) and iv < spec.video_steps
            vs = v_sig[iv] if iv < spec.video_steps else 0.0

            v_cond, a_cond = self.velocity(cond, video_x, vs, action_x, a_sig[ia])

            if advance_video:
                if spec.uses_cfg:
                    v_unc, _ = self.velocity(
                        cond, video_x, vs, action_x, a_sig[ia], text_emb=uncond_text
                    )
                    v_use = v_unc + spec.cfg_scale * (v_cond - v_unc)
                else:
                    v_use = v_cond
                if record_x0:
                    traj["sigmas_video"].append(vs)
                    traj["x0_video"].append((video_x - vs * v_use).to(torch.float16).cpu())
                video_x = video_x + v_use * (v_next[iv] - vs)
                iv += 1

            if record_x0:
                traj["sigmas_action"].append(a_sig[ia])
                traj["x0_action"].append(
                    ((action_x - a_sig[ia] * a_cond) * mask).to(torch.float16).cpu()
                )
            action_x = (action_x + a_cond * (a_next[ia] - a_sig[ia])) * mask

            if progress is not None:
                progress(ia + 1, spec.action_steps)

        if iv != spec.video_steps:
            raise RuntimeError(f"video ODE took {iv} of {spec.video_steps} steps")
        out = {"video": video_x, "action": action_x}
        if record_x0:
            out["trajectory"] = traj
        return out

    @torch.no_grad()
    def student_one_step(self, cond: Condition, video_x: Tensor, action_x: Tensor) -> dict:
        """The student's single forward at sigma=1, where video_x *is* epsilon.

        Returns two readouts of the video branch, and they must not be confused.

        `video_x0 = x_t - sigma * v` is what the network predicts. `video` is the
        consistency function `c_skip * x_t + c_out * pred_x0` that Flash-WAM
        trains, which at sigma=1 with sigma_data=0.5 is `0.2 * epsilon + 0.447 *
        pred_x0`. That skip term is an *analytic* function of epsilon, so it
        contributes a fixed variance of c_skip^2 = 0.04 per element no matter what
        the network does. Measured on real weights the student's whole video
        v_intra is ~0.044, i.e. essentially all of it is the skip term. Reading
        epsilon-sensitivity off `video` would therefore report the parametrisation
        rather than the model, and would pass H7 even for a totally
        mean-collapsed student. H7 is judged on `video_x0`.

        The action branch uses the plain x0 form (`action_distill_mode="x0"`), so
        it has no skip term and the two readouts coincide.
        """
        from consistency import scalings_for_boundary_conditions  # noqa: PLC0415

        sigma = 1.0
        mask = cond.actions_mask.float()
        v_video, v_action = self.velocity(cond, video_x, sigma, action_x, sigma)
        c_skip, c_out = scalings_for_boundary_conditions(sigma, self.spec.sigma_data)
        video_x0 = video_x - sigma * v_video
        action_x0 = (action_x - sigma * v_action) * mask
        return {
            "video": c_skip * video_x + c_out * video_x0,
            "action": action_x0,
            "video_x0": video_x0,
            "action_x0": action_x0,
            "c_skip": c_skip,
            "c_out": c_out,
        }


# ---------------------------------------------------------------- controls


@torch.no_grad()
def check_determinism(
    sampler: Sampler, cond: Condition, video_x: Tensor, action_x: Tensor
) -> dict:
    """V2: the same epsilon evaluated twice must agree bit for bit.

    A mismatch means some hidden randomness (dropout left on, a non-deterministic
    reduction seeded per call) is inflating every dispersion number downstream.
    """
    v1, a1 = sampler.velocity(cond, video_x, 1.0, action_x, 1.0)
    v2, a2 = sampler.velocity(cond, video_x, 1.0, action_x, 1.0)
    return {
        "video_identical": bool(torch.equal(v1, v2)),
        "action_identical": bool(torch.equal(a1, a2)),
        "video_max_diff": (v1 - v2).abs().max().item(),
        "action_max_diff": (a1 - a2).abs().max().item(),
    }


@torch.no_grad()
def check_modality_decoupling(
    sampler: Sampler, cond: Condition, generator: torch.Generator
) -> dict:
    """V4: the branches must be blind to each other's noisy tokens.

    The interleaved schedule reuses one forward for a video step and an action
    step taken at unrelated sigmas, which is only sound if perturbing the noisy
    action stream leaves the video velocity untouched and vice versa. A nonzero
    response means the two noisy streams do attend to one another and the video
    and action ODEs can no longer share forwards.
    """
    v0, a0 = initial_noise(cond, generator)
    v_ref, a_ref = sampler.velocity(cond, v0, 1.0, a0, 1.0)

    a_pert = torch.empty_like(a0).normal_(generator=generator) * cond.actions_mask.float()
    v_after_a, _ = sampler.velocity(cond, v0, 1.0, a_pert, 1.0)

    v_pert = torch.empty_like(v0).normal_(generator=generator)
    _, a_after_v = sampler.velocity(cond, v_pert, 1.0, a0, 1.0)

    return {
        "video_response_to_action_perturbation": (v_after_a - v_ref).abs().max().item(),
        "action_response_to_video_perturbation": (a_after_v - a_ref).abs().max().item(),
        "video_velocity_scale": v_ref.abs().mean().item(),
        "action_velocity_scale": a_ref.abs().mean().item(),
        "decoupled": bool(
            torch.equal(v_after_a, v_ref) and torch.equal(a_after_v, a_ref)
        ),
    }


@torch.no_grad()
def check_chunk_independence(
    sampler: Sampler, cond: Condition, generator: torch.Generator
) -> dict:
    """Each chunk must denoise independently of later chunks' noisy tokens.

    Perturbing the noisy video of the final chunk may not change the velocity of
    any earlier chunk. This is what licenses treating the `ceil(F/2)` chunks of
    one forward as separate conditions rather than one joint sample.
    """
    v0, a0 = initial_noise(cond, generator)
    v_ref, _ = sampler.velocity(cond, v0, 1.0, a0, 1.0)

    last = cond.n_chunks - 1
    last_frames = shapes.video_frames_of_chunk(last, cond.n_frames)
    v_pert = v0.clone()
    v_pert[:, :, last_frames] = torch.empty_like(v_pert[:, :, last_frames]).normal_(
        generator=generator
    )
    v_new, _ = sampler.velocity(cond, v_pert, 1.0, a0, 1.0)

    earlier = [f for f in range(cond.n_frames) if f not in last_frames]
    if not earlier:
        return {"n_chunks": cond.n_chunks, "skipped": True}
    delta_earlier = (v_new[:, :, earlier] - v_ref[:, :, earlier]).abs().max().item()
    delta_last = (v_new[:, :, last_frames] - v_ref[:, :, last_frames]).abs().max().item()
    return {
        "n_chunks": cond.n_chunks,
        "earlier_chunk_response": delta_earlier,
        "perturbed_chunk_response": delta_last,
        "independent": bool(torch.equal(v_new[:, :, earlier], v_ref[:, :, earlier])),
    }
