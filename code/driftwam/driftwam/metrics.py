"""Conditional-dispersion statistics (M1-M8) and validity controls (V1-V3).

Everything operates on a stack of K samples drawn under one fixed condition,
flattened to `(K, D)`. Callers slice out cameras / action channels / chunks and
call in per subset; nothing here knows about the model.

All accumulation is float64. The quantities of interest are ratios of small
numbers, and the whole point of the experiment is to tell a genuinely small
variance apart from a bf16 rounding artefact, so accumulating in the sample dtype
would beg the question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

Tensor = torch.Tensor


def _as_2d(x: Tensor) -> Tensor:
    if x.ndim < 2:
        raise ValueError(f"need at least (K, ...), got {tuple(x.shape)}")
    return x.reshape(x.shape[0], -1).to(torch.float64)


#: Above this many elements, `_pdist` walks row by row instead of letting cdist
#: materialise the full (n, n, D) difference tensor. DINO features push D past
#: 700k, where the one-shot form would ask for tens of gigabytes.
_PDIST_ELEM_BUDGET = 1 << 29


def _pdist(x: Tensor) -> Tensor:
    """Pairwise distances, always by direct subtraction.

    cdist's default `use_mm_for_euclid_dist_if_necessary` switches to
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b once either side has more than 25 rows.
    That expansion cancels away the leading digits whenever the separation is
    small next to the norms, which is precisely the regime being measured here.
    K <= 16 stays on the direct path either way, but M2 compares hundreds of
    condition means and would silently cross the threshold.

    The row-by-row fallback for wide inputs computes the same direct subtraction,
    so it is exact in the same way; only the peak allocation differs.
    """
    n, d = x.shape
    if n * n * d <= _PDIST_ELEM_BUDGET:
        return torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")
    out = torch.empty((n, n), dtype=x.dtype, device=x.device)
    for i in range(n):
        out[i] = (x - x[i]).pow(2).sum(dim=1).sqrt()
    return out


@dataclass
class ConditionStats:
    """Per-condition dispersion. Field names map onto the plan's M-numbers."""

    k: int
    d: int
    v_intra: float  # M1 unbiased per-element variance
    scale: float  # s_c, per-element RMS of the samples
    rho: float  # M1 sqrt(v_intra) / s_c
    participation_ratio: float  # M4
    min_pair_ratio: float  # M5
    dead_frac: float = 0.0  # fraction of dimensions with zero spread across K
    bias: float | None = None  # M6 ||mean - truth|| / (sqrt(D) * s_c)
    eigenvalues: list[float] = field(default_factory=list)
    mean: Tensor | None = None  # kept for M2; float32 to bound memory

    def as_row(self) -> dict:
        return {
            "K": self.k,
            "D": self.d,
            "v_intra": self.v_intra,
            "scale": self.scale,
            "rho": self.rho,
            "PR": self.participation_ratio,
            "min_pair_ratio": self.min_pair_ratio,
            "dead_frac": self.dead_frac,
            "bias": self.bias,
        }


def condition_stats(
    samples: Tensor, truth: Tensor | None = None, keep_mean: bool = True
) -> ConditionStats:
    """M1, M4, M5 and (given `truth`) M6 for one condition.

    `samples` is (K, ...) with K >= 2; any trailing shape is flattened.
    """
    x = _as_2d(samples)
    k, d = x.shape
    if k < 2:
        raise ValueError("need K >= 2 samples to estimate a within-condition variance")

    mean = x.mean(dim=0)
    centred = x - mean

    # M1: unbiased, per element of D.
    v_intra = (centred.pow(2).sum() / ((k - 1) * d)).item()
    scale = math.sqrt((x.pow(2).sum() / (k * d)).item())
    rho = math.sqrt(v_intra) / scale if scale > 0 else float("nan")

    # M4: eigenvalues of the K x K Gram matrix carry the full spectrum, since
    # the centred matrix has rank <= K-1.
    gram = centred @ centred.T / (k - 1)
    eig = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    tr = eig.sum()
    pr = (tr.pow(2) / eig.pow(2).sum()).item() if tr > 0 else float("nan")

    # M5: closest pair relative to the typical pair.
    dist = _pdist(x)
    off = ~torch.eye(k, dtype=torch.bool, device=x.device)
    dmin, dmean = dist[off].min().item(), dist[off].mean().item()
    min_pair = dmin / dmean if dmean > 0 else float("nan")

    # Dimensions on which every draw agreed bit for bit. Reported rather than
    # filtered out: a dimension whose ground truth is constant (an idle arm, a
    # gripper that never opens) can still receive spread from the model, so
    # dropping it would discard a real output dimension. Knowing what fraction is
    # frozen says directly how much of the output is degenerate.
    #
    # Compared against min/max rather than against the mean: summing K identical
    # full-mantissa doubles can round (3v needs one bit more than v), so a frozen
    # dimension does not always satisfy x == mean exactly.
    dead_frac = (x.amax(dim=0) == x.amin(dim=0)).to(torch.float64).mean().item()

    bias = None
    if truth is not None:
        t = truth.reshape(-1).to(torch.float64)
        if t.numel() != d:
            raise ValueError(f"truth has {t.numel()} elements, samples have D={d}")
        bias = ((mean - t).norm() / (math.sqrt(d) * scale)).item() if scale > 0 else float("nan")

    return ConditionStats(
        k=k,
        d=d,
        v_intra=v_intra,
        scale=scale,
        rho=rho,
        participation_ratio=pr,
        min_pair_ratio=min_pair,
        dead_frac=dead_frac,
        bias=bias,
        eigenvalues=eig.tolist(),
        mean=mean.to(torch.float32) if keep_mean else None,
    )


def _mean_sq_pair_dist(means: Tensor, pair_mask: Tensor | None = None) -> tuple[float, int]:
    """mean over ordered pairs i != j of ||m_i - m_j||^2 / D."""
    m = means.to(torch.float64)
    n, d = m.shape
    sq = _pdist(m).pow(2)
    off = ~torch.eye(n, dtype=torch.bool, device=m.device)
    if pair_mask is not None:
        off = off & pair_mask
    npairs = int(off.sum().item())
    if npairs == 0:
        return float("nan"), 0
    return (sq[off].sum() / (npairs * d)).item(), npairs


def inter_condition_variance(
    means: Tensor, episode_ids: list[int] | None = None
) -> dict:
    """M2 in three flavours, plus the strict between-group variance.

    The plan's estimator is the mean squared distance between condition means,
    restricted to pairs from *different* episodes: chunks of one episode are
    nested (chunk 1's condition contains chunk 0's), so they are not independent
    draws of a condition.

    Mean pairwise squared distance equals twice the between-group variance, so
    `v_inter_anova` is reported alongside to keep the factor of two visible when
    R is compared against a variance-fraction intuition.
    """
    means = means.to(torch.float64)
    n = means.shape[0]
    out: dict = {"n_conditions": n}

    if episode_ids is not None:
        if len(episode_ids) != n:
            raise ValueError("episode_ids length must match number of conditions")
        ep = torch.as_tensor(episode_ids, device=means.device)
        same = ep[:, None] == ep[None, :]
        v_cross, n_cross = _mean_sq_pair_dist(means, ~same)
        v_same, n_same = _mean_sq_pair_dist(means, same)
        out["v_inter_cross_episode"] = v_cross
        out["n_pairs_cross_episode"] = n_cross
        out["v_inter_same_episode"] = v_same
        out["n_pairs_same_episode"] = n_same

    v_all, n_all = _mean_sq_pair_dist(means)
    out["v_inter_all_pairs"] = v_all
    out["n_pairs_all"] = n_all

    grand = means.mean(dim=0)
    out["v_inter_anova"] = (
        (means - grand).pow(2).sum() / ((n - 1) * means.shape[1])
    ).item() if n > 1 else float("nan")

    out["v_inter"] = out.get("v_inter_cross_episode", v_all)
    return out


def variance_ratio(v_intra_mean: float, v_inter: float) -> float:
    """M3: R = v_intra / (v_intra + v_inter). R -> 0 means Dirac-like."""
    denom = v_intra_mean + v_inter
    return v_intra_mean / denom if denom > 0 else float("nan")


def rho_from_R(r: float) -> float:
    """Conditional std as a fraction of marginal std, the anchor used in the plan."""
    return math.sqrt(r / (1.0 - r)) if 0.0 <= r < 1.0 else float("nan")


def summarise(
    stats: list[ConditionStats],
    episode_ids: list[int] | None = None,
    noise_floor: float | None = None,
    floor_ratio_required: float = 10.0,
) -> dict:
    """Pool per-condition stats into M1-M6 plus the V3 verdict."""
    if not stats:
        return {"n_conditions": 0}

    v = torch.tensor([s.v_intra for s in stats], dtype=torch.float64)
    out = {
        "n_conditions": len(stats),
        "K": stats[0].k,
        "D": stats[0].d,
        "v_intra_mean": v.mean().item(),
        "v_intra_median": v.median().item(),
        "v_intra_p10": v.quantile(0.10).item(),
        "v_intra_p90": v.quantile(0.90).item(),
        "rho_mean": _nanmean([s.rho for s in stats]),
        "scale_mean": _nanmean([s.scale for s in stats]),
        "PR_mean": _nanmean([s.participation_ratio for s in stats]),
        "PR_median": _nanmedian([s.participation_ratio for s in stats]),
        # The centred K x D matrix has rank <= K-1, so PR cannot exceed K-1 no
        # matter how isotropic the spread is. Comparing raw PR across runs with
        # different K is meaningless; the fraction of the attainable maximum is
        # the interpretable quantity (1 = isotropic in the sampled subspace,
        # 1/(K-1) = rank one).
        "PR_ceiling": stats[0].k - 1,
        "PR_frac_mean": _nanmean(
            [s.participation_ratio / (s.k - 1) for s in stats]
        ),
        "min_pair_ratio_mean": _nanmean([s.min_pair_ratio for s in stats]),
        "min_pair_ratio_min": min(s.min_pair_ratio for s in stats),
        "dead_frac_mean": _nanmean([s.dead_frac for s in stats]),
        "dead_frac_max": max(s.dead_frac for s in stats),
    }
    biases = [s.bias for s in stats if s.bias is not None]
    if biases:
        out["bias_mean"] = _nanmean(biases)
        out["bias_median"] = _nanmedian(biases)

    if all(s.mean is not None for s in stats):
        means = torch.stack([s.mean for s in stats])
        inter = inter_condition_variance(means, episode_ids)
        out.update(inter)
        out["R"] = variance_ratio(out["v_intra_mean"], inter["v_inter"])
        out["R_anova"] = variance_ratio(out["v_intra_mean"], inter["v_inter_anova"])
        out["cond_std_frac_of_marginal"] = rho_from_R(out["R"])

        # Each condition mean is itself estimated from K draws, so it carries
        # sampling error v_intra/K. Two independent noisy means therefore sit
        # 2*v_intra/K further apart in expectation than the true means do, which
        # inflates v_inter and biases R downward. The correction is exact and
        # matters most in the regime where the two terms are comparable.
        k = stats[0].k
        deb = inter["v_inter"] - 2.0 * out["v_intra_mean"] / k
        out["v_inter_debiased"] = deb
        out["R_debiased"] = variance_ratio(out["v_intra_mean"], max(deb, 0.0))

    if noise_floor is not None:
        out["v_floor"] = noise_floor
        ratio = out["v_intra_mean"] / noise_floor if noise_floor > 0 else float("inf")
        out["floor_ratio"] = ratio
        out["measurable"] = bool(ratio >= floor_ratio_required)
        s = out["scale_mean"]
        out["rho_floor"] = math.sqrt(noise_floor) / s if s > 0 else float("nan")
    return out


def _nanmean(xs: list[float]) -> float:
    t = torch.tensor(xs, dtype=torch.float64)
    t = t[~t.isnan()]
    return t.mean().item() if t.numel() else float("nan")


def _nanmedian(xs: list[float]) -> float:
    t = torch.tensor(xs, dtype=torch.float64)
    t = t[~t.isnan()]
    return t.median().item() if t.numel() else float("nan")


# ---------------------------------------------------------------- M8


def dispersion_trajectory(x0_traj: Tensor) -> dict:
    """M8: how the spread of the predicted x0 develops along the reverse ODE.

    `x0_traj` is (T, K, ...) ordered by decreasing sigma. Returns per-step
    dispersion (mean pairwise distance) normalised by its final value.
    """
    t = x0_traj.shape[0]
    disp = []
    for i in range(t):
        x = _as_2d(x0_traj[i])
        k = x.shape[0]
        dist = _pdist(x)
        off = ~torch.eye(k, dtype=torch.bool, device=x.device)
        disp.append(dist[off].mean().item())
    final = disp[-1]
    return {
        "dispersion": disp,
        "dispersion_normalised": [d / final if final > 0 else float("nan") for d in disp],
        "final": final,
    }


def trajectory_alignment(x0_traj: Tensor) -> dict:
    """When along the reverse ODE the samples' *identities* become settled.

    The raw dispersion of `x0_hat = x_t - sigma * v` cannot answer this. At high
    sigma the model cannot fully denoise, so `x0_hat` still carries most of the
    input noise and its spread is large for a trivial reason -- on the action
    branch it is measured at 2.3x the final spread, i.e. it *falls* along the
    trajectory. Any "fraction of final dispersion" threshold is then crossed at
    the first step no matter what the model does.

    What has content, for a deterministic sampler whose output is a fixed function
    of epsilon, is when the *difference between two samples* settles into its final
    direction. So this returns, per step, the mean cosine between the pairwise
    difference `x0_hat_i - x0_hat_j` and that same pair's final difference. It is 1
    by construction at the last step, and near 0 while the differences are still
    dominated by residual noise, which is orthogonal to the data manifold.

    `x0_traj` is (T, K, ...) ordered by decreasing sigma.
    """
    t = x0_traj.shape[0]
    final = _as_2d(x0_traj[-1])
    k = final.shape[0]
    if k < 2:
        raise ValueError("need K >= 2 samples to compare pairwise differences")
    iu = torch.triu_indices(k, k, offset=1)
    ref = final[iu[0]] - final[iu[1]]
    ref = ref / ref.norm(dim=1, keepdim=True).clamp_min(1e-300)
    fnorm = final.norm(dim=1).clamp_min(1e-300)

    align, resid = [], []
    for i in range(t):
        x = _as_2d(x0_traj[i])
        d = x[iu[0]] - x[iu[1]]
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-300)
        align.append((d * ref).sum(dim=1).mean().item())
        resid.append(((x - final).norm(dim=1) / fnorm).mean().item())
    return {"alignment": align, "residual_to_final": resid}


def sigma_star(sigmas: list[float], dispersion: list[float], frac: float = 0.5) -> float:
    """Largest sigma at which dispersion has reached `frac` of its final value.

    `sigmas` is decreasing and aligned with `dispersion`.
    """
    if len(sigmas) != len(dispersion):
        raise ValueError("sigmas and dispersion must be the same length")
    target = frac * dispersion[-1]
    hits = [s for s, d in zip(sigmas, dispersion) if d >= target]
    return max(hits) if hits else float("nan")


# ---------------------------------------------------------------- V1-V3


def sampler_fidelity(mean: Tensor, truth: Tensor) -> float:
    """V1: relative L2 between the condition mean and ground truth."""
    m, t = mean.reshape(-1).to(torch.float64), truth.reshape(-1).to(torch.float64)
    n = t.norm()
    return ((m - t).norm() / n).item() if n > 0 else float("nan")


def bitwise_identical(a: Tensor, b: Tensor) -> bool:
    """V2: two runs of the same epsilon must agree exactly."""
    return a.shape == b.shape and bool(torch.equal(a, b))


def noise_floor(replicates: Tensor) -> dict:
    """V3: dispersion among outputs that *should* be identical.

    `replicates` is (K, ...): the same epsilon evaluated at K different positions
    in the batch. bf16 accumulation and flex_attention's block-sparse reduction
    order make this nonzero, and it manufactures apparent conditional variance.
    Any branch whose measured v_intra is not comfortably above this floor is
    unmeasurable rather than low-variance.
    """
    s = condition_stats(replicates, keep_mean=False)
    return {
        "v_floor": s.v_intra,
        "rho_floor": s.rho,
        "scale": s.scale,
        "max_abs_dev": (
            (_as_2d(replicates) - _as_2d(replicates).mean(dim=0)).abs().max().item()
        ),
    }
