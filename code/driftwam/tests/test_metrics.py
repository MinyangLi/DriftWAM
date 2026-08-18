"""Self-tests for the dispersion statistics, on data with known answers.

Run: python -m pytest tests/ -q     (or: python tests/test_metrics.py)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import metrics as M  # noqa: E402
from driftwam import shapes as S  # noqa: E402


def test_isotropic_gaussian_recovers_variance():
    torch.manual_seed(0)
    k, d, sigma = 16, 4096, 0.37
    x = torch.randn(k, d, dtype=torch.float64) * sigma
    s = M.condition_stats(x)
    assert abs(s.v_intra - sigma**2) / sigma**2 < 0.05, s.v_intra
    # Isotropic in D >> K: the K-1 nonzero eigenvalues are near-equal, so the
    # participation ratio should sit just under K-1.
    assert s.participation_ratio > 0.8 * (k - 1), s.participation_ratio


def test_rank_one_spread_has_unit_participation_ratio():
    torch.manual_seed(0)
    k, d = 16, 4096
    direction = torch.randn(d, dtype=torch.float64)
    coeff = torch.randn(k, 1, dtype=torch.float64)
    x = coeff * direction + torch.randn(1, d, dtype=torch.float64) * 10.0
    s = M.condition_stats(x)
    assert abs(s.participation_ratio - 1.0) < 1e-6, s.participation_ratio


def test_identical_samples_have_zero_variance():
    x = torch.randn(1, 512, dtype=torch.float64).repeat(8, 1)
    s = M.condition_stats(x)
    # Not exactly 0: the mean of eight identical float64 values can land an ulp
    # off, so v_intra bottoms out around eps^2.
    assert s.v_intra < 1e-28, s.v_intra
    assert s.rho < 1e-14, s.rho
    assert s.min_pair_ratio != s.min_pair_ratio  # nan: every pair distance is 0


def test_pairwise_distance_survives_matmul_threshold():
    """M2 compares hundreds of condition means that may sit very close together.

    Above 25 rows cdist expands ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b, cancelling
    the leading digits when the separation is small next to the norms. That is the
    regime of clustered condition means, so _pdist must not take that path.
    """
    torch.manual_seed(0)
    n, d = 64, 4096  # n > 25 triggers cdist's matmul path
    base = torch.randn(1, d, dtype=torch.float64)
    x = base.repeat(n, 1) + torch.randn(n, d, dtype=torch.float64) * 1e-4
    x32 = x.to(torch.float32)
    truth = M._pdist(x)
    off = ~torch.eye(n, dtype=torch.bool)
    err_mm = (torch.cdist(x32, x32)[off].double() - truth[off]).abs().max().item()
    err_direct = (M._pdist(x32)[off].double() - truth[off]).abs().max().item()
    assert err_direct < err_mm / 10, (err_direct, err_mm)


def test_duplicate_pair_is_flagged_by_min_pair_ratio():
    torch.manual_seed(0)
    x = torch.randn(8, 512, dtype=torch.float64)
    x[3] = x[2]  # exact duplicate -> a 1/d repulsion kernel would blow up
    s = M.condition_stats(x)
    assert s.min_pair_ratio < 1e-9, s.min_pair_ratio


def test_variance_ratio_matches_constructed_split():
    """Build groups with known within/between variance and check R."""
    torch.manual_seed(0)
    n_cond, k, d = 40, 16, 1024
    v_intra_true, v_inter_scale = 0.01, 1.0
    means = torch.randn(n_cond, d, dtype=torch.float64) * v_inter_scale
    stats = []
    for c in range(n_cond):
        x = means[c] + torch.randn(k, d, dtype=torch.float64) * math.sqrt(v_intra_true)
        stats.append(M.condition_stats(x))
    # every condition in its own episode -> all pairs are cross-episode
    out = M.summarise(stats, episode_ids=list(range(n_cond)))
    assert abs(out["v_intra_mean"] - v_intra_true) / v_intra_true < 0.05
    # mean pairwise squared distance is 2x the between-group variance
    assert abs(out["v_inter_cross_episode"] / out["v_inter_anova"] - 2.0) < 0.1
    expected_R = out["v_intra_mean"] / (out["v_intra_mean"] + out["v_inter_cross_episode"])
    assert abs(out["R"] - expected_R) < 1e-12, (out["R"], expected_R)
    # with v_intra 100x below the between-condition spread, R lands near 0.005
    assert 0.003 < out["R"] < 0.008, out["R"]


def test_debiasing_removes_the_finite_K_inflation_of_v_inter():
    """Means estimated from K draws sit 2*v_intra/K too far apart.

    Constructed so the inflation is large: v_intra is the same size as the true
    between-condition spread and K is small, which is exactly when the raw R is
    most misleading.
    """
    torch.manual_seed(0)
    n_cond, k, d = 60, 4, 2048
    v_intra_true, v_between_true = 0.05, 0.05
    means = torch.randn(n_cond, d, dtype=torch.float64) * math.sqrt(v_between_true)
    stats = [
        M.condition_stats(
            means[c] + torch.randn(k, d, dtype=torch.float64) * math.sqrt(v_intra_true)
        )
        for c in range(n_cond)
    ]
    out = M.summarise(stats, episode_ids=list(range(n_cond)))

    # true mean pairwise squared distance is 2 * v_between
    inflation = 2.0 * out["v_intra_mean"] / k
    assert out["v_inter"] > 2 * v_between_true + 0.5 * inflation
    assert abs(out["v_inter_debiased"] - 2 * v_between_true) < 0.1 * 2 * v_between_true
    # removing an inflation of v_inter raises R
    assert out["R_debiased"] > out["R"]
    true_R = v_intra_true / (v_intra_true + 2 * v_between_true)
    assert abs(out["R_debiased"] - true_R) < 0.02, (out["R_debiased"], true_R)


def test_dead_frac_counts_frozen_dimensions():
    torch.manual_seed(0)
    k, d = 8, 1000
    x = torch.randn(k, d, dtype=torch.float64)
    x[:, :250] = x[0, :250]  # a quarter of the dimensions never move
    s = M.condition_stats(x)
    assert abs(s.dead_frac - 0.25) < 1e-12, s.dead_frac
    assert M.condition_stats(torch.randn(k, d, dtype=torch.float64)).dead_frac == 0.0


def test_pr_frac_is_comparable_across_K():
    """PR is capped at K-1, so only the fraction of that ceiling compares."""
    torch.manual_seed(0)
    d = 4096
    for k in (4, 8, 16):
        iso = [M.condition_stats(torch.randn(k, d, dtype=torch.float64))]
        out = M.summarise(iso)
        assert out["PR_ceiling"] == k - 1
        assert out["PR_frac_mean"] > 0.8, (k, out["PR_frac_mean"])

        direction = torch.randn(d, dtype=torch.float64)
        rank1 = [M.condition_stats(torch.randn(k, 1, dtype=torch.float64) * direction)]
        out1 = M.summarise(rank1)
        assert abs(out1["PR_frac_mean"] - 1.0 / (k - 1)) < 1e-6, out1["PR_frac_mean"]


def test_R_anchors_from_plan():
    """The plan quotes R=0.02 -> 14% and R=0.10 -> 33% of marginal std."""
    assert abs(M.rho_from_R(0.02) - 0.1429) < 1e-3
    assert abs(M.rho_from_R(0.10) - 0.3333) < 1e-3


def test_same_and_cross_episode_pairs_are_separated():
    torch.manual_seed(0)
    means = torch.randn(6, 128, dtype=torch.float64)
    episode_ids = [0, 0, 0, 1, 1, 1]
    out = M.inter_condition_variance(means, episode_ids)
    assert out["n_pairs_same_episode"] == 12  # 2 episodes x 3x2 ordered pairs
    assert out["n_pairs_cross_episode"] == 18
    assert out["n_pairs_all"] == 30


def test_noise_floor_and_measurability_gate():
    torch.manual_seed(0)
    k, d = 8, 2048
    floor = M.noise_floor(torch.randn(k, d, dtype=torch.float64) * 1e-3)
    assert abs(floor["v_floor"] - 1e-6) / 1e-6 < 0.1
    stats = [M.condition_stats(torch.randn(k, d, dtype=torch.float64) * 1e-3) for _ in range(5)]
    out = M.summarise(stats, noise_floor=floor["v_floor"], floor_ratio_required=10.0)
    assert not out["measurable"]  # same magnitude as the floor
    big = [M.condition_stats(torch.randn(k, d, dtype=torch.float64) * 1e-2) for _ in range(5)]
    out2 = M.summarise(big, noise_floor=floor["v_floor"], floor_ratio_required=10.0)
    assert out2["measurable"] and out2["floor_ratio"] > 50


def test_trajectory_alignment_rises_from_noise_to_one():
    """Differences start as noise and rotate into the final direction."""
    torch.manual_seed(0)
    t, k, d = 10, 6, 2048
    final = torch.randn(k, d, dtype=torch.float64)
    traj = torch.empty(t, k, d, dtype=torch.float64)
    for i in range(t):
        w = i / (t - 1)  # 0 at high sigma, 1 at the end
        traj[i] = w * final + (1 - w) * torch.randn(k, d, dtype=torch.float64) * 5
    out = M.trajectory_alignment(traj)
    assert abs(out["alignment"][-1] - 1.0) < 1e-12, out["alignment"][-1]
    assert abs(out["residual_to_final"][-1]) < 1e-12
    assert out["alignment"][0] < 0.2, out["alignment"][0]
    # monotone up to sampling noise, and it must cross the halfway mark
    assert out["alignment"][-2] > out["alignment"][1]
    assert M.sigma_star(
        [1.0 - i / (t - 1) for i in range(t)], out["alignment"], 0.5
    ) < 1.0


def test_alignment_is_blind_to_dispersion_magnitude():
    """A trajectory that only shrinks, never rotates, stays fully aligned.

    This is the case the plan's dispersion-based sigma_star mishandles: the spread
    falls by 10x yet the sample identities were fixed from the first step.
    """
    torch.manual_seed(0)
    t, k, d = 8, 5, 512
    base = torch.randn(k, d, dtype=torch.float64)
    traj = torch.stack([base * (10.0 - i) for i in range(t)])
    out = M.trajectory_alignment(traj)
    assert min(out["alignment"]) > 1.0 - 1e-9, min(out["alignment"])
    disp = M.dispersion_trajectory(traj)
    assert disp["dispersion_normalised"][0] > 3.0  # dispersion says the opposite


def test_sigma_star_picks_largest_sigma_above_half():
    sigmas = [1.0, 0.8, 0.6, 0.4, 0.2]
    disp = [0.1, 0.2, 0.55, 0.9, 1.0]
    assert M.sigma_star(sigmas, disp) == 0.6
    disp_early = [0.9, 0.95, 0.98, 0.99, 1.0]
    assert M.sigma_star(sigmas, disp_early) == 1.0


def test_dispersion_trajectory_is_normalised_to_final():
    torch.manual_seed(0)
    traj = torch.stack([torch.randn(8, 256, dtype=torch.float64) * s for s in (0.1, 0.5, 1.0)])
    out = M.dispersion_trajectory(traj)
    assert abs(out["dispersion_normalised"][-1] - 1.0) < 1e-12
    assert out["dispersion_normalised"][0] < out["dispersion_normalised"][1]


def test_bias_is_zero_when_mean_equals_truth():
    torch.manual_seed(0)
    x = torch.randn(9, 512, dtype=torch.float64)
    s = M.condition_stats(x, truth=x.mean(dim=0))
    assert s.bias < 1e-12


# ---------------------------------------------------------------- shapes


def test_camera_layout_tiles_the_grid_exactly():
    area = sum(c.n_cells for c in S.CAMERAS)
    assert area == S.GRID_H * S.GRID_W == 480
    cover = torch.zeros(S.GRID_H, S.GRID_W, dtype=torch.int32)
    for c in S.CAMERAS:
        cover[c.rows, c.cols] += 1
    assert bool((cover == 1).all()), "camera crops must tile without overlap or gap"


def test_split_cameras_matches_pixel_grids():
    x = torch.randn(S.LATENT_CHANNELS, 5, S.GRID_H, S.GRID_W)
    parts = S.split_cameras(x)
    for c in S.CAMERAS:
        assert parts[c.name].shape == (S.LATENT_CHANNELS, 5, c.latent_h, c.latent_w)
        # DINOv3/16 patch grid must coincide with the latent grid
        assert c.pixel_h // S.DINO_PATCH == c.latent_h
        assert c.pixel_w // S.DINO_PATCH == c.latent_w


def test_chunk_indexing_and_frame_ids():
    assert S.n_chunks(5) == 3 and S.n_chunks(4) == 2 and S.n_chunks(21) == 11
    assert S.frame_ids(5, action=False) == [0, 0, 2, 2, 4]
    assert S.frame_ids(5, action=True) == [1, 1, 3, 3, 5]
    assert S.video_frames_of_chunk(1, 5) == [2, 3]
    assert S.video_frames_of_chunk(2, 5) == [4]  # trailing partial chunk


def test_action_pad_frame_is_excluded():
    assert S.action_frame_slice(5) == [1, 2, 3, 4]


def test_latent_frame_count_from_video_frames():
    assert S.expected_latent_frames(17) == 5
    assert S.expected_latent_frames(21) == 6


def test_sequence_length_at_F5():
    sl = S.sequence_length(5)
    assert sl["video"] == 600 and sl["action"] == 80
    assert sl["total"] == 1360


def test_valid_action_channels_match_config_permutation():
    used = list(range(0, 7)) + list(range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
    assert S.VALID_ACTION_CHANNELS == sorted(used)
    assert len(S.VALID_ACTION_CHANNELS) == 16
    assert S.CHANNEL_GROUP[28] == "left_gripper" and S.CHANNEL_GROUP[29] == "right_gripper"
    assert S.CHANNEL_GROUP[0] == "left_arm_trans" and S.CHANNEL_GROUP[7] == "right_arm_trans"


def test_action_norm_roundtrip_and_scales():
    q01 = [-0.0617, -3.67e-05, -0.0878, -1, -1, -1, -1,
           -0.3547, -1.31e-06, -0.1198, -1, -1, -1, -1] + [0.0] * 16
    q99 = [0.3463, 0.3997, 0.1475, 1, 1, 1, 1,
           0.0342, 0.3914, 0.1792, 1, 1, 1, 1] + [0.0] * 14 + [1.0, 1.0]
    norm = S.ActionNorm({"q01": q01, "q99": q99})
    raw = torch.rand(S.ACTION_DIM, 5, 16, 1, dtype=torch.float64) * 0.1
    back = norm.denormalise(
        (raw - norm.q01.view(-1, 1, 1, 1)) / norm.span.view(-1, 1, 1, 1) * 2 - 1
    )
    active = S.VALID_ACTION_CHANNELS
    assert torch.allclose(back[active], raw[active], atol=1e-9)
    # one normalised unit is ~0.20 m of left-arm x but 0.5 of gripper travel
    assert abs(norm.scale[0].item() - 0.2040) < 1e-3
    assert abs(norm.scale[28].item() - 0.5) < 1e-6
    assert abs(norm.scale[3].item() - 1.0) < 1e-6


def test_active_channels_drop_a_frozen_arm():
    torch.manual_seed(0)
    a = torch.zeros(4, S.ACTION_DIM, 5, 16, 1)
    moving = [0, 1, 2, 3, 4, 5, 6, 28]  # left arm + its gripper
    a[:, moving] = torch.randn(4, len(moving), 5, 16, 1)
    assert S.active_action_channels(a) == moving


def _main() -> int:
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fails = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_main())
