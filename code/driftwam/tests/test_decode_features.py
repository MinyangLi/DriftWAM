"""Self-tests for the pixel/feature path, on data with known answers.

These cover the parts that no downstream number would reveal as wrong: an
off-by-one in the latent-to-pixel frame mapping, a swap batch that leaks frames it
should not, or a temporal pooling that averages across the wrong boundary would all
produce a perfectly plausible variance ratio.

Run: python -m pytest tests/ -q     (or: python tests/test_decode_features.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import decode as D  # noqa: E402
from driftwam import metrics as M  # noqa: E402
from driftwam import shapes as S  # noqa: E402


def test_chunked_pdist_matches_the_direct_form():
    """The wide-input fallback must be a memory trick, not a different answer.

    Agreement is to a few ulps rather than bitwise: both branches subtract
    directly, but they accumulate the sum of squares in a different order.
    """
    torch.manual_seed(0)
    x = torch.randn(12, 500, dtype=torch.float64)
    direct = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")
    saved = M._PDIST_ELEM_BUDGET
    try:
        M._PDIST_ELEM_BUDGET = 0          # force the row-by-row branch
        chunked = M._pdist(x)
    finally:
        M._PDIST_ELEM_BUDGET = saved
    rel = ((direct - chunked).abs() / direct.clamp_min(1e-300)).max()
    assert rel < 1e-14, rel


def test_chunked_pdist_keeps_precision_on_near_duplicates():
    """The reason the direct form exists at all: tiny separations, large norms."""
    torch.manual_seed(0)
    base = torch.randn(1, 4096, dtype=torch.float64) * 1e3
    x = base.repeat(8, 1)
    x[1:] += torch.randn(7, 4096, dtype=torch.float64) * 1e-9
    saved = M._PDIST_ELEM_BUDGET
    try:
        M._PDIST_ELEM_BUDGET = 0
        chunked = M._pdist(x)
    finally:
        M._PDIST_ELEM_BUDGET = saved
    expect = (x[1] - x[2]).norm()
    assert abs(chunked[1, 2] - expect) < 1e-15 * expect, (chunked[1, 2], expect)


def test_pixel_frame_mapping_tiles_the_sequence_exactly_once():
    n_lat = 8
    seen: list[int] = []
    for f in range(n_lat):
        seen.extend(D.pixel_frames_of_latent_frame(f))
    assert seen == list(range(D.n_pixel_frames(n_lat))), seen
    # Frame 0 stands alone; every later frame covers the full temporal ratio.
    assert len(D.pixel_frames_of_latent_frame(0)) == 1
    assert len(D.pixel_frames_of_latent_frame(3)) == D.VAE_TEMPORAL_RATIO


def test_chunk_pixel_frames_partition_by_chunk():
    n_lat = 8
    all_frames: list[int] = []
    for c in range(S.n_chunks(n_lat)):
        all_frames.extend(D.pixel_frames_of_chunk(c, n_lat))
    assert sorted(all_frames) == all_frames
    assert all_frames == list(range(D.n_pixel_frames(n_lat)))


def test_splice_history_touches_only_its_own_chunk():
    torch.manual_seed(0)
    k, c_ch, n_lat = 4, 6, 8
    sample = torch.randn(k, c_ch, n_lat, 4, 5)
    truth = torch.randn(c_ch, n_lat, 4, 5)
    for chunk in range(S.n_chunks(n_lat)):
        out = D.splice_history(sample, truth, chunk)
        own = S.video_frames_of_chunk(chunk, n_lat)
        for f in range(n_lat):
            if f in own:
                assert torch.equal(out[:, :, f], sample[:, :, f]), (chunk, f)
            else:
                # every draw must carry identical history, or the history itself
                # would contribute dispersion
                assert torch.equal(out[:, :, f], truth[None, :, f].expand(k, -1, -1, -1)), (chunk, f)


def test_swap_batch_has_one_row_per_draw_plus_one_truth_row():
    torch.manual_seed(0)
    k, n_lat = 8, 8
    sample = torch.randn(k, 6, n_lat, 4, 5)
    truth = torch.randn(6, n_lat, 4, 5)
    chunks = [0, 1, 2, 3]
    batch, labels = D.build_swap_batch(sample, truth, chunks)
    assert batch.shape[0] == len(chunks) * k + 1 == len(labels)
    assert labels[-1] == (-1, -1)
    assert torch.equal(batch[-1], truth)
    # rows are grouped by chunk, draws in order, which is what the decode loop
    # relies on to label its outputs
    assert labels[:k] == [(0, i) for i in range(k)]
    for row, (c, draw) in enumerate(labels[:-1]):
        own = S.video_frames_of_chunk(c, n_lat)
        assert torch.equal(batch[row][:, own[0]], sample[draw][:, own[0]]), row


def test_swap_rows_differ_from_truth_only_inside_the_chunk():
    torch.manual_seed(0)
    k, n_lat = 3, 8
    truth = torch.zeros(6, n_lat, 4, 5)
    sample = torch.ones(k, 6, n_lat, 4, 5)
    batch, labels = D.build_swap_batch(sample, truth, [1, 2])
    for row, (c, _) in enumerate(labels[:-1]):
        nonzero = [f for f in range(n_lat) if batch[row][:, f].abs().sum() > 0]
        assert nonzero == S.video_frames_of_chunk(c, n_lat), (row, nonzero)


def test_pooling_averages_within_a_latent_frame_not_across():
    from extract_dino import chunk_groups, pool_to_latent_frames  # noqa: PLC0415

    n_lat = 8
    # chunk 0 owns latent frames 0 and 1 -> 1 + 4 pixel frames
    assert chunk_groups(0, n_lat) == [1, 4]
    assert chunk_groups(2, n_lat) == [4, 4]

    feats = torch.zeros(5, 2, 2, 3)
    feats[0] = 1.0                      # latent frame 0
    feats[1:5] = torch.tensor([2.0, 4.0, 6.0, 8.0]).view(4, 1, 1, 1)
    pooled = pool_to_latent_frames(feats, [1, 4])
    assert pooled.shape == (2, 2, 2, 3)
    assert torch.allclose(pooled[0], torch.ones(2, 2, 3))
    assert torch.allclose(pooled[1], torch.full((2, 2, 3), 5.0))


def test_dino_input_normalisation_maps_the_endpoints():
    from driftwam import features as FT  # noqa: PLC0415

    x = torch.tensor([-1.0, 1.0]).view(1, 1, 1, 2).expand(1, 3, 1, 2).contiguous()
    out = FT.to_dino_input(x)
    mean = torch.tensor(FT.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(FT.IMAGENET_STD).view(1, 3, 1, 1)
    assert torch.allclose(out[..., 0], (-mean / std).squeeze(-1), atol=1e-6)
    assert torch.allclose(out[..., 1], ((1 - mean) / std).squeeze(-1), atol=1e-6)


def test_patch_grid_equals_latent_grid_for_every_camera():
    """The reason for choosing patch16 over DINOv2's patch14."""
    from driftwam import features as FT  # noqa: PLC0415

    for cam in S.CAMERAS:
        assert FT.expected_patch_grid(cam) == FT.latent_grid_of(cam), cam.name


def test_channel_standardisation_leaves_a_variance_ratio_alone():
    """Whitening rescales dimensions; on already-isotropic data R must not move."""
    from driftwam import features as FT  # noqa: PLC0415

    torch.manual_seed(0)
    x = torch.randn(64, 4, 4, 8, dtype=torch.float64)
    w = FT.standardise_channels(x)
    per_channel = w.reshape(-1, 8).std(dim=0)
    assert torch.allclose(per_channel, torch.ones(8, dtype=torch.float64), atol=1e-12)


def _run_all() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    raise SystemExit(_run_all())
