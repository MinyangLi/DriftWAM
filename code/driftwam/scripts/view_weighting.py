"""Can weighting the three camera views raise R another notch?

There is an exact answer, and it is a bound rather than a number.

Both pooled variances are per-element means over the concatenated feature vector,
and a squared distance splits across a concatenation, so for per-view weights w_c:

    v_intra(w) = sum_c w_c^2 D_c vi_c / sum_c w_c^2 D_c
    v_inter(w) = sum_c w_c^2 D_c ve_c / sum_c w_c^2 D_c
    R(w)       = sum_c b_c R_c / sum_c b_c ,  b_c = w_c^2 D_c (vi_c + ve_c)

So R(w) is a *convex combination* of the per-view R_c. It can never leave
[min_c R_c, max_c R_c] no matter how the weights are chosen: view weighting can
only slide R between the worst and best single view, and the ceiling is the best
view on its own. The script verifies the decomposition against the stored combined
numbers, then reports where each candidate weighting lands.

Two follow-ups, both CPU-only:

* the same bound under per-channel whitening, which needs per-view whitened R.
  Computed by streaming accumulation using
  sum_{i!=j} ||m_i - m_j||^2 = 2n sum_i ||m_i||^2 - 2||sum_i m_i||^2,
  which avoids ever forming an (n, n, D) tensor;
* whether the wrist views' higher R is independent diversity or just geometric
  amplification of the same small trajectory differences. If it is amplification,
  per-condition wrist dispersion should track action dispersion more closely than
  the overhead view's does, and re-weighting toward the wrists would be buying a
  larger number rather than more diversity.

  python scripts/view_weighting.py
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import sys
from pathlib import Path

import torch


def _rss_mb() -> float:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * 4096 / 2**20

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Deliberately no `bootstrap.setup()`: nothing here touches a model, and pulling in
# the Flash-WAM stack costs a few hundred MB of imports. In no-GPU mode the container
# is capped at 2 GB with most of it already spoken for, so the difference decides
# whether this script runs at all.
from driftwam import metrics as M  # noqa: E402
from driftwam import paths  # noqa: E402
from driftwam import shapes as S  # noqa: E402

torch.set_num_threads(1)

CAMS = [c.name for c in S.CAMERAS]


def decomposition_check(pooled: dict, block: str) -> dict:
    """The additive split must reproduce the stored combined figures exactly."""
    per = {c: pooled[c][block]["with_history"] for c in CAMS}
    comb = pooled["combined"][block]["with_history"]
    d = {c: per[c]["D"] for c in CAMS}
    tot = sum(d.values())
    vi = sum(d[c] * per[c]["v_intra_mean"] for c in CAMS) / tot
    ve = sum(d[c] * per[c]["v_inter"] for c in CAMS) / tot
    return {
        "D_sum": tot, "D_stored": comb["D"],
        "v_intra_rebuilt": vi, "v_intra_stored": comb["v_intra_mean"],
        "v_inter_rebuilt": ve, "v_inter_stored": comb["v_inter"],
        "R_rebuilt": M.variance_ratio(vi, ve), "R_stored": comb["R"],
        "rel_err_R": abs(M.variance_ratio(vi, ve) - comb["R"]) / comb["R"],
    }


def weighted_R(vi: dict, ve: dict, d: dict, w: dict) -> float:
    num = sum(w[c] ** 2 * d[c] * vi[c] for c in w)
    den = sum(w[c] ** 2 * d[c] * (vi[c] + ve[c]) for c in w)
    return num / den if den else float("nan")


def weighting_scan(vi: dict, ve: dict, d: dict) -> dict:
    """Candidate weightings, plus the bound they all obey."""
    r_each = {c: M.variance_ratio(vi[c], ve[c]) for c in CAMS}
    wrists = [c for c in CAMS if "wrist" in c]
    best = max(r_each, key=r_each.get)

    schemes = {
        # plain concatenation: cam_high dominates simply by having 4x the patches
        "concat (as measured)": {c: 1.0 for c in CAMS},
        # equal contribution per camera regardless of patch count
        "equal per camera": {c: d[CAMS[0]] ** 0.5 / d[c] ** 0.5 for c in CAMS},
        "wrists only": {c: (1.0 if c in wrists else 0.0) for c in CAMS},
        "cam_high only": {c: (1.0 if c == "cam_high" else 0.0) for c in CAMS},
        f"{best} only (ceiling)": {c: (1.0 if c == best else 0.0) for c in CAMS},
    }
    out = {"R_per_view": r_each,
           "bound": [min(r_each.values()), max(r_each.values())],
           "schemes": {}}
    base = weighted_R(vi, ve, d, schemes["concat (as measured)"])
    for name, w in schemes.items():
        r = weighted_R(vi, ve, d, {c: v for c, v in w.items() if v > 0})
        out["schemes"][name] = {"R": r, "vs_concat": r / base}
    return out


class Accum:
    """Streaming accumulators for one (view, block).

    Everything here is O(positions x channels) or smaller, so the 4.8 GB of feature
    files can be walked one at a time. The container has a 2 GB memory limit, which
    rules out holding the condition means for all episodes at once.
    """

    def __init__(self) -> None:
        self.sq = None          # (C,)   sum_i sum_p m_i[p, ch]^2
        self.sum = None         # (P, C) sum_i m_i[p, ch]
        self.same = None        # (C,)   within-episode pair sums, to be removed
        self.within: list[torch.Tensor] = []
        self.n = 0
        self.n_same_pairs = 0
        self.n_pos = 0

    def add_episode(self, p: dict) -> None:
        e_sq, e_sum, g = p["e_sq"], p["e_sum"], p["g"]
        self.n_pos = e_sum.shape[0]
        self.sq = e_sq if self.sq is None else self.sq + e_sq
        self.sum = e_sum if self.sum is None else self.sum + e_sum
        # same identity applied inside the episode, so those pairs can be subtracted
        s = 2 * g * e_sq - 2 * e_sum.pow(2).sum(dim=0)
        self.same = s if self.same is None else self.same + s
        self.n += g
        self.n_same_pairs += g * (g - 1)
        self.within.extend(p["within"])

    def inter_per_channel(self) -> tuple[torch.Tensor, int]:
        total = 2 * self.n * self.sq - 2 * self.sum.pow(2).sum(dim=0)
        n_cross = self.n * (self.n - 1) - self.n_same_pairs
        return (total - self.same) / (n_cross * self.n_pos), n_cross

    def within_per_channel(self) -> torch.Tensor:
        return torch.stack(self.within).mean(dim=0)


def episode_partials(path: Path, blocks: list[int], chunks_min: int = 1) -> dict:
    """Reduce one feature file to the few small quantities the accumulators need.

    Each file is ~260 MB and the container is capped at 2 GB with roughly 300 MB
    free, so this runs in a throwaway child process: the mapped pages are only
    guaranteed to be handed back when the process exits.
    """
    it = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    chunks = [c for c in it["chunks"] if c >= chunks_min]
    out: dict = {"ep": f"{it['task']}#{it['segment']}", "chunks": chunks,
                 "views": {}, "gchan": {}, "v_intra": {}}
    for b in blocks:
        ch = it["chan"]["combined"][b]
        n = sum(ch[c]["n"] for c in chunks)
        out["gchan"][b] = (n,
                           sum(ch[c]["sum"] for c in chunks).to(torch.float64),
                           sum(ch[c]["sumsq"] for c in chunks).to(torch.float64))
        for view in CAMS:
            nc = it["chan"][view][b][chunks[0]]["sum"].numel()
            e_sq = e_sum = None
            for c in chunks:
                m = it["stats"][view][b][c].mean.to(torch.float64).reshape(-1, nc)
                sq = m.pow(2).sum(dim=0)
                e_sq = sq if e_sq is None else e_sq + sq
                e_sum = m.clone() if e_sum is None else e_sum + m
                del m
                out["v_intra"].setdefault(c, {})[f"{view}_b{b}"] = \
                    it["stats"][view][b][c].v_intra
            out["views"][(view, b)] = {
                "e_sq": e_sq, "e_sum": e_sum, "g": len(chunks),
                "within": [it["chan"][view][b][c]["within_var"].to(torch.float64)
                           for c in chunks],
            }
    del it
    gc.collect()
    return out


def _child(path: str, blocks: list[int], out: str) -> None:
    torch.save(episode_partials(Path(path), blocks), out)


def _in_subprocess(path: Path, blocks: list[int], scratch: Path) -> dict:
    """Run `episode_partials` in a child and hand the result back through a file.

    A multiprocessing queue would pass the tensors as shared-memory descriptors,
    which stop being readable once the child exits -- and the child exiting is the
    whole point here.
    """
    tmp = scratch / f"{path.stem}.partial.pt"
    ctx = mp.get_context("fork")
    p = ctx.Process(target=_child, args=(str(path), blocks, str(tmp)))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"worker for {path.name} exited {p.exitcode} "
                           "(137 means it hit the memory cap)")
    try:
        return torch.load(tmp, map_location="cpu", weights_only=False)
    finally:
        tmp.unlink(missing_ok=True)


def stream_features(ddir: Path, blocks: list[int], scratch: Path,
                    chunks_min: int = 1, isolate: bool = True) -> dict:
    """One pass over the feature files, accumulating everything needed.

    Uses `sum_{i!=j} (a_i - a_j)^2 = 2n sum a^2 - 2 (sum a)^2` so no (n, n, D)
    tensor is ever formed and no more than one episode is ever resident.
    """
    acc: dict[tuple[str, int], Accum] = {}
    gsum: dict[int, torch.Tensor] = {}
    gsq: dict[int, torch.Tensor] = {}
    gn: dict[int, int] = {}
    per_cond: dict[tuple[str, int], dict] = {}
    files = sorted(ddir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no feature files in {ddir}")

    for i, f in enumerate(files):
        p = (_in_subprocess(f, blocks, scratch) if isolate
             else episode_partials(f, blocks, chunks_min))
        for b in blocks:
            n, s, sq = p["gchan"][b]
            gn[b] = gn.get(b, 0) + n
            gsum[b] = s if b not in gsum else gsum[b] + s
            gsq[b] = sq if b not in gsq else gsq[b] + sq
        for key, part in p["views"].items():
            acc.setdefault(key, Accum()).add_episode(part)
        for c, row in p["v_intra"].items():
            per_cond[(p["ep"], c)] = row
        del p
        gc.collect()
        print(f"  [{i + 1}/{len(files)}] rss {_rss_mb():.0f} MB", flush=True)

    gvar = {b: (gsq[b] / gn[b] - (gsum[b] / gn[b]).pow(2)).clamp_min(1e-12)
            for b in blocks}
    return {"acc": acc, "gvar": gvar, "per_cond": per_cond, "n_items": len(files)}


def whitened_per_view(st: dict, block: int) -> dict:
    """Per-view R after dividing each channel by its global std.

    The scale is the one the combined view uses, shared across cameras, so the
    per-view numbers stay on the same footing as the headline whitened figure.
    """
    gvar = st["gvar"][block]
    out = {}
    for view in CAMS:
        a = st["acc"][(view, block)]
        inter_ch, n_cross = a.inter_per_channel()
        within = a.within_per_channel()
        vi_w = (within / gvar).mean().item()
        ve_w = (inter_ch / gvar).mean().item()
        out[view] = {
            "v_intra_white": vi_w, "v_inter_white": ve_w,
            "R_white": M.variance_ratio(vi_w, ve_w),
            # unwhitened rebuild from the same stream, as a check on the identity
            "v_inter_raw_rebuilt": inter_ch.mean().item(),
            "v_intra_raw_rebuilt": within.mean().item(),
            "n_cross_pairs": n_cross, "n_conditions": a.n,
        }
    return {"per_view": out, "gvar_channels": int(gvar.numel())}


def action_coupling(st: dict, sample_dir: Path, block: int) -> dict:
    """Does per-condition wrist dispersion track action dispersion?"""
    dino = {k: {v: row[f"{v}_b{block}"] for v in CAMS}
            for k, row in st["per_cond"].items()}

    act: dict[tuple[str, int], float] = {}
    for f in sorted(sample_dir.glob("*.pt")):
        rec = torch.load(f, map_location="cpu")
        a = rec["action"].float()
        key0 = f"{rec['task']}#{rec['segment']}"
        for c in range(S.n_chunks(rec["F"])):
            frames = [fr for fr in S.video_frames_of_chunk(c, rec["F"])
                      if fr >= S.ACTION_PAD_FRAMES]
            if not frames:
                continue
            x = a[:, S.VALID_ACTION_CHANNELS][:, :, frames]
            act[(key0, c)] = M.condition_stats(x).v_intra
        del rec, a

    keys = [k for k in dino if k in act]
    if len(keys) < 8:
        return {}
    av = torch.tensor([act[k] for k in keys], dtype=torch.float64)
    out = {"n": len(keys)}
    for v in CAMS:
        dv = torch.tensor([dino[k][v] for k in keys], dtype=torch.float64)
        out[v] = {"pearson": _corr(dv, av), "spearman": _corr(_rank(dv), _rank(av))}
    return out


def _rank(x: torch.Tensor) -> torch.Tensor:
    r = torch.empty_like(x)
    r[x.argsort()] = torch.arange(x.numel(), dtype=x.dtype)
    return r


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return (a @ b / den).item() if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=str(Path(paths.EXP01_DIR) / "dino_nopool_stats.json"))
    ap.add_argument("--dino", default=str(Path(paths.EXP01_DIR) / "dino" /
                                          "teacher_cfg5_F8_K8_nopool"))
    ap.add_argument("--samples", default=str(Path(paths.EXP01_DIR) / "samples" /
                                             "teacher_cfg5_F8_K8"))
    ap.add_argument("--blocks", default="2,8")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="only the analytic part, no reading of the feature files")
    ap.add_argument("--scratch", default="/root/autodl-tmp/wam/scratch",
                    help="local disk for the per-episode partial sums")
    ap.add_argument("--out", default=str(Path(paths.EXP01_DIR) / "view_weighting.json"))
    args = ap.parse_args()

    stats = json.loads(Path(args.stats).read_text())
    pooled = stats["pooled"]
    blocks = [int(b) for b in args.blocks.split(",")]
    result: dict = {"source": args.stats, "blocks": blocks}

    for b in blocks:
        key = str(b)
        chk = decomposition_check(pooled, key)
        per = {c: pooled[c][key]["with_history"] for c in CAMS}
        vi = {c: per[c]["v_intra_mean"] for c in CAMS}
        ve = {c: per[c]["v_inter"] for c in CAMS}
        d = {c: per[c]["D"] for c in CAMS}
        scan = weighting_scan(vi, ve, d)
        result[key] = {"decomposition": chk, "scan": scan}

        print(f"\n=== block {b} ===")
        print(f"decomposition check: R rebuilt {chk['R_rebuilt']:.6g} vs stored "
              f"{chk['R_stored']:.6g}  (rel err {chk['rel_err_R']:.2e})")
        print(f"{'view':18s} {'D':>9s} {'v_intra':>9s} {'v_inter':>9s} {'R':>9s}")
        for c in CAMS:
            print(f"{c:18s} {d[c]:9d} {vi[c]:9.4f} {ve[c]:9.4f} "
                  f"{scan['R_per_view'][c]:9.5f}")
        lo, hi = scan["bound"]
        print(f"\nR is a convex combination of those -> any weighting lands in "
              f"[{lo:.5f}, {hi:.5f}]")
        print(f"{'weighting':26s} {'R':>9s} {'vs concat':>10s}")
        for name, v in scan["schemes"].items():
            print(f"{name:26s} {v['R']:9.5f} {v['vs_concat']:9.3f}x")

    if not args.skip_heavy:
        print(f"\nstreaming feature files from {args.dino} ...", flush=True)
        scratch = Path(args.scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        st = stream_features(Path(args.dino), blocks, scratch)

        for b in blocks:
            w = whitened_per_view(st, b)
            result[str(b)]["whitened"] = w
            scan_w = weighting_scan(
                {c: w["per_view"][c]["v_intra_white"] for c in CAMS},
                {c: w["per_view"][c]["v_inter_white"] for c in CAMS},
                {c: pooled[c][str(b)]["with_history"]["D"] for c in CAMS},
            )
            result[str(b)]["scan_whitened"] = scan_w

            print(f"\n=== block {b}, per-channel whitened ===")
            print(f"{'view':18s} {'R raw':>9s} {'R white':>9s} "
                  f"{'v_inter rebuilt':>16s} {'v_inter stored':>15s}")
            for c in CAMS:
                v = w["per_view"][c]
                print(f"{c:18s} {result[str(b)]['scan']['R_per_view'][c]:9.5f} "
                      f"{v['R_white']:9.5f} {v['v_inter_raw_rebuilt']:16.4f} "
                      f"{pooled[c][str(b)]['with_history']['v_inter']:15.4f}")
            lo, hi = scan_w["bound"]
            print(f"whitened: any weighting lands in [{lo:.5f}, {hi:.5f}]; "
                  f"combined whitened (stored) {stats['standardised'][str(b)]['R']:.5f}")
            print(f"{'weighting':26s} {'R':>9s} {'vs concat':>10s}")
            for name, v in scan_w["schemes"].items():
                print(f"{name:26s} {v['R']:9.5f} {v['vs_concat']:9.3f}x")

        print("\nreading action samples for the coupling test ...", flush=True)
        cpl = action_coupling(st, Path(args.samples), blocks[-1])
        result["action_coupling"] = cpl
        if cpl:
            print(f"per-condition correlation of DINO v_intra with action v_intra "
                  f"(block {blocks[-1]}, n = {cpl['n']}):")
            print(f"{'view':18s} {'pearson':>9s} {'spearman':>9s}")
            for c in CAMS:
                print(f"{c:18s} {cpl[c]['pearson']:9.3f} {cpl[c]['spearman']:9.3f}")

    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
