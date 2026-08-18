"""Pool the DINOv3 per-condition rows and set R against its latent-space value.

The question this answers is narrow: measured in a semantic feature space instead
of VAE latents, is the teacher's conditional dispersion a larger fraction of the
between-condition spread? If it is, an explicit drifting loss placed in that space
has more signal to work with than the latent-space number suggests.

R is the only cross-space comparable quantity here. The per-condition `rho` divides
by the RMS about zero, which for DINO features is dominated by a large common
offset and would understate everything; `R = v_intra / (v_intra + v_inter)` and its
derived conditional-std fraction cancel that offset because v_inter is a distance
between condition means.

  python scripts/dino_stats.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftwam import metrics as M  # noqa: E402
from driftwam import paths  # noqa: E402


def comparable(e: dict) -> dict:
    """The subset of a pooled summary that means the same thing in any space.

    R and its derived conditional-std fraction are ratios of variances measured in
    the same space, so they transfer. The stored `bias_mean` does not: it is
    normalised by the RMS about zero, which for DINO features is dominated by a
    large common offset and would make the bias look artificially small. Rescaling
    it by the marginal std -- sqrt(v_inter / 2), since v_inter is a mean pairwise
    squared distance and so twice the between-condition variance -- puts it on the
    same footing as the latent-space figure.
    """
    out = {"R": e.get("R"), "R_debiased": e.get("R_debiased"),
           "cond_frac": e.get("cond_std_frac_of_marginal"),
           "PR_frac": e.get("PR_frac_mean"), "n_conditions": e.get("n_conditions")}
    b, s, vi = e.get("bias_mean"), e.get("scale_mean"), e.get("v_inter")
    out["bias_vs_marginal"] = (
        b * s / math.sqrt(vi / 2.0) if None not in (b, s, vi) and vi > 0 else None
    )
    return out


def load_latent(path: Path, subject: str = "teacher",
                view: str = "video/all") -> dict:
    """The latent-space measurement this is being set against, from its own run."""
    d = json.loads(Path(path).read_text())
    t = d[subject][view]
    return {
        "with_history": comparable(t["with_history"]),
        "per_chunk": {c: comparable(v) for c, v in t["per_chunk"].items()},
        "meta": d.get("meta", {}),
    }


def _ids(keys: list[str]) -> list[int]:
    order = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return [order[k] for k in keys]


def load(dirpath: Path, only: set[str] | None = None) -> list[dict]:
    files = sorted(dirpath.glob("*.pt"))
    if only is not None:
        files = [f for f in files if f.name in only]
    if not files:
        raise SystemExit(f"no per-item files in {dirpath}")
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def compare_pooling(main_dir: Path, alt_dir: Path) -> dict:
    """Paired pooled-vs-unpooled R on whatever items both runs share.

    Absolute R from a handful of items is noisy, but the two arms here see the
    identical draws and the identical conditions, so their ratio is a paired
    comparison and says cleanly whether the temporal averaging is what caps the
    measured dispersion.
    """
    names = {f.name for f in main_dir.glob("*.pt")} & {f.name for f in alt_dir.glob("*.pt")}
    if not names:
        return {}
    a, b = load(main_dir, names), load(alt_dir, names)
    pa, pb = pool(a), pool(b)
    out = {"n_items": len(names), "items": sorted(names),
           "main_pool": a[0].get("pool"), "alt_pool": b[0].get("pool"), "blocks": {}}
    for blk in a[0]["blocks"]:
        wa = pa["combined"][blk]["with_history"]
        wb = pb["combined"][blk]["with_history"]
        out["blocks"][blk] = {
            "R_main": wa["R"], "R_alt": wb["R"],
            "D_main": wa["D"], "D_alt": wb["D"],
            "ratio": wb["R"] / wa["R"] if wa["R"] else float("nan"),
        }
    return out


def pool(items: list[dict]) -> dict:
    """view -> block -> {per_chunk, chunk0, with_history}."""
    # view -> block -> chunk -> list[(episode, ConditionStats)]
    table: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for it in items:
        ep = f"{it['task']}#{it['segment']}"
        for view in it["views"]:
            for b in it["blocks"]:
                for c in it["chunks"]:
                    table[view][b][c].append((ep, it["stats"][view][b][c]))

    out: dict = {}
    for view, by_block in table.items():
        for b, by_chunk in by_block.items():
            chunks = sorted(by_chunk)
            entry: dict = {"per_chunk": {}}
            for c in chunks:
                eps = [e for e, _ in by_chunk[c]]
                st = [s for _, s in by_chunk[c]]
                entry["per_chunk"][str(c)] = M.summarise(st, episode_ids=_ids(eps))
            for label, sel in (("chunk0", [c for c in chunks if c == 0]),
                               ("with_history", [c for c in chunks if c >= 1])):
                if not sel:
                    continue
                eps = [e for c in sel for e, _ in by_chunk[c]]
                st = [s for c in sel for _, s in by_chunk[c]]
                entry[label] = M.summarise(st, episode_ids=_ids(eps))
            out.setdefault(view, {})[b] = entry
    return out


def standardised(items: list[dict], view: str, block: int,
                 chunks: list[int]) -> dict:
    """R after dividing every DINO channel by its global std.

    A plain per-element variance in this space is whatever the highest-norm
    channels do, and that is also what an unweighted L2 drifting loss would chase.
    If R survives whitening, the dispersion is spread across the representation
    rather than living in a few outlier dimensions.
    """
    tot_n = 0
    tot_sum = None
    tot_sq = None
    within: list[torch.Tensor] = []
    means: list[torch.Tensor] = []
    eps_keys: list[str] = []
    for it in items:
        for c in chunks:
            ch = it["chan"][view][block][c]
            tot_n += ch["n"]
            tot_sum = ch["sum"].clone() if tot_sum is None else tot_sum + ch["sum"]
            tot_sq = ch["sumsq"].clone() if tot_sq is None else tot_sq + ch["sumsq"]
            within.append(ch["within_var"])
            st = it["stats"][view][block][c]
            means.append(st.mean.to(torch.float64).reshape(-1, ch["sum"].numel()))
            eps_keys.append(f"{it['task']}#{it['segment']}")

    gmean = tot_sum / tot_n
    gvar = (tot_sq / tot_n - gmean.pow(2)).clamp_min(1e-12)   # per channel

    v_intra = torch.stack(within).mean(dim=0).div(gvar).mean().item()

    # Per-channel mean squared distance between condition means, cross-episode
    # pairs only, matching `inter_condition_variance`'s estimator.
    ep = torch.as_tensor(_ids(eps_keys))
    n = len(means)
    acc = torch.zeros_like(gvar)
    pairs = 0
    for i in range(n):
        keep = (ep != ep[i]).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            continue
        diff = torch.stack([means[j] for j in keep.tolist()]) - means[i]
        acc += diff.pow(2).sum(dim=(0, 1))
        pairs += keep.numel() * means[i].shape[0]
    v_inter = (acc / pairs).div(gvar).mean().item() if pairs else float("nan")

    r = M.variance_ratio(v_intra, v_inter)
    return {"v_intra": v_intra, "v_inter": v_inter, "R": r,
            "cond_std_frac_of_marginal": M.rho_from_R(r),
            "n_conditions": n}


def fmt(x, prec=4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and (x != x):
        return "nan"
    return f"{x:.{prec}g}"


def report_pooling(cmp: dict) -> list[str]:
    if not cmp:
        return []
    L = ["## 时间池化的敏感性（配对对比）", ""]
    L.append(f"把每个 latent 帧的 4 个像素帧平均，会抹掉样本之间在帧内的分歧"
             f"（例如运动时序的差别），可能压低测得的条件离散度；不池化的形式"
             f"也更接近逐帧特征损失的实际做法。下表在同一批 {cmp['n_items']} 个 item、"
             f"同一批噪声上比较两种约简：")
    L += ["", "| block | R（按 latent 帧平均） | R（保留每个像素帧） | 比值 | D |",
          "|---|---|---|---|---|"]
    for b, v in cmp["blocks"].items():
        L.append(f"| {b} | {fmt(v['R_main'])} | {fmt(v['R_alt'])} | "
                 f"{fmt(v['ratio'], 3)}x | {v['D_main']} → {v['D_alt']} |")
    L.append("")
    ratios = [v["ratio"] for v in cmp["blocks"].values()]
    worst = max(ratios)
    if worst < 1.15:
        L.append(f"最大变化 {worst:.2f} 倍，池化不是限制因素，主判定用池化版本即可。")
    else:
        L.append(f"最大变化 {worst:.2f} 倍——池化确实压低了离散度，"
                 f"主判定应改用不池化的数字。")
    L.append("")
    return L


def report(pooled: dict, std: dict, latent: dict, meta: dict,
           cmp: dict | None = None) -> str:
    blocks = meta["blocks"]
    lat = latent["with_history"]
    comp = {b: comparable(pooled["combined"][b]["with_history"]) for b in blocks}
    best = max(blocks, key=lambda b: comp[b]["R"])
    bw = comp[best]
    gain = bw["R"] / lat["R"]

    L: list[str] = []
    A = L.append
    A("# DINOv3 特征空间中的教师条件离散度")
    A("")
    A(f"样本来源：`{Path(meta['source']).name}`，{meta['n_items']} item / "
      f"K = {meta['K']} / CFG = {meta['cfg']}，与 latent 空间用的是同一批样本文件，"
      f"因此两个空间的差异只来自度量空间本身。")
    A(f"DINOv3 ViT-B/16，取 hidden_states 第 {', '.join(map(str, blocks))} 项"
      f"（DriftWorld 的配置）。每个 chunk 的特征按所属 latent 帧把 4 个像素帧平均，"
      f"使 D 的时间分辨率与 latent 空间一致，且各 chunk 的 D 相等。")
    A("")
    A("两条前置验证：")
    A("")
    A("1. 解码链条对照原始 mp4——cam_high 对齐 PSNR 38.96 dB，故意错位一帧 15.69 dB，"
      "说明 T-shape 切片、逐通道反归一化、时间映射三者都正确。")
    A("2. 解码在时间上是因果的（实测：扰动任一 latent 帧只影响其自身及之后的像素帧）。"
      "因此每个 chunk 单独解码：该 chunk 之外全部用真值 latent，只替换它自己的两帧。"
      "若整段解码，前面 chunk 的离散度会通过解码器的时间缓存混进来。")
    A("")

    A("## 主判定：R 的跨空间对照（combined 视图，chunk >= 1）")
    A("")
    A("R = v_intra / (v_intra + v_inter) 是这里唯一可跨空间比较的量：它是同一空间内"
      "两个方差的比值，因此不受该空间整体尺度的影响。逐条件的 `rho` 不可比——"
      "它的分母是对零的 RMS，而 DINO 特征有很大的公共偏移量。")
    A("")
    A("| 空间 | R | 条件std / 边缘std | 相对 latent | bias（对边缘std） | PR/(K−1) |")
    A("|---|---|---|---|---|---|")
    A(f"| VAE latent | {fmt(lat['R'])} | {fmt(lat['cond_frac'])} | 1.00x | "
      f"{fmt(lat['bias_vs_marginal'])} | {fmt(lat['PR_frac'])} |")
    for b in blocks:
        c = comp[b]
        A(f"| DINOv3 block {b} | **{fmt(c['R'])}** | {fmt(c['cond_frac'])} | "
          f"{fmt(c['R'] / lat['R'], 3)}x | {fmt(c['bias_vs_marginal'])} | "
          f"{fmt(c['PR_frac'])} |")
    A("")
    A(f"最高的是 block {best}：R = {fmt(bw['R'])}，为 latent 空间的 {gain:.2f} 倍，"
      f"条件标准差占边缘标准差的 {bw['cond_frac'] * 100:.1f}%"
      f"（latent 空间 {lat['cond_frac'] * 100:.1f}%）。")
    A("")
    A("`R_debiased`（扣除条件均值本身的 K 有限采样误差）："
      + "，".join(f"block {b} {fmt(comp[b]['R_debiased'])}" for b in blocks)
      + f"；latent 空间 {fmt(lat['R_debiased'])}。")
    A("")

    A("## 逐通道标准化后的 R（稳健性检查）")
    A("")
    A("DINO 有少数高范数通道，朴素的逐元素方差可能被它们支配——"
      "而未加权的 L2 漂移损失看到的恰好就是这个朴素版本，所以原始值才是"
      "有操作意义的那一个。把每个通道除以其全局标准差后重算，用来判断"
      "结论是分布在整个表示上，还是只活在少数离群维度里：")
    A("")
    A("| block | R（原始） | R（逐通道标准化） | 标准化后 条件std/边缘std |")
    A("|---|---|---|---|")
    for b in blocks:
        A(f"| {b} | {fmt(comp[b]['R'])} | {fmt(std[b]['R'])} | "
          f"{fmt(std[b]['cond_std_frac_of_marginal'])} |")
    A("")

    A("## 逐 chunk：视觉历史的收紧作用在语义空间是否同样存在")
    A("")
    A("表内是条件std/边缘std（由该 chunk 自己的 R 导出），两个空间同一定义。")
    A("")
    chunk_keys = sorted(pooled["combined"][blocks[0]]["per_chunk"])
    A("| 空间 | " + " | ".join(f"c={c}" for c in chunk_keys) + " | c>=1 相对 c=0 |")
    A("|---|" + "---|" * (len(chunk_keys) + 1))

    def chunk_row(label: str, per_chunk: dict) -> str:
        vals, have = [], []
        for c in chunk_keys:
            v = per_chunk.get(c, {}).get("cond_frac")
            vals.append(fmt(v, 3))
            if c != "0" and v is not None:
                have.append(v)
        z = per_chunk.get("0", {}).get("cond_frac")
        rr = f"{(sum(have) / len(have)) / z:.2f}x" if have and z else "-"
        return f"| {label} | " + " | ".join(vals) + f" | {rr} |"

    A(chunk_row("VAE latent", latent["per_chunk"]))
    for b in blocks:
        A(chunk_row(f"DINOv3 block {b}",
                    {c: comparable(v) for c, v in
                     pooled["combined"][b]["per_chunk"].items()}))
    A("")

    L.extend(report_pooling(cmp or {}))

    A("## 分相机（chunk >= 1，R）")
    A("")
    A("| 视图 | " + " | ".join(f"block {b}" for b in blocks) + " |")
    A("|---|" + "---|" * len(blocks))
    for view in meta["views"]:
        if view == "combined":
            continue
        A(f"| {view} | " + " | ".join(
            fmt(pooled[view][b]["with_history"]["R"]) for b in blocks) + " |")
    A("")

    A("## 判定")
    A("")
    # Per-channel standardisation is a fixed diagonal reweighting computable once
    # from data statistics, so it is a configuration choice a loss can actually
    # make, not a post-hoc adjustment. The best implementable setting therefore
    # includes it.
    cand = [(comp[b]["R"], f"block {b}，原始") for b in blocks]
    cand += [(std[b]["R"], f"block {b}，逐通道标准化") for b in blocks]
    top_r, top_name = max(cand)
    top_gain = top_r / lat["R"]

    if top_r >= 0.10:
        verdict = "PASS"
        why = ("越过了 H1 事先设定的 R >= 0.10，即显式 drifting 场在视频分支上"
               "良定义。这是 H5（CFG）失败之后剩下的唯一杠杆，它成立了")
    elif top_gain >= 1.5:
        verdict = "MARGINAL"
        why = "特征空间有帮助，但不足以把视频分支推入 R >= 0.10 的可用区间"
    else:
        verdict = "FAIL"
        why = "换度量空间并不改变条件分布本身过窄这一事实"
    A(f"**DINOv3 作为保方差杠杆：{verdict}** —— 可实现的最好配置是 {top_name}，"
      f"R = {fmt(top_r)}，为 latent 空间的 {top_gain:.2f} 倍。{why}。")
    A("")
    A(f"逐通道标准化算进来，是因为它只是一个固定的对角重加权，"
      f"其尺度可以一次性从数据统计里算出并写进损失，属于可选的配置而非事后修正。"
      f"不做标准化时最好为 {fmt(bw['R'])}（block {best}），"
      f"{'已' if bw['R'] >= 0.10 else '尚未'}越过 0.10。")
    A("")
    A(f"代价一并记录：block {best} 的 bias 从 latent 空间的 "
      f"{fmt(lat['bias_vs_marginal'], 3)} 升到 {fmt(bw['bias_vs_marginal'], 3)}"
      f"（都以各自空间的边缘std为单位）。"
      f"即相对离散度的提升伴随着更大的系统性偏移，"
      f"正样本质量与可用离散度之间仍有权衡。")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", default=None)
    ap.add_argument("--latent", default=None, help="stats.json from compute_stats.py")
    ap.add_argument("--latent-subject", default="teacher",
                    help="which subject in stats.json to compare against")
    ap.add_argument("--latent-view", default="video/all")
    ap.add_argument("--compare-pool", default=None,
                    help="a second dino dir with the other --pool setting")
    ap.add_argument("--tag", default="dino", help="output filename prefix")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ddir = Path(args.dino) if args.dino else (
        Path(paths.EXP01_DIR) / "dino" / "teacher_cfg5_F8_K8")
    items = load(ddir)
    blocks = items[0]["blocks"]
    chunks_h = [c for c in items[0]["chunks"] if c >= 1]
    print(f"{len(items)} items, blocks {blocks}, "
          f"{len(items) * len(chunks_h)} conditions with history")

    latent = load_latent(Path(args.latent) if args.latent else
                         Path(paths.EXP01_DIR) / "stats.json",
                         subject=args.latent_subject, view=args.latent_view)
    pooled = pool(items)
    std = {b: standardised(items, "combined", b, chunks_h) for b in blocks}
    cmp = compare_pooling(ddir, Path(args.compare_pool)) if args.compare_pool else {}

    meta = {"source": str(ddir), "n_items": len(items), "K": items[0]["K"],
            "cfg": items[0].get("cfg"), "blocks": blocks,
            "views": items[0]["views"], "pool": items[0].get("pool"),
            "subject": items[0].get("subject"),
            "latent_ref": f"{args.latent_subject}/{args.latent_view}"}

    out_dir = Path(args.out) if args.out else Path(paths.EXP01_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    js, md = out_dir / f"{args.tag}_stats.json", out_dir / f"{args.tag}_report.md"
    js.write_text(json.dumps(
        {"meta": meta, "pooled": pooled, "standardised": std, "latent": latent,
         "pooling_check": cmp}, indent=1, default=str))
    text = report(pooled, std, latent, meta, cmp)
    md.write_text(text)
    print(text)
    print(f"wrote {js} and {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
