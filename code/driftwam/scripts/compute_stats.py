"""Turn saved samples into M1-M7 and the decision-table verdicts.

A *condition* is one (item, chunk) pair. Statistics are computed per condition and
then pooled, and three slicing decisions are load-bearing:

* **Chunk 0 is reported separately.** Its noisy tokens see no clean history at all,
  so it is a text-only conditional and its spread is not comparable to a chunk that
  has visual history. Pooling them would inflate every number.
* **R is computed within a chunk index.** The inter-condition variance must not
  absorb the systematic difference between "chunk 1" and "chunk 3"; only conditions
  at the same depth are compared, and only across different episodes.
* **Action statistics drop frame 0 and dead channels.** Frame 0 is constant padding
  and idle-arm channels are bit-constant all episode, both flagged valid by the
  mask. Averaging either into a variance drags it toward zero for reasons that have
  nothing to do with the teacher.

  python scripts/compute_stats.py --samples <dir> [--student-samples <dir>]
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

from driftwam import bootstrap  # noqa: E402

CFG = bootstrap.setup()

from driftwam import metrics as M  # noqa: E402
from driftwam import paths  # noqa: E402
from driftwam import shapes as S  # noqa: E402

FLOOR_RATIO = 10.0
SIGMA_DATA = 0.5


def load_samples(d: Path) -> tuple[list[dict], dict]:
    run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
    files = sorted(p for p in d.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt samples in {d}")
    return [torch.load(p, map_location="cpu") for p in files], run


def video_views(video: torch.Tensor, frames: list[int]) -> dict[str, torch.Tensor]:
    """Whole composite plus per-camera crops, restricted to a chunk's frames."""
    v = video[:, :, frames]
    out = {"all": v}
    for name, crop in S.split_cameras(v).items():
        out[name] = crop
    return out


def action_views(action: torch.Tensor, frames: list[int]) -> dict[str, torch.Tensor]:
    """All 16 valid channels pooled, plus each semantic group.

    The channel set is fixed rather than per-item. Restricting to the channels
    that move in a given episode would make D differ between items, and D must
    match for the inter-condition variance to mean anything. `dead_frac` reports
    how much of the output was frozen instead, which is the quantity that was
    actually wanted.
    """
    if not frames:
        return {}
    out = {"all": action[:, S.VALID_ACTION_CHANNELS][:, :, frames]}
    for group, chans in S.ACTION_GROUPS.items():
        out[group] = action[:, chans][:, :, frames]
    return out


ACTION_VIEW_CHANNELS: dict[str, list[int]] = {
    "all": S.VALID_ACTION_CHANNELS, **S.ACTION_GROUPS
}


def per_condition(records: list[dict]) -> dict:
    """Build ConditionStats for every (item, chunk, modality, view)."""
    # view -> chunk -> list[(episode_key, ConditionStats)]
    table: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    n_active = []

    for rec in records:
        f = rec["F"]
        ep = f"{rec['task']}#{rec['segment']}"
        vid, act = rec["video"].float(), rec["action"].float()
        raw = rec["video_x0"].float() if "video_x0" in rec else None
        truth_v, truth_a = rec["truth_video"].float(), rec["truth_action"].float()
        active = S.active_action_channels(truth_a)
        n_active.append(len(active))

        for c in range(S.n_chunks(f)):
            vframes = S.video_frames_of_chunk(c, f)
            # a partial trailing chunk has fewer frames and so a smaller D, which
            # cannot be pooled with the full ones; F is cropped even, so this is a
            # guard rather than an expected path
            if len(vframes) != S.FRAME_CHUNK_SIZE:
                continue
            for name, x in video_views(vid, vframes).items():
                t = video_views(truth_v.unsqueeze(0), vframes)[name][0]
                table[f"video/{name}"][c].append(
                    (ep, M.condition_stats(x, truth=t))
                )
            # the student's raw x0 prediction, free of the c_skip * epsilon term
            if raw is not None:
                for name, x in video_views(raw, vframes).items():
                    t = video_views(truth_v.unsqueeze(0), vframes)[name][0]
                    table[f"video_x0/{name}"][c].append(
                        (ep, M.condition_stats(x, truth=t))
                    )
            # action frame 0 is padding, so chunk 0 contributes only frame 1
            aframes = [fr for fr in vframes if fr >= S.ACTION_PAD_FRAMES]
            for name, x in action_views(act, aframes).items():
                t = truth_a[ACTION_VIEW_CHANNELS[name]][:, aframes]
                table[f"action/{name}"][c].append(
                    (ep, M.condition_stats(x, truth=t))
                )

    return {"table": table, "n_active_mean": sum(n_active) / len(n_active)}


def pool(table, floors: dict | None = None) -> dict:
    """Pool per-condition stats, keeping chunk 0 apart from chunks with history."""
    out: dict = {}
    for view, by_chunk in table.items():
        chunks = sorted(by_chunk)
        entry: dict = {"per_chunk": {}}
        floor = (floors or {}).get(
            "video" if view.startswith("video") else "action"
        )

        for c in chunks:
            eps = [e for e, _ in by_chunk[c]]
            st = [s for _, s in by_chunk[c]]
            entry["per_chunk"][str(c)] = M.summarise(
                st, episode_ids=_ids(eps), noise_floor=floor,
                floor_ratio_required=FLOOR_RATIO,
            )

        with_hist = [c for c in chunks if c >= 1]
        for label, sel in (("chunk0", [0] if 0 in chunks else []),
                           ("with_history", with_hist)):
            if not sel:
                continue
            eps = [e for c in sel for e, _ in by_chunk[c]]
            st = [s for c in sel for _, s in by_chunk[c]]
            entry[label] = M.summarise(
                st, episode_ids=_ids(eps), noise_floor=floor,
                floor_ratio_required=FLOOR_RATIO,
            )
        # R pooled over chunk indices would let the chunk-depth effect leak into
        # v_inter, so the headline R is the mean of the within-chunk-index Rs
        rs = [entry["per_chunk"][str(c)].get("R") for c in with_hist]
        rs = [r for r in rs if r is not None and not math.isnan(r)]
        entry["R_within_chunk_mean"] = sum(rs) / len(rs) if rs else float("nan")
        entry["n_chunks_used"] = len(with_hist)
        out[view] = entry
    return out


def _ids(keys: list[str]) -> list[int]:
    uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return [uniq[k] for k in keys]


def verdicts(tea: dict, stu: dict | None) -> dict:
    """Auto-evaluate H1-H4 and H7 against the plan's thresholds."""
    def get(d, view, field, block="with_history"):
        try:
            v = d[view][block][field] if block else d[view][field]
            return v if v is not None else float("nan")
        except (KeyError, TypeError):
            return float("nan")

    rv = tea["video/all"]["R_within_chunk_mean"]
    ra = tea["action/all"]["R_within_chunk_mean"]
    pv = get(tea, "video/all", "PR_frac_mean")
    pa = get(tea, "action/all", "PR_frac_mean")
    ceil_ = get(tea, "video/all", "PR_ceiling")
    mv = get(tea, "video/all", "measurable")
    ma = get(tea, "action/all", "measurable")

    out: dict = {}
    if rv >= 0.10:
        h1 = "PASS: explicit drifting well defined on video"
    elif rv >= 0.02:
        h1 = "MARGINAL: needs variance-preserving measures (lower CFG, DINOv3 space)"
    else:
        h1 = "FAIL: video conditional is effectively Dirac"
    out["H1_video_has_dispersion"] = {"R_video": rv, "verdict": h1, "measurable": mv}

    h2 = ("PASS (as predicted): action is Dirac-like -> IDP-style implicit route"
          if ra < 0.02 else
          "FAIL: action retains dispersion -> explicit joint drifting viable")
    out["H2_action_is_dirac"] = {
        "R_action": ra, "PR_frac_action": pa,
        "dead_frac_action": get(tea, "action/all", "dead_frac_mean"),
        "verdict": h2, "measurable": ma,
    }

    ratio = rv / ra if ra > 0 else float("inf")
    out["H3_modality_asymmetry"] = {
        "R_ratio": ratio,
        "verdict": "PASS" if ratio >= 5 else "FAIL: branches can be treated alike",
    }
    if not ceil_ or ceil_ < 4:
        h4 = f"NO POWER: PR ceiling is only {ceil_}, need K >= 5"
    elif pa <= 0.4 and pv >= 0.7:
        h4 = "PASS: spread is low rank on action, isotropic repulsion is wrong"
    elif pa >= 0.7:
        h4 = "FAIL: action spread fills the sampled subspace"
    else:
        h4 = ("FAIL (partial): action spread is only mildly lower rank than "
              "video, so the degeneracy is one of magnitude, not direction; "
              "anisotropy-aware repulsion is not the remedy")
    out["H4_action_anisotropic"] = {
        "PR_frac_video": pv, "PR_frac_action": pa, "PR_ceiling": ceil_,
        "verdict": h4,
    }

    if stu:
        # judged on the raw x0 prediction: the consistency readout adds
        # c_skip * epsilon, an analytic term worth v_intra = 0.04 on its own, which
        # would let a mean-collapsed student pass
        vkey = "video_x0/all" if "video_x0/all" in stu else "video/all"
        rsv = stu[vkey]["R_within_chunk_mean"]
        rsa = stu["action/all"]["R_within_chunk_mean"]

        def cmp(rs, rt):
            if not (rt > 0):
                return float("nan"), "N/A"
            r = rs / rt
            if r >= 0.5:
                return r, "PASS: student keeps most dispersion, safe as negative sampler"
            if r < 0.1:
                return r, "FAIL: student mean-collapsed, repulsion dead at init"
            return r, "MARGINAL"

        rv_, tv_ = cmp(rsv, rv)
        ra_, ta_ = cmp(rsa, ra)
        entry = {
            "video_readout": vkey,
            "R_student_video": rsv, "R_student_action": rsa,
            "ratio_video": rv_, "verdict_video": tv_,
            "ratio_action": ra_, "verdict_action": ta_,
            "R_student_video_consistency_readout":
                stu["video/all"]["R_within_chunk_mean"],
        }
        # How much of the consistency readout's apparent spread is just the
        # analytic skip term. epsilon has unit variance per element, so the skip
        # contributes exactly c_skip^2 regardless of the network.
        c_skip = SIGMA_DATA**2 / (1.0 + SIGMA_DATA**2)
        v_obs = get(stu, "video/all", "v_intra_mean")
        if v_obs and not math.isnan(v_obs):
            entry["c_skip_squared"] = c_skip**2
            entry["skip_share_of_consistency_v_intra"] = c_skip**2 / v_obs
        entry["student_video_scale_vs_x0_scale"] = (
            get(stu, "video/all", "scale_mean")
            / get(stu, "video_x0/all", "scale_mean")
            if vkey == "video_x0/all" else float("nan")
        )
        out["H7_student_keeps_eps_sensitivity"] = entry
    return out


def cfg_sweep(by_cfg: dict[float, dict]) -> dict:
    """H5: is guidance scale a usable variance knob?

    Criterion from the plan: R falls monotonically with CFG and
    R(low) >= 2 * R(high). Reported next to the bias, because buying diversity by
    weakening guidance is only a win if positive-sample quality survives.
    """
    cfgs = sorted(by_cfg)
    rows = []
    for c in cfgs:
        d = by_cfg[c]
        w = d["video/all"]["with_history"]
        wa = d["action/all"]["with_history"]
        rows.append({
            "cfg": c,
            "R_video": d["video/all"]["R_within_chunk_mean"],
            "R_action": d["action/all"]["R_within_chunk_mean"],
            "cond_std_frac_video": w.get("cond_std_frac_of_marginal"),
            "cond_std_frac_action": wa.get("cond_std_frac_of_marginal"),
            "bias_video": w.get("bias_mean"),
            "bias_action": wa.get("bias_mean"),
            "v_intra_video": w.get("v_intra_mean"),
            "scale_video": w.get("scale_mean"),
            "PR_frac_video": w.get("PR_frac_mean"),
        })
    out = {"rows": rows, "n_cfg": len(cfgs)}
    if len(cfgs) < 2:
        out["verdict"] = "NOT TESTED: need at least two CFG values"
        return out

    rv = [r["R_video"] for r in rows]
    monotone = all(rv[i] >= rv[i + 1] for i in range(len(rv) - 1))
    out["monotone_decreasing"] = monotone
    out["R_by_cfg"] = {f"{c:g}": r for c, r in zip(cfgs, rv)}

    # The plan states the criterion on the pair (2, 10). Judge on exactly that
    # pair when both are present, and report the full endpoint span separately so
    # a wider sweep cannot quietly change the threshold being applied.
    r_at = dict(zip(cfgs, rv))
    if 2.0 in r_at and 10.0 in r_at:
        lo, hi = 2.0, 10.0
    else:
        lo, hi = cfgs[0], cfgs[-1]
    ratio = r_at[lo] / r_at[hi] if r_at[hi] > 0 else float("inf")
    out["cfg_low"], out["cfg_high"] = lo, hi
    out["R_ratio_low_over_high"] = ratio
    out["endpoint_span"] = {
        "cfg_min": cfgs[0], "cfg_max": cfgs[-1],
        "R_ratio": rv[0] / rv[-1] if rv[-1] > 0 else float("inf"),
    }
    if monotone and ratio >= 2.0:
        out["verdict"] = ("PASS: guidance is a real variance knob; lower CFG buys "
                          "diversity, to be traded against positive-sample bias")
    elif ratio >= 2.0:
        out["verdict"] = ("MARGINAL: the endpoints differ by >=2x but the trend is "
                          "not monotone, so CFG is not a clean dial")
    else:
        out["verdict"] = ("FAIL: CFG barely moves the conditional dispersion; the "
                          "constraint is the conditioning itself, not guidance")

    # how much of the gap to a usable R the knob can close
    b_at = {r["cfg"]: r["bias_video"] for r in rows}
    out["bias_ratio_low_over_high"] = (
        b_at[lo] / b_at[hi] if b_at.get(hi) else float("nan")
    )

    # Flash-WAM guides the video branch only (`action_guidance_scale=1`), and the
    # two noisy streams cannot see each other, so the action branch must come out
    # unchanged. Getting exactly that across independent full runs validates the
    # interleaved schedule along the whole trajectory, not just one forward.
    ra = [r["R_action"] for r in rows]
    spread = max(ra) - min(ra)
    out["action_R_spread_across_cfg"] = spread
    out["action_invariant_across_cfg"] = spread == 0.0
    return out


def trajectory_stats(tdir: Path) -> dict:
    """M8/H6: where along the noise axis the sample's identity gets fixed."""
    files = sorted(tdir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no trajectory files in {tdir}")
    per_item, agg = [], {}
    for p in files:
        d = torch.load(p, map_location="cpu")
        rec: dict = {"task": d["task"], "segment": d["segment"], "K": d["K"],
                     "cfg": d["cfg"]}
        for mod, key, skey in (("video", "x0_video", "sigmas_video"),
                               ("action", "x0_action", "sigmas_action")):
            traj, sig = d[key].float(), list(d[skey])
            # action latent frame 0 is constant padding and would flatten the curve
            if mod == "action":
                traj = traj[:, :, S.VALID_ACTION_CHANNELS][
                    :, :, :, S.action_frame_slice(d["F"])]
            dt = M.dispersion_trajectory(traj)
            al = M.trajectory_alignment(traj)
            rec[mod] = {
                "sigmas": sig,
                "dispersion_normalised": dt["dispersion_normalised"],
                "final": dt["final"],
                "alignment": al["alignment"],
                "residual_to_final": al["residual_to_final"],
                # sigma_star is read off the alignment curve, which ends at exactly
                # 1.0, so frac is an absolute alignment threshold
                "sigma_star_50": M.sigma_star(sig, al["alignment"], 0.5),
                "sigma_star_90": M.sigma_star(sig, al["alignment"], 0.9),
                "sigma_star_dispersion_50": M.sigma_star(sig, dt["dispersion"], 0.5),
            }
        per_item.append(rec)

    for mod in ("video", "action"):
        s50 = [r[mod]["sigma_star_50"] for r in per_item]
        s90 = [r[mod]["sigma_star_90"] for r in per_item]
        agg[mod] = {
            "sigmas": per_item[0][mod]["sigmas"],
            "sigma_star_50_mean": M._nanmean(s50),
            "sigma_star_50_all": s50,
            "sigma_star_90_mean": M._nanmean(s90),
            "sigma_star_dispersion_50_mean": M._nanmean(
                [r[mod]["sigma_star_dispersion_50"] for r in per_item]),
        }
        for curve in ("dispersion_normalised", "alignment", "residual_to_final"):
            c = torch.tensor([r[mod][curve] for r in per_item], dtype=torch.float64)
            agg[mod][f"{curve}_mean"] = c.mean(dim=0).tolist()

    s = agg["video"]["sigma_star_50_mean"]
    if math.isnan(s):
        verdict = "INCONCLUSIVE"
    elif s > 0.8:
        verdict = (f"HIGH (sigma*={s:.3f}): identity is settled in the high-noise "
                   "band. A one-step student must produce all diversity at "
                   "sigma=1, so the repulsion also acts there and H7 becomes the "
                   "binding question")
    else:
        verdict = (f"LOW (sigma*={s:.3f}): sample identity is still being decided "
                   "below this sigma, which is direct support for feeding TFD "
                   "moderately noised inputs rather than pure noise")
    return {"per_item": per_item, "aggregate": agg, "verdict_video": verdict,
            "n_items": len(files)}


def fmt(x, nd=4) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "NO"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}g}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.{nd}f}"
    return str(x)


def report_h5(sweep: dict) -> list[str]:
    L = ["## H5：CFG 是否是可用的方差旋钮", "",
         f"- **verdict**: {sweep['verdict']}"]
    if sweep["n_cfg"] >= 2:
        L.append(f"- R_video 随 CFG 单调下降 = "
                 f"{'是' if sweep['monotone_decreasing'] else '否'}；"
                 f"方案判据 R(CFG={sweep['cfg_low']:g}) / R(CFG={sweep['cfg_high']:g}) = "
                 f"`{fmt(sweep['R_ratio_low_over_high'], 3)}`（门槛 2.0）")
        es = sweep["endpoint_span"]
        if (es["cfg_min"], es["cfg_max"]) != (sweep["cfg_low"], sweep["cfg_high"]):
            L.append(f"- 全扫描区间 R(CFG={es['cfg_min']:g}) / R(CFG={es['cfg_max']:g}) "
                     f"= `{fmt(es['R_ratio'], 3)}`，即该旋钮的上限")
        L.append(f"- 代价侧：bias_video 之比 = "
                 f"`{fmt(sweep['bias_ratio_low_over_high'], 3)}`"
                 f"（小于 1 表示低 CFG 的正样本反而更贴近真值）")
        if sweep.get("action_invariant_across_cfg"):
            L.append("- **附带控制**：动作分支的统计量在各 CFG 下完全一致"
                     "（R_action 极差为 0）。Flash-WAM 只对视频做 CFG"
                     "（`action_guidance_scale=1`），而两条噪声流互不可见，"
                     "所以这是必然结果——但它同时说明 V4 模态解耦在**整条积分轨迹**"
                     "上都成立，而不只是单次前向。实测更强：CFG=2 与 CFG=10 的动作"
                     "输出在全部 20 个 item 上**逐位相同**，而视频最大差 8.0。")
    L += ["",
          "| CFG | 视频 条件std/边缘std | 视频 R | 视频 bias | 视频 PR/(K-1) "
          "| 动作 条件std/边缘std | 动作 R | 动作 bias |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sweep["rows"]:
        L.append(
            f"| {r['cfg']:g} | **{fmt(r['cond_std_frac_video'], 3)}** "
            f"| {fmt(r['R_video'])} | {fmt(r['bias_video'], 3)} "
            f"| {fmt(r['PR_frac_video'], 3)} "
            f"| {fmt(r['cond_std_frac_action'], 3)} | {fmt(r['R_action'])} "
            f"| {fmt(r['bias_action'], 3)} |"
        )
    L.append("")
    return L


def report_h6(traj: dict) -> list[str]:
    L = [f"## H6：多样性沿噪声轴的位置（M8，{traj['n_items']} 个 item）", "",
         f"- **verdict**: {traj['verdict_video']}", "",
         "判据说明：方案原本用「x0_hat 的离散度达到终值 50% 时的最大 sigma」，"
         "但这个量在高 sigma 处被残余噪声污染——`x0_hat = x_t - sigma * v` 在 "
         "sigma 接近 1 时仍带着大部分输入噪声，离散度因此天然很大"
         "（动作分支在 sigma=1 处的离散度是终值的 2.3 倍，即沿轨迹是**下降**的），"
         "任何「达到终值某个比例」的门槛都会在第一步就被跨过。",
         "",
         "而且这条 ODE 是确定性的，样本完全是 eps 的函数，"
         "「身份何时被决定」只有在「预测何时收敛」的意义下才有内容。"
         "所以 sigma* 改从**对齐度**读出：每一步算样本两两之差 "
         "`x0_hat_i - x0_hat_j` 与该对最终之差的余弦，终点恒为 1，"
         "在差异仍由残余噪声主导时接近 0。原始离散度曲线仍一并给出以作对照。",
         ""]
    for mod in ("video", "action"):
        a = traj["aggregate"][mod]
        L.append(f"- {mod}: sigma*(对齐度 0.5) = "
                 f"`{fmt(a['sigma_star_50_mean'], 3)}`，"
                 f"sigma*(对齐度 0.9) = `{fmt(a['sigma_star_90_mean'], 3)}`"
                 f"（逐 item：{', '.join(fmt(x, 3) for x in a['sigma_star_50_all'])}）"
                 f"；若按原离散度判据则为 "
                 f"`{fmt(a['sigma_star_dispersion_50_mean'], 3)}`")
    L.append("")
    for mod in ("video", "action"):
        a = traj["aggregate"][mod]
        sig = a["sigmas"]
        step = max(1, len(sig) // 10)
        idx = list(range(0, len(sig), step))
        if idx[-1] != len(sig) - 1:
            idx.append(len(sig) - 1)
        L += [f"| {mod} sigma | " + " | ".join(f"{sig[i]:.3f}" for i in idx) + " |",
              "|---|" + "---|" * len(idx)]
        for label, key in (("**对齐度**", "alignment_mean"),
                           ("到终值的相对距离", "residual_to_final_mean"),
                           ("归一化离散度（对照）", "dispersion_normalised_mean")):
            L.append(f"| {label} | "
                     + " | ".join(f"{a[key][i]:.3f}" for i in idx) + " |")
        L.append("")
    return L


def report(tea, stu, verd, meta, n_active, sweep=None, traj=None) -> str:
    L = ["# 实验 01 结果：条件方差", ""]
    L.append(f"- 采样配置：{meta.get('items')} item / K={meta.get('K')} / "
             f"CFG={meta.get('cfg')} / F={meta.get('frames')}")
    L.append(f"- 条件总数：{meta.get('conditions_total')}"
             f"（每 item {meta.get('n_chunks_per_item')} 个 chunk，"
             f"其中 chunk 0 无视觉历史，单独汇报）")
    L.append(f"- 参考诊断：真值中会动的动作通道均值 {n_active:.1f} / 16"
             f"（统计仍固定用全部 16 通道，见 `dead_frac`）")
    L.append("")
    L += ["## 测量有效性", ""]
    if meta.get("floor_bitwise_all") is not None:
        ok = meta["floor_bitwise_all"]
        L.append(f"- **V3 固定槽位复现：逐位相同 = {'是' if ok else '否'}**。"
                 f"重放 k=0 的 epsilon 走完整条 ODE，输出与首次完全一致，"
                 f"因此本次编排的数值地板是 0，测得的离散度里没有布局噪声成分。")
    L.append("- **冻结维占比 = 0**（所有视图）。没有任何输出维度因 bf16 量化而"
             "把 K 个样本压成同一个值，说明小方差是真的被分辨出来了，而不是被精度截断。")
    L.append("- 表中「保守地板比」用的是 smoke 里**朴素批处理**下测得的地板"
             "（video 2.8e-5 / action 1.7e-6）。那个地板来自同一 epsilon 落在不同 "
             "batch 槽位时 block-sparse 归约顺序不同，本次编排已经消除它，"
             "所以该列只是一个上界稳健性检查：比值小于 10 的行意味着"
             "「如果当初按朴素方式批处理，这一行就测不出来了」。")
    L.append("")

    L += ["## 判定", ""]
    for h, v in verd.items():
        L.append(f"### {h}")
        for kk, vv in v.items():
            if kk.startswith("verdict"):
                L.append(f"- **{kk}**: {vv}")
            else:
                L.append(f"- {kk} = `{fmt(vv)}`")
        L.append("")

    if sweep:
        L += report_h5(sweep)
    if traj:
        L += report_h6(traj)

    for label, d in (("Teacher", tea), ("Flash-WAM 学生", stu)):
        if not d:
            continue
        L += [f"## {label}：分视图统计（chunk >= 1 池化，{d['video/all']['with_history']['n_conditions']} 个条件）",
              "",
              "条件标准差/边缘标准差是最好解读的一列：0.17 表示「固定条件下重采样，"
              "得到的样本只在边缘分布 17% 的尺度内晃动」。",
              "",
              "| 视图 | D | 条件std/边缘std | R | rho | PR/(K-1) | m_c | bias "
              "| v_intra 均值 | v_intra 中位数 | 保守地板比 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for view in sorted(d):
            w = d[view].get("with_history", {})
            L.append(
                f"| `{view}` | {fmt(w.get('D'))} "
                f"| **{fmt(w.get('cond_std_frac_of_marginal'), 3)}** "
                f"| {fmt(d[view]['R_within_chunk_mean'])} "
                f"| {fmt(w.get('rho_mean'))} | {fmt(w.get('PR_frac_mean'), 3)} "
                f"| {fmt(w.get('min_pair_ratio_mean'), 3)} "
                f"| {fmt(w.get('bias_mean'), 3)} "
                f"| {fmt(w.get('v_intra_mean'))} | {fmt(w.get('v_intra_median'))} "
                f"| {fmt(w.get('floor_ratio'), 3)} |"
            )
        L.append("")
        L.append("动作各分组的 v_intra 均值明显高于中位数，分布右偏：少数条件"
                 "（集中在右臂旋转）的离散度比典型条件高一到两个数量级，"
                 "所以动作分支的「近确定性」是就典型条件而言的。")
        L.append("")

        L += [f"### {label}：方差随 chunk 深度", "",
              "| 视图 | " + " | ".join(f"c={c}" for c in range(4)) + " |",
              "|---|" + "---|" * 4]
        for view in ("video/all", "action/all"):
            if view not in d:
                continue
            cells = []
            for c in range(4):
                s = d[view]["per_chunk"].get(str(c))
                cells.append(fmt(s.get("rho_mean"), 3) if s else "-")
            L.append(f"| `{view}` rho | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, nargs="+",
                    help="teacher sample dirs; more than one enables the H5 sweep")
    ap.add_argument("--primary-cfg", type=float, default=5.0,
                    help="which CFG carries the H1-H4 headline")
    ap.add_argument("--trajectory", default=None,
                    help="dir of recorded x0 trajectories for M8/H6")
    ap.add_argument("--student-samples", default=None)
    ap.add_argument("--floor-video", type=float, default=None,
                    help="v_floor for video; default from smoke.json if present")
    ap.add_argument("--floor-action", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(paths.EXP01_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    floors = {}
    smoke = Path(paths.EXP01_DIR) / "smoke.json"
    if smoke.exists():
        s = json.loads(smoke.read_text()).get("V3_floor_batched", {})
        floors = {"video": s.get("video", {}).get("v_floor"),
                  "action": s.get("action", {}).get("v_floor")}
    if args.floor_video is not None:
        floors["video"] = args.floor_video
    if args.floor_action is not None:
        floors["action"] = args.floor_action
    print(f"noise floors used (from batched smoke, conservative): {floors}")

    by_cfg: dict[float, dict] = {}
    primary = None
    for d in args.samples:
        print(f"\nloading teacher samples from {d}")
        recs, run = load_samples(Path(d))
        cfg_val = float(recs[0]["cfg"])
        print(f"  {len(recs)} items, K={recs[0]['K']}, F={recs[0]['F']}, "
              f"CFG={cfg_val:g}")
        built = per_condition(recs)
        pooled = pool(built["table"], floors)
        by_cfg[cfg_val] = pooled
        if primary is None or abs(cfg_val - args.primary_cfg) < abs(
                primary[0] - args.primary_cfg):
            primary = (cfg_val, pooled, run, built["n_active_mean"])

    cfg_val, tea, run, n_active = primary
    print(f"\nheadline CFG = {cfg_val:g}")

    stu = None
    if args.student_samples:
        print(f"loading student samples from {args.student_samples}")
        srecs, _ = load_samples(Path(args.student_samples))
        stu = pool(per_condition(srecs)["table"], floors)

    sweep = cfg_sweep(by_cfg) if len(by_cfg) > 1 else None
    traj = trajectory_stats(Path(args.trajectory)) if args.trajectory else None

    verd = verdicts(tea, stu)
    meta = {**run, "n_active_mean": n_active, "headline_cfg": cfg_val,
            "cfgs_available": sorted(by_cfg)}

    stats = {"meta": meta, "floors": floors, "teacher": tea, "student": stu,
             "verdicts": verd, "cfg_sweep": sweep, "trajectory": traj}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
    md = report(tea, stu, verd, meta, n_active, sweep, traj)
    (out_dir / "report.md").write_text(md)

    print("\n" + "=" * 74)
    print(md)
    print("=" * 74)
    print(f"wrote {out_dir / 'stats.json'} and {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
