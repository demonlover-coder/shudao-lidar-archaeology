# -*- coding: utf-8 -*-
"""反哺实验：用已收敛的识别轴线/交付结果，回头改进
    ① 滤波保留（替换"石板平坦"假设，改为轴线先验）；
    ② 分辨率自适应（细档只铺在轴线上，而不是整片 P≥0.5）。

两部分都用 SHP 真值/源 DEM 做**最后评分**，不参与决策。

A. 滤波保留（点级，真值 = {site}_label.npy 的 SHP 面内点）
    geom_only            P≥0.2 + 法向<30° + 邻域Zσ≤0.1 m（原方案）
    axis_band            P≥0.2 + 距轴线≤3 m（纯轴线先验）
    axis+geom_strict     轴线带内仍用法向/平坦阈值
    axis+geom_relaxed    轴线带内放宽到 55°/0.35 m

B. 分辨率自适应（1 m 规划图）
    prob 驱动            P≥0.5→0.1 m，0.1≤P<0.5→0.5 m，距核心≤5 m→1 m，其余 4 m（原方案）
    axis 驱动            距轴线≤2 m→0.1 m，≤4 m→0.5 m，≤5 m→1 m，其余 4 m
   评分：样本量、细档覆盖率（真值走廊被 ≤0.5 m 覆盖的比例）、
         50 m 窗口内 RMSE 与细档梯度保留（从源 DEM 重新渲染比较）。

输出：
    results/axis_feedback.json
    figures/axis_feedback.png

用法：
    python tools/axis_feedback.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial import cKDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import p2_dem as P  # noqa: E402
from eval_new_shp import RES  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from p2_adaptive_gsd import LEVELS, local_std  # noqa: E402
from p2_export_all import project_covered  # noqa: E402
from fig_recognition import load_truth  # noqa: E402
from adaptive_gsd_render import (block_mean, compose, expand_codes, read_source,  # noqa: E402
                                 to_native, uniform_1m)
from p2_adaptive_gsd import FIG  # noqa: E402

REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")
SITES = ("杨家坝", "西坝村")
CH = 4_000_000
WIN = 50.0


# ================================================================ A 滤波
def geom_pass(xyz, k=32, r=0.5, min_nb=8, chunk=150_000):
    """KD-Tree 邻域 PCA：返回 (nb, angle_deg, z_std)。"""
    tree = cKDTree(xyz)
    n = len(xyz)
    nb = np.zeros(n, np.int32)
    ang = np.full(n, 90.0, np.float32)
    sd = np.full(n, 9.9, np.float32)
    for s0 in range(0, n, chunk):
        e0 = min(s0 + chunk, n)
        d, ii = tree.query(xyz[s0:e0], k=k, distance_upper_bound=r, workers=-1)
        ok = np.isfinite(d)
        m = ok.sum(axis=1)
        P3 = np.where(ok[..., None], xyz[np.minimum(ii, n - 1)], 0.0)
        w = ok.astype(np.float64)
        cnt = np.maximum(m, 1)
        mu = P3.sum(axis=1) / cnt[:, None]
        Q = (P3 - mu[:, None, :]) * w[..., None]
        C = np.einsum("nki,nkj->nij", Q, Q) / cnt[:, None, None]
        ev, evec = np.linalg.eigh(C)
        nv = evec[:, :, 0]
        ang[s0:e0] = np.degrees(np.arccos(np.clip(np.abs(nv[:, 2]), 0, 1)))
        z = P3[..., 2]
        mu_z = (z * w).sum(axis=1) / cnt
        var = ((z - mu_z[:, None]) ** 2 * w).sum(axis=1) / cnt
        sd[s0:e0] = np.sqrt(np.clip(var, 0, None))
        nb[s0:e0] = m
    return nb, ang, sd


def eval_keep(keep, label, n_all):
    tp = int((keep & label).sum())
    fp = int((keep & ~label).sum())
    fn = int(((~keep) & label).sum())
    return dict(keep=int(keep.sum()), tp=tp, fp=fp, fn=fn,
                P=round(tp / max(tp + fp, 1), 4), R=round(tp / max(tp + fn, 1), 4),
                F1=round(2 * tp / max(2 * tp + fp + fn, 1), 4))


def experiment_filter(site, prob, axis, log):
    meta = json.load(open(os.path.join(ROOT, "data", f"meta_{site}.json"),
                          encoding="utf-8"))
    gsd, x0, y1 = meta["gsd_m"], meta["x0"], meta["y1"]
    H, W = prob.shape
    xyz_f = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    o = np.load(os.path.join(REAL, f"{site}_origin.npy")).astype(np.float64)
    label = np.asarray(np.load(os.path.join(REAL, f"{site}_label.npy"), mmap_mode="r"),
                       dtype=bool)
    n_all = xyz_f.shape[0]
    d2 = distance_transform_edt(~axis)

    idxs, pvs, d2s = [], [], []
    t0 = time.time()
    for s0 in range(0, n_all, CH):
        e0 = min(s0 + CH, n_all)
        p = np.asarray(xyz_f[s0:e0], dtype=np.float64) + o[None, :]
        c = np.round((p[:, 0] - x0) / gsd).astype(np.int32)
        r = np.round((y1 - p[:, 1]) / gsd).astype(np.int32)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        cc, rr = np.clip(c, 0, W - 1), np.clip(r, 0, H - 1)
        pv = np.zeros(len(p), np.float32)
        pv[ok] = prob[rr[ok], cc[ok]]
        sel = np.flatnonzero(pv >= 0.2)
        if sel.size:
            idxs.append((sel + s0).astype(np.int64))
            pvs.append(pv[sel])
            d2s.append(d2[rr[sel], cc[sel]].astype(np.float32))
    idx = np.concatenate(idxs)
    pv = np.concatenate(pvs)
    dv = np.concatenate(d2s)
    log(f"  候选（P≥0.2）：{len(idx):,} 点（{len(idx) / n_all * 100:.2f}%），"
        f"扫描 {time.time() - t0:.0f}s")

    pts = np.asarray(xyz_f[idx], dtype=np.float64)
    nb, ang, sd = geom_pass(pts)
    log(f"  几何：邻域中位 {np.median(nb):.0f}，法向中位 {np.median(ang):.1f}°，"
        f"Zσ中位 {np.median(sd):.3f} m")
    del pts

    out = {}
    def add(name, sel_c):
        keep_c = np.zeros(len(idx), bool)
        keep_c[sel_c] = True
        keep = np.zeros(n_all, bool)
        keep[idx[keep_c]] = True
        out[name] = eval_keep(keep, label, n_all)
        log(f"    {name:<18s} keep {out[name]['keep']:>8,}  "
            f"P {out[name]['P']:.4f}  R {out[name]['R']:.4f}  F1 {out[name]['F1']:.4f}")

    strict = (nb >= 8) & (ang < 30) & (sd <= 0.1)
    relax = (nb >= 5) & (ang < 55) & (sd <= 0.35)
    add("geom_only", strict)
    add("axis_band", dv <= 3.0)
    add("axis+geom_strict", (dv <= 3.0) & strict)
    add("axis+geom_relaxed", (dv <= 3.0) & relax)
    return dict(n_all=n_all, cand=int(len(idx)), truth=int(label.sum()), variants=out)


# ================================================================ B 自适应
def build_axis_map(site, prob, axis, valid, dem, covered):
    """轴线驱动的 GSD 规划图（与 prob 驱动同档位、同物理门控）。"""
    H, W = prob.shape
    d = distance_transform_edt(~axis)
    R = local_std(dem, valid)
    g = np.full((H, W), LEVELS[3], np.float32)
    g[d <= 5.0] = LEVELS[2]
    g[(d > 2.0) & (d <= 4.0)] = LEVELS[1]
    g[d <= 2.0] = LEVELS[0]
    bg = (d > 5.0) & (R > 0.4)
    g[bg] = np.maximum(g[bg], LEVELS[2])
    blind = ~covered
    g[blind] = LEVELS[3]
    g[blind & (d <= 2.0)] = LEVELS[2]
    return g, d


def build_prob_map(prob, valid, dem, covered):
    H, W = prob.shape
    core = (prob >= 0.5) & valid
    band = (prob >= 0.1) & (prob < 0.5) & valid
    dist = distance_transform_edt(~core)
    R = local_std(dem, valid)
    g = np.full((H, W), LEVELS[3], np.float32)
    g[dist <= 5.0] = LEVELS[2]
    g[band] = LEVELS[1]
    g[core] = LEVELS[0]
    bg = (dist > 5.0) & (R > 0.4)
    g[bg] = np.maximum(g[bg], LEVELS[2])
    blind = ~covered
    g[blind] = LEVELS[3]
    g[blind & core] = LEVELS[2]
    return g, dist


def map_stats(g, gt, dmap, core_thr):
    px = {str(v): int((g == v).sum()) for v in LEVELS}
    samples = float(sum((g == v).sum() / (v * v) for v in LEVELS))
    fine = (g <= 0.5) & gt
    return dict(px=px, samples=round(samples, 1),
                samples_uniform_1m=int(g.size),
                fine_cover_truth=round(float(fine.sum() / max(gt.sum(), 1)), 4),
                core_px=int((dmap <= core_thr).sum()))


def window_render(site, g, src, src_gsd, x0_src, y1_src, gt, dmap, log):
    """50 m 窗口：按规划图从源 DEM 合成，算 RMSE 与细档梯度保留。"""
    H, W = g.shape
    ys, xs = np.nonzero(gt)
    if len(ys) == 0:
        return {}
    k = len(ys) // 2
    K = int(WIN)
    r1 = int(np.clip(round(ys[k] / 4) * 4 - K // 2, 0, H - K))
    c1 = int(np.clip(round(xs[k] / 4) * 4 - K // 2, 0, W - K))
    # 源 DEM 裁剪：窗口起点(米) / 源 GSD
    px0 = int(round(c1 / src_gsd))
    py0 = int(round(r1 / src_gsd))
    px1 = px0 + int(round(K / src_gsd))
    py1 = py0 + int(round(K / src_gsd))
    crop = src[py0:py1, px0:px1]
    nat = to_native(crop, src_gsd, (K * 10, K * 10))
    gw = g[r1:r1 + K, c1:c1 + K]
    codes = expand_codes(gw, (K * 10, K * 10))
    ada = compose(nat, codes)
    uni = uniform_1m(nat)
    gtw = np.repeat(np.repeat(gt[r1:r1 + K, c1:c1 + K], 10, 0), 10, 1)
    fin = np.isfinite(nat)

    def rmse(a, m):
        e = (a - nat)[m]
        e = e[np.isfinite(e)]
        return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")

    gy, gx = np.gradient(np.where(fin, nat, np.nan))
    gya, gxa = np.gradient(np.where(fin, ada, np.nan))
    fine = (codes == 1) & fin
    ga = np.hypot(gx, gy)
    gb = np.hypot(gxa, gya)
    with np.errstate(invalid="ignore"):
        ret = float(np.nanmean(gb[fine]) / max(np.nanmean(ga[fine]), 1e-9))
    log(f"    窗口 ({r1},{c1})：RMSE 自适应 {rmse(ada, fin & gtw):.3f} m，"
        f"均匀 1 m {rmse(uni, fin & gtw):.3f} m，细档梯度保留 {ret:.3f}")
    return dict(window=[int(r1), int(c1)], rmse_adaptive=round(rmse(ada, fin & gtw), 4),
                rmse_uniform1m=round(rmse(uni, fin & gtw), 4),
                fine_grad_retention=round(ret, 4))


# ================================================================ 主
def main():
    os.makedirs(FIG, exist_ok=True)
    lf = open(os.path.join(RES, "axis_feedback.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n")
        lf.flush()

    res = {}
    for site in SITES:
        log("\n" + "=" * 74)
        log(f"[{site}]")
        x0, y1, wm, hm, gsd, d = P.dem_window(site)
        H, W = d["dem_mean"].shape
        valid = d["valid"] == 1
        dem = fill_nodata(d["dem_min"], valid)
        z = np.load(os.path.join(RES, f"cold_start_{site}_lc6.npz"))
        prob = np.where(valid, z["prob"].astype(np.float64), 0.0)
        axis = z["path"].astype(bool)
        gt = load_truth(site, valid, x0, y1, gsd, H, W)

        log("  ── A 滤波保留")
        fA = experiment_filter(site, prob, axis, log)

        log("  ── B 分辨率自适应")
        covered = project_covered(site, x0, y1, gsd, H, W, dilate=1)
        gp, dp = build_prob_map(prob, valid, dem, covered)
        ga, da = build_axis_map(site, prob, axis, valid, dem, covered)
        sP = map_stats(gp, gt, dp, 1.0)
        sA = map_stats(ga, gt, da, 2.0)
        log(f"    prob 驱动：样本 {sP['samples']:,.0f}，细档覆盖真值 "
            f"{sP['fine_cover_truth'] * 100:.1f}%")
        log(f"    axis 驱动：样本 {sA['samples']:,.0f}，细档覆盖真值 "
            f"{sA['fine_cover_truth'] * 100:.1f}%")

        log("    渲染校验（50 m 窗口）")
        src, src_gsd, xs, ys = read_source(site)
        wP = window_render(site, gp, src, src_gsd, xs, ys, gt, dp, log)
        wA = window_render(site, ga, src, src_gsd, xs, ys, gt, da, log)
        del src

        res[site] = dict(filter=fA, prob_map=sP, axis_map=sA,
                         window=dict(prob=wP, axis=wA))

    json.dump(res, open(os.path.join(RES, "axis_feedback.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("写出 results/axis_feedback.json")

    # ---------------- 图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    names = ["geom_only", "axis_band", "axis+geom_strict", "axis+geom_relaxed"]
    lab = ["原方案\n几何滤波", "轴线带\n(纯先验)", "轴线带+\n严格几何", "轴线带+\n放宽几何"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), constrained_layout=True)
    x = np.arange(len(names))
    bw = 0.13
    cols = {"杨家坝": ("#2c7bb6", "#d7191c", "#5aae61"),
            "西坝村": ("#80b1d3", "#fb8072", "#b3de69")}
    for i, site in enumerate(SITES):
        v = res[site]["filter"]["variants"]
        for j, key in enumerate(("P", "R", "F1")):
            off = (i * 3 + j - 2.5) * bw
            axes[0].bar(x + off, [v[n][key] for n in names], bw,
                        color=cols[site][j],
                        label=f"{site}-{key}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lab, fontsize=8)
    axes[0].set_title("A 滤波保留：轴线先验 vs 原几何滤波", fontsize=11)
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3)

    for ax, key, title, unit in (
            (axes[1], "fine_cover_truth", "B 细档对真值走廊覆盖率", "比例"),
            (axes[2], "samples", "B 采样点数（相对均匀 1 m）", "比例")):
        vals = []
        for site in SITES:
            for mk in ("prob_map", "axis_map"):
                s = res[site][mk]
                vals.append(s[key] if key == "fine_cover_truth"
                            else s["samples"] / s["samples_uniform_1m"])
        bars = ax.bar([f"{s}\n{m}" for s in SITES for m in ("prob", "axis")], vals,
                      color=["#999999", "#d7191c"] * 2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=9)
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("反哺实验：用识别轴线改进滤波保留与分辨率自适应", fontsize=13)
    out = os.path.join(FIG, "axis_feedback.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    log(f"写出 {out}")


if __name__ == "__main__":
    main()
