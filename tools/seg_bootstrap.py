# -*- coding: utf-8 -*-
"""分段时间块 bootstrap：轴线按弧长切 10 段，重采样给 P/R/IoU 的 95% CI。

口径与 `eval_freeaxis.py` 完全一致（axis=skel）：
    轴线      = 概率图 P≥0.5 掩膜闭运算 3×3 → skeletonize（1 m 网格）
    候选      = P≥0.5
    交付      = 3D 管距 d3 ≤ d3_site 且 AGL ≤ 2.0 m
    公平口径  TP/FP 只统计 AGL≤0.3 的地面点；FN = 全部真值地面点 − TP
              （含 P<0.5 从未进入候选的真值地面点）

分块：把骨架主轴（双 BFS 直径路径）按弧长等分为 10 段，每个轴线像元归到
最近的弧长位置；LAS 点按"最近轴线像元"的段号归段，逐段累计 TP/FP/FN。

bootstrap：对 10 个段做 1000 次有放回重采样，按段汇总后重算 P/R/IoU，
用百分位法取 95% CI。**只有 10 个空间块，样本量小，CI 偏窄，文中必须注明。**

真值（仅评分）：杨家坝 = results/mask_1m_杨家坝.npy；西坝村 = 新 SHP 主板。
SHP 不参与任何筛选。

用法：
    python tools/seg_bootstrap.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, label as cc_lab
from scipy.spatial import cKDTree
from skimage.morphology import disk, skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import p2_dem as P  # noqa: E402
from eval_new_shp import LAS, RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from fig_pred_vs_truth import order_path  # noqa: E402

CH = 3_000_000
SS = 4
N_SEG = 10
N_BOOT = 1000
SEED = 0
SITES = ("杨家坝", "西坝村")
CFG = {
    "杨家坝": {"prob": "cold_start_杨家坝_lc6.npz", "d3": 3.5, "truth": "old"},
    "西坝村": {"prob": "cold_start_西坝村_lc6.npz", "d3": 3.5, "truth": "new_main"},
}


def load_axis(site, prob, valid):
    """P≥0.5 闭运算 → 骨架，再按弧长切 N_SEG 段，返回 (ax, d2, zax, segmap, seg_len)。"""
    H, W = prob.shape
    core = (prob >= 0.5) & valid
    ax = skeletonize(binary_closing(core, np.ones((3, 3), bool)))
    if not ax.any():
        raise RuntimeError(f"{site} 轴线为空")
    d2, idx = distance_transform_edt(~ax, return_indices=True)

    ordered = order_path(ax)
    if ordered is None or len(ordered) < N_SEG:
        raise RuntimeError(f"{site} 轴线无法排序（{0 if ordered is None else len(ordered)} px）")
    ordered = np.asarray(ordered, np.float64)
    ds = np.hypot(*np.diff(ordered, axis=0).T)          # 8 邻域步长 1 / √2
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1])
    # 每段在弧长上的边界：0, L/10, ..., L
    seg_edges = np.linspace(0.0, total, N_SEG + 1)

    # 轴线每个像元 → 最近的"有序主轴点" → 段号
    tree = cKDTree(ordered)
    _, k = tree.query(np.argwhere(ax))
    seg_at_main = np.clip(np.floor(s/(total/max(N_SEG, 1))).astype(int), 0, N_SEG-1)
    seg_full = np.full(H * W, -1, np.int8)
    seg_full[np.ravel_multi_index(tuple(np.argwhere(ax).T), (H, W))] = seg_at_main[k]
    segmap = seg_full.reshape(H, W)
    if (segmap[ax] < 0).any():
        raise RuntimeError("轴线像元未被分配段号")

    seg_len = np.array([float(((segmap == i) & ax).sum()) for i in range(N_SEG)])
    info = dict(total_len_m=round(total, 2),
                seg_px=[int(v) for v in seg_len],
                seg_len_m=[round(float(seg_edges[i+1]-seg_edges[i]), 2)
                           for i in range(N_SEG)])
    return ax, d2, idx, segmap, info


def truth_mask(site, valid, x0, y1, gsd, H, W):
    if CFG[site]["truth"] == "old":
        return np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
    _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
    mh = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
    gt = (mh.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5) & valid
    lb, n = cc_lab(gt)
    if n > 1:
        sz = np.bincount(lb.ravel())
        sz[0] = 0
        gt = lb == sz.argmax()
    return gt


def stream_segments(site, prob, valid, dem, d2, idx, zax, segmap, gt):
    """流式 LAS，逐段累计 TP/FP/FN（公平口径）。"""
    H, W = prob.shape
    x0, y1, gsd = None, None, None
    x0, y1, wm, hm, gsd, _ = P.dem_window(site)
    d3s = CFG[site]["d3"]
    tp = np.zeros(N_SEG, np.int64)
    fp = np.zeros(N_SEG, np.int64)
    fn_gt = np.zeros(N_SEG, np.int64)   # 全部真值地面点（按段）
    idx0, idx1 = idx[0], idx[1]
    n_pts = 0
    import laspy
    with laspy.open(LAS[site]) as f:
        for pts in f.chunk_iterator(CH):
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            Z = np.asarray(pts.z, np.float64)
            c = np.floor((X - x0) / gsd).astype(np.int64)
            r = np.floor((y1 - Y) / gsd).astype(np.int64)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            if not ok.any():
                continue
            X, Y, Z, c, r = X[ok], Y[ok], Z[ok], c[ok], r[ok]
            n_pts += int(Z.size)
            agl = Z - dem[r, c]
            tr = gt[r, c]
            seg = segmap[idx0[r, c], idx1[r, c]].astype(np.int64)

            # FN 分母：全部真值地面点（含未进候选者）
            gtm = tr & (agl <= 0.3)
            if gtm.any():
                fn_gt += np.bincount(seg[gtm], minlength=N_SEG)

            cand = prob[r, c] >= 0.5
            if not cand.any():
                continue
            X, Y, Z, c, r = X[cand], Y[cand], Z[cand], c[cand], r[cand]
            agl = agl[cand]
            tr = tr[cand]
            seg = seg[cand]
            d3 = np.hypot(d2[r, c], Z - zax[r, c])
            keep = (d3 <= d3s) & (agl <= 2.0)
            ground = agl <= 0.3
            kg = keep & ground
            if kg.any():
                tp += np.bincount(seg[kg & tr], minlength=N_SEG)
                fp += np.bincount(seg[kg & ~tr], minlength=N_SEG)
    return dict(tp=tp, fp=fp, fn=fn_gt - tp, n_window_points=n_pts)


def prf(tp, fp, fn):
    tp, fp, fn = float(tp), float(fp), float(fn)
    p = tp / max(tp + fp, 1.0)
    r = tp / max(tp + fn, 1.0)
    return {"P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-12),
            "IoU": tp / max(tp + fp + fn, 1.0)}


def bootstrap(tp, fp, fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    out = {"P": [], "R": [], "IoU": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(tp), len(tp))
        m = prf(tp[i].sum(), fp[i].sum(), fn[i].sum())
        for k in out:
            out[k].append(m[k])
    ci = {}
    for k, v in out.items():
        v = np.asarray(v)
        ci[k] = [round(float(np.percentile(v, 2.5)), 4),
                 round(float(np.percentile(v, 97.5)), 4)]
    return ci


def make_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    metrics = ["P", "R", "IoU"]
    colors = {"P": "#4c72b0", "R": "#55a868", "IoU": "#c44e52"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    for ax, site in zip(axes, SITES):
        r = results[site]
        x = np.arange(len(metrics))
        vals = [r["point_estimate"][m] for m in metrics]
        lo = [r["point_estimate"][m] - r["ci95"][m][0] for m in metrics]
        hi = [r["ci95"][m][1] - r["point_estimate"][m] for m in metrics]
        ax.bar(x, vals, 0.55, color=[colors[m] for m in metrics], alpha=0.85)
        ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor="black",
                    capsize=6, lw=1.3)
        for xi, v, m in zip(x, vals, metrics):
            ax.text(xi, min(v + 0.06, 1.03), f"{v:.4f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Precision", "Recall", "IoU"])
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.set_ylabel("值（误差棒 = 95% CI）", fontsize=9)
        ax.set_title(f"{site}（10 段空间块 bootstrap，{N_BOOT} 次）", fontsize=10.5)
    fig.suptitle("分段时间块 bootstrap：P/R/IoU 点估计与 95% 置信区间"
                 "（10 段样本量小，CI 偏窄）", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(ROOT, "figures", "seg_bootstrap.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"写出 {out}")


def main():
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    results = {}
    for site in SITES:
        print(f"\n[{site}] 读取网格/概率/真值 ...", flush=True)
        x0, y1, wm, hm, gsd, d = P.dem_window(site)
        H, W = d["dem_mean"].shape
        valid = d["valid"] == 1
        dem = fill_nodata(d["dem_min"], valid)
        prob = np.load(os.path.join(RES, CFG[site]["prob"]))["prob"].astype(np.float64)
        prob = np.where(valid, prob, 0.0)
        gt = truth_mask(site, valid, x0, y1, gsd, H, W)
        print(f"  网格 {H}×{W}，真值 {int(gt.sum()):,} px，d3={CFG[site]['d3']} m")

        ax, d2, idx, segmap, ainfo = load_axis(site, prob, valid)
        zax = dem[tuple(idx)]
        print(f"  轴线 {int(ax.sum()):,} px，主轴长 {ainfo['total_len_m']:.1f} m，"
              f"分段像元 {ainfo['seg_px']}")

        cnt = stream_segments(site, prob, valid, dem, d2, idx, zax, segmap, gt)
        tp, fp, fn = cnt["tp"], cnt["fp"], cnt["fn"]
        segs = []
        for i in range(N_SEG):
            m = prf(tp[i], fp[i], fn[i])
            segs.append(dict(seg=i, len_m=ainfo["seg_len_m"][i], axis_px=ainfo["seg_px"][i],
                             tp=int(tp[i]), fp=int(fp[i]), fn=int(fn[i]),
                             P=round(m["P"], 4), R=round(m["R"], 4), IoU=round(m["IoU"], 4)))
        # 点估计：全段汇总（与 eval_freeaxis 的 point_fair 同口径）
        est_raw = prf(tp.sum(), fp.sum(), fn.sum())
        est = {k: round(v, 4) for k, v in est_raw.items()}
        ci = bootstrap(tp, fp, fn)
        results[site] = dict(
            site=site, axis="skel", prob_file=CFG[site]["prob"], d3_m=CFG[site]["d3"],
            agl_m=2.0, n_seg=N_SEG, n_bootstrap=N_BOOT, seed=SEED,
            truth_source=("results/mask_1m_%s.npy" % site
                          if CFG[site]["truth"] == "old" else
                          "SHP %s → poly_mask_hr → 1 m 最大连通域" % SHP_NEW[site]["poly"]),
            axis_px=int(ax.sum()), total_len_m=ainfo["total_len_m"],
            segments=segs,
            point_estimate=est,
            ci95=ci,
            n_window_points=cnt["n_window_points"],
            definitions=dict(
                tp="交付且真值内且 AGL≤0.3 的点数",
                fp="交付且真值外且 AGL≤0.3 的点数",
                fn="全部真值地面点（AGL≤0.3）− TP（含 P<0.5 未进候选的点）",
                ci="10 个空间段有放回重采样 1000 次的 2.5/97.5 百分位"),
            note=("空间块 bootstrap 只有 10 个段：样本量小，重采样分布只能作量级参考，"
                  "不是严格置信区间；相邻段共享端点与地形，并非独立。"
                  "实测块间异质性很强（如杨家坝第 2、9 段 IoU 仅 0.11 / 0.20），"
                  "因此本数据的 CI 并未因段数少而偏窄，反而偏宽、不稳；"
                  "文中应注明 10 段样本量小，CI 不可解读为模型稳定性的强证据。"))
        out = os.path.join(RES, f"seg_bootstrap_{site}.json")
        json.dump(results[site], open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"  点估计 {est}")
        print(f"  95% CI  {ci}")
        print(f"  写出 {out}")

    make_figure(results)

    print("\n### 分段时间块 bootstrap（点估计与 95% CI）\n")
    print("| 站点 | P | P 95% CI | R | R 95% CI | IoU | IoU 95% CI |")
    print("|---|---|---|---|---|---|---|")
    for s in SITES:
        r = results[s]
        e, c = r["point_estimate"], r["ci95"]
        print(f"| {s} | {e['P']:.4f} | [{c['P'][0]:.4f}, {c['P'][1]:.4f}] | "
              f"{e['R']:.4f} | [{c['R'][0]:.4f}, {c['R'][1]:.4f}] | "
              f"{e['IoU']:.4f} | [{c['IoU'][0]:.4f}, {c['IoU'][1]:.4f}] |")


if __name__ == "__main__":
    main()
