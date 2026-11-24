# -*- coding: utf-8 -*-
"""走廊引导滤波后的 DEM 重建：1 m 基准 + 走廊自适应 0.5 m（含 2 m 线性羽化）。

输入：`corridor_guided` 方案的保留点（与本项目 `corridor_filter.py` 完全同一判据，
逐点重算，不落中间 LAS）。

产物：
    results/dem_adaptive_{site}.npz   dem(0.5 m 自适应, NaN=无点) + gsd_grid + dem_1m
    results/dem_rebuild_{site}.json   DEM 级指标
    figures/dem_rebuild.png           上行杨家坝 / 下行西坝村：
                                     左=走廊 DEM 晕渲，右=沿轴线高程剖面

自适应规则：
    1 m 基准     每格取保留点最小 Z；无点格 = NaN
    0.5 m 子格   走廊内（水平距离 d2 <= d3_site+2 m）每 1 m 格 2×2 子格，各自取最小 Z
    羽化         走廊外侧 2 m 环带（d3_site < d2 <= d3_site+2）线性融合 0.5 m 与 1 m
                 高程，权重 w = (d3_site+2-d2)/2（外缘 0、内缘 1）；1 m 场双线性上采样；
                 子格无点时回退父格 1 m 值

DEM 级评价（参考面 = 1 m `dem_min`，本区最密的原始地面基准）：
    corridor_void_rate   真值掩膜内无值格占比（1 m / 0.5 m 自适应 / raw 0.5 m 三口径）
    corridor_rmse        真值内两套 DEM 与 dem_min 的高差 RMSE
    seam_step            走廊档位边界内外相邻格高差绝对值中位数（羽化前/后）
    profile_rmse         沿骨架轴线剖面的 RMSE（1 m / 0.5 m 自适应）

用法：
    python tools/dem_rebuild.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import convolve, zoom
from scipy.spatial import cKDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import corridor_filter as CF  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from eval_new_shp import LAS, RES  # noqa: E402
from p2_adaptive_gsd import FIG  # noqa: E402

CH = 3_000_000
FEATHER_M = 2.0
SUB_MARGIN_M = 2.0

DEFINITIONS = {
    "dem_1m": "每 1 m 格取 corridor_guided 保留点的最小 Z；无点格 NaN",
    "dem_adaptive": "水平距 d2<=d3_site+2 m 的走廊用 0.5 m 子格（2×2）；外侧 1 m；"
                    "d2∈(d3_site, d3_site+2] 环带按 w=(d3_site+2-d2)/2 融合 0.5 m 与 1 m；"
                    "子格无点时回退父格 1 m 值",
    "corridor_void_rate_1m": "真值掩膜内 dem_1m 为 NaN 的 1 m 格占比",
    "corridor_void_rate_adaptive": "真值 1 m 格对应的 0.5 m 子格中 dem_adaptive 为 NaN 的占比",
    "corridor_void_rate_adaptive_raw": "同上，但用未填洞的 0.5 m 子格（无点即 NaN）",
    "corridor_rmse_1m": "真值内 dem_1m 与 dem_min 均有值格的 RMSE（m）",
    "corridor_rmse_adaptive": "真值 0.5 m 子格上 dem_adaptive 与 dem_min(按父格重复) 的 RMSE（m）",
    "corridor_rmse_adaptive_raw": "同上，但只用未填洞的 0.5 m 子格",
    "seam_step_hard": "走廊 0.5 m 区外缘两侧相邻 0.5 m 格高差绝对值中位数，羽化前",
    "seam_step_feather": "同上，羽化后",
    "seam_step_residual_hard": "同上，但高差取 DEM 相对 dem_min 的残差之差（剔除地形梯度）",
    "seam_step_residual_feather": "同上，羽化后的残差口径",
    "profile_rmse_1m": "骨架轴线像元处 dem_1m 与 dem_min 的 RMSE（m，配对有限值）",
    "profile_rmse_adaptive": "轴线像元中心所在 0.5 m 子格与 dem_min 的 RMSE（m）",
}


# ---------------------------------------------------------------- 剖面排序
def order_axis(ax):
    """骨架像元按最近邻贪心串成一条链（近似沿轴顺序），返回 (rows, cols)。"""
    rr, cc = np.nonzero(ax)
    if rr.size <= 2:
        return rr, cc
    a = ax.astype(np.uint8)
    nb = convolve(a, np.ones((3, 3), np.uint8), mode="constant") - a
    ends = np.argwhere(ax & (nb <= 1))
    start = (ends[0, 0], ends[0, 1]) if len(ends) else (rr[0], cc[0])
    pts = np.column_stack([rr, cc]).astype(np.float64)
    tree = cKDTree(pts)
    used = np.zeros(rr.size, bool)
    cur = int(np.flatnonzero((rr == start[0]) & (cc == start[1]))[0])
    order = []
    for _ in range(rr.size):
        order.append(cur)
        used[cur] = True
        _, idx = tree.query(pts[cur], k=min(32, rr.size))
        nxt = next((int(j) for j in np.atleast_1d(idx) if not used[j]), None)
        if nxt is None:
            rest = np.flatnonzero(~used)
            if rest.size == 0:
                break
            nxt = int(rest[np.argmin(np.hypot(*(pts[rest] - pts[cur]).T))])
        cur = nxt
    order = np.asarray(order)
    return rr[order], cc[order]


# ---------------------------------------------------------------- 点云 → 格网
def build_grids(pr, gmr):
    """流式累加 1 m 与走廊 0.5 m 的最小高程。返回 (dem1, sub, n_keep, n_las)。"""
    x0, y1, gs, H, W = pr["x0"], pr["y1"], pr["gsd"], pr["H"], pr["W"]
    gs05 = gs / 2.0
    H2, W2 = H * 2, W * 2
    dem, d2, zax, d3s = pr["dem"], pr["d2"], pr["zax"], pr["d3_site"]
    g2 = gmr[1]                       # r=2，本文走廊外判据
    dem1 = np.full((H, W), np.inf, np.float32)
    sub = np.full((H2, W2), np.inf, np.float32)
    n_keep = n_las = 0

    import laspy
    t0 = time.time()
    with laspy.open(LAS[pr["site"]]) as f:
        for ci, pts in enumerate(f.chunk_iterator(CH), start=1):
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            Z = np.asarray(pts.z, np.float64)
            n_las += X.size
            col = np.floor((X - x0) / gs).astype(np.int64)
            row = np.floor((y1 - Y) / gs).astype(np.int64)
            ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
            if not ok.any():
                continue
            X, Y, Z = X[ok], Y[ok], Z[ok]
            row, col = row[ok], col[ok]

            agl = Z - dem[row, col]
            d3 = np.hypot(d2[row, col], Z - zax[row, col])
            cor = d3 <= d3s
            keep = np.where(cor, agl <= 0.5, Z - g2[row, col] <= 0.3)
            if not keep.any():
                continue
            kk = np.flatnonzero(keep)
            n_keep += kk.size
            np.minimum.at(dem1, (row[kk], col[kk]), Z[kk])

            # 走廊 0.5 m 子格（含 2 m 羽化环带）
            m = d3[kk] <= d3s + SUB_MARGIN_M
            if m.any():
                ii = kk[m]
                rs = np.floor((y1 - Y[ii]) / gs05).astype(np.int64)
                cs = np.floor((X[ii] - x0) / gs05).astype(np.int64)
                inb = (rs >= 0) & (rs < H2) & (cs >= 0) & (cs < W2)
                np.minimum.at(sub, (rs[inb], cs[inb]), Z[ii][inb])
            if ci % 5 == 0:
                print(f"    {pr['site']} 块 {ci}：保留 {n_keep:,} 点，"
                      f"{time.time()-t0:.0f}s", flush=True)

    dem1[~np.isfinite(dem1)] = np.nan
    sub[~np.isfinite(sub)] = np.nan
    return dem1, sub, n_keep, n_las


# ---------------------------------------------------------------- 自适应融合
def adaptive_dem(pr, dem1, sub):
    """生成 0.5 m 自适应 DEM。返回 (demA, hard, zone05)。"""
    d2, d3s = pr["d2"], pr["d3_site"]
    base05 = np.repeat(np.repeat(dem1, 2, axis=0), 2, axis=1)
    zone1 = d2 <= d3s + SUB_MARGIN_M
    zone05 = np.repeat(np.repeat(zone1, 2, axis=0), 2, axis=1)
    w1 = np.clip((d3s + SUB_MARGIN_M - d2) / FEATHER_M, 0.0, 1.0).astype(np.float32)
    w05 = np.clip(zoom(w1, 2, order=1), 0.0, 1.0) * zone05
    # 子格无点 → 回退父格 1 m 值，保证交付 DEM 连续
    blend = np.where(np.isfinite(sub), w05 * sub + (1.0 - w05) * base05, base05)
    demA = np.where(zone05, blend, base05)
    hard = np.where(zone05 & np.isfinite(sub), sub, base05)
    return demA, hard, zone05


# ---------------------------------------------------------------- 指标
def seam_step(dem, zone05):
    """档位边界（zone05 内外相邻格）高差绝对值中位数。"""
    vals = []
    for axis in (0, 1):
        if axis == 0:
            a, b = dem[:-1, :], dem[1:, :]
            za, zb = zone05[:-1, :], zone05[1:, :]
        else:
            a, b = dem[:, :-1], dem[:, 1:]
            za, zb = zone05[:, :-1], zone05[:, 1:]
        m = (za != zb) & np.isfinite(a) & np.isfinite(b)
        if m.any():
            vals.append(np.abs(a[m] - b[m]))
    if not vals:
        return float("nan"), 0
    v = np.concatenate(vals)
    return float(np.median(v)), int(v.size)


def rmse(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def evaluate(pr, gt, dem1, demA, sub, hard, zone05):
    H, W = dem1.shape
    ref = pr["dem"]
    gt05 = np.repeat(np.repeat(gt, 2, axis=0), 2, axis=1)
    ref05 = np.repeat(np.repeat(ref, 2, axis=0), 2, axis=1)

    m1 = gt & np.isfinite(dem1) & np.isfinite(ref)
    m05 = gt05 & np.isfinite(demA) & np.isfinite(ref05)
    mraw = gt05 & np.isfinite(sub) & np.isfinite(ref05)

    seam_h, n_seam = seam_step(hard, zone05)
    seam_f, _ = seam_step(demA, zone05)
    # 残差口径：先扣掉 dem_min 的地形梯度，剩下的才是"档位切换"造成的台阶
    seam_rh, _ = seam_step(hard - ref05, zone05)
    seam_rf, _ = seam_step(demA - ref05, zone05)

    rr, cc = order_axis(pr["ax"])
    ax_ref = ref[rr, cc]
    ax_1m = dem1[rr, cc]
    r2 = np.clip(2 * rr + 1, 0, 2 * H - 1)
    c2 = np.clip(2 * cc + 1, 0, 2 * W - 1)
    ax_a = demA[r2, c2]
    p1 = np.isfinite(ax_1m) & np.isfinite(ax_ref)
    pa = np.isfinite(ax_a) & np.isfinite(ax_ref)

    res = {
        "n_truth_px_1m": int(gt.sum()),
        "corridor_void_rate_1m": round(
            float((gt & ~np.isfinite(dem1)).sum() / max(gt.sum(), 1)), 4),
        "corridor_void_rate_adaptive": round(
            float((gt05 & ~np.isfinite(demA)).sum() / max(gt05.sum(), 1)), 4),
        "corridor_void_rate_adaptive_raw": round(
            float((gt05 & ~np.isfinite(sub)).sum() / max(gt05.sum(), 1)), 4),
        "corridor_rmse_1m": round(rmse(dem1[m1] - ref[m1]), 4),
        "corridor_rmse_adaptive": round(rmse(demA[m05] - ref05[m05]), 4),
        "corridor_rmse_adaptive_raw": round(rmse(sub[mraw] - ref05[mraw]), 4),
        "seam_step_hard": round(seam_h, 4),
        "seam_step_feather": round(seam_f, 4),
        "seam_step_residual_hard": round(seam_rh, 4),
        "seam_step_residual_feather": round(seam_rf, 4),
        "seam_pairs": int(n_seam),
        "profile_n": int(p1.sum()),
        "profile_rmse_1m": round(rmse(ax_1m[p1] - ax_ref[p1]), 4),
        "profile_rmse_adaptive": round(rmse(ax_a[pa] - ax_ref[pa]), 4),
        "profile_coverage_adaptive": round(float(pa.mean()), 4),
    }
    return res, (rr, cc, ax_ref, ax_1m, ax_a)


# ---------------------------------------------------------------- 出图
def make_figure(panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    for i, p in enumerate(panels):
        pr, gt, demA, zone05, prof = p
        site, x0, y1, gs = pr["site"], pr["x0"], pr["y1"], pr["gsd"]
        H2, W2 = demA.shape
        rows, cols = np.nonzero(np.repeat(np.repeat(pr["d2"] <= pr["d3_site"] + 20, 2, 0),
                                          2, 1))
        pad = 30
        r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, H2)
        c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, W2)
        crop = demA[r0:r1, c0:c1].astype(np.float32)
        fin = np.isfinite(crop)
        filled = fill_nodata(np.where(fin, crop, 0.0), fin)
        shade = LightSource(azdeg=315, altdeg=45).hillshade(
            filled, vert_exag=1.0, dx=gs / 2, dy=gs / 2)
        ext = [x0 + c0 * gs / 2, x0 + c1 * gs / 2, y1 - r1 * gs / 2, y1 - r0 * gs / 2]

        ax = axes[i, 0]
        im = ax.imshow(filled, cmap="terrain", extent=ext, origin="upper")
        ax.imshow(shade, cmap="gray", alpha=0.45, extent=ext, origin="upper")
        ar, ac = np.nonzero(pr["ax"])
        ax.plot(x0 + (ac + 0.5) * gs, y1 - (ar + 0.5) * gs, ".", color="white",
                ms=1.6, label="骨架轴线")
        g05 = np.repeat(np.repeat(gt, 2, axis=0), 2, axis=1)[r0:r1, c0:c1].astype(float)
        yy, xx = np.mgrid[0:g05.shape[0], 0:g05.shape[1]]
        ax.contour(x0 + (c0 + xx + 0.5) * gs / 2, y1 - (r0 + yy + 0.5) * gs / 2,
                   g05, levels=[0.5], colors="#00e5ff", linewidths=0.9)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{site}：走廊自适应 DEM（0.5 m 走廊 + 1 m 外围 + 2 m 羽化）",
                     fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right")
        plt.colorbar(im, ax=ax, fraction=0.032, pad=0.02, label="高程 (m)")

        ax = axes[i, 1]
        rr, cc, ax_ref, ax_1m, ax_a = prof
        xs = x0 + (cc + 0.5) * gs
        ys = y1 - (rr + 0.5) * gs
        dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
        ax.plot(dist, ax_ref, "-", color="#333333", lw=1.6, label="dem_min（参考）")
        ax.plot(dist, ax_1m, "-", color="#1f77b4", lw=1.1, label="1 m DEM")
        ax.plot(dist, ax_a, "-", color="#d62728", lw=1.1, label="自适应 DEM (0.5 m)")
        ax.set_xlabel("沿轴线累计距离 (m)", fontsize=9)
        ax.set_ylabel("高程 (m)", fontsize=9)
        ax.grid(ls=":", alpha=0.55)
        ax.legend(fontsize=8)
        ax.set_title(f"{site}：沿轴线高程剖面", fontsize=11)

    fig.suptitle("走廊引导滤波后的 DEM 重建与沿轴剖面", fontsize=14, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIG, "dem_rebuild.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"写出 {out}")


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    results, panels = {}, []
    for site in CF.SITES:
        t0 = time.time()
        print(f"\n[{site}] 读取先验 ...", flush=True)
        pr = CF.load_prior(site)
        gt = CF.truth_mask(site, pr)
        gmr = CF.gmorph_ref(site, pr["dem"])
        print(f"  网格 {pr['H']}×{pr['W']}，真值 {int(gt.sum()):,} px，"
              f"d3={pr['d3_site']} m", flush=True)

        dem1, sub, n_keep, n_las = build_grids(pr, gmr)
        print(f"  保留 {n_keep:,} / {n_las:,} 点，1 m 有值 "
              f"{int(np.isfinite(dem1).sum()):,} 格（{time.time()-t0:.0f}s）", flush=True)

        demA, hard, zone05 = adaptive_dem(pr, dem1, sub)
        res, prof = evaluate(pr, gt, dem1, demA, sub, hard, zone05)
        res.update({
            "site": site, "n_las_points": n_las, "n_keep_points": n_keep,
            "keep_ratio": round(n_keep / max(n_las, 1), 4),
            "d3_site_m": pr["d3_site"],
            "feather_band_m": FEATHER_M,
            "sub_margin_m": SUB_MARGIN_M,
            "dem_min_reference": f"data/{site}_1m.npz dem_min（valid 外最近邻填充）",
            "adaptive_npz": f"results/dem_adaptive_{site}.npz",
            "definitions": DEFINITIONS,
            "elapsed_s": round(time.time() - t0, 1),
        })
        results[site] = res

        gsd_grid = np.where(zone05, 0.5, 1.0).astype(np.float32)
        npz = os.path.join(RES, f"dem_adaptive_{site}.npz")
        np.savez_compressed(npz, dem=demA.astype(np.float32), gsd_grid=gsd_grid,
                            dem_1m=dem1.astype(np.float32), hard=hard.astype(np.float32),
                            x0=np.float64(pr["x0"]), y1=np.float64(pr["y1"]),
                            gsd=np.float64(pr["gsd"]))
        out = os.path.join(RES, f"dem_rebuild_{site}.json")
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  写出 {npz}\n  写出 {out}（{res['elapsed_s']}s）", flush=True)
        panels.append((pr, gt, demA, zone05, prof))

    make_figure(panels)

    print("\n### DEM 重建指标（参考面 = 1 m dem_min）\n")
    print("| 站点 | 保留点 | 保留率 | 真空率 1m | 真空率 0.5m | 真空率 raw | "
          "RMSE 1m | RMSE 自适应 | seam 羽化前 | seam 羽化后 | 剖面 RMSE 1m | 剖面 RMSE 自适应 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for site in CF.SITES:
        r = results[site]
        print(f"| {site} | {r['n_keep_points']:,} | {r['keep_ratio']:.4f} | "
              f"{r['corridor_void_rate_1m']:.4f} | {r['corridor_void_rate_adaptive']:.4f} | "
              f"{r['corridor_void_rate_adaptive_raw']:.4f} | {r['corridor_rmse_1m']:.4f} | "
              f"{r['corridor_rmse_adaptive']:.4f} | {r['seam_step_hard']:.4f} | "
              f"{r['seam_step_feather']:.4f} | {r['profile_rmse_1m']:.4f} | "
              f"{r['profile_rmse_adaptive']:.4f} |")


if __name__ == "__main__":
    main()
