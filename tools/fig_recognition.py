# -*- coding: utf-8 -*-
"""蜀道识别效果展示图（两站）：地形底图 + 识别走廊 vs SHP 真值。

每站一行：
    (左) 全局：Hillshade 底图 + 交付点（红）+ 真值走廊（绿虚线）+ 模型轴线（青）
    (右) 放大窗 150 m：同一套图层，看清走廊贴合情况

数据来源（全部是既有产物，不重跑模型）：
    results/cold_start_{site}_lc6.npz          概率图 + Dijkstra 轴线
    results/cold_start_{site}_lc6_keep_idx.npy D+ 交付点索引
    data/{site}_1m.npz                         DEM（画 Hillshade）
    真值：杨家坝 = 旧 SHP 栅格；西坝村 = 新 SHP 主板（最大连通域）

用法：
    python tools/fig_recognition.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, distance_transform_edt,
                           gaussian_filter, label as ccl)
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import p2_dem as P  # noqa: E402
from eval_new_shp import RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from p2_adaptive_gsd import FIG  # noqa: E402

REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")
SITES = ("杨家坝", "西坝村")
DIST3 = {"杨家坝": 3.5, "西坝村": 3.5}      # 文献口径：半通行宽度 2 m + 1.5 m 配准容差
ZOOM_M = 150.0
SS = 4


def load_truth(site, valid, x0, y1, gsd, H, W):
    if site == "杨家坝":
        gt = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
    else:
        _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        mh = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
        gt = (mh.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5) & valid
        lb, n = ccl(gt)
        if n > 1:
            sz = np.bincount(lb.ravel())
            sz[0] = 0
            gt = lb == sz.argmax()
    return gt


def delivered_raster(site, idx, x0, y1, gsd, H, W, axis, dist3, dem, agl_max=2.0):
    """交付点索引 → 1 m 占用栅格，只保留距轴线 ≤ dist3 且 AGL ≤ agl_max 的点。

    与评分口径（候选 P≥0.5 ∧ 距轴 ≤ dist3 ∧ AGL≤2.0）一致；距离用 2D 近似，
    对"带有多宽"的展示足够，指标仍以 eval_freeaxis 的 3D 口径为准。
    """
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    o = np.load(os.path.join(REAL, f"{site}_origin.npy")).astype(np.float64)
    d2 = distance_transform_edt(~axis)
    step = max(1, int(len(idx) // 1_500_000))
    sub = idx[::step]
    occ = np.zeros((H, W), bool)
    for s in range(0, len(sub), 2_000_000):
        p = np.asarray(xyz[sub[s:s + 2_000_000]], dtype=np.float64) + o[None, :]
        c = ((p[:, 0] - x0) / gsd).astype(np.int32)
        r = ((y1 - p[:, 1]) / gsd).astype(np.int32)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        r, c, zz = r[ok], c[ok], p[ok, 2]
        keep = (d2[r, c] <= dist3) & ((zz - dem[r, c]) <= agl_max)
        occ[r[keep], c[keep]] = True
    return occ, len(sub)


def hillshade_img(dem, valid, az=315.0, alt=45.0):
    z = gaussian_filter(np.where(valid, dem, np.nanmedian(dem[valid])), 1.0)
    gy, gx = np.gradient(z, 1.0, 1.0)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    a, h = np.radians(az), np.radians(alt)
    hs = np.sin(h) * np.cos(slope) + np.cos(h) * np.sin(slope) * np.cos(a - aspect)
    hs = np.clip(hs, 0.0, 1.0)
    return np.where(valid, hs, np.nan)


def metrics(site):
    f = os.path.join(RES, f"eval_lc6_skel_{site}_skel.json")
    if not os.path.exists(f):
        return None
    e = json.load(open(f, encoding="utf-8"))
    return e["point_fair"], e["skel_f1"]["2"]


def pick_window(gt, occ, H, W, K):
    """在真值范围内找交付覆盖最好的 K×K 窗口（避免选到真值边缘）。"""
    sk = skeletonize(binary_closing(gt, np.ones((3, 3), bool)))
    labs, n = ccl(sk)
    if n > 1:
        sz = np.bincount(labs.ravel())
        sz[0] = 0
        sk = labs == sz.argmax()
    from scipy.ndimage import uniform_filter
    score = uniform_filter((sk & binary_dilation(occ, iterations=8)).astype(np.float32), K)
    score[~binary_dilation(gt, iterations=K // 4)] = -1
    r0 = int(np.clip(np.argmax(score) // W - K // 2, 0, H - K))
    c0 = int(np.clip(np.argmax(score) % W - K // 2, 0, W - K))
    return r0, c0, sk


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    rows = []
    for site in SITES:
        x0, y1, wm, hm, gsd, d = P.dem_window(site)
        H, W = d["dem_mean"].shape
        valid = d["valid"] == 1
        dem = fill_nodata(d["dem_min"], valid)
        hs = hillshade_img(dem, valid)
        z = np.load(os.path.join(RES, f"cold_start_{site}_lc6.npz"))
        axis = z["path"].astype(bool)
        prob = np.where(valid, z["prob"].astype(np.float64), 0.0)
        idx = np.load(os.path.join(RES, f"cold_start_{site}_lc6_keep_idx.npy"))
        occ, n_draw = delivered_raster(site, idx, x0, y1, gsd, H, W, axis, DIST3[site], dem)
        gt = load_truth(site, valid, x0, y1, gsd, H, W)
        print(f"[{site}] 交付点 {len(idx):,}（抽样 {n_draw:,}）→ 栅格 {int(occ.sum()):,}px；"
              f"真值 {int(gt.sum()):,}px")
        K = int(round(ZOOM_M / gsd))
        r0, c0, gt_skel = pick_window(gt, occ, H, W, K)
        rows.append(dict(site=site, x0=x0, y1=y1, gsd=gsd, H=H, W=W, hs=hs, occ=occ,
                         gt=gt, axis=axis, prob=prob, r0=r0, c0=c0, K=K,
                         m=metrics(site)))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 14.5),
                             gridspec_kw={"width_ratios": [1.25, 1], "hspace": 0.16,
                                          "wspace": 0.05})
    RED, GREEN, CYAN = "#e02424", "#0b9d58", "#00c2d1"
    from matplotlib.colors import ListedColormap
    cm_red, cm_green, cm_cyan = (ListedColormap([RED]), ListedColormap([GREEN]),
                                 ListedColormap([CYAN]))

    def draw_axis(ax, mask, ext, grow=1):
        m = binary_dilation(mask, iterations=grow) if grow else mask
        ax.imshow(np.where(m, 1.0, np.nan), cmap=cm_cyan, vmin=0, vmax=1,
                  extent=ext, interpolation="nearest", alpha=0.95)

    for k, R in enumerate(rows):
        x0, y1, gsd, H, W = R["x0"], R["y1"], R["gsd"], R["H"], R["W"]
        ext = [x0, x0 + W * gsd, y1 - H * gsd, y1]

        # ---------------- 左：全局
        ax = axes[k][0]
        ax.imshow(R["hs"], cmap="gray", vmin=0.05, vmax=0.85, extent=ext,
                  interpolation="nearest", alpha=0.95)
        ax.imshow(np.where(R["occ"], 1.0, np.nan), cmap=cm_red, vmin=0, vmax=1,
                  extent=ext, interpolation="nearest", alpha=0.55)
        ax.imshow(np.where(R["gt"], 1.0, np.nan), cmap=cm_green, vmin=0, vmax=1,
                  extent=ext, interpolation="nearest", alpha=0.22)
        ax.contour(np.linspace(ext[0], ext[1], W), np.linspace(ext[3], ext[2], H),
                   R["gt"].astype(float), levels=[0.5], colors=[GREEN],
                   linewidths=1.2, linestyles="--")
        draw_axis(ax, R["axis"], ext)
        ax.plot([], [], color=CYAN, lw=2.2, label="识别轴线")
        ax.plot([], [], color=GREEN, ls="--", lw=1.4, label="SHP 真值走廊")
        ax.plot([], [], color=RED, lw=6, alpha=0.55, label="D+ 交付带")
        rect = Rectangle((x0 + R["c0"] * gsd, y1 - (R["r0"] + R["K"]) * gsd),
                         R["K"] * gsd, R["K"] * gsd, fill=False, ec="#ffd400",
                         lw=1.8, ls="-")
        ax.add_patch(rect)
        ax.text(x0 + (R["c0"] + R["K"]) * gsd,
                y1 - (R["r0"] - 0.015 * R["K"]) * gsd,
                f"放大窗 {ZOOM_M:.0f} m", fontsize=8.5, color="#b8860b",
                va="bottom", ha="right", clip_on=False,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        mt = R["m"]
        sub = (f"P={mt[0]['P']:.3f}  R={mt[0]['R']:.3f}  IoU={mt[0]['IoU']:.3f}\n"
               f"骨架 F1@2px={mt[1]['F1']:.3f}") if mt else ""
        ax.set_title(f"{R['site']}：全局识别效果", fontsize=12)
        ax.text(0.015, 0.02, sub, transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(fc="white", ec="#999999", alpha=0.88, pad=4))
        # 比例尺 100 m（右下）+ 指北针
        sx, sy = ext[1] - 0.30 * (ext[1] - ext[0]), ext[2] + 0.045 * (ext[3] - ext[2])
        ax.plot([sx, sx + 100], [sy, sy], color="black", lw=3)
        ax.text(sx + 50, sy + 4, "100 m", ha="center", fontsize=8)
        ax.annotate("N", xy=(ext[1] - 0.05 * (ext[1] - ext[0]),
                             ext[3] - 0.045 * (ext[3] - ext[2])),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

        # ---------------- 右：放大窗
        ax = axes[k][1]
        sl = (slice(R["r0"], R["r0"] + R["K"]), slice(R["c0"], R["c0"] + R["K"]))
        e2 = [x0 + R["c0"] * gsd, x0 + (R["c0"] + R["K"]) * gsd,
              y1 - (R["r0"] + R["K"]) * gsd, y1 - R["r0"] * gsd]
        ax.imshow(R["hs"][sl], cmap="gray", vmin=0.05, vmax=0.85, extent=e2,
                  interpolation="nearest")
        ax.imshow(np.where(R["occ"][sl], 1.0, np.nan), cmap=cm_red, vmin=0, vmax=1,
                  extent=e2, alpha=0.55, interpolation="nearest")
        ax.imshow(np.where(R["gt"][sl], 1.0, np.nan), cmap=cm_green, vmin=0, vmax=1,
                  extent=e2, alpha=0.22, interpolation="nearest")
        ax.contour(np.linspace(e2[0], e2[1], R["K"]),
                   np.linspace(e2[3], e2[2], R["K"]),
                   R["gt"][sl].astype(float), levels=[0.5], colors=[GREEN],
                   linewidths=1.3, linestyles="--")
        draw_axis(ax, R["axis"][sl], e2, grow=0)
        ax.set_xlim(e2[0], e2[1])
        ax.set_ylim(e2[2], e2[3])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_title(f"{R['site']}：放大窗 {ZOOM_M:.0f} m（底图 1 m DEM 晕渲）",
                     fontsize=12)
        sx, sy = e2[1] - 0.30 * (e2[1] - e2[0]), e2[2] + 0.045 * (e2[3] - e2[2])
        ax.plot([sx, sx + 50], [sy, sy], color="black", lw=3)
        ax.text(sx + 25, sy + 4, "50 m", ha="center", fontsize=8)

    fig.suptitle("蜀道识别效果：D+ 交付带（红）与 SHP 真值走廊（绿虚线）叠加于 1 m DEM 晕渲",
                 fontsize=15, y=0.975)
    out = os.path.join(FIG, "shudao_recognition.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
