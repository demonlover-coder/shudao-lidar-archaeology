# -*- coding: utf-8 -*-
"""汇报用三张对比图（plot_final_results.py）。

图 1 `figures/fig1_3d_compare.png`
    左：9 月纯回波法保留点（杨 2111 万 → 随机降采样 100 万展示）
    右：本文 2D 引导法保留点（P≥0.5，杨 138.6 万，全部展示）
    画法：手工正交投影（azim/elev 俯视），按离地高度 AGL 着色
    （棕=地面/石板，绿=树冠），按深度排序 + 深度明暗模拟 3D 渲染。
    左右两块用同一个投影中心与视角，保证可直接对照。

图 2 `figures/fig2_metric_boost.png`
    Precision / F1 双面板分组柱状图，每站两根柱（9 月回波法、本文 2D 法），
    柱上标数值，2D 柱上方标精度提升倍数。

图 3 `figures/fig3_funnel.png`
    两站各自的漏斗图：原始 LAS → 回波法初筛 → 2D 概率门控 P≥0.5，
    宽度按 sqrt(点数) 缩放（否则第三层几乎不可见）。

数据来源（全部为已实测口径，不重算滤波）
    点云：  data_real/{site}_xyz_local.npy, {site}_origin.npy
    回波法保留： data_real/{site}_filtered_回波法_idx.npy
    概率图： results/model_guided_prob_{site}.npz（fusion_train CNN 全图推理）
    真值：  results/mask_1m_{site}.npy（本脚本不读，指标直接取 results/*.json 实测值）

用法
----
    python tools/plot_final_results.py
    python tools/plot_final_results.py --site 杨家坝 --azim -50 --elev 58
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from fusion_train import load_terrain                                    # noqa: E402
from model_guided_3d_filter import bilinear                              # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")
SITES = ("杨家坝", "西坝村")
CHUNK = 4_000_000

# 离地高度色标：低=地面/石板（棕），高=树冠（绿）
CMAP = LinearSegmentedColormap.from_list("agl", [
    (0.00, "#5d3f22"), (0.10, "#8a6234"), (0.22, "#b99a63"),
    (0.36, "#8aa64a"), (0.60, "#31963f"), (1.00, "#0b6b26"),
])

# ---- 实测指标（来源 results/joint_filter.json、results/3d_tube_filter_*.json）
METRIC = {
    "杨家坝": dict(echo_p=0.0314, echo_f1=0.0604, d2_p=0.3762, d2_f1=0.4650,
                   n_raw=29_181_425, n_echo=21_111_512, n_2d=1_386_142),
    "西坝村": dict(echo_p=0.0307, echo_f1=0.0587, d2_p=0.4520, d2_f1=0.3842,
                   n_raw=91_599_004, n_echo=70_503_043, n_2d=2_320_389),
}


def style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#f7f8f9",
        "font.size": 14,
        "axes.titlesize": 16,
        "savefig.dpi": 300,
    })


# ---------------------------------------------------------------- 数据
def load_meta(site):
    import json
    m = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
    return m, float(m["x0"]), float(m["y1"]), float(m["gsd_m"])


def project_prob(site, log):
    """逐点双线性取 2D 概率；域外或 valid 外置 0。"""
    z = np.load(os.path.join(RES, f"model_guided_prob_{site}.npz"))
    prob, vmask = z["prob"].astype(np.float64), z["valid"] == 1
    H, W = prob.shape
    _, x0, y1, gsd = load_meta(site)
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    n = xyz.shape[0]
    pv = np.zeros(n, dtype=np.float32)
    vf = vmask.astype(np.float64)
    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        p = np.asarray(xyz[s:e], dtype=np.float64)
        c = (p[:, 0] + o[0] - x0) / gsd - 0.5
        r = (y1 - p[:, 1] - o[1]) / gsd - 0.5
        ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
        ii = np.flatnonzero(ins)
        if ii.size:
            v = bilinear(prob, r[ii], c[ii])
            vb = bilinear(vf, r[ii], c[ii])
            pv[s + ii] = np.where(vb > 0.5, v, 0.0)
    log(f"  概率投影 {n:,} 点（{time.time() - t0:.0f}s），P≥0.5 命中 "
        f"{int((pv >= 0.5).sum()):,}")
    return pv


def agl_of(idx, xyz, dem, meta, o):
    """给定点下标，返回离地高度（需先排序索引以加速 memmap 随机读）。"""
    x0, y1, gsd = float(meta["x0"]), float(meta["y1"]), float(meta["gsd_m"])
    order = np.argsort(idx)
    srt = idx[order]
    p = np.asarray(xyz[srt], dtype=np.float64)
    p += np.asarray(o, dtype=np.float64)[None, :]
    c = (p[:, 0] - x0) / gsd - 0.5
    r = (y1 - p[:, 1]) / gsd - 0.5
    agl = np.empty(idx.size, dtype=np.float64)
    agl[order] = p[:, 2] - bilinear(dem, r, c)
    return agl


# ---------------------------------------------------------------- 图 1
def ortho(P, center, azim, elev):
    """手工正交投影：返回 (u, v, depth)。depth 越大越靠近相机。"""
    a, e = np.radians(azim), np.radians(elev)
    X = P[:, 0] - center[0]
    Y = P[:, 1] - center[1]
    Z = P[:, 2] - center[2]
    u = -X * np.sin(a) + Y * np.cos(a)
    v = -X * np.cos(a) * np.sin(e) - Y * np.sin(a) * np.sin(e) + Z * np.cos(e)
    d = X * np.cos(a) * np.cos(e) + Y * np.sin(a) * np.cos(e) + Z * np.sin(e)
    return u, v, d


def draw_points(ax, P, agl, center, azim, elev, s, vmax=14.0, alpha=0.9):
    u, v, d = ortho(P, center, azim, elev)
    order = np.argsort(d)                     # 画家算法：远→近
    rgba = CMAP(np.clip(agl / vmax, 0.0, 1.0))
    dd = (d - d.min()) / max(float(np.ptp(d)), 1e-9)
    shade = 0.68 + 0.32 * dd                  # 远处略暗，制造纵深
    rgba[:, :3] = np.clip(rgba[:, :3] * shade[:, None], 0.0, 1.0)
    rgba[:, 3] = alpha
    ax.scatter(u[order], v[order], s=s, c=rgba[order], marker="o",
               linewidths=0, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return u, v


def fig1(a, log):
    site = a.site
    meta, _, _, _ = load_meta(site)
    dem, _, valid = load_terrain(site)
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")

    pv = project_prob(site, log)

    echo_idx = np.asarray(np.load(os.path.join(REAL,
                                               f"{site}_filtered_回波法_idx.npy"),
                                  mmap_mode="r"))
    rng = np.random.default_rng(a.seed)
    m = min(a.n_left, echo_idx.size)
    sub = rng.choice(echo_idx.size, size=m, replace=False)
    idx_l = np.sort(echo_idx[sub])
    idx_r = np.flatnonzero(pv >= 0.5)
    if idx_r.size > a.n_right:
        idx_r = np.sort(rng.choice(idx_r, size=a.n_right, replace=False))
    log(f"  左图 {m:,} 点（回波法 {echo_idx.size:,} 降采样）；"
        f"右图 {idx_r.size:,} 点（P≥0.5 全量 {int((pv >= 0.5).sum()):,}）")

    agl_l = agl_of(idx_l, xyz, dem, meta, o)
    agl_r = agl_of(idx_r, xyz, dem, meta, o)
    P_l = np.asarray(xyz[idx_l], dtype=np.float64) + o[None, :]
    P_r = np.asarray(xyz[idx_r], dtype=np.float64) + o[None, :]

    # 两块共用同一个投影中心（有效区几何中心 + 最低高程），视角可直接对照
    H_, W_ = dem.shape
    gsd = float(meta["gsd_m"])
    center = (float(meta["x0"]) + W_ * gsd / 2.0,
              float(meta["y1"]) - H_ * gsd / 2.0,
              float(np.nanmin(dem[valid])))

    fig, ax = plt.subplots(1, 2, figsize=(15.0, 7.2))
    ul, vl = draw_points(ax[0], P_l, agl_l, center, a.azim, a.elev, s=a.pt_size)
    ur, vr = draw_points(ax[1], P_r, agl_r, center, a.azim, a.elev, s=a.pt_size)
    # 两幅图共用同一地理窗口，左右可直接对照（右图只剩一条走廊带）
    pad = 0.03 * max(np.ptp(ul), np.ptp(vl))
    ax[0].set_xlim(min(ul.min(), ur.min()) - pad, max(ul.max(), ur.max()) + pad)
    ax[0].set_ylim(min(vl.min(), vr.min()) - pad, max(vl.max(), vr.max()) + pad)
    ax[1].set_xlim(*ax[0].get_xlim()); ax[1].set_ylim(*ax[0].get_ylim())

    ax[0].set_title(f"9 月纯回波法 · 保留 {echo_idx.size / 1e4:.0f} 万点"
                    f"（随机展示 {m / 1e4:.0f} 万）\n"
                    f"树冠与地面混成一片，看不出任何路的痕迹", fontsize=15)
    ax[1].set_title(f"本文 2D 概率引导 · 保留 {idx_r.size / 1e4:.1f} 万点\n"
                    f"只剩一条蜿蜒走廊：棕=路面点，绿=漏网的树冠", fontsize=15)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 14))
    cb = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01)
    cb.set_label("离地高度 AGL (m)：棕=地面/石板，绿=树冠", fontsize=12)
    fig.suptitle(f"{site} · 点云滤波效果 3D 俯视对比（视角 azim={a.azim:g}°，"
                 f"elev={a.elev:g}°，两图同一地理窗口）", fontsize=17)
    fp = os.path.join(FIG, "fig1_3d_compare.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存 {os.path.relpath(fp, ROOT)}")
    return fp


# ---------------------------------------------------------------- 图 2
def fig2(log):
    sites = list(SITES)
    panels = [("精确率 Precision", "p", "echo_p", "d2_p"),
              ("F1 分数", "f1", "echo_f1", "d2_f1")]
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 6.0))
    x = np.arange(len(sites))
    w = 0.34
    for k, (title, _, ke, kd) in enumerate(panels):
        ve = np.array([METRIC[s][ke] for s in sites])
        vd = np.array([METRIC[s][kd] for s in sites])
        b1 = ax[k].bar(x - w / 2, ve, w, color="#9aa0a6", edgecolor="#5f6368",
                       label="9 月纯回波法")
        b2 = ax[k].bar(x + w / 2, vd, w, color="#e8590c", edgecolor="#a33d05",
                       label="本文 2D 引导法")
        for bars, vals in ((b1, ve), (b2, vd)):
            for b, v in zip(bars, vals):
                ax[k].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                           ha="center", fontsize=13, fontweight="bold")
        if k == 0:                            # 只在精确率面板标提升倍数
            for xi, e, d in zip(x, ve, vd):
                ax[k].annotate(f"精度提升 {d / max(e, 1e-9):.1f} 倍",
                               xy=(xi + w / 2, d), xytext=(xi + w / 2, min(0.92, d + 0.16)),
                               ha="center", fontsize=13, color="#c92a2a",
                               fontweight="bold",
                               arrowprops=dict(arrowstyle="->", color="#c92a2a", lw=1.4))
        ax[k].set_xticks(x); ax[k].set_xticklabels(sites, fontsize=14)
        ax[k].set_ylim(0, 1.0); ax[k].set_ylabel(title, fontsize=14)
        ax[k].set_title(title, fontsize=16, pad=10)
        ax[k].grid(axis="y", alpha=0.25)
        ax[k].set_axisbelow(True)
        if k == 0:
            ax[k].legend(loc="upper left", fontsize=12, framealpha=0.9)
    fig.suptitle("纯 2D 引导 vs 9 月纯回波法：精确率与 F1", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fp = os.path.join(FIG, "fig2_metric_boost.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存 {os.path.relpath(fp, ROOT)}")
    return fp


# ---------------------------------------------------------------- 图 3
def fig3(log):
    layers = [("原始 LAS 点云", "n_raw", ""),
              ("回波法初筛", "n_echo", "几乎没去掉东西，几乎全是树"),
              ("2D 概率门控 P≥0.5", "n_2d", "剔除约 95% 树冠，只留走廊骨架")]
    colors = ["#343a40", "#868e96", "#e8590c"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))
    for ax, site in zip(axes, SITES):
        d = METRIC[site]
        ns = [d[k] for _, k, _ in layers]
        # 宽度按 sqrt 缩放（纯线性会让第三层细到看不见）；数值一律写在漏斗右侧
        ws = [np.sqrt(n / ns[0]) for n in ns]
        top = 3.0
        for i, (name, key, note) in enumerate(layers):
            n = ns[i]
            y1_, y0_ = top - i, top - i - 1
            wtop, wbot = ws[i], (ws[i + 1] if i + 1 < len(ws) else ws[i] * 0.75)
            poly = Polygon([(-wtop, y1_), (wtop, y1_), (wbot, y0_), (-wbot, y0_)],
                           closed=True, facecolor=colors[i], edgecolor="white",
                           linewidth=2.0, alpha=0.92, zorder=3)
            ax.add_patch(poly)
            ym = (y0_ + y1_) / 2.0
            ax.text(-1.14, ym, name, ha="right", va="center", fontsize=13)
            cnt = (f"{n / 1e4:,.0f} 万点" if n >= 1e7 else f"{n / 1e4:,.1f} 万点")
            ax.text(1.14, ym + (0.16 if note else 0.0), f"{cnt} · {n / ns[0] * 100:.1f}%",
                    ha="left", va="center", fontsize=13.5, fontweight="bold",
                    color="#212529")
            if note:
                ax.text(1.14, ym - 0.20, note, ha="left", va="center",
                        fontsize=11.5, color="#c92a2a")
        ax.set_xlim(-2.75, 2.55); ax.set_ylim(-0.12, 3.12)
        ax.axis("off")
        ax.set_title(f"{site}（原始 {ns[0] / 1e4:,.0f} 万点）", fontsize=16)
        ax.text(0, -0.28, "漏斗宽度按点数开方缩放（示意，非等比）",
                ha="center", fontsize=10.5, color="#868e96")
    fig.suptitle("2D 概率引导 3D 滤波的数据提纯漏斗", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fp = os.path.join(FIG, "fig3_funnel.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存 {os.path.relpath(fp, ROOT)}")
    return fp


def main():
    p = argparse.ArgumentParser(description="汇报用三张对比图")
    p.add_argument("--site", default="杨家坝", help="图 1 使用的站点")
    p.add_argument("--azim", type=float, default=-52.0)
    p.add_argument("--elev", type=float, default=56.0)
    p.add_argument("--n-left", type=int, default=1_000_000)
    p.add_argument("--n-right", type=int, default=1_500_000)
    p.add_argument("--pt-size", type=float, default=1.1, help="散点面积 pt²")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.makedirs(FIG, exist_ok=True)
    style()
    lf = open(os.path.join(RES, "plot_final_results.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True); lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 72)
    log(f"汇报图 · {time.strftime('%Y-%m-%d %H:%M:%S')}  站点={a.site}  "
        f"azim={a.azim:g} elev={a.elev:g}")
    fig1(a, log)
    fig2(log)
    fig3(log)
    lf.close()


if __name__ == "__main__":
    main()
