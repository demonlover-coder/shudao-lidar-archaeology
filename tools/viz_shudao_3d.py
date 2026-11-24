# -*- coding: utf-8 -*-
"""杨家坝 / 西坝村 三维蜀道图。

数据源（只读，不重算任何因子）：
  data/{site}_1m.npz 的 dem_mean（仅 valid=1 处有效）、label（走廊真值 0/1）

做法：
  * 只在真实测点附近 25 m 内用最近有效像元补值，更远处整片透明——
    避免最近邻外推造出"假平台"，把没有证据的区域画成一片平地；
  * 按站点自适应降采样（dem 块均值、label 块内存在即真），保留细长走廊；
  * 光照贴图（LightSource）给地形上色，走廊真值以骨架中心线高亮；
  * 两站并排、同一视角、同一高程色标，便于目视对照。

用法：
  python tools/viz_shudao_3d.py
  python tools/viz_shudao_3d.py --out results/shudao_3d.png --dpi 220
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LightSource, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ROOT, "data")

SITES = ["杨家坝", "西坝村"]
ROAD_COLOR = "#E8463A"          # 走廊真值高亮色
# 土色地形渐变：低处深绿 → 坡地黄绿 → 梁峁土黄 → 高处褐
TERRAIN_CMAP = LinearSegmentedColormap.from_list("shudao", [
    "#1E4A3B", "#2F6046", "#4F7C4B", "#7D9C55",
    "#A9B36A", "#CDBD8C", "#C39A6B", "#9E7350",
])
MAX_FILL_M = 25                 # 只在距真实测点 25 m 内补值，更远处不画
VERT_EXAG = 3.0                 # 纵向夸张倍数（同一倍数，两站可比）


def load_site(site: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (显示用 DEM, 可见掩码, 走廊标签)，没有证据的区域用掩码排除。"""
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    dem = d["dem_mean"].astype(np.float32)
    obs = (d["valid"] == 1) & np.isfinite(dem)
    label = d["label"].astype(bool)
    dist = distance_transform_edt(~obs)
    vis = dist <= MAX_FILL_M
    idx = distance_transform_edt(~obs, return_distances=False, return_indices=True)
    dem_filled = dem[tuple(idx)].astype(np.float32)
    return dem_filled, vis, label


def block_reduce(a: np.ndarray, k: int, mode: str) -> np.ndarray:
    """块内聚合：'mean' 均值（地形）、'any' 存在即真（走廊）、'max' 最大（掩码）。"""
    H, W = a.shape
    H2, W2 = H // k * k, W // k * k
    b = a[:H2, :W2].reshape(H2 // k, k, W2 // k, k)
    if mode == "mean":
        return b.mean(axis=(1, 3))
    if mode == "any":
        return b.any(axis=(1, 3))
    return b.max(axis=(1, 3))


def render_site(ax, site: str, norm: Normalize, ls: LightSource) -> dict:
    dem, vis, label = load_site(site)
    ys, xs = np.where(vis)
    r0, r1 = max(ys.min() - 12, 0), min(ys.max() + 13, dem.shape[0])
    c0, c1 = max(xs.min() - 12, 0), min(xs.max() + 13, dem.shape[1])
    dem = dem[r0:r1, c0:c1]
    vis = vis[r0:r1, c0:c1]
    label = label[r0:r1, c0:c1]

    k = max(2, int(round(max(dem.shape) / 190)))
    z = block_reduce(dem, k, "mean")
    cover = block_reduce(vis.astype(np.float32), k, "max")
    corr = block_reduce(label.astype(np.float32), k, "any")
    H, W = z.shape
    X, Y = np.meshgrid(np.arange(W) * k, np.arange(H) * k)      # 米（1 m GSD）
    zmin, zmax = float(z.min()), float(z.max())

    # 光照贴图：按纵向夸张倍数计算坡向明暗
    rgb = ls.shade(z, cmap=plt.get_cmap(TERRAIN_CMAP), vert_exag=VERT_EXAG,
                   norm=norm, blend_mode="soft")
    # 走廊带：面片着色 + 轻微抬升，保证斜视时不被山体遮挡
    lift = 0.010 * (zmax - zmin)
    zz = np.where(corr, z + lift, z)
    rgb[corr] = matplotlib.colors.to_rgba(ROAD_COLOR)
    # 无证据区透明：不画"假平台"
    rgb[..., 3] = np.where(cover, 1.0, 0.0)

    ax.plot_surface(X, Y, zz, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    # 骨架中心线：让 1 px 宽的走廊在斜视下也读得出来（在降采样后的走廊带上取骨架）
    sk = skeletonize(corr)
    sy, sx = np.where(sk)
    if sy.size:
        ax.scatter(sx * k, sy * k, z[sy, sx] + lift * 2.0, s=4,
                   c=ROAD_COLOR, depthshade=False, linewidths=0)

    ax.view_init(elev=37, azim=-62)
    ax.set_box_aspect((W * k, H * k, (zmax - zmin) * VERT_EXAG), zoom=1.58)
    ax.set_axis_off()

    ax.set_title(f"{site}\n高程 {zmin:.0f}–{zmax:.0f} m ｜ 走廊真值 {int(label.sum()):,} 像元",
                 fontsize=12, pad=6, linespacing=1.5)
    return dict(zmin=zmin, zmax=zmax, n_corr=int(label.sum()),
                cover=float(vis.mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "shudao_3d.png"))
    ap.add_argument("--dpi", type=int, default=220)
    a = ap.parse_args()

    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    dems = [load_site(s)[0] for s in SITES]
    # 每站独立色标：两站海拔差 300 m，共用色标会把杨家坝压成一团深色
    norms = [Normalize(vmin=float(d.min()), vmax=float(d.max())) for d in dems]
    norms = [Normalize(vmin=math.floor(float(d.min()) / 20) * 20,
                       vmax=math.ceil(float(d.max()) / 20) * 20) for d in dems]
    ls = LightSource(azdeg=315, altdeg=42)
    cmap = plt.get_cmap(TERRAIN_CMAP)

    fig = plt.figure(figsize=(15.5, 7.6), facecolor="white")
    fig.subplots_adjust(top=0.885, bottom=0.150, left=0.0, right=1.0, wspace=0.0)
    for i, site in enumerate(SITES):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.set_facecolor("white")
        print(f"[{site}] 渲染中…")
        render_site(ax, site, norms[i], ls)
        cax = fig.add_axes([0.215 + i * 0.40, 0.088, 0.180, 0.015])
        cb = fig.colorbar(ScalarMappable(norm=norms[i], cmap=cmap),
                          cax=cax, orientation="horizontal")
        cb.set_label("高程 (m)", fontsize=10)
        cb.ax.tick_params(labelsize=9)

    fig.text(0.5, 0.968, "杨家坝 · 西坝村  三维蜀道地形", ha="center",
             fontsize=18, fontweight="bold")
    fig.text(0.5, 0.930,
             "1 m 网格 DEM ｜ 光照贴图 ｜ 红色 = 走廊真值（SHP）｜ 纵向夸张 ×3 ｜ 各站独立色标",
             ha="center", fontsize=10, color="#52525B")
    fig.text(0.5, 0.022,
             "测点覆盖仅 27–29%：距实测点 25 m 以内用最近邻补值，之外留空（不画无证据的平地）",
             ha="center", fontsize=9, color="#71717A")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    print(f"已保存: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
