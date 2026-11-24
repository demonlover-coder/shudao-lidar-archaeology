# -*- coding: utf-8 -*-
"""P2 三大能力合成图：多分辨率 DEM / 5 通道地形因子 / 分辨率自适应。

一张 3×5 图：
  Row 1  L0~L3 多分辨率 DEM（同一窗口，内部不同分辨率）
  Row 2  slope / sin_aspect / cos_aspect / curvature / roughness（同窗口 L0 反演）
  Row 3  同一 UTM 窗口在 gsd ∈ {0.25,0.5,1,2,4} 下的 L0（分辨率自适应）

用法：
    python tools/p2_combined_figure.py --site 杨家坝
    python tools/p2_combined_figure.py --site 西坝村
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import p2_dem as P                      # noqa: E402
from p2_terrain import CHANNELS, terrain_stack   # noqa: E402

FIG = os.path.join(ROOT, "figures")
GSD_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]
# Row 2 只展示**主线 5 通道**；地形增强新增的 profile_curv / tpi15 / tpi31 不在此图展示。
# 用显式主线常量替代原先硬编码的 range(5)，并与 CHANNELS 前 5 个做一致性校验。
MAIN_CHANNELS = ("slope", "sin_aspect", "cos_aspect", "curvature", "roughness")
assert tuple(CHANNELS[:len(MAIN_CHANNELS)]) == MAIN_CHANNELS, (
    f"主线通道应为 CHANNELS 的前 {len(MAIN_CHANNELS)} 个，实际 CHANNELS={CHANNELS}")
N_MAIN = len(MAIN_CHANNELS)
ROW_TITLES = (
    "P2 多分辨率 DEM（L0~L3，内部不同分辨率）",
    f"{N_MAIN} 通道地形因子（同窗口）",
    "分辨率宽度自适应（同一 UTM 窗口）",
)
ROW_LABELS = ("① 多分辨率 DEM", f"② 地形因子（{N_MAIN} 通道）", "③ 分辨率自适应")


def main():
    ap = argparse.ArgumentParser(description="P2 三合一成图（多分辨率 / 多通道 / 自适应）")
    ap.add_argument("--site", default="杨家坝", choices=["杨家坝", "西坝村"])
    a = ap.parse_args()
    site = a.site

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(FIG, exist_ok=True)

    x0, y1, wm, hm, gsd, _ = P.dem_window(site)
    print(f"[{site}] 窗口 x0={x0:.1f} y1={y1:.1f} {wm:.0f}×{hm:.0f} m @ {gsd} m，设备 {P.DEV}")

    model = P.load_model(site)

    # Row 1：一次采样得 4 个分辨率层级
    packed = model.pack_dem(x0, y1, wm, hm, gsd, n_levels=4, device=P.DEV)
    levels = [P.to_utm(site, packed[i]).cpu().numpy().astype(np.float64) for i in range(4)]
    # Row 2：从 L0 反演 8 通道地形因子（本行只展示主线 5 通道 = MAIN_CHANNELS）
    stack = terrain_stack(levels[0], gsd=gsd)
    # Row 3：同一窗口遍历 gsd，各取 L0
    sweep = []
    for g in GSD_LIST:
        p1 = model.pack_dem(x0, y1, wm, hm, g, n_levels=1, device=P.DEV)
        sweep.append((g, P.to_utm(site, p1[0]).cpu().numpy().astype(np.float64)))
        print(f"  gsd={g:<5} → L0 {sweep[-1][1].shape}")

    fig, axes = plt.subplots(3, 5, figsize=(25, 15))

    for c in range(5):
        ax = axes[0][c]
        if c < 4:
            im = ax.imshow(levels[c], cmap="terrain")
            ax.set_title(f"L{c}   1/{2 ** c}   {levels[c].shape}", fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046, label="高程 (m)")
        ax.axis("off")

    for c, name in enumerate(MAIN_CHANNELS):
        ax = axes[1][c]
        arr = stack[CHANNELS.index(name)]
        lo, hi = np.nanpercentile(arr, (2, 98))
        im = ax.imshow(arr, cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(name, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")

    for c, (g, z) in enumerate(sweep):
        ax = axes[2][c]
        im = ax.imshow(z, cmap="terrain")
        ax.set_title(f"gsd={g} m\n{z.shape}", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, label="高程 (m)")
        ax.axis("off")

    fig.suptitle(f"P2 点云 → 多分辨率多通道 DEM（{site}，窗口 {wm:.0f}×{hm:.0f} m）",
                 fontsize=18)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.885, bottom=0.03,
                        hspace=0.40, wspace=0.32)

    # 行标题（居中置于每行上方）+ 行标签（每行左侧，竖排）
    for r in range(3):
        pos = axes[r][0].get_position()
        fig.text(0.5, pos.y1 + 0.028, ROW_TITLES[r], ha="center", va="bottom",
                 fontsize=14, fontweight="bold")
        fig.text(0.015, (pos.y0 + pos.y1) / 2, ROW_LABELS[r], rotation=90,
                 ha="center", va="center", fontsize=13, fontweight="bold")

    out = os.path.join(FIG, f"p2_combined_{site}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"写出 {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
