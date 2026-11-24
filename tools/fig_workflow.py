# -*- coding: utf-8 -*-
"""图 1 工作流示意图：支路 A（宏观寻路，地形物理约束）+ 支路 B（微观弱监督 CNN）。

输出：figures/workflow.png（300 dpi，投稿用）

用法：
    python tools/fig_workflow.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

C_A = "#1f4fd8"      # 支路 A
C_B = "#e3001b"      # 支路 B
C_OUT = "#1a7f37"    # 交付
C_REF = "#6b6b6b"    # 参考（仅评分）
C_BG_A = "#eef2fd"
C_BG_B = "#fdeef0"
C_BG_OUT = "#eef7f0"


def box(ax, x, y, w, h, fc, ec, lw=1.4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))


def arrow(ax, p0, p1, color, lw=2.0, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, ls=ls, zorder=3,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # ── 支路 A ──────────────────────────────────────────────
    box(ax, 2, 34, 45, 26, C_BG_A, C_A)
    ax.text(24.5, 57.4, "支路 A：宏观寻路（地形物理约束）", ha="center", va="center",
            fontsize=11, color=C_A, weight="bold")
    for txt, y in ((r"机载 LiDAR → 1 m DEM（$dem_{min}$）", 52.5),
                   (r"openness 代价面 $c$", 47.0),
                   (r"方向余弦惩罚 $\lambda_c\cdot pen$，$\lambda_c=6$", 42.5),
                   ("状态增广 Dijkstra（端点由实地给定）", 37.6)):
        ax.text(24.5, y, txt, ha="center", va="center", fontsize=9.2)
    for y in (54.6, 49.2, 44.6, 40.36):
        arrow(ax, (24.5, y + 0.35), (24.5, y - 0.45), C_A, lw=1.2)
    ax.text(24.5, 35.5, "→ 宏观路径（循等高线）", ha="center", va="center",
            fontsize=9.2, color=C_A, weight="bold")

    # ── 支路 B ──────────────────────────────────────────────
    box(ax, 53, 34, 45, 26, C_BG_B, C_B)
    ax.text(75.5, 57.4, "支路 B：微观弱监督分割（1.5k CNN）", ha="center", va="center",
            fontsize=11, color=C_B, weight="bold")
    for txt, y in (("4 通道：Hillshade / CSSC / 距离场 / openness", 52.5),
                   ("96×96 patch，BCE + 正样本加权，Adam", 47.0),
                   ("整图滑窗推理 → 概率图 $P$", 41.5),
                   ("$P\\geq 0.5$ → 骨架轴线", 37.0)):
        ax.text(75.5, y, txt, ha="center", va="center", fontsize=9.2)
    for y in (54.6, 49.2, 43.8, 39.2):
        arrow(ax, (75.5, y + 0.35), (75.5, y - 0.45), C_B, lw=1.2)

    # ── 支路 A → B 的物理先验传递 ───────────────────────────
    arrow(ax, (47.4, 47.0), (52.6, 47.0), "#111111", lw=2.6)
    ax.text(50.0, 50.8, "物理先验", ha="center", va="center", fontsize=9.5, weight="bold",
            bbox=dict(fc="white", ec="none", pad=1.6), zorder=4)
    ax.text(50.0, 43.2, "伪标签\n+ 距离场", ha="center", va="center", fontsize=8.2,
            color="#333333", linespacing=1.25,
            bbox=dict(fc="white", ec="none", pad=1.4), zorder=4)

    # ── 交付与产品 ──────────────────────────────────────────
    box(ax, 43, 20, 55, 9.5, C_BG_OUT, C_OUT)
    ax.text(70.5, 24.75, r"交付：3D 距轴 $\leq d_3=3.5$ m 且 AGL $\leq 2.0$ m",
            ha="center", va="center", fontsize=9.8, color=C_OUT, weight="bold")
    arrow(ax, (75.5, 33.6), (75.5, 29.7), C_OUT)

    box(ax, 2, 20, 35, 9.5, "white", C_OUT, ls="--")
    ax.text(19.5, 24.75, "走廊引导保路滤波 + 自适应 DEM", ha="center", va="center",
            fontsize=9.5, color=C_OUT)
    arrow(ax, (43.0, 23.2), (37.4, 23.2), C_OUT, lw=1.4, ls="--")

    ax.text(24.5, 16.6, "植被去除 >99.9%，路面点保留 91–92%", ha="center", va="center",
            fontsize=8.4, color="#333333")
    ax.text(24.5, 13.4, "走廊 0.5 m 子格 DEM（保路槽、剔树冠）", ha="center", va="center",
            fontsize=8.4, color="#333333")

    # ── 参考走廊只用于评分 ──────────────────────────────────
    box(ax, 2, 3.0, 42, 7.5, "#fafafa", C_REF, ls="--")
    ax.text(23.0, 6.75, "参考走廊（SHP）：仅用于最终评分", ha="center", va="center",
            fontsize=9.5, color=C_REF, weight="bold")
    ax.text(23.0, 4.6, "不进入特征 / 寻路 / 伪标签 / 训练 / 交付", ha="center", va="center",
            fontsize=8.2, color=C_REF)

    box(ax, 53, 3.0, 45, 7.5, "white", C_REF, ls="--")
    ax.text(75.5, 6.75, "评价：公平点级 IoU / 骨架 F1 / 线 IoU", ha="center", va="center",
            fontsize=9.5, color="#222222", weight="bold")
    arrow(ax, (44.2, 6.75), (52.6, 6.75), C_REF, lw=1.4, ls="--")
    arrow(ax, (75.5, 19.6), (75.5, 11.0), C_REF, lw=1.4, ls="--")

    fig.tight_layout()
    out = os.path.join(FIG, "workflow.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
