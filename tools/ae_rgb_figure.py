# -*- coding: utf-8 -*-
"""AE 前 3 维瓶颈 → RGB 合成图（叠加 SHP 走廊真值轮廓 + 各维命名/AUC 图例）。

目的
----
把 `tools/train_ae.py` 学到的 [N, latent] 瓶颈取前 3 维，valid 内分位映射到 R/G/B，
与稀疏 PCA 的 `figures/shudao_pc_rgb_{site}.png` 同风格对照：看非线性 AE 的
低维特征空间在走廊处是否有可见结构。

说明
----
  - 每维独立做 valid 内**分位（秩）映射** → 三通道铺满，不受长尾挤压；
  - R/G/B = dim0/dim1/dim2（顺序固定，图例标明每维命名与 AUC）；
  - 白色轮廓 = SHP 走廊真值（results/mask_1m_{site}.npy）；
  - valid 之外留白。

用法
----
    python tools/ae_rgb_figure.py --site 西坝村 --latent 8
    python tools/ae_rgb_figure.py --site 杨家坝 --latent 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

DIMS = (0, 1, 2)                      # 前 3 维 → RGB
RGB_TAG = ("R", "G", "B")


def main():
    ap = argparse.ArgumentParser(description="AE 前 3 维 RGB 合成图")
    ap.add_argument("--site", required=True, choices=["西坝村", "杨家坝"])
    ap.add_argument("--latent", type=int, default=8)
    a = ap.parse_args()
    site = a.site

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(FIG, exist_ok=True)

    npz = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    valid = npz["valid"] == 1
    H, W = valid.shape
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    Z = np.load(os.path.join(DATA, f"ae_latent_{site}_L{a.latent}.npy"))
    rows = np.flatnonzero(valid.ravel())

    with open(os.path.join(RES, f"ae_interpret_L{a.latent}.json"), encoding="utf-8") as f:
        interp = json.load(f)[site]
    meta_path = os.path.join(RES, f"ae_meta_{site}_L{a.latent}.json")
    r2 = json.load(open(meta_path, encoding="utf-8"))["r2"] if os.path.exists(meta_path) else float("nan")

    rgb = np.ones((H, W, 3), dtype=np.float64)
    legend_rows = []
    for c, d in enumerate(DIMS):
        # 对 valid 行做分位映射，再回填栅格
        ch = np.full(H * W, 0.5, dtype=np.float64)
        r = rankdata(Z[:, d], method="average")
        ch[rows] = (r - 1.0) / max(len(r) - 1.0, 1.0)
        rgb[..., c] = ch.reshape(H, W)
        dim = interp["dims"][d]
        loads = " ".join(f"{t['load']:+.2f}·{t['factor']}" for t in dim["top3"])
        legend_rows.append(f"{RGB_TAG[c]} = dim{d}  AUC={dim['auc']:.4f}\n     {loads}")
    rgb[~valid] = 1.0

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.25], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb)
    if label.any():
        ax.contour(label.astype(float), levels=[0.5], colors="white", linewidths=0.9)
    ax.set_title(f"AE 瓶颈前 3 维 RGB 合成图（{site}，latent={a.latent}，重建 R²={r2:.4f}）\n"
                 f"R/G/B = dim{DIMS[0]}/dim{DIMS[1]}/dim{DIMS[2]}（各维 valid 内分位映射）",
                 fontsize=13)
    ax.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    handles = [Line2D([], [], color="white", lw=2, label="白色轮廓 = SHP 走廊真值")]
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 0.98),
               fontsize=10, frameon=False)
    txt = ("AE 瓶颈维命名 / 走廊 AUC\n\n" + "\n\n".join(legend_rows) +
           "\n\ntpi31 基准（修正口径）：\n  西坝村 0.6691 / 杨家坝 0.7148\n\n"
           "判据：无任一维两站 AUC 均 > tpi31\n  → 主通道仍 tpi31，AE 作辅助特征")
    ax2.text(0.0, 0.86, txt, fontsize=9.5, va="top", ha="left",
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.7"))

    out = os.path.join(FIG, f"ae_rgb_{site}_L{a.latent}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"写出 {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
