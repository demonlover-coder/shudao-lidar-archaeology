# -*- coding: utf-8 -*-
"""审稿防御实验汇总图：捷径 / 跨站 / 地形置换 / 种子稳定性。

输入：results/exp_shortcut.json, exp_cross*.json, exp_shuffle.json, exp_seeds_cnn.json
输出：figures/review_pack.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_new_shp import RES  # noqa: E402
from p2_adaptive_gsd import FIG  # noqa: E402


def load(name):
    return json.load(open(os.path.join(RES, name), encoding="utf-8"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    sc = load("exp_shortcut.json")
    cr = load("exp_cross.json") if os.path.exists(
        os.path.join(RES, "exp_cross.json")) else load("exp_cross_cnn.json")
    cru = load("exp_cross_unet.json")
    sh = load("exp_shuffle.json")
    sd = load("exp_seeds_cnn.json")
    mc = load("model_compare_summary.json")
    in_site = {s: [r for r in mc[s] if r["net"] == "cnn"][0]["IoU"] for s in ("杨家坝", "西坝村")}
    unet_site = {s: [r for r in mc[s] if r["net"] == "unet"][0]["IoU"] for s in ("杨家坝", "西坝村")}

    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    sites = ("杨家坝", "西坝村")

    # ---- (a) 去掉距离场
    x = np.arange(2)
    w = 0.2
    vals = {"CNN-4ch": [in_site[s] for s in sites],
            "CNN-3ch": [sc[f"{s}_cnn"]["IoU"] for s in sites],
            "U-Net-4ch": [unet_site[s] for s in sites],
            "U-Net-3ch": [sc[f"{s}_unet"]["IoU"] for s in sites]}
    cols = ["#1f4fd8", "#9ecae1", "#e3001b", "#fcaeae"]
    for i, (k, v) in enumerate(vals.items()):
        b = ax[0][0].bar(x + (i - 1.5) * w, v, w, color=cols[i], label=k)
        for bb, vv in zip(b, v):
            ax[0][0].text(bb.get_x() + bb.get_width() / 2, vv, f"{vv:.2f}",
                          ha="center", va="bottom", fontsize=7)
    ax[0][0].set_xticks(x)
    ax[0][0].set_xticklabels(sites)
    ax[0][0].set_ylim(0, 1)
    ax[0][0].set_title("① 去掉距离场通道：纯视觉盲测", fontsize=11)
    ax[0][0].grid(axis="y", alpha=0.3)
    ax[0][0].legend(fontsize=8, ncol=2, loc="upper left")

    # ---- (b) 跨站
    x = np.arange(2)
    lab = ["杨→西", "西→杨"]
    v4 = [cr["杨家坝->西坝村_4ch"]["IoU"], cr["西坝村->杨家坝_4ch"]["IoU"]]
    v3 = [cr["杨家坝->西坝村_3ch"]["IoU"], cr["西坝村->杨家坝_3ch"]["IoU"]]
    u3 = [cru["杨家坝->西坝村_3ch"]["IoU"], cru["西坝村->杨家坝_3ch"]["IoU"]]
    for i, (k, v, c) in enumerate((("4ch（含距离场）", v4, "#2c7bb6"),
                                   ("3ch CNN", v3, "#e02424"),
                                   ("3ch U-Net", u3, "#f4a582"))):
        b = ax[0][1].bar(x + (i - 1) * 0.26, v, 0.26, color=c, label=k)
        for bb, vv in zip(b, v):
            ax[0][1].text(bb.get_x() + bb.get_width() / 2, max(vv, 0.01), f"{vv:.2f}",
                          ha="center", va="bottom", fontsize=7)
    ax[0][1].axhline(np.mean(list(in_site.values())), color="k", ls=":", lw=1)
    ax[0][1].set_xticks(x)
    ax[0][1].set_xticklabels(lab)
    ax[0][1].set_ylim(0, 1)
    ax[0][1].set_title("② 零样本跨站（虚线=本站训练均值）", fontsize=11)
    ax[0][1].grid(axis="y", alpha=0.3)
    ax[0][1].legend(fontsize=8)

    # ---- (c) 地形置换
    x = np.arange(2)
    real = [sh[s]["real"]["IoU"] for s in sites]
    shuf = [sh[s]["shuffled"]["IoU"] for s in sites]
    b1 = ax[1][0].bar(x - 0.2, real, 0.38, color="#00a6b8", label="真实地形")
    b2 = ax[1][0].bar(x + 0.2, shuf, 0.38, color="#e02424", label="高程随机打乱")
    for bb, v in list(zip(b1, real)) + list(zip(b2, shuf)):
        ax[1][0].text(bb.get_x() + bb.get_width() / 2, v, f"{v:.2f}", ha="center",
                      va="bottom", fontsize=8)
    for i, s in enumerate(sites):
        d = (real[i] - shuf[i]) * 100
        ax[1][0].annotate(f"-{d:.0f} pt", (x[i], max(real[i], shuf[i]) + 0.05),
                          ha="center", fontsize=9, color="#b00000")
    ax[1][0].set_xticks(x)
    ax[1][0].set_xticklabels(sites)
    ax[1][0].set_ylim(0, 1)
    ax[1][0].set_title("③ 地形置换：λc=6 在乱码 DEM 上失效", fontsize=11)
    ax[1][0].grid(axis="y", alpha=0.3)
    ax[1][0].legend(fontsize=8)

    # ---- (d) 种子
    for i, s in enumerate(sites):
        ious = [r["IoU"] for r in sd[s]["runs"]]
        ax[1][1].scatter([i] * len(ious), ious, s=70, color=["#1f4fd8", "#e3001b"][i])
        ax[1][1].errorbar(i, sd[s]["mean"], yerr=sd[s]["std"], color="k", capsize=8)
        ax[1][1].text(i + 0.12, sd[s]["mean"],
                      f"{sd[s]['mean']:.3f}±{sd[s]['std']:.3f}", fontsize=9, va="center")
    ax[1][1].set_xticks([0, 1])
    ax[1][1].set_xticklabels(sites)
    ax[1][1].set_ylim(0.5, 0.85)
    ax[1][1].set_title("④ 种子 42/2023/2024：公平 IoU", fontsize=11)
    ax[1][1].grid(alpha=0.3)

    fig.suptitle("审稿防御实验：捷径洗清 / 跨站 / 地形置换 / 种子稳定性", fontsize=14)
    out = os.path.join(FIG, "review_pack.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
