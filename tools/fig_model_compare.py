# -*- coding: utf-8 -*-
"""模型容量对照汇总：读 eval_freeaxis 的评分 json，出同口径指标表与对比图。

输入
----
    results/segcmp_{net}_{site}_skel.json   公平点级 P/R/IoU、骨架 F1、Buffer IoU
    results/segcmp_{site}_{net}.json        参数量、训练耗时、epoch

输出
----
    results/model_compare_summary.json
    figures/model_compare_{site}.png

用法：
    python tools/fig_model_compare.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_new_shp import RES  # noqa: E402
from p2_adaptive_gsd import FIG  # noqa: E402

NETS = ("cnn", "unet", "deeplabv3plus", "lineformer")
LABEL = {"cnn": "CNN 基线", "unet": "U-Net",
         "deeplabv3plus": "DeepLabV3+", "lineformer": "线状网络"}
SITES = ("杨家坝", "西坝村")


def _pseudo_iou(site, prob, valid):
    """模型 P≥0.5 与其训练伪标签的像元 IoU（说明"模型只是复刻标签"）。"""
    pz = np.load(os.path.join(RES, f"cold_start_{site}_lc6.npz"))
    ps = (pz["pseudo"] >= 0.5) & valid
    pr = (prob >= 0.5) & valid
    return float((pr & ps).sum() / max((pr | ps).sum(), 1))


def _row(site, key, label, valid=None):
    fe = os.path.join(RES, f"segcmp_{key}_{site}_skel.json")
    fm = os.path.join(RES, f"segcmp_{site}_{key}.json")
    if not (os.path.exists(fe) and os.path.exists(fm)):
        return None
    e, m = json.load(open(fe, encoding="utf-8")), json.load(open(fm, encoding="utf-8"))
    pz = os.path.join(RES, f"segcmp_{site}_{key}.npz")
    pi = _pseudo_iou(site, np.load(pz)["prob"], valid) if valid is not None else float("nan")
    return dict(
        net=key, label=label, params=m["params"] / 1e3, epochs=m["epochs"],
        seconds=m["seconds"], pseudoIoU=pi,
        IoU=e["point_fair"]["IoU"], P=e["point_fair"]["P"], R=e["point_fair"]["R"],
        F1=e["point_fair"]["F1"],
        skelF1=e["skel_f1"]["2"]["F1"], skelP=e["skel_f1"]["2"]["P"],
        skelR=e["skel_f1"]["2"]["R"],
        bufIoU=e["buffer_iou_5m_band"]["1"])


def collect(site, valid=None):
    return [r for r in (_row(site, n, LABEL[n], valid) for n in NETS) if r]


def supplementary(site, valid=None):
    """DeepLabV3+ 的原生 256px 规模试跑（patch 96 对它结构不利，单独列出）。"""
    r = _row(site, "deeplabv3plus_p256", "DeepLabV3+ @256px†", valid)
    return [r] if r else []


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    summary = {}
    for site in SITES:
        valid = np.load(os.path.join(os.path.dirname(RES), "data", f"{site}_1m.npz"))["valid"] == 1
        rows = collect(site, valid)
        if not rows:
            continue
        summary[site] = rows + supplementary(site, valid)
        print(f"\n[{site}] 同口径指标（骨架轴线 + D+ 交付）")
        hdr = f"  {'模型':<18s}{'参数量':>9s}{'IoU':>8s}{'P':>8s}{'R':>8s}" \
              f"{'骨架F1':>8s}{'Buffer':>8s}{'vs标签':>8s}{'epoch':>7s}{'耗时s':>8s}"
        print(hdr)
        for r in summary[site]:
            print(f"  {r['label']:<18s}{r['params']:8.1f}k{r['IoU']:8.3f}{r['P']:8.3f}"
                  f"{r['R']:8.3f}{r['skelF1']:8.3f}{r['bufIoU']:8.3f}"
                  f"{r['pseudoIoU']:8.3f}"
                  f"{r['epochs']:7d}{r['seconds']:8.0f}")

    json.dump(summary, open(os.path.join(RES, "model_compare_summary.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---------------- 图：每个站点两个面板（指标分组柱 + 容量-精度散点）
    metrics = [("IoU", "公平 IoU"), ("P", "Precision"), ("R", "Recall"),
               ("skelF1", "骨架 F1@2px"), ("bufIoU", "Buffer IoU@1px")]
    colors = ["#1f4fd8", "#2ca02c", "#e3001b", "#8e44ad"]
    sites = [s for s in SITES if s in summary]
    fig, axes = plt.subplots(len(sites), 2, figsize=(13, 4.6 * len(sites)),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    if len(sites) == 1:
        axes = [axes]
    for k, site in enumerate(sites):
        rows = summary[site]
        main = [r for r in rows if r["net"] != "deeplabv3plus_p256"]
        ax = axes[k][0]
        x = range(len(metrics))
        w = 0.8 / max(len(main), 1)
        for i, r in enumerate(main):
            vals = [r[m] for m, _ in metrics]
            ax.bar([v + (i - (len(main) - 1) / 2) * w for v in x], vals, w,
                   color=colors[i], label=r["label"])
            for j, v in enumerate(vals):
                ax.text(j + (i - (len(main) - 1) / 2) * w, v + 0.012, f"{v:.2f}",
                        ha="center", fontsize=7)
        ax.set_xticks(list(x))
        ax.set_xticklabels([n for _, n in metrics], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{site}：同输入 / 同标签 / 同预算下的指标", fontsize=11)
        ax.legend(fontsize=8, ncol=2)

        ax2 = axes[k][1]
        off = {"cnn": (8, 6), "unet": (8, 8), "lineformer": (8, -14),
               "deeplabv3plus": (8, 6)}
        for i, r in enumerate(main):
            ax2.scatter(r["params"], r["IoU"], s=90, color=colors[i], zorder=3)
            ax2.annotate(r["label"], (r["params"], r["IoU"]), textcoords="offset points",
                         xytext=off.get(r["net"], (6, 5)), fontsize=8, color=colors[i])
        for r in rows:
            if r["net"] == "deeplabv3plus_p256":
                ax2.scatter(r["params"], r["IoU"], s=90, facecolors="none",
                            edgecolors=colors[2], zorder=3)
                ax2.annotate(r["label"], (r["params"], r["IoU"]),
                             textcoords="offset points", xytext=(6, -12),
                             fontsize=8, color=colors[2])
        ax2.set_xscale("log")
        ax2.set_xlabel("参数量（k，对数轴）")
        ax2.set_ylabel("公平点级 IoU")
        ax2.grid(alpha=0.3)
        ax2.set_title(f"{site}：容量 vs 精度", fontsize=11)

    fig.tight_layout()
    out = os.path.join(FIG, "model_compare.png")
    fig.savefig(out, dpi=160)
    print(f"\n写出 {out}")
    print(f"写出 {os.path.join(RES, 'model_compare_summary.json')}")


if __name__ == "__main__":
    main()
