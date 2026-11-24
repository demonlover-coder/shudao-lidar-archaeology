# -*- coding: utf-8 -*-
"""λc 敏感性曲线（西坝村）：公平 IoU 与路径纵坡 P95 随 λc 的变化。

数据来源：results/eval_lc{4,5,6,7}_skel_*.json、results/eval_lcs{1,2,3,8,9,10}_*.json
纵坡 P95 由 diag_path_grade 同款算法现场计算（只读 DEM 与路径，不读 SHP）。

用法：
    python tools/fig_lambda_c_sensitivity.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import RES  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from fig_pred_vs_truth import order_path  # noqa: E402
from p2_adaptive_gsd import FIG  # noqa: E402


def load_iou():
    pts = {}
    for f in glob.glob(os.path.join(RES, "eval_lc*_西坝村_skel.json")):
        tag = os.path.basename(f)[len("eval_"):].split("_西坝村")[0].replace("_skel", "")
        m = re.fullmatch(r"lcs?(\d+(?:\.\d+)?)", tag)
        if not m:
            continue
        lc = float(m.group(1))
        pts[lc] = json.load(open(f, encoding="utf-8"))["point_fair"]["IoU"]
    return pts


def grade_p95(site, npz):
    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    dem = fill_nodata(d["dem_min"], d["valid"] == 1)
    z = np.load(npz)["path"]
    pl = order_path(z.astype(bool))
    p = np.asarray(pl, np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    si = np.arange(0.0, s[-1], 1.0)
    rr, cc = np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1])
    zz = dem[np.clip(np.round(rr).astype(int), 0, H - 1),
             np.clip(np.round(cc).astype(int), 0, W - 1)]
    g = np.abs(zz[10:] - zz[:-10]) / 10.0
    return float(np.percentile(g, 95) * 100)


def main():
    iou = load_iou()
    files = {}
    for f in glob.glob(os.path.join(RES, "cold_start_西坝村_lc*.npz")):
        tag = os.path.basename(f).split("_")[-1].replace(".npz", "")   # lc4 / lcs3
        m = re.fullmatch(r"lcs?(\d+(?:\.\d+)?)", tag)
        if m:
            files[float(m.group(1))] = f
    lcs = sorted(iou)
    grades = {l: grade_p95("西坝村", files[l]) for l in lcs if l in files}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [l for l in lcs if l in iou]
    ax.plot(xs, [iou[l] for l in xs], "o-", color="#1f4fd8", lw=2, ms=6,
            label="公平点级 IoU")
    for x, y in zip(xs, [iou[l] for l in xs]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="#1f4fd8")
    ax.axvspan(5, 10, color="#9ad0f5", alpha=0.25, label="平台区 5–10")
    ax.axvline(6, color="#e3001b", ls="--", lw=1, label="锁定值 λc=6")
    ax.set_xlabel("λc（方向余弦惩罚权重）")
    ax.set_ylabel("公平点级 IoU", color="#1f4fd8")
    ax.tick_params(axis="y", labelcolor="#1f4fd8")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    gx = [l for l in xs if l in grades]
    ax2.plot(gx, [grades[l] for l in gx], "s--", color="#2ca02c", lw=1.4,
             ms=5, label="路径纵坡 P95")
    ax2.axhline(20, color="#2ca02c", ls=":", lw=1, alpha=0.7)
    ax2.set_ylabel("10 m 窗纵坡 P95 (%)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax.set_title("西坝村 λc 敏感性：平台区 5–10（IoU 0.740–0.753）", fontsize=11)
    out = os.path.join(FIG, "lambda_c_sensitivity_西坝村.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"写出 {out}")
    print("λc → IoU / P95")
    for l in xs:
        print(f"  {l:>4g}  {iou[l]:.4f}  {grades.get(l, float('nan')):.0f}%")


if __name__ == "__main__":
    main()
