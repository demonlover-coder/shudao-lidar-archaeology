# -*- coding: utf-8 -*-
"""线评价（中心线缓冲区口径）：真值骨干线 vs 预测骨干线。

对应审稿意见里的"参考结果匹配"图：
    · 真值骨干线  = SHP 面骨架
    · 预测骨干线  = P≥0.5 掩膜骨架（与 eval_freeaxis 的 skel 轴同源）
    · 缓冲区半径 r（米）：以对方骨干线做缓冲区，落在其中视为"匹配"

三个指标（长度均按 1 m 像元计）：
    COM 完整性 = 落在预测缓冲区的真值长度 / 真值总长
    COR 正确性 = 落在真值缓冲区的预测长度 / 预测总长
    线 IoU     = TP / (TP + FP + FN)
                 TP = 预测落在真值缓冲区的长度，FP = 预测在缓冲区外，
                 FN = 真值落在预测缓冲区外的长度

同时给出既有面积口径（公平 IoU）作参照。

输出：results/line_metric.json、results/line_metric.csv、figures/line_metric.png

用法：
    python tools/eval_line_met.py                 # 两站，r=1..5 m
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import p2_dem as P                                            # noqa: E402
from eval_new_shp import RES                                  # noqa: E402
from fig_recognition import load_truth                        # noqa: E402
from p2_adaptive_gsd import FIG                               # noqa: E402

SITES = ("杨家坝", "西坝村")
RADII = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def main():
    out = {}
    for site in SITES:
        x0, y1, wm, hm, gsd, d = P.dem_window(site)
        H, W = d["dem_mean"].shape
        valid = d["valid"] == 1
        z = np.load(os.path.join(RES, f"cold_start_{site}_lc6.npz"))
        prob = np.where(valid, z["prob"].astype(np.float64), 0.0)
        pred_sk = skeletonize(binary_closing((prob >= 0.5) & valid,
                                             np.ones((3, 3), bool))) & valid
        gt = load_truth(site, valid, x0, y1, gsd, H, W)
        gt_sk = skeletonize(gt) & valid
        dp = distance_transform_edt(~pred_sk) * gsd     # 到预测骨干线距离（米）
        dg = distance_transform_edt(~gt_sk) * gsd       # 到真值骨干线距离（米）
        lp, lg = int(pred_sk.sum()), int(gt_sk.sum())
        rows = []
        for r in RADII:
            tp = int((pred_sk & (dg <= r)).sum())
            com = float((gt_sk & (dp <= r)).sum()) / max(lg, 1)
            cor = tp / max(lp, 1)
            fn = lg - int((gt_sk & (dp <= r)).sum())
            iou = tp / max(tp + (lp - tp) + fn, 1)
            rows.append(dict(r=r, COM=round(com, 4), COR=round(cor, 4),
                             line_IoU=round(iou, 4), len_pred=lp, len_truth=lg))
        out[site] = dict(pred_px=lp, truth_px=lg, rows=rows)
        print(f"[{site}] 预测骨干 {lp:,} px，真值骨干 {lg:,} px")
        for q in rows:
            print(f"  r={q['r']:.1f} m  COM {q['COM']:.4f}  COR {q['COR']:.4f}  "
                  f"线IoU {q['line_IoU']:.4f}")

    json.dump(out, open(os.path.join(RES, "line_metric.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(RES, "line_metric.csv"), "w",
              encoding="utf-8-sig") as f:
        f.write("site,r_m,COM,COR,line_IoU,len_pred_px,len_truth_px\n")
        for s in SITES:
            for q in out[s]["rows"]:
                f.write(f"{s},{q['r']},{q['COM']},{q['COR']},{q['line_IoU']},"
                        f"{q['len_pred']},{q['len_truth']}\n")

    # ---------------- 图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), constrained_layout=True)
    for ax, site in zip(axes, SITES):
        q = out[site]["rows"]
        r = [v["r"] for v in q]
        ax.plot(r, [v["COM"] for v in q], "o-", color="#0b9d58", label="COM 完整性")
        ax.plot(r, [v["COR"] for v in q], "s-", color="#e02424", label="COR 正确性")
        ax.plot(r, [v["line_IoU"] for v in q], "^-", color="#1f4fd8",
                label="线 IoU")
        ax.set_title(f"{site}：中心线缓冲区评价", fontsize=11)
        ax.set_xlabel("缓冲区半径 (m)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("线评价：真实骨干线/预测骨干线的缓冲区匹配", fontsize=13)
    fp = os.path.join(FIG, "line_metric.png")
    fig.savefig(fp, dpi=140, bbox_inches="tight")
    print(f"写出 results/line_metric.json / .csv 与 {fp}")


if __name__ == "__main__":
    main()
