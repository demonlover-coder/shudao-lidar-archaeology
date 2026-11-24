# -*- coding: utf-8 -*-
"""主结果汇总：面积口径 + 线评价口径同时保留。

面积口径（eval_lc6_skel_*）：点级公平 IoU、骨架 F1@1/2px、Buffer IoU@0/1/2px
线口径（line_metric.json）：COM / COR / 线 IoU（r=1/2/3 m 等）

输出：
    results/main_metrics.json / main_metrics.csv
    figures/main_metrics.png

用法：
    python tools/summarize_metrics.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from eval_new_shp import RES  # noqa: E402
from p2_adaptive_gsd import FIG  # noqa: E402

SITES = ("杨家坝", "西坝村")
SRC = {"杨家坝": "eval_lc6_skel_杨家坝_skel.json",
       "西坝村": "eval_lc6_skel_西坝村_skel.json"}
R_MAIN = 2.0


def main():
    line = json.load(open(os.path.join(RES, "line_metric.json"), encoding="utf-8"))
    rows = []
    for site in SITES:
        e = json.load(open(os.path.join(RES, SRC[site]), encoding="utf-8"))
        pf, sf, bi = e["point_fair"], e["skel_f1"], e["buffer_iou_5m_band"]
        lr = {q["r"]: q for q in line[site]["rows"]}
        q = lr[R_MAIN]
        rows.append(dict(
            site=site,
            # 面积 / 交付口径
            P=pf["P"], R=pf["R"], F1=pf["F1"], IoU=pf["IoU"],
            skelF1_1px=sf["1"]["F1"], skelF1_2px=sf["2"]["F1"],
            bufferIoU_0px=bi["0"], bufferIoU_1px=bi["1"], bufferIoU_2px=bi["2"],
            # 线口径（r=2 m）
            line_r=float(R_MAIN), line_COM=q["COM"], line_COR=q["COR"],
            line_IoU=q["line_IoU"],
            line_IoU_r1=lr[1.0]["line_IoU"], line_IoU_r3=lr[3.0]["line_IoU"],
            len_pred_px=q["len_pred"], len_truth_px=q["len_truth"],
            source=[SRC[site], "line_metric.json"],
        ))

    json.dump(rows, open(os.path.join(RES, "main_metrics.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    keys = ["site", "P", "R", "F1", "IoU", "skelF1_1px", "skelF1_2px",
            "bufferIoU_0px", "bufferIoU_1px", "bufferIoU_2px",
            "line_r", "line_COM", "line_COR", "line_IoU",
            "line_IoU_r1", "line_IoU_r3", "len_pred_px", "len_truth_px"]
    with open(os.path.join(RES, "main_metrics.csv"), "w", encoding="utf-8-sig") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    names = ["公平 IoU\n(点级/面积)", "骨架 F1\n@2px", "Buffer IoU\n@1px",
             "线 IoU\n(r=2 m)"]
    vals = {s: [r["IoU"], r["skelF1_2px"], r["bufferIoU_1px"], r["line_IoU"]]
            for s, r in zip(SITES, rows)}
    x = np.arange(len(names))
    w = 0.36
    for i, s in enumerate(SITES):
        b = ax.bar(x + (i - 0.5) * w, vals[s], w,
                   color=["#2c7bb6", "#d7191c"][i], label=s)
        for bb, v in zip(b, vals[s]):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("指标值")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("主结果：面积口径与线评价口径并列（西坝村=主板真值）", fontsize=12)
    fp = os.path.join(FIG, "main_metrics.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"写出 results/main_metrics.json / .csv 与 {fp}")


if __name__ == "__main__":
    main()
