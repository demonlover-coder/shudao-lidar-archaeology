# -*- coding: utf-8 -*-
"""用文献口径（AGL=2.0 m，d3=3.5 m）重评模型对比，不重训。

读 results/segcmp_{site}_{net}.npz → eval_freeaxis → 回写 model_compare_summary.json
并重绘 figures/model_compare.png。

用法：
    python tools/reeval_models.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RES = os.path.join(ROOT, "results")
SITES = ("杨家坝", "西坝村")
ARGS = {"杨家坝": ["--truth", "old", "--dist3", "3.5"],
        "西坝村": ["--truth", "new", "--truth-main", "--dist3", "3.5"]}


def main():
    sm = os.path.join(RES, "model_compare_summary.json")
    d = json.load(open(sm, encoding="utf-8"))
    for site in SITES:
        for row in d[site]:
            net = row["net"]
            npz = os.path.join(RES, f"segcmp_{site}_{net}.npz")
            if not os.path.exists(npz):
                print(f"跳过（无 npz）：{site} {net}")
                continue
            out = f"segcmp_{net}"
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, "eval_freeaxis.py"),
                   "--site", site, "--axis", "skel",
                   "--npz", f"segcmp_{site}_{net}.npz",
                   "--agl", "2.0", "--out", out] + ARGS[site]
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            f = os.path.join(RES, f"segcmp_{net}_{site}_skel.json")
            if p.returncode != 0 or not os.path.exists(f):
                print(p.stdout[-400:], p.stderr[-400:])
                raise SystemExit(1)
            e = json.load(open(f, encoding="utf-8"))
            pf = e["point_fair"]
            row.update(IoU=pf["IoU"], P=pf["P"], R=pf["R"], F1=pf["F1"],
                       skelF1=e["skel_f1"]["2"]["F1"],
                       bufIoU=e["buffer_iou_5m_band"]["1"],
                       eval_agl=2.0, eval_dist3=float(ARGS[site][-1]))
            print(f"{site} {net:<18s} IoU {row['IoU']:.4f}  P {row['P']:.4f}  "
                  f"R {row['R']:.4f}")
    json.dump(d, open(sm, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("回写 results/model_compare_summary.json")
    subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "fig_model_compare.py")],
                   cwd=ROOT)


if __name__ == "__main__":
    main()
