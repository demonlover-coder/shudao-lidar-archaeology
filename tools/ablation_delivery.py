# -*- coding: utf-8 -*-
"""消融 2：交付链（P≥0.5 → 3D 距离 → AGL）。

同一概率图与轴线，只切换交付条件，看每一步的 P/R/IoU 贡献。
全部走 eval_freeaxis，口径与主表一致（公平点级：FN 计全部真值地面点）。

    raw        仅 P≥0.5（d3、AGL 都不限）
    +d3        加 3D 管线距（d3=3.5 m，两站一致）
    +agl       加 AGL≤2.0 m
    full       两者都加（= 主表）

用法：
    python tools/ablation_delivery.py
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
CFG = {"杨家坝": dict(npz="cold_start_杨家坝_lc6.npz", d3=3.5,
                     truth=["--truth", "old"]),
       "西坝村": dict(npz="cold_start_西坝村_lc6.npz", d3=3.5,
                     truth=["--truth", "new", "--truth-main"])}
STEPS = (("raw 仅 P≥0.5", 9999.0, 999.0),
         ("+d3", None, 999.0),
         ("+AGL", 9999.0, 2.0),
         ("full", None, 2.0))


def main():
    tab = {}
    for site, c in CFG.items():
        print(f"\n=== {site}")
        row = {}
        j0 = json.load(open(os.path.join(RES, f"eval_lc6_skel_{site}_skel.json"),
                            encoding="utf-8"))
        full, full_all = j0["point_fair"], j0["point_all"]
        for name, d3, agl in STEPS:
            if name == "full":
                row[name] = {"fair": {k: full[k] for k in ("P", "R", "F1", "IoU")},
                             "all": {k: full_all[k] for k in ("P", "R", "F1", "IoU")}}
                print(f"  {name:<14s} fair {row[name]['fair']['IoU']:.4f}  "
                      f"all {row[name]['all']['IoU']:.4f}")
                continue
            d3 = c["d3"] if d3 is None else d3
            tag = f"ablate_{site}_{name.split()[0].replace('+', '')}"
            fjson = os.path.join(RES, f"{tag}_{site}_skel.json")
            if not os.path.exists(fjson):
                cmd = [sys.executable, os.path.join(SCRIPT_DIR, "eval_freeaxis.py"),
                       "--site", site, "--axis", "skel", "--npz", c["npz"],
                       "--dist3", str(d3), "--agl", str(agl), "--out", tag] + c["truth"]
                p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
                if p.returncode != 0:
                    print(p.stdout[-500:], p.stderr[-500:])
                    raise SystemExit(1)
            d = json.load(open(fjson, encoding="utf-8"))
            row[name] = {"fair": {k: d["point_fair"][k] for k in ("P", "R", "F1", "IoU")},
                         "all": {k: d["point_all"][k] for k in ("P", "R", "F1", "IoU")}}
            print(f"  {name:<14s} fair {row[name]['fair']['IoU']:.4f}  "
                  f"all {row[name]['all']['IoU']:.4f}  "
                  f"allP {row[name]['all']['P']:.4f}")
        tab[site] = row

    json.dump(tab, open(os.path.join(RES, "ablation_delivery.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    steps = [s[0] for s in STEPS]
    with open(os.path.join(RES, "ablation_delivery.csv"), "w", encoding="utf-8-sig") as f:
        f.write("step," + ",".join(f"{s}_{m}_{k}" for s in SITES
                                   for m in ("fair", "all") for k in ("P", "R", "IoU"))
                + "\n")
        for st in steps:
            f.write(st + "," + ",".join(
                f"{tab[s][st][m][k]:.4f}" for s in SITES
                for m in ("fair", "all") for k in ("P", "R", "IoU")) + "\n")
    print("\n写出 results/ablation_delivery.json / .csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=150, sharey=True)
    xs = list(range(len(steps)))
    for ax, mode, title in zip(axes, ("fair", "all"),
                               ("公平口径（仅真值地面点）", "全点口径（含树冠真值点）")):
        for i, s in enumerate(SITES):
            vals = [tab[s][st][mode]["IoU"] for st in steps]
            ax.bar([x + (i - 0.5) * 0.38 for x in xs], vals, 0.36,
                   label=s, color=("#4C78A8", "#F58518")[i])
            for x, v in zip(xs, vals):
                ax.text(x + (i - 0.5) * 0.38, v + 0.01, f"{v:.2f}",
                        ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(steps, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 0.9)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("点级 IoU")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("交付链消融：P≥0.5 → 3D 管距 → AGL", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "ablation_delivery.png"))
    print("写出 figures/ablation_delivery.png")


if __name__ == "__main__":
    main()
