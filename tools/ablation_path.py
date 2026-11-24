# -*- coding: utf-8 -*-
"""消融 1：寻路代价项与基线（不重训模型）。

在同一 openness + h1 相位场、同一端点、同一膨胀宽度下，只改代价项，比较
1 m 掩膜 IoU（路径膨胀后 vs SHP 真值面）。用来回答"物理先验的哪一项在起作用"。

配置
----
    straight      两站端点直连（无地形，最朴素基线）
    otsu_open     对 openness 做 Otsu 阈值（无寻路的经典基线）
    openness      仅 openness 代价的 Dijkstra（无方向/转角/坡度项）
    +dir          加 h1 方向先验（λ_dir=1.5）
    +dir+turn     再加转角惩罚（λ_turn=0.2）
    +dir+turn+λz  再加高差惩罚（λz=2.0，历史口径）
    +dir+turn+λc  再加方向余弦惩罚（λc=6.0，最终口径）

SHP 只用于最后的 IoU 评分。

用法：
    python tools/ablation_path.py
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label
from skimage.draw import line as skline
from skimage.filters import threshold_otsu
from skimage.morphology import disk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import cold_start_transfer as C                                      # noqa: E402
import p2_dem as P                                                   # noqa: E402
from eval_new_shp import RES, SHP_BASE, SHP_NEW, read_shp_any        # noqa: E402
from eval_line_iou import poly_mask_hr                               # noqa: E402
from path_extraction import find_endpoints, merge_and_pick_pair      # noqa: E402

SITES = ("杨家坝", "西坝村")
GATE = {"杨家坝": 3, "西坝村": 5}          # 与伪标签/最终训练一致的膨胀半径
SS = 4
BASE = dict(dir_lambda=1.5, turn_lambda=0.2, turn_min=1, slope_lambda=0.0,
            max_grade=0.0, grade_lag=5)
CONFIGS = [
    ("openness",      dict(lam_c=0.0, z_lambda=0.0, dir_lambda=0.0, turn_lambda=0.0)),
    ("+dir",          dict(lam_c=0.0, z_lambda=0.0, turn_lambda=0.0)),
    ("+dir+turn",     dict(lam_c=0.0, z_lambda=0.0)),
    ("+dir+turn+λz",  dict(lam_c=0.0, z_lambda=2.0)),
    ("+dir+turn+λc",  dict(lam_c=6.0, z_lambda=0.0)),
]


def load_truth(site, valid, x0, y1, gsd, H, W):
    if site == "杨家坝":
        gt = np.load(os.path.join(RES, "mask_1m_杨家坝.npy")).astype(bool) & valid
    else:
        _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        hr = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
        gt = hr.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5
        gt &= valid
    lb, n = cc_label(gt)
    if n > 1:                                   # 主板口径
        sz = np.bincount(lb.ravel())
        sz[0] = 0
        gt = lb == sz.argmax()
    return gt


def iou(mask, gt):
    u = int((mask | gt).sum())
    return float((mask & gt).sum() / u) if u else 0.0


def main():
    out = {}
    for site in SITES:
        print(f"\n=== {site}")
        dem, valid = C.load_dem(site, log=print)
        x0, y1, wm, hm, gsd, d = P.dem_window(site)
        H, W = dem.shape
        gt = load_truth(site, valid, x0, y1, gsd, H, W)
        print(f"  真值主板 {int(gt.sum()):,} px")

        op, n_row, n_col, conf, _ = C.macro_fields(dem, valid)
        pat = np.load(os.path.join(RES, f"cold_start_{site}_lc6.npz"))["path"] > 0
        ends = find_endpoints(pat)
        p, q = merge_and_pick_pair(np.argwhere(ends))
        ep = ((float(p[0]), float(p[1])), (float(q[0]), float(q[1])))
        g = GATE[site]
        k = disk(g).astype(bool)

        row = {}
        # 基线 1：端点直连
        ln = np.zeros((H, W), bool)
        rr, cc = skline(int(round(p[0])), int(round(p[1])),
                        int(round(q[0])), int(round(q[1])))
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        ln[rr[ok], cc[ok]] = True
        row["straight 端点直连"] = iou(binary_dilation(ln, k) & valid, gt)
        print(f"  straight          IoU {row['straight 端点直连']:.4f}")

        # 基线 2：openness 的 Otsu 阈值
        thr = float(threshold_otsu(op[valid]))
        row["otsu 阈值(openness)"] = iou((op < thr) & valid, gt)
        print(f"  otsu(openness)    IoU {row['otsu 阈值(openness)']:.4f}  thr={thr:.3f}")

        for name, kw in CONFIGS:
            a = SimpleNamespace(**{**BASE, **kw})
            path, _, _, _ = C.macro_path(op, valid, ep, n_row, n_col, conf, a,
                                         dem=dem, log=print)
            row[name] = iou(binary_dilation(path, k) & valid, gt)
            print(f"  {name:<16s} IoU {row[name]:.4f}")
        out[site] = row

    json.dump(out, open(os.path.join(RES, "ablation_path.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    names = list(out[SITES[0]].keys())
    with open(os.path.join(RES, "ablation_path.csv"), "w", encoding="utf-8-sig") as f:
        f.write("config," + ",".join(SITES) + "\n")
        for n in names:
            f.write(f"{n}," + ",".join(f"{out[s][n]:.4f}" for s in SITES) + "\n")
    print("\n写出 results/ablation_path.json / .csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    xs = np.arange(len(names))
    colors = ("#4C78A8", "#F58518")
    for i, s in enumerate(SITES):
        vals = [out[s][n] for n in names]
        ax.bar(xs + (i - 0.5) * 0.38, vals, 0.36, label=s, color=colors[i])
        for x, v in zip(xs + (i - 0.5) * 0.38, vals):
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("1 m 掩膜 IoU（路径/阈值膨胀后）")
    ax.set_ylim(0, 0.75)
    ax.set_title("寻路代价项消融与基线（SHP 仅用于评分）", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "ablation_path.png"))
    print("写出 figures/ablation_path.png")


if __name__ == "__main__":
    main()
