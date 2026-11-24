# -*- coding: utf-8 -*-
"""真值宽度复查：标注多边形到底有多宽，模型命中它的哪一层（仅评分，无训练）。

问题背景：西坝村真值宽 13.7 m，而模型概率带只有 7 m。到底是"模型太窄"，
还是"标注里混入了路肩/台地"？

做法（纯像素级，不读点云）：
  1. 统计真值面沿骨架的横向宽度分布（2×距离变换最大值）；
  2. 对真值做不同半径的腐蚀，得到"路面核心"（r=0 即原始标注）；
  3. 用同一个预测掩膜 P≥0.5，分别对 r=0..4 m 的核心算 IoU / 召回。

如果 IoU 随 r 增大而显著上升，说明模型命中的是标注的"核心带"，
之前的低分主要来自标注宽度定义，而不是模型漏检。

用法：
    python tools/diag_truth_core.py --site 西坝村 --truth new --truth-main
    python tools/diag_truth_core.py --site 杨家坝 --truth old
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402

SS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--truth", default="new", choices=("new", "old"))
    ap.add_argument("--truth-main", action="store_true")
    ap.add_argument("--npz", default="")
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    site = a.site

    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1

    if a.truth == "old":
        gt = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
    else:
        _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        mh = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
        gt = mh.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5
        gt &= valid
    if a.truth_main:
        from scipy.ndimage import label as ccl
        lb, n = ccl(gt)
        if n > 1:
            sz = np.bincount(lb.ravel())
            sz[0] = 0
            gt = lb == sz.argmax()

    # ---- 真值宽度分布
    dt = distance_transform_edt(gt) * gsd          # 面内到边界距离
    sk = skeletonize(gt)
    w_axis = 2.0 * dt[sk]
    print(f"[{site}] 真值 {int(gt.sum()):,}px；沿骨架宽度："
          f"中位 {np.median(w_axis):.2f} m，均值 {w_axis.mean():.2f} m，"
          f"P10/P90 {np.percentile(w_axis, 10):.2f}/{np.percentile(w_axis, 90):.2f} m，"
          f"最大 {w_axis.max():.2f} m")
    area_w = gt.sum() / max(sk.sum(), 1) * gsd
    print(f"  面积/骨架长度 = {area_w:.2f} m（平均带宽）")

    # ---- 预测掩膜
    npz = a.npz or f"cold_start_{site}_ep.npz"
    if not os.path.isabs(npz):
        npz = os.path.join(RES, npz)
    prob = np.load(npz)["prob"].astype(np.float64)
    pred = (np.where(valid, prob, 0.0) >= a.thr) & valid
    print(f"  预测 {os.path.basename(npz)}：{int(pred.sum()):,}px")

    rows = []
    for r in (0, 1, 2, 3, 4, 5):
        core = binary_erosion(gt, iterations=r) if r else gt
        if not core.any():
            continue
        inter = int((pred & core).sum())
        iou = inter / max(int((pred | core).sum()), 1)
        rec = inter / max(int(core.sum()), 1)
        prec = inter / max(int(pred.sum()), 1)
        rows.append(dict(erode_m=r, core_px=int(core.sum()),
                         width_m=round(2 * float(np.median(dt[core])) * 1.0, 2),
                         P=round(prec, 4), R=round(rec, 4), IoU=round(iou, 4)))
        print(f"  腐蚀 {r} m → 核心 {int(core.sum()):,}px "
              f"（中位宽 {rows[-1]['width_m']:.1f} m）："
              f"P={prec:.4f} R={rec:.4f} IoU={iou:.4f}")

    out = os.path.join(RES, f"diag_truth_core_{site}.json")
    json.dump(dict(site=site, truth=a.truth, truth_main=bool(a.truth_main),
                   width_median=round(float(np.median(w_axis)), 3),
                   width_mean=round(float(w_axis.mean()), 3),
                   width_p90=round(float(np.percentile(w_axis, 90)), 3),
                   area_per_len_m=round(float(area_w), 3),
                   rows=rows), open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  写出 {out}")


if __name__ == "__main__":
    main()
