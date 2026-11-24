# -*- coding: utf-8 -*-
"""概率带宽度诊断：沿模型轴切法向断面，量 P 带有多宽、偏在哪一侧。

不读 SHP。输出三个量的分布（中位/均值）：
    w50   ：P≥0.5 连续段（含轴心那一段）的物理宽度（米）
    fwhm  ：以轴心处峰值为基准的半高全宽
    c20   ：P>0.2 区间加权质心相对轴心的偏移（米，正=法向正方向）
    rad   ：二值概率图距离变换在轴心处的值（局部半径，米）
另给出"轴心本身不在 P≥0.5 内"的断面占比，用于判断轴是否已偏出带外。

用法：
    python tools/diag_band_width.py --site 杨家坝 --npz cold_start_杨家坝_cb.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import RES  # noqa: E402
from fig_pred_vs_truth import order_path  # noqa: E402


def resample(path, step_m=5.0, gsd=1.0):
    """按弧长 step_m 重采样有序折线，返回 (r, c)。"""
    d = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.arange(0.0, s[-1], step_m)
    return (np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--step", type=float, default=5.0, help="断面间隔（米）")
    ap.add_argument("--reach", type=float, default=20.0, help="断面半长（米）")
    a = ap.parse_args()

    site = a.site
    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    valid = d["valid"] == 1
    H, W = d["dem_mean"].shape
    npz = a.npz or f"cold_start_{site}_ep.npz"
    if not os.path.isabs(npz):
        npz = os.path.join(RES, npz)
    prob = np.where(valid, np.load(npz)["prob"].astype(np.float64), 0.0)

    path = order_path(prob >= 0.5)
    if path is None:
        raise RuntimeError("概率带骨架为空")
    rr, cc = resample(path, a.step, gsd)
    off = np.arange(-a.reach, a.reach + 1e-9, 0.5)

    dt = distance_transform_edt(prob >= 0.5) * gsd
    w50, fwhm, c20, rad = [], [], [], []
    outside = 0
    multi = 0
    for i in range(len(rr)):
        j0, j1 = max(0, i - 1), min(len(rr) - 1, i + 1)
        tr, tc = rr[j1] - rr[j0], cc[j1] - cc[j0]
        n = np.hypot(tr, tc)
        if n < 1e-9:
            continue
        nr, nc = -tc / n, tr / n           # 法向（行, 列）
        sR = rr[i] + (off / gsd) * nr
        sC = cc[i] + (off / gsd) * nc
        pr = map_coordinates(prob, [sR, sC], order=1, mode="nearest")
        k0 = int(round(a.reach / 0.5))
        if pr[k0] < 0.5:
            outside += 1
            continue
        # P≥0.5 的连续段（含轴心）
        lo = k0
        while lo > 0 and pr[lo - 1] >= 0.5:
            lo -= 1
        hi = k0
        while hi < len(pr) - 1 and pr[hi + 1] >= 0.5:
            hi += 1
        w50.append((hi - lo) * 0.5)
        pk = pr[k0]
        lo = k0
        while lo > 0 and pr[lo - 1] >= pk / 2:
            lo -= 1
        hi = k0
        while hi < len(pr) - 1 and pr[hi + 1] >= pk / 2:
            hi += 1
        fwhm.append((hi - lo) * 0.5)
        m = pr > 0.2
        if m.any():
            c20.append(float((off[m] * pr[m]).sum() / pr[m].sum()))
            # 统计 P>0.2 的独立段数：>1 说明法向断面切到了相邻路段，质心会被污染
            seg = np.diff(np.concatenate([[0], m.astype(int), [0]]))
            if int((seg == 1).sum()) > 1:
                multi += 1
        rad.append(float(dt[int(round(rr[i])), int(round(cc[i]))]))

    def st(v):
        v = np.asarray(v, dtype=float)
        return dict(n=int(v.size), median=round(float(np.median(v)), 2),
                    mean=round(float(v.mean()), 2),
                    p10=round(float(np.percentile(v, 10)), 2),
                    p90=round(float(np.percentile(v, 90)), 2))

    out = dict(site=site, npz=os.path.basename(npz), n_station=len(rr),
               axis_outside_band=outside,
               multi_limb_frac=round(multi / max(len(c20), 1), 4),
               w50=st(w50), fwhm=st(fwhm), c20=st(c20), radius=st(rad),
               half_w50=round(float(np.median(w50)) / 2, 2),
               half_fwhm=round(float(np.median(fwhm)) / 2, 2),
               half_radius=round(float(np.median(rad)), 2))
    print(f"[{site}] 断面 {len(rr)} 个，轴心落在 P<0.5 的 {outside} 个")
    print(f"  w50 宽度      {out['w50']}  → 半宽 {out['half_w50']} m")
    print(f"  FWHM          {out['fwhm']}  → 半宽 {out['half_fwhm']} m")
    print(f"  P>0.2 质心偏移 {out['c20']} m（正=法向正向）")
    print(f"  断面切到多条带（质心会被污染）占比 {out['multi_limb_frac'] * 100:.1f}%")
    print(f"  距离变换半径   {out['radius']}  → {out['half_radius']} m")
    p = os.path.join(RES, f"diag_band_width_{site}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 {p}")


if __name__ == "__main__":
    main()
