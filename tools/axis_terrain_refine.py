# -*- coding: utf-8 -*-
"""横向地形台地检测修轴（无 SHP）。

古法修路会在横坡上削出一条平缓台地：沿当前轴切 ±reach 的法向地形剖面，
平滑后算局部坡度，取"坡度最小的连续区间中心"作为候选新轴心，限幅后
沿程中值平滑，最后把轴平移到台地中心。

只读 DEM（dem_min），不读 SHP。

用法：
    python tools/axis_terrain_refine.py --site 西坝村 \
        --npz cold_start_西坝村_cb.npz --out_tag _ref
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates, median_filter, uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import RES  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from fig_pred_vs_truth import order_path  # noqa: E402
from diag_band_width import resample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--out_tag", default="_ref")
    ap.add_argument("--step", type=float, default=2.0, help="断面间隔（米）")
    ap.add_argument("--reach", type=float, default=8.0, help="法向半长（米）")
    ap.add_argument("--max_shift", type=float, default=4.0, help="单点最大横移（米）")
    ap.add_argument("--win", type=float, default=3.0, help="坡度平均窗（米）")
    a = ap.parse_args()

    site = a.site
    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    valid = d["valid"] == 1
    H, W = d["dem_mean"].shape
    dem = fill_nodata(d["dem_min"], valid)
    npz = a.npz or f"cold_start_{site}_ep.npz"
    if not os.path.isabs(npz):
        npz = os.path.join(RES, npz)
    z = np.load(npz)
    prob = np.where(valid, z["prob"].astype(np.float64), 0.0)
    path = z["path"]
    if path.ndim == 2 and path.shape[1] == 2:
        m = np.zeros((H, W), bool)
        m[np.round(path[:, 0]).astype(int), np.round(path[:, 1]).astype(int)] = True
        path = m
    pts = order_path(path.astype(bool))
    rr, cc = resample(pts, a.step, gsd)

    off = np.arange(-a.reach, a.reach + 1e-9, 0.5)
    nw = max(1, int(round(a.win / 0.5)))
    shifts = np.zeros(len(rr))
    k0 = int(np.argmin(np.abs(off)))
    for i in range(len(rr)):
        j0, j1 = max(0, i - 2), min(len(rr) - 1, i + 2)
        tr, tc = rr[j1] - rr[j0], cc[j1] - cc[j0]
        n = np.hypot(tr, tc)
        if n < 1e-9:
            continue
        nr, nc = -tc / n, tr / n
        sR = rr[i] + (off / gsd) * nr
        sC = cc[i] + (off / gsd) * nc
        zz = map_coordinates(dem, [sR, sC], order=1, mode="nearest")
        zs = uniform_filter1d(zz, 5, mode="nearest")          # 2.5 m 平滑
        slope = np.gradient(zs, 0.5)
        flat = uniform_filter1d(np.abs(slope), nw, mode="nearest")
        k = int(np.argmin(flat))
        shifts[i] = np.clip(off[k], -a.max_shift, a.max_shift)

    shifts = median_filter(shifts, size=5, mode="nearest")
    # 平滑后按法向重建轴（逐站取整栅格化）
    newm = np.zeros((H, W), bool)
    for i in range(len(rr)):
        j0, j1 = max(0, i - 2), min(len(rr) - 1, i + 2)
        tr, tc = rr[j1] - rr[j0], cc[j1] - cc[j0]
        n = np.hypot(tr, tc)
        if n < 1e-9:
            continue
        nr, nc = -tc / n, tr / n
        r_ = int(round(rr[i] + (shifts[i] / gsd) * nr))
        c_ = int(round(cc[i] + (shifts[i] / gsd) * nc))
        if 0 <= r_ < H and 0 <= c_ < W:
            newm[r_, c_] = True

    tag = a.out_tag
    out = os.path.join(RES, f"cold_start_{site}{tag}.npz")
    np.savez(out, prob=prob, path=newm)
    info = dict(site=site, src=os.path.basename(npz),
                n_station=int(len(rr)),
                shift_abs_median=round(float(np.median(np.abs(shifts))), 2),
                shift_abs_p90=round(float(np.percentile(np.abs(shifts), 90)), 2),
                shift_median=round(float(np.median(shifts)), 2),
                n_shifted=int((np.abs(shifts) > 0.5).sum()),
                max_shift=a.max_shift)
    print(f"[{site}] 断面 {len(rr)}，横移 |中位| {info['shift_abs_median']} m，"
          f"p90 {info['shift_abs_p90']} m，>0.5 m 的站 {info['n_shifted']}")
    p = os.path.join(RES, f"axis_terrain_refine_{site}{tag}.json")
    json.dump(info, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 {out}\n  写出 {p}")


if __name__ == "__main__":
    main()
