# -*- coding: utf-8 -*-
"""路径纵坡统计（无 SHP）：总爬升、10 m 窗纵坡 P95 / 最大值、长度。

用法：
    python tools/diag_path_grade.py --site 西坝村 --npz cold_start_西坝村_g5.npz cold_start_西坝村_lc6.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import RES  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from fig_pred_vs_truth import order_path  # noqa: E402


def resample(path, step=1.0):
    p = np.asarray(path, np.float64)
    dl = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(dl)])
    si = np.arange(0.0, s[-1], step)
    return (np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1]), si)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--npz", nargs="+", required=True)
    a = ap.parse_args()
    site = a.site
    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    dem = fill_nodata(d["dem_min"], d["valid"] == 1)

    print(f"{'配置':<26}{'长度':>8}{'总爬升':>10}{'P95':>8}{'最大':>8}")
    for fn in a.npz:
        f = fn if os.path.isabs(fn) else os.path.join(RES, fn)
        z = np.load(f)
        p = z["path"]
        if p.ndim == 2:
            pl = order_path(p.astype(bool))
        else:
            pl = order_path(_mask(p, H, W))
        rr, cc, s = resample(pl, 1.0)
        zz = dem[np.clip(np.round(rr).astype(int), 0, H - 1),
                 np.clip(np.round(cc).astype(int), 0, W - 1)]
        dz = np.diff(zz)
        win = 10
        if zz.size > win:
            g = np.abs(zz[win:] - zz[:-win]) / win
        else:
            g = np.array([0.0])
        print(f"{os.path.basename(fn):<26}{s[-1]:>7.0f}m"
              f"{float(dz[dz > 0].sum()):>9.1f}m"
              f"{np.percentile(g, 95) * 100:>7.0f}%{g.max() * 100:>7.0f}%")


def _mask(pts, H, W):
    m = np.zeros((H, W), bool)
    rr = np.round(pts[:, 0]).astype(int)
    cc = np.round(pts[:, 1]).astype(int)
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    m[rr[ok], cc[ok]] = True
    return m


if __name__ == "__main__":
    main()
