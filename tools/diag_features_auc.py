# -*- coding: utf-8 -*-
"""候选 2D 特征的像素级可分性检验（无 SHP 参与计算，仅用真值分组评分）。

检验四个特征能否把"真值带"和"背景"分开：
    curv_perp   横断面（垂直坡向）二阶导 —— 路堑凹槽
    curv_along  纵断面二阶导
    tpi5/10/30  多尺度局部高差
    chm         1 m 冠层高度（max Z − dem_min）
    chm_tophat  5 m 底帽响应（局部凹陷的林冠走廊）
    gap5        chm 与 5 m 邻域均值的差

用法：
    python tools/diag_features_auc.py --site 西坝村 --npz cold_start_西坝村_g5.npz --reach 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import LAS, RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_freeaxis import axis_field, fill_nodata, poly_mask_hr  # noqa: E402
from diag_geo_separability import auc  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

CH = 4_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--truth", default="new", choices=("new", "old"))
    ap.add_argument("--truth-main", action="store_true")
    ap.add_argument("--reach", type=float, default=20.0)
    a = ap.parse_args()

    site = a.site
    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem = fill_nodata(d["dem_min"], valid)
    npz = a.npz or f"cold_start_{site}_ep.npz"
    if not os.path.isabs(npz):
        npz = os.path.join(RES, npz)
    prob = np.where(valid, np.load(npz)["prob"].astype(np.float64), 0.0)
    d2, _ = axis_field(site, prob, valid, H, W, gsd, x0, y1, None, dem, "path", npz)

    if a.truth == "old":
        gt = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
    else:
        SS = 4
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
    dgt = distance_transform_edt(~skeletonize(gt))

    # ---- 地形特征 ----
    zs = gaussian_filter(dem, 1.0)
    gy, gx = np.gradient(zs)
    s = np.hypot(gx, gy)
    ux, uy = gx / np.maximum(s, 1e-9), gy / np.maximum(s, 1e-9)
    nxp, nyp = -uy, ux                              # 垂直坡向（横向）
    zyy, zxy = np.gradient(gy)
    zyx, zxx = np.gradient(gx)
    zxy = 0.5 * (zxy + zyx)
    curv_perp = nxp ** 2 * zxx + 2 * nxp * nyp * zxy + nyp ** 2 * zyy
    curv_along = ux ** 2 * zxx + 2 * ux * uy * zxy + uy ** 2 * zyy
    tpi = {w: zs - uniform_filter(zs, w, mode="nearest") for w in (5, 11, 31)}

    # ---- CHM（流式取每像元最高点） ----
    maxz = np.full((H, W), -np.inf, np.float32)
    import laspy
    with laspy.open(LAS[site]) as f:
        for pts in f.chunk_iterator(CH):
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            Z = np.asarray(pts.z, np.float32)
            c = np.floor((X - x0) / gsd).astype(np.int64)
            r = np.floor((y1 - Y) / gsd).astype(np.int64)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            if not ok.any():
                continue
            np.maximum.at(maxz, (r[ok], c[ok]), Z[ok])
    has = np.isfinite(maxz) & valid
    chm = np.where(has, np.clip(maxz - dem, 0, 60), np.nan)
    chm_f = np.where(has, chm, 0.0)
    chm_gap = chm_f - uniform_filter(chm_f, 5, mode="nearest")   # 局部林冠凹陷
    chm_top = uniform_filter(chm_f, 11, mode="nearest") - chm_f   # 底帽

    feats = {"curv_perp": curv_perp, "curv_along": curv_along,
             "tpi5": tpi[5], "tpi10": tpi[11], "tpi30": tpi[31],
             "chm": chm_f, "chm_gap5": chm_gap, "chm_tophat11": chm_top}

    # ---- 分组：走廊内 真值带 vs 背景 ----
    cor = (d2 <= a.reach) & valid & has
    GT = cor & gt
    BG = cor & (~gt) & (dgt > 10)
    print(f"[{site}] 走廊 {int(cor.sum()):,} px；真值 {int(GT.sum()):,} / 背景 {int(BG.sum()):,}")
    print(f"  真值面中 P<0.5 的像素 {int((GT & (prob < 0.5)).sum()):,}"
          f"（占真值 {float((GT & (prob < 0.5)).sum()) / max(int(GT.sum()), 1):.1%}）")

    out = {"site": site, "npz": os.path.basename(npz), "reach": a.reach,
           "n_gt": int(GT.sum()), "n_bg": int(BG.sum()), "auc": {},
           "med": {}}
    for k, f_ in feats.items():
        v = f_[cor]
        m = np.isfinite(v)
        y = np.concatenate([np.ones(int(GT.sum()), bool), np.zeros(int(BG.sum()), bool)])
        vv = f_[GT].tolist() + f_[BG].tolist()
        vv = np.asarray(vv, np.float64)
        okv = np.isfinite(vv)
        au = auc(vv[okv], y[okv])
        # 绝对幅度版本（曲率/差值类看"偏离大小"）
        va = np.abs(f_[GT]).tolist() + np.abs(f_[BG]).tolist()
        va = np.asarray(va, np.float64)
        oka = np.isfinite(va)
        au_abs = auc(va[oka], y[oka])
        out["auc"][k] = round(au, 3) if au is not None else None
        out["auc"][k + "_abs"] = round(au_abs, 3) if au_abs is not None else None
        ga = f_[GT][np.isfinite(f_[GT])]
        ba = f_[BG][np.isfinite(f_[BG])]
        out["med"][k] = [round(float(np.median(ga)), 3),
                         round(float(np.median(ba)), 3)]
        print(f"  {k:12s} GT中位 {out['med'][k][0]:8.3f}  BG中位 {out['med'][k][1]:8.3f}"
              f"   AUC {out['auc'][k]:5}   |x|AUC {out['auc'][k + '_abs']}")

    p = os.path.join(RES, f"diag_features_auc_{site}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 {p}")


if __name__ == "__main__":
    main()
