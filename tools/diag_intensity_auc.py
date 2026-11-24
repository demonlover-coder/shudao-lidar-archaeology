# -*- coding: utf-8 -*-
"""强度 / 密度 / 回波 特征的可分性检验（不改流程，只测）。

用途：判断"点云自身的光学与回波信息"能否把古道与周围林地分开。
分组（真值只用于评分分组）：
    GT   真值面内像元
    RING 距真值骨架 1~8 m（决定带宽能否外扩的关键对照）
    BG   距真值骨架 8~20 m

特征（1 m 网格，全部来自 LAS 本身，与 DEM 几何正交）：
    int_mean  平均强度      int_std  强度标准差
    dens      点数（log1p） single   单次回波比例
    first     首回波比例

用法：
    python tools/diag_intensity_auc.py --site 杨家坝 --truth old
    python tools/diag_intensity_auc.py --site 西坝村 --truth new --truth-main
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import LAS, RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402

CH = 4_000_000
SS = 4


def auc(x_pos, x_neg):
    """Mann-Whitney AUC（不依赖 sklearn）。"""
    x_pos = np.asarray(x_pos, np.float64)
    x_neg = np.asarray(x_neg, np.float64)
    if x_pos.size == 0 or x_neg.size == 0:
        return float("nan")
    allv = np.concatenate([x_pos, x_neg])
    order = allv.argsort()
    ranks = np.empty(allv.size, np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # 并列取平均秩
    s = allv[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r1 = ranks[:x_pos.size].sum()
    return float((r1 - x_pos.size * (x_pos.size + 1) / 2) / (x_pos.size * x_neg.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--truth", default="new", choices=("new", "old"))
    ap.add_argument("--truth-main", action="store_true")
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

    skel = skeletonize(gt)
    dgt = distance_transform_edt(~skel) * gsd
    m_gt = gt
    m_ring = (~gt) & (dgt > 1.0) & (dgt <= 8.0) & valid
    m_bg = (dgt > 8.0) & (dgt <= 20.0) & valid
    print(f"[{site}] GT {int(m_gt.sum()):,}px  RING {int(m_ring.sum()):,}px  "
          f"BG {int(m_bg.sum()):,}px")

    cnt = np.zeros((H, W), np.float64)
    s_i = np.zeros((H, W), np.float64)
    s_i2 = np.zeros((H, W), np.float64)
    n_first = np.zeros((H, W), np.float64)
    n_single = np.zeros((H, W), np.float64)

    import laspy
    has_int = True
    with laspy.open(LAS[site]) as f:
        for pts in f.chunk_iterator(CH):
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            c = np.floor((X - x0) / gsd).astype(np.int64)
            r = np.floor((y1 - Y) / gsd).astype(np.int64)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            if not ok.any():
                continue
            c, r = c[ok], r[ok]
            try:
                it = np.asarray(pts.intensity, np.float64)[ok]
            except Exception:
                has_int = False
                it = np.zeros(r.size)
            try:
                rn = np.asarray(pts.return_number, np.int16)[ok]
                nr = np.asarray(pts.number_of_returns, np.int16)[ok]
                first = (rn == 1)
                single = (nr == 1)
            except Exception:
                first = np.zeros(r.size, bool)
                single = np.zeros(r.size, bool)
            np.add.at(cnt, (r, c), 1.0)
            np.add.at(s_i, (r, c), it)
            np.add.at(s_i2, (r, c), it * it)
            np.add.at(n_first, (r, c), first.astype(np.float64))
            np.add.at(n_single, (r, c), single.astype(np.float64))

    if not has_int:
        print("  警告：LAS 无 intensity 字段，强度特征将被跳过")

    occ = cnt > 0
    safe = np.maximum(cnt, 1)
    feats = {
        "int_mean": s_i / safe,
        "int_std": np.sqrt(np.maximum(s_i2 / safe - (s_i / safe) ** 2, 0)),
        "dens": np.log1p(cnt),
        "single_frac": n_single / safe,
        "first_frac": n_first / safe,
    }

    rows = []
    for k, v in feats.items():
        if k.startswith("int") and not has_int:
            continue
        vv = np.where(occ, v, np.nan)
        # 只在有点的像元上比较（密度特征本身在无点处=0，单独处理）
        base = occ if k != "dens" else np.ones_like(occ)
        g = vv[m_gt & base]
        rg = vv[m_ring & base]
        bg = vv[m_bg & base]
        g = g[np.isfinite(g)]
        rg = rg[np.isfinite(rg)]
        bg = bg[np.isfinite(bg)]
        rows.append(dict(
            feat=k,
            gt=round(float(np.median(g)), 3) if g.size else None,
            ring=round(float(np.median(rg)), 3) if rg.size else None,
            bg=round(float(np.median(bg)), 3) if bg.size else None,
            auc_gt_bg=round(auc(g, bg), 4),
            auc_gt_ring=round(auc(g, rg), 4),
            n=dict(gt=int(g.size), ring=int(rg.size), bg=int(bg.size)),
        ))

    print(f"\n{'feature':<12}{'GT':>9}{'RING':>9}{'BG':>9}{'AUC(GT,BG)':>12}"
          f"{'AUC(GT,RING)':>14}")
    for r in rows:
        print(f"{r['feat']:<12}{str(r['gt']):>9}{str(r['ring']):>9}{str(r['bg']):>9}"
              f"{r['auc_gt_bg']:>12}{r['auc_gt_ring']:>14}")

    out = os.path.join(RES, f"diag_intensity_auc_{site}.json")
    json.dump(dict(site=site, truth=a.truth, truth_main=bool(a.truth_main),
                   has_intensity=has_int, rows=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
