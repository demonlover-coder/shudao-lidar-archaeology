# -*- coding: utf-8 -*-
"""几何可分性检验：法向倾角 / 平面性 能否区分 路面 / 带外真值 / FP / 背景地面。

只做统计，不产出模型。分组标签仅用于评分：
    TP   交付 ∩ 真值 ∩ 地面
    FNLP 真值 ∩ 地面 ∩ P<0.5（角度一要救的那批）
    FNAX 真值 ∩ 地面 ∩ P≥0.5 ∩ 距轴>d3
    FP   交付 ∩ 非真值 ∩ 地面
    BG   非真值 ∩ 地面 ∩ 距真值骨架 > 10 m

用法：
    python tools/diag_geo_separability.py --site 西坝村 \
        --npz cold_start_西坝村_g5.npz --dist3 5 --reach 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import LAS, RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_freeaxis import axis_field, fill_nodata, poly_mask_hr  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

CH = 3_000_000


def auc(score, pos):
    """Mann-Whitney AUC：score 越大越像 pos。"""
    s = np.asarray(score, np.float64)
    y = np.asarray(pos, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s)
    rk = np.empty(len(s), np.float64)
    rk[order] = np.arange(1, len(s) + 1)
    # 并列取平均秩
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.bincount(inv, weights=rk)
        rk = (sums / cnt)[inv]
    return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def q(v, name):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return {name: None}
    return {name: dict(n=int(v.size),
                       p25=round(float(np.percentile(v, 25)), 3),
                       p50=round(float(np.percentile(v, 50)), 3),
                       p75=round(float(np.percentile(v, 75)), 3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--truth", default="new", choices=("new", "old"))
    ap.add_argument("--truth-main", action="store_true")
    ap.add_argument("--dist3", type=float, default=4.0)
    ap.add_argument("--agl", type=float, default=2.0)
    ap.add_argument("--reach", type=float, default=25.0, help="走廊半宽（米）")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--maxpts", type=int, default=3_500_000)
    ap.add_argument("--nsample", type=int, default=300_000)
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
    gtsk = skeletonize(gt)
    dgt = distance_transform_edt(~gtsk)

    d2, zax = axis_field(site, prob, valid, H, W, gsd, x0, y1, None, dem, "path", npz)

    # ---- 采集走廊内的地面点 ----
    PP, GG, LL = [], [], []
    import laspy
    with laspy.open(LAS[site]) as f:
        for pts in f.chunk_iterator(CH):
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            Z = np.asarray(pts.z, np.float64)
            c = np.floor((X - x0) / gsd).astype(np.int64)
            r = np.floor((y1 - Y) / gsd).astype(np.int64)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            if not ok.any():
                continue
            X, Y, Z, c, r = X[ok], Y[ok], Z[ok], c[ok], r[ok]
            m = d2[r, c] <= a.reach
            if not m.any():
                continue
            X, Y, Z, c, r = X[m], Y[m], Z[m], c[m], r[m]
            agl = Z - dem[r, c]
            m = agl <= a.agl
            X, Y, Z, c, r, agl = X[m], Y[m], Z[m], c[m], r[m], agl[m]
            PP.append(np.stack([X, Y, Z], 1).astype(np.float64))
            GG.append(gt[r, c])
            LL.append(np.stack([agl, prob[r, c], d2[r, c], dgt[r, c]], 1)
                      .astype(np.float32))
    P3 = np.concatenate(PP)
    G = np.concatenate(GG)
    L = np.concatenate(LL)
    del PP, GG, LL
    print(f"走廊内点 {len(P3):,}（真值 {int(G.sum()):,}）")
    if len(P3) > a.maxpts:
        idx = np.random.default_rng(0).choice(len(P3), a.maxpts, replace=False)
        P3, G, L = P3[idx], G[idx], L[idx]
        print(f"  下采样到 {len(P3):,}")

    # ---- 邻域 PCA ----
    rng = np.random.default_rng(1)
    qi = rng.choice(len(P3), min(a.nsample, len(P3)), replace=False)
    tree = cKDTree(P3)
    _, nn = tree.query(P3[qi], k=a.k, workers=-1)
    nb = P3[nn] - P3[qi][:, None, :]
    cov = np.einsum("nki,nkj->nij", nb, nb) / a.k
    w, v = np.linalg.eigh(cov)                    # 升序 λ
    lam = np.clip(w[:, ::-1], 1e-12, None)        # λ1≥λ2≥λ3
    planarity = (lam[:, 1] - lam[:, 2]) / lam[:, 0]
    normal = np.abs(v[:, :, 0])                    # λ 最小对应法向
    ang = np.degrees(np.arccos(np.clip(normal[:, 2], 0.0, 1.0)))

    agl, pv, d2q, dgtq = L[qi].T
    g = agl <= 0.3
    tr = G[qi]
    keep = (pv >= 0.5) & (d2q <= a.dist3) & g
    TP = keep & tr
    FP = keep & ~tr
    FNLP = tr & g & (pv < 0.5)
    FNAX = tr & g & (pv >= 0.5) & (d2q > a.dist3)
    BG = (~tr) & g & (dgtq > 10)
    print(f"  分组：TP {int(TP.sum()):,}  FP {int(FP.sum()):,}  "
          f"FNLP {int(FNLP.sum()):,}  FNAX {int(FNAX.sum()):,}  BG {int(BG.sum()):,}")

    out = {"site": site, "npz": os.path.basename(npz), "dist3": a.dist3,
           "groups": {k: int(v_.sum()) for k, v_ in
                      dict(TP=TP, FP=FP, FNLP=FNLP, FNAX=FNAX, BG=BG).items()}}

    for name, m in (("TP", TP), ("FNLP", FNLP), ("FP", FP), ("BG", BG)):
        out.update(q(planarity[m], f"planarity_{name}"))
        out.update(q(ang[m], f"normal_deg_{name}"))

    # 两组 AUC 需要单独构造
    def two_auc(m1, m2, score):
        s = np.concatenate([score[m1], score[m2]])
        y = np.concatenate([np.ones(int(m1.sum()), bool),
                            np.zeros(int(m2.sum()), bool)])
        return auc(s, y)

    pairs = {}
    if TP.sum() > 50 and BG.sum() > 50:
        pairs["planarity_TP_vs_BG"] = two_auc(TP, BG, planarity)
        pairs["angle_TP_vs_BG"] = two_auc(TP, BG, -ang)
    if TP.sum() > 50 and FNLP.sum() > 50:
        pairs["planarity_FNLP_vs_BG"] = two_auc(FNLP, BG, planarity)
        pairs["angle_FNLP_vs_BG"] = two_auc(FNLP, BG, -ang)
    if TP.sum() > 50 and FP.sum() > 50:
        pairs["angle_TP_vs_FP"] = two_auc(TP, FP, -ang)
        pairs["planarity_TP_vs_FP"] = two_auc(TP, FP, planarity)
    out["auc"] = {k: (round(v, 3) if v is not None else None)
                  for k, v in pairs.items()}

    print("\n  平面性 planarity（越大越平）")
    for n_ in ("TP", "FNLP", "FP", "BG"):
        print(f"    {n_:5s} {out.get(f'planarity_{n_}', {}).get('p50', 'NA')}"
              f"  [{out.get(f'planarity_{n_}', {}).get('p25', 'NA')}, "
              f"{out.get(f'planarity_{n_}', {}).get('p75', 'NA')}]")
    print("  法向与竖直夹角（度）")
    for n_ in ("TP", "FNLP", "FP", "BG"):
        print(f"    {n_:5s} {out.get(f'normal_deg_{n_}', {}).get('p50', 'NA')}"
              f"  [{out.get(f'normal_deg_{n_}', {}).get('p25', 'NA')}, "
              f"{out.get(f'normal_deg_{n_}', {}).get('p75', 'NA')}]")
    print("\n  AUC（0.5=无区分力）")
    for k, v in out["auc"].items():
        print(f"    {k:26s} {v}")

    p = os.path.join(RES, f"diag_geo_separability_{site}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 {p}")


if __name__ == "__main__":
    main()
