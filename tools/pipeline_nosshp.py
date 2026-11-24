# -*- coding: utf-8 -*-
"""无 SHP 参与的收窄 + 骨架评估（pipeline_nosshp.py）。

硬约束：**SHP 只用于最后评分，不参与任何筛选/训练/后处理**。
因此"到轴线距离 ≤ d"里的轴线必须由模型自己产生，两种来源：

    path     : 双分支支路 1 的 Dijkstra 路径（由 openness + h1 相位推出，无 SHP）
    skel     : 模型概率图 P≥`--p` 的掩膜骨架（闭运算后 skeletonize）

评估时才读取 SHP：
    GT 骨架 = 本体范围面 → 0.25 m skeletonize → 1 m

3D 距离用"到轴线栅格的最近格 + 高差"近似：
    d3 = sqrt(dist2d² + (z − z_axis)²)，z_axis 取该最近格的 dem_min

用法：
    python tools/pipeline_nosshp.py --site 杨家坝
    python tools/pipeline_nosshp.py --site 杨家坝 --axis both --dist 3,4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402
from eval_new_shp import RES, ROOT, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_physical_width import up_skeleton  # noqa: E402

SS = 4


def axis_field(mask: np.ndarray, gsd: float, dem_min: np.ndarray):
    """轴线掩膜 → (到轴线 2D 距离场, 最近轴线格的 dem_min)。"""
    from scipy.ndimage import distance_transform_edt
    d2, idx = distance_transform_edt(~mask, sampling=gsd, return_indices=True)
    return d2, dem_min[idx[0], idx[1]]


def eval_skeleton(sk, gt, gsd, tol):
    from scipy.ndimage import binary_dilation, distance_transform_edt
    Dg = distance_transform_edt(~gt, sampling=gsd)
    Dp = distance_transform_edt(~sk, sampling=gsd)
    pr = float((Dg[sk] <= tol).mean()) if sk.any() else 0.0
    rc = float((Dp[gt] <= tol).mean())
    f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    bp = binary_dilation(sk, iterations=int(np.ceil(tol)))
    bg = binary_dilation(gt, iterations=int(np.ceil(tol)))
    return {"precision": round(pr, 4), "recall": round(rc, 4),
            "f1": round(f1, 4),
            "buffer_iou": round(float((bp & bg).sum() / max((bp | bg).sum(), 1)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="杨家坝")
    ap.add_argument("--axis", default="both", choices=["path", "skel", "both", "none"])
    ap.add_argument("--dist", default="3")
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--close", type=int, default=3)
    a = ap.parse_args()
    site = a.site
    from scipy.ndimage import binary_closing, distance_transform_edt, label
    from sklearn.cluster import KMeans

    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem_min = d["dem_min"].astype(np.float64)

    zc = np.load(os.path.join(RES, f"cold_start_{site}_ep.npz"))
    prob = zc["prob"].astype(np.float64)
    prob = np.where((zc["valid"] == 1) if "valid" in zc.files else (prob > 0), prob, 0.0)
    path = (zc["path"] > 0) & valid if "path" in zc.files else None

    cand = np.load(os.path.join(RES, f"narrow_cand_{site}.npz"))
    X, Y, Z, agl = cand["x"], cand["y"], cand["z"], cand["agl"]
    print(f"[{site}] 候选 P≥{a.p:g}：{X.size:,} 点（缓存，无 SHP 参与）")

    rs = np.random.default_rng(0)
    sub = agl[rs.choice(agl.size, min(300_000, agl.size), replace=False)]
    cen = np.sort(KMeans(n_clusters=2, random_state=0, n_init=10)
                  .fit(sub.reshape(-1, 1)).cluster_centers_.ravel())
    split = float(cen.mean())
    print(f"  AGL KMeans 分界 {split:.2f} m（中心 {cen[0]:.2f} / {cen[1]:.2f}）")

    # ---- 仅评估用的真值骨架 ----
    _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
    gt = up_skeleton(poly_mask_hr(polys, x0, y1, gsd, H, W, SS), H, W, SS) & valid
    print(f"  GT 骨架（仅评分）：{int(gt.sum())} px")

    # ---- 模型自产轴线 ----
    axes = {}
    if path is not None:
        axes["path"] = path
    msk = (prob >= a.p) & valid
    if a.close:
        msk = binary_closing(msk, np.ones((3, 3), dtype=bool), iterations=a.close)
    from skimage.morphology import skeletonize
    axes["skel"] = skeletonize(np.ascontiguousarray(msk)) & valid

    rows = []
    for name, mask in ([] if a.axis == "none" else
                       [(k, v) for k, v in axes.items()
                        if a.axis in ("both", k)]):
        if mask.sum() == 0:
            print(f"  [跳过] 轴线 {name} 为空")
            continue
        d2, zax = axis_field(mask, gsd, dem_min)
        cr = np.clip(np.rint((y1 - Y) / gsd - 0.5), 0, H - 1).astype(np.int64)
        cl = np.clip(np.rint((X - x0) / gsd - 0.5), 0, W - 1).astype(np.int64)
        d3 = np.hypot(d2[cr, cl], Z - zax[cr, cl])
        for dd in [float(t) for t in a.dist.split(",") if t.strip()]:
            keep = (d3 <= dd) & (agl <= split)
            if keep.sum() < 100:
                continue
            occ = np.zeros((H * SS, W * SS), dtype=bool)
            cc = np.floor((X[keep] - x0) / gsd * SS).astype(np.int64)
            rr = np.floor((y1 - Y[keep]) / gsd * SS).astype(np.int64)
            ok = (cc >= 0) & (cc < W * SS) & (rr >= 0) & (rr < H * SS)
            occ[rr[ok], cc[ok]] = True
            if a.close:
                occ = binary_closing(occ, np.ones((3, 3), dtype=bool), iterations=a.close)
            sk = up_skeleton(occ, H, W, SS) & valid
            m1, m2 = eval_skeleton(sk, gt, gsd, 1.0), eval_skeleton(sk, gt, gsd, 2.0)
            nseg = label(sk)[1]
            rows.append({"axis": name, "dist_m": dd, "keep": int(keep.sum()),
                         "skeleton_px": int(sk.sum()), "components": int(nseg),
                         "at_1px": m1, "at_2px": m2})
            print(f"  轴={name:<4} 轴距≤{dd:g} m：保留 {int(keep.sum()):>7,} 点 → "
                  f"骨架 {int(sk.sum()):>4} px/{nseg:>3} 段 | "
                  f"P@1 {m1['precision']:.4f} R@1 {m1['recall']:.4f} F1@1 {m1['f1']:.4f} | "
                  f"F1@2 {m2['f1']:.4f}")

    # 不设轴对照
    keep = agl <= split
    occ = np.zeros((H * SS, W * SS), dtype=bool)
    cc = np.floor((X[keep] - x0) / gsd * SS).astype(np.int64)
    rr = np.floor((y1 - Y[keep]) / gsd * SS).astype(np.int64)
    ok = (cc >= 0) & (cc < W * SS) & (rr >= 0) & (rr < H * SS)
    occ[rr[ok], cc[ok]] = True
    if a.close:
        occ = binary_closing(occ, np.ones((3, 3), dtype=bool), iterations=a.close)
    sk = up_skeleton(occ, H, W, SS) & valid
    m1, m2 = eval_skeleton(sk, gt, gsd, 1.0), eval_skeleton(sk, gt, gsd, 2.0)
    rows.append({"axis": "none", "dist_m": None, "keep": int(keep.sum()),
                 "skeleton_px": int(sk.sum()), "components": int(label(sk)[1]),
                 "at_1px": m1, "at_2px": m2})
    print(f"  轴=无（仅 P+AGL）：保留 {int(keep.sum()):>7,} 点 → 骨架 {int(sk.sum()):>4} px | "
          f"F1@1 {m1['f1']:.4f} | F1@2 {m2['f1']:.4f}")

    best = max([r for r in rows if r["dist_m"] is not None], key=lambda r: r["at_1px"]["f1"],
               default=None)
    if best:
        print(f"\n[无 SHP 最优] 轴={best['axis']} 轴距≤{best['dist_m']:g} m："
              f"F1@1 {best['at_1px']['f1']:.4f} / F1@2 {best['at_2px']['f1']:.4f}"
              f"（对照：SHP 轴线泄漏版 0.6314 / 0.7812，已废弃）")

    res = {"site": site, "note": "SHP 仅用于评分；轴线由模型自产",
           "kmeans_split_m": round(split, 2), "gt_skeleton_px": int(gt.sum()),
           "rows": rows}
    out = os.path.join(RES, f"pipeline_nosshp_{site}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 results/{os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
