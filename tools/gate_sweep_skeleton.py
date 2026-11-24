# -*- coding: utf-8 -*-
"""放宽交付闸门后的骨架评估扫描（3D 轴距 × 高度簇）。

闸门组合：
    3D 轴距 d ∈ {3, 5, 6} m
    高度模式 ∈ {low: AGL ≤ KMeans 分界, mid: AGL ≤ 分界+2 m, all: 不限}

每个组合：候选点 → 0.25 m 占用 → 闭运算×3 → 骨架 → 降到 1 m，
与真值骨架比 1 px / 2 px 容差下的 P / R / F1 与 Buffer IoU。

候选点（P≥0.5 且落在有效区内）缓存到 results/narrow_cand_{site}.npz，
之后改闸门无需再读 LAS。3D 距离只用 2D 距离 ≤10 m 的点计算（其余记 999）。

用法：
    python tools/gate_sweep_skeleton.py --site 杨家坝
    python tools/gate_sweep_skeleton.py --site 杨家坝 --dist 3,5,6 --rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from diag_shp_truth import read_line_z  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402
from eval_new_shp import (LAS, RES, SHP_BASE, SHP_NEW, bilinear,  # noqa: E402
                          line_dist_map, parts_of, read_shp_any)
from eval_physical_width import up_skeleton  # noqa: E402
from line_metrics_narrow import _zparts, seg_dist3  # noqa: E402

SS = 4
PRE2D = 10.0      # 只对 2D 距离 ≤10 m 的点算 3D 折线距离


def build_cache(site, x0, y1, gsd, H, W, valid, dem_min, prob, chunk=3_000_000):
    import laspy
    _, line_shapes = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["line"]))
    zl = read_line_z(os.path.join(SHP_BASE, SHP_NEW[site]["line"]))
    segs3 = []
    for sh in zl:
        for xy, zz in zip(parts_of(sh), _zparts(sh)):
            for k in range(len(xy) - 1):
                segs3.append((tuple(xy[k]) + (float(zz[k]),),
                              tuple(xy[k + 1]) + (float(zz[k + 1]),)))
    d2map = line_dist_map(line_shapes, x0, y1, gsd, H, W)
    print(f"  3D 折线段 {len(segs3)} 条")

    X_, Y_, Z_, A_, D_ = [], [], [], [], []
    with laspy.open(LAS[site]) as f:
        for pts in f.chunk_iterator(chunk):
            X = np.asarray(pts.x, dtype=np.float64)
            Y = np.asarray(pts.y, dtype=np.float64)
            Z = np.asarray(pts.z, dtype=np.float64)
            c = (X - x0) / gsd - 0.5
            r = (y1 - Y) / gsd - 0.5
            ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
            pb = bilinear(prob, r, c)
            vb = bilinear(valid.astype(np.float32), r, c)
            m = ins & (vb > 0.5) & (pb >= 0.5)
            ii = np.flatnonzero(m)
            if ii.size == 0:
                continue
            cr = np.clip(np.rint(r[ii]), 0, H - 1).astype(np.int64)
            cl = np.clip(np.rint(c[ii]), 0, W - 1).astype(np.int64)
            d3 = np.full(ii.size, 999.0, dtype=np.float32)
            near = d2map[cr, cl] <= PRE2D
            if near.any():
                sel = ii[near]
                d3[near] = seg_dist3(X[sel], Y[sel], Z[sel], segs3)
            X_.append(X[ii].astype(np.float32))
            Y_.append(Y[ii].astype(np.float32))
            Z_.append(Z[ii].astype(np.float32))
            A_.append((Z[ii] - dem_min[cr, cl]).astype(np.float32))
            D_.append(d3)
    np.savez_compressed(os.path.join(RES, f"narrow_cand_{site}.npz"),
                        x=np.concatenate(X_), y=np.concatenate(Y_),
                        z=np.concatenate(Z_), agl=np.concatenate(A_),
                        d3=np.concatenate(D_))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="杨家坝")
    ap.add_argument("--dist", default="3,5,6")
    ap.add_argument("--close", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    site = a.site
    from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, label

    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem_min = d["dem_min"].astype(np.float64)
    zp = np.load(os.path.join(RES, f"cold_start_{site}_ep.npz"))
    prob = zp["prob"].astype(np.float64)
    prob = np.where((zp["valid"] == 1) if "valid" in zp.files else (prob > 0), prob, 0.0)

    cache = os.path.join(RES, f"narrow_cand_{site}.npz")
    if a.rebuild or not os.path.exists(cache):
        print("  构建候选点缓存（扫 LAS 一次，约 1 分钟）…")
        build_cache(site, x0, y1, gsd, H, W, valid, dem_min, prob)
    z = np.load(cache)
    X, Y, agl, d3 = z["x"], z["y"], z["agl"], z["d3"]
    print(f"[{site}] 候选 P≥0.5：{X.size:,} 点；3D 轴距 ≤3/5/6 m："
          + " / ".join(f"{int((d3 <= t).sum()):,}" for t in (3, 5, 6)))

    from sklearn.cluster import KMeans
    rs = np.random.default_rng(0)
    sub = agl[rs.choice(agl.size, min(300_000, agl.size), replace=False)]
    cen = np.sort(KMeans(n_clusters=2, random_state=0, n_init=10)
                  .fit(sub.reshape(-1, 1)).cluster_centers_.ravel())
    split = float(cen.mean())
    print(f"  AGL KMeans 中心 {cen[0]:.2f} / {cen[1]:.2f} m → 分界 {split:.2f} m")

    _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
    gt = up_skeleton(poly_mask_hr(polys, x0, y1, gsd, H, W, SS), H, W, SS) & valid
    Dg = distance_transform_edt(~gt, sampling=gsd)

    def ev(sk, tol):
        Dp = distance_transform_edt(~sk, sampling=gsd)
        P_ = float((Dg[sk] <= tol).mean()) if sk.any() else 0.0
        R_ = float((Dp[gt] <= tol).mean())
        F_ = 2 * P_ * R_ / max(P_ + R_, 1e-9)
        bp = binary_dilation(sk, iterations=int(np.ceil(tol)))
        bg = binary_dilation(gt, iterations=int(np.ceil(tol)))
        return {"precision": round(P_, 4), "recall": round(R_, 4), "f1": round(F_, 4),
                "buffer_iou": round(float((bp & bg).sum() / max((bp | bg).sum(), 1)), 4)}

    modes = [("low", "AGL≤分界", split),
             ("mid", "AGL≤分界+2m", split + 2.0),
             ("all", "不限高度", np.inf)]
    rows = []
    print(f"\n{'轴距':>4} {'高度模式':<12} {'点数':>9} {'骨架px':>6} {'段':>4} "
          f"{'P@1':>7} {'R@1':>7} {'F1@1':>7} {'F1@2':>7} {'IoU@1':>7}")
    for dd in [float(t) for t in a.dist.split(",") if t.strip()]:
        for key, name, thr in modes:
            keep = (d3 <= dd) & (agl <= thr)
            if keep.sum() == 0:
                continue
            Xk, Yk = X[keep], Y[keep]
            occ = np.zeros((H * SS, W * SS), dtype=bool)
            cc = np.floor((Xk - x0) / gsd * SS).astype(np.int64)
            rr = np.floor((y1 - Yk) / gsd * SS).astype(np.int64)
            ok = (cc >= 0) & (cc < W * SS) & (rr >= 0) & (rr < H * SS)
            occ[rr[ok], cc[ok]] = True
            if a.close:
                occ = binary_closing(occ, np.ones((3, 3), dtype=bool), iterations=a.close)
            sk = up_skeleton(occ, H, W, SS) & valid
            m1, m2 = ev(sk, 1.0), ev(sk, 2.0)
            nseg = label(sk)[1]
            rows.append({"dist3d_m": dd, "height_mode": key,
                         "height_thr_m": (None if np.isinf(thr) else round(thr, 2)),
                         "keep": int(keep.sum()), "skeleton_px": int(sk.sum()),
                         "components": int(nseg), "at_1px": m1, "at_2px": m2})
            print(f"{dd:>4g} {name:<12} {int(keep.sum()):>9,} {int(sk.sum()):>6} {nseg:>4} "
                  f"{m1['precision']:>7.4f} {m1['recall']:>7.4f} {m1['f1']:>7.4f} "
                  f"{m2['f1']:>7.4f} {m1['buffer_iou']:>7.4f}")

    best1 = max(rows, key=lambda r: r["at_1px"]["f1"])
    best2 = max(rows, key=lambda r: r["at_2px"]["f1"])
    print(f"\n[最优 1 px] 轴距 {best1['dist3d_m']:g} m / {best1['height_mode']}："
          f"F1 {best1['at_1px']['f1']:.4f}  "
          f"（{'突破 0.70' if best1['at_1px']['f1'] >= 0.70 else '未达 0.70'}）")
    print(f"[最优 2 px] 轴距 {best2['dist3d_m']:g} m / {best2['height_mode']}："
          f"F1 {best2['at_2px']['f1']:.4f}  "
          f"（{'突破 0.70' if best2['at_2px']['f1'] >= 0.70 else '未达 0.70'}）")

    res = {"site": site, "grid": [H, W], "gsd_m": gsd, "skel_res_m": gsd / SS,
           "close_iter": a.close,
           "kmeans_centers_m": [round(float(cen[0]), 2), round(float(cen[1]), 2)],
           "kmeans_split_m": round(split, 2), "gt_skeleton_px": int(gt.sum()),
           "rows": rows, "best_1px": best1, "best_2px": best2}
    out = os.path.join(RES, f"gate_sweep_skeleton_{site}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 results/{os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
