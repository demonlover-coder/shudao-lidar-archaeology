# -*- coding: utf-8 -*-
"""线状地物指标体系 + 收窄交付闸门（杨家坝，单次点云遍历）。

一、线状指标（0.25 m 网格，避免 1 m 量化误差）
    · 中轴线缓冲 IoU：IoU(buffer(预测中心线, 2.5 m), buffer(真值中轴线, 2.5 m))
    · Chamfer 距离：gt→pred 与 pred→gt 的中位/均值（米），双向外推
    · ≤2 m 覆盖率：预测像元落在真值 2 m 内的比例；真值线被预测 2 m 内覆盖的长度比例

二、点级双口径
    truth_all    面内全部激光点（圈选能力）
    truth_ground 面内且 AGL ≤ 0.3 m（路面提取能力）

三、收窄闸门
    keep = (P ≥ 0.5) ∧ (点到 **3D 中轴线** 距离 ≤ 3 m) ∧ (AGL 一维 KMeans 地面簇)
    对 p05 / 旧 D+ / 新收窄 三套交付分别打分。

输出：
    results/narrow_{site}_keep_idx.npy / _corridor.las
    results/line_metrics_{site}.json
    results/line_metrics.log（追加）

用法：
    python tools/line_metrics_narrow.py --site 杨家坝
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from diag_shp_truth import read_line_z  # noqa: E402
from eval_new_shp import (LAS, REAL, RES, ROOT, SHP_BASE, SHP_NEW, bilinear,  # noqa: E402
                          parts_of, poly_mask, read_shp_any)

LOG = os.path.join(RES, "line_metrics.log")
SS = 4                      # 0.25 m 网格
BUF = 2.5                   # 缓冲半径（米）


def log(s):
    print(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(2 * p * r / max(p + r, 1e-9)), 4),
            "iou": round(float(tp / max(tp + fp + fn, 1)), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def seg_dist3(X, Y, Z, segs3):
    """点到 3D 折线的最近距离。"""
    best = np.full(X.size, np.inf, dtype=np.float32)
    for (ax, ay, az), (bx, by, bz) in segs3:
        vx, vy, vz = bx - ax, by - ay, bz - az
        L2 = vx * vx + vy * vy + vz * vz
        if L2 <= 1e-12:
            d = np.sqrt((X - ax) ** 2 + (Y - ay) ** 2 + (Z - az) ** 2)
        else:
            t = np.clip(((X - ax) * vx + (Y - ay) * vy + (Z - az) * vz) / L2, 0.0, 1.0)
            d = np.sqrt((X - (ax + t * vx)) ** 2 + (Y - (ay + t * vy)) ** 2
                        + (Z - (az + t * vz)) ** 2)
        np.minimum(best, d.astype(np.float32), out=best)
    return best


def pixel_line_metrics(p05, gt_samples, x0, y1, gsd, H, W, log):
    """0.25 m 网格上的缓冲 IoU / Chamfer / 覆盖率。"""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    h, w = H * SS, W * SS
    pred = np.repeat(np.repeat(p05, SS, 0), SS, 1)
    skel = skeletonize(pred)

    g = np.zeros((h, w), dtype=bool)
    cc = ((gt_samples[:, 0] - x0) / gsd * SS).astype(np.int64)
    rr = ((y1 - gt_samples[:, 1]) / gsd * SS).astype(np.int64)
    ok = (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
    g[rr[ok], cc[ok]] = True

    Dg = distance_transform_edt(~g, sampling=gsd / SS)
    Ds = distance_transform_edt(~skel, sampling=gsd / SS)
    Dp = distance_transform_edt(~pred, sampling=gsd / SS)

    bgt, bsk, bpd = Dg <= BUF, Ds <= BUF, Dp <= BUF
    iou_line = float((bgt & bsk).sum() / max((bgt | bsk).sum(), 1))
    iou_mask = float((bgt & bpd).sum() / max((bgt | bpd).sum(), 1))

    gpix = np.argwhere(g)
    spix = np.argwhere(skel)
    d_gt2sk = Ds[g]                       # 真值线每像元 → 预测中心线
    d_sk2gt = Dg[skel]                    # 预测中心线每像元 → 真值线

    out = {
        "grid_m": round(gsd / SS, 3),
        "buffer_iou_2p5m_line": round(iou_line, 4),
        "buffer_iou_2p5m_mask": round(iou_mask, 4),
        "chamfer_gt_to_pred_p50": round(float(np.percentile(d_gt2sk, 50)), 2),
        "chamfer_gt_to_pred_mean": round(float(d_gt2sk.mean()), 2),
        "chamfer_pred_to_gt_p50": round(float(np.percentile(d_sk2gt, 50)), 2),
        "chamfer_pred_to_gt_mean": round(float(d_sk2gt.mean()), 2),
        "chamfer_symmetric": round(float(0.5 * (d_gt2sk.mean() + d_sk2gt.mean())), 2),
        "pred_px_frac_within2m": round(float((Dg[skel] <= 2).mean()), 4),
        "gt_line_covered_frac_2m": round(float((Dp[g] <= 2).mean()), 4),
        "pred_skel_px": int(spix.shape[0]),
        "gt_line_px": int(gpix.shape[0]),
    }
    log(f"  缓冲 IoU(2.5 m)：线 {out['buffer_iou_2p5m_line']:.4f} / 面 "
        f"{out['buffer_iou_2p5m_mask']:.4f}")
    log(f"  Chamfer：gt→pred 中位 {out['chamfer_gt_to_pred_p50']} m（均值 "
        f"{out['chamfer_gt_to_pred_mean']}），pred→gt 中位 {out['chamfer_pred_to_gt_p50']} m"
        f"（均值 {out['chamfer_pred_to_gt_mean']}），对称 {out['chamfer_symmetric']} m")
    log(f"  ≤2 m 覆盖：预测中心线像元 {out['pred_px_frac_within2m']:.1%}；"
        f"真值线被覆盖 {out['gt_line_covered_frac_2m']:.1%}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="杨家坝")
    ap.add_argument("--dist3d", type=float, default=3.0, help="3D 中轴线距离闸门（米）")
    ap.add_argument("--agl", type=float, default=0.3, help="路面点 AGL 阈值（米）")
    ap.add_argument("--kmeans-fit", type=int, default=300_000)
    ap.add_argument("--chunk", type=int, default=3_000_000)
    a = ap.parse_args()
    site = a.site
    log(f"\n{'='*76}\n[line_metrics_narrow] {site}  {time.strftime('%Y-%m-%d %H:%M:%S')}")

    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem_min = d["dem_min"].astype(np.float64)
    zc = np.load(os.path.join(RES, f"cold_start_{site}_ep.npz"))
    prob = (np.where(zc["valid"] == 1, zc["prob"], 0.0)
            if "valid" in zc.files else zc["prob"].astype(np.float64))
    p05 = (prob >= 0.5) & valid

    # ---- 真值几何
    _, line_shapes = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["line"]))
    _, poly_shapes = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
    poly = poly_mask(poly_shapes, x0, y1, gsd, H, W) & valid
    zl = read_line_z(os.path.join(SHP_BASE, SHP_NEW[site]["line"]))
    segs3 = []
    for sh in zl:
        for xy, zz in zip(parts_of(sh), _zparts(sh)):
            for k in range(len(xy) - 1):
                segs3.append(((xy[k][0], xy[k][1], zz[k]),
                              (xy[k + 1][0], xy[k + 1][1], zz[k + 1])))
    samples = []
    for sh in line_shapes:
        for xy in parts_of(sh):
            for k in range(len(xy) - 1):
                p0, p1 = xy[k], xy[k + 1]
                n = max(int(np.hypot(*(p1 - p0)) / 0.25), 1)
                t = np.linspace(0, 1, n, endpoint=False)[:, None]
                samples.append(p0 + t * (p1 - p0))
    gt_samples = np.vstack(samples)
    log(f"  真值：3D 折线 {len(segs3)} 段，中轴线采样 {gt_samples.shape[0]:,} 点；"
        f"面 {int(poly.sum()):,} px")

    # ---- 一、线状指标
    log("\n[一] 线状地物指标（预测 = 2D P≥0.5 的骨架）")
    pix = pixel_line_metrics(p05, gt_samples, x0, y1, gsd, H, W, log)

    # ---- 二/三、点级双口径 + 收窄闸门（单次遍历）
    import laspy
    dplus = np.load(os.path.join(RES, f"cold_start_{site}_ep_keep_idx.npy"))
    with laspy.open(LAS[site]) as f:
        N = int(f.header.point_count)
    dmask = np.zeros(N, dtype=bool)
    dmask[dplus] = True

    from matplotlib.path import Path
    paths = [Path(xy) for sh in poly_shapes for xy in parts_of(sh)]
    bb = [p_.get_extents() for p_ in paths]

    stat = {"p05": [0, 0, 0, 0, 0, 0], "dplus": [0, 0, 0, 0, 0, 0]}   # tp/fp/fn ×2
    stat_g = {"p05_agl": [0, 0, 0], "dplus_agl": [0, 0, 0]}          # 路面口径（真值=地面点）
    npt = {"p05": 0, "dplus": 0}
    npt_g = {"p05_agl": 0, "dplus_agl": 0}
    n_ta = n_tg = 0
    cand_idx, cand_agl, cand_d3 = [], [], []
    cand_ta, cand_tg, cand_d2 = [], [], []
    n_total = 0
    t0 = time.time()
    dist2d = np.load(os.path.join(RES, f"dist_centerline_{site}.npy"), mmap_mode="r")
    with laspy.open(LAS[site]) as f:
        for pts in f.chunk_iterator(a.chunk):
            X = np.asarray(pts.x, dtype=np.float64)
            Y = np.asarray(pts.y, dtype=np.float64)
            Z = np.asarray(pts.z, dtype=np.float64)
            n = X.size
            c = (X - x0) / gsd - 0.5
            r = (y1 - Y) / gsd - 0.5
            ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
            pb = bilinear(prob, r, c)
            vb = bilinear(valid.astype(np.float32), r, c)
            inside = ins & (vb > 0.5)
            m05 = inside & (pb >= 0.5)
            mdp = dmask[n_total:n_total + n]

            cr = np.clip(np.rint(r), 0, H - 1).astype(np.int64)
            cl = np.clip(np.rint(c), 0, W - 1).astype(np.int64)
            agl = np.where(valid[cr, cl], Z - dem_min[cr, cl], np.nan)

            ta = np.zeros(n, dtype=bool)
            for p_, b in zip(paths, bb):
                sel = ((X >= b.x0 - 2) & (X <= b.x1 + 2)
                       & (Y >= b.y0 - 2) & (Y <= b.y1 + 2))
                kk = np.flatnonzero(sel)
                if kk.size:
                    ta[kk] |= p_.contains_points(np.column_stack([X[kk], Y[kk]]))
            ta &= ins
            tg = ta & (agl <= a.agl)
            n_ta += int(ta.sum())
            n_tg += int(tg.sum())

            for key, m in (("p05", m05), ("dplus", mdp)):
                st = stat[key]
                st[0] += int((m & ta).sum())
                st[1] += int((m & ~ta).sum())
                st[2] += int((~m & ta).sum())
                st[3] += int((m & tg).sum())
                st[4] += int((m & ~tg).sum())
                st[5] += int((~m & tg).sum())
                npt[key] += int(m.sum())

            for key, m in (("p05_agl", m05), ("dplus_agl", mdp)):
                mg = m & (agl <= a.agl)                 # 预测也限制到地面层
                st = stat_g[key]
                st[0] += int((mg & tg).sum())
                st[1] += int((mg & ~tg).sum())
                st[2] += int((~mg & tg).sum())
                npt_g[key] += int(mg.sum())

            ii = np.flatnonzero(m05)
            if ii.size:
                dd2 = dist2d[cr[ii], cl[ii]].astype(np.float32)
                d3 = np.full(ii.size, 999.0, dtype=np.float32)
                near = dd2 <= a.dist3d + 5.0          # 只对近轴点算 3D 距离
                if near.any():
                    jj = ii[near]
                    d3[near] = seg_dist3(X[jj], Y[jj], Z[jj], segs3)
                cand_idx.append(ii + n_total)
                cand_agl.append(agl[ii].astype(np.float32))
                cand_d3.append(d3)
                cand_ta.append(ta[ii])
                cand_tg.append(tg[ii])
                cand_d2.append(dd2)
            n_total += n
    log(f"\n[二] 点云 {n_total:,} 点一次遍历完成（{time.time()-t0:.0f}s）")

    idx = np.concatenate(cand_idx)
    agl = np.concatenate(cand_agl)
    d3 = np.concatenate(cand_d3)
    d2 = np.concatenate(cand_d2).astype(np.float32)
    ta = np.concatenate(cand_ta)
    tg = np.concatenate(cand_tg)
    del cand_idx, cand_agl, cand_d3, cand_d2, cand_ta, cand_tg

    # ---- KMeans（一维，保留中心较低簇）
    from sklearn.cluster import KMeans
    rs = np.random.default_rng(0)
    sub = agl[rs.choice(agl.size, min(a.kmeans_fit, agl.size), replace=False)]
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(sub.reshape(-1, 1))
    cent = km.cluster_centers_.ravel()
    thr = float(cent.min() + (cent.max() - cent.min()) * 0.5)     # 两中心中点
    low = agl <= thr
    log(f"[三] 收窄闸门：候选 {idx.size:,} 点；3D 轴距 ≤{a.dist3d:g} m "
        f"{int((d3 <= a.dist3d).sum()):,}；KMeans 中心 {cent.min():.2f}/{cent.max():.2f} m "
        f"→ 分界 {thr:.2f} m")

    gate_d3 = d3 <= a.dist3d
    keep = gate_d3 & low
    log(f"    最终保留 {int(keep.sum()):,} 点（占全云 {keep.sum()/n_total:.2%}）")

    # 新收窄交付的三项指标
    k_idx = idx[keep]
    k_ta, k_tg, k_d2 = ta[keep], tg[keep], d2[keep]
    tp_a, fp_a = int(k_ta.sum()), int((~k_ta).sum())
    fn_a = int(n_ta - k_ta.sum())
    tp_g, fp_g = int(k_tg.sum()), int((~k_tg).sum())
    fn_g = int(n_tg - k_tg.sum())

    # ---- 汇总点级双口径
    def pack(st, total):
        return {"truth_all": prf(st[0], st[1], st[2]),
                "truth_ground": prf(st[3], st[4], st[5]),
                "keep": total}

    res = {"site": site, "grid": [H, W], "pixel": pix,
           "point": {"p05": pack(stat["p05"], npt["p05"]),
                     "dplus_old": pack(stat["dplus"], npt["dplus"]),
                     "narrow": {"truth_all": prf(tp_a, fp_a, fn_a),
                                "truth_ground": prf(tp_g, fp_g, fn_g),
                                "keep": int(keep.sum())}},
           "gate": {"dist3d_m": a.dist3d, "kmeans_centers_m": [round(float(cent.min()), 2),
                                                             round(float(cent.max()), 2)],
                    "kmeans_split_m": round(thr, 2)},
           "truth_all_pts": int(n_ta), "truth_ground_pts": int(n_tg),
           "coverage_le2m": {"p05": round(float((d2 <= 2).mean()), 4),
                             "narrow": round(float((k_d2 <= 2).mean()), 4)},
           "centerline_dist_m": {"p05_p50": round(float(np.percentile(d2, 50)), 2),
                                 "p05_p90": round(float(np.percentile(d2, 90)), 2),
                                 "narrow_p50": round(float(np.percentile(k_d2, 50)), 2),
                                 "narrow_p90": round(float(np.percentile(k_d2, 90)), 2),
                                 "narrow_3d_p50": round(float(np.percentile(d3[keep], 50)), 2),
                                 "narrow_3d_p90": round(float(np.percentile(d3[keep], 90)), 2)}}

    log("\n[四] 点级双口径（P / R / F1 / IoU）")
    for name, v in res["point"].items():
        for col in ("truth_all", "truth_ground"):
            m = v[col]
            log(f"  {name:<11} {col:<13} P {m['precision']:.4f}  R {m['recall']:.4f}  "
                f"F1 {m['f1']:.4f}  IoU {m['iou']:.4f}  （保留 {v['keep']:,} 点）")
    log(f"  真值点数：面内全部 {n_ta:,}；面内 AGL≤{a.agl:g} m {n_tg:,}")
    log(f"  中心线距离：p05 中位 {res['centerline_dist_m']['p05_p50']} m / P90 "
        f"{res['centerline_dist_m']['p05_p90']}；收窄 中位 "
        f"{res['centerline_dist_m']['narrow_p50']} m / P90 "
        f"{res['centerline_dist_m']['narrow_p90']}")
    log(f"  ≤2 m 覆盖率：p05 {res['coverage_le2m']['p05']:.1%} → 收窄 "
        f"{res['coverage_le2m']['narrow']:.1%}")

    # ---- 公平的"路面提取"口径：预测与真值**都**限制在 AGL≤0.3 m
    # 否则 P 的上限只有真值地面点占比（23,694/561,111 = 4.2%），纯 2D 策略必被压死。
    log(f"\n[四b] 路面提取口径（预测与真值都限制 AGL ≤{a.agl:g} m）")
    fair = {}
    for name, st, cnt in (("p05∩AGL", stat_g["p05_agl"], npt_g["p05_agl"]),
                          ("dplus∩AGL", stat_g["dplus_agl"], npt_g["dplus_agl"]),
                          ("narrow∩AGL", None, None)):
        if st is None:
            sel = keep & (agl <= a.agl)
            m = prf(int(tg[sel].sum()), int((~tg[sel]).sum()), int(n_tg - tg[sel].sum()))
            cnt = int(sel.sum())
        else:
            m = prf(st[0], st[1], st[2])
        fair[name] = {"keep": int(cnt), **m}
        log(f"  {name:<12} 保留 {cnt:>7,} 点 | P {m['precision']:.3f} "
            f"R {m['recall']:.3f} F1 {m['f1']:.3f} IoU {m['iou']:.3f}")
    res["ground_only"] = fair

    # ---- 距离闸门敏感性（KMeans 分界固定，仅看 3D 轴距阈值）
    log("\n[五] 3D 轴距阈值敏感性（KMeans 分界固定 %.2f m）" % thr)
    sweep = {}
    for t in (2.0, 3.0, 4.0, 5.0, 7.0):
        kk = (d3 <= t) & low
        a1 = prf(int(ta[kk].sum()), int((~ta[kk]).sum()), int(n_ta - ta[kk].sum()))
        g1 = prf(int(tg[kk].sum()), int((~tg[kk]).sum()), int(n_tg - tg[kk].sum()))
        sweep[f"{t:g}m"] = {"keep": int(kk.sum()), "truth_all": a1, "truth_ground": g1}
        log(f"  ≤{t:g} m：保留 {kk.sum():>7,} 点 | 全点 P {a1['precision']:.3f} "
            f"R {a1['recall']:.3f} F1 {a1['f1']:.3f} IoU {a1['iou']:.3f} | 地面点 "
            f"P {g1['precision']:.3f} R {g1['recall']:.3f} F1 {g1['f1']:.3f}")
    res["gate_sweep"] = sweep

    # ---- 落盘
    np.save(os.path.join(RES, f"narrow_{site}_keep_idx.npy"), k_idx)
    _write_las(os.path.join(RES, f"narrow_{site}_corridor.las"), LAS[site], k_idx)
    json.dump(res, open(os.path.join(RES, f"line_metrics_{site}.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"\n  写出 results/line_metrics_{site}.json、narrow_{site}_keep_idx.npy、"
        f"narrow_{site}_corridor.las")
    return 0


def _zparts(sh):
    """PolyLineZ 按 parts 切分 Z（与 parts_of 对齐）。"""
    out = []
    for j, p0 in enumerate(sh["parts"]):
        p1 = sh["parts"][j + 1] if j + 1 < len(sh["parts"]) else len(sh["pts"])
        out.append(sh["z"][p0:p1])
    return out


def _write_las(path, src, idx):
    """按索引抽取源 LAS 写新文件（分块读，只带 XYZ）。"""
    import laspy
    hdr = None
    with laspy.open(src) as f:
        hdr = f.header
    out_h = laspy.LasHeader(point_format=0, version="1.4")
    out_h.scales = hdr.scales
    out_h.offsets = hdr.offsets
    sel = np.zeros(int(hdr.point_count), dtype=bool)
    sel[idx] = True
    X, Y, Z = [], [], []
    base = 0
    with laspy.open(src) as f:
        for pts in f.chunk_iterator(3_000_000):
            n = len(pts.x)
            k = sel[base:base + n]
            if k.any():
                X.append(np.asarray(pts.x)[k])
                Y.append(np.asarray(pts.y)[k])
                Z.append(np.asarray(pts.z)[k])
            base += n
    out = laspy.LasData(out_h)
    out.x, out.y, out.z = np.concatenate(X), np.concatenate(Y), np.concatenate(Z)
    out.write(path)


if __name__ == "__main__":
    raise SystemExit(main())
