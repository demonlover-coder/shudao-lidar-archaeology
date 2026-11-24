# -*- coding: utf-8 -*-
"""IoU 天花板诊断：为什么公平口径 IoU 停在 0.5 附近，到不了 0.7。

诊断对象：当前定稿流程（端点给定 + 模型轴线 + ±3 m + AGL 门 + 公平点级口径）。

输出三块证据：
1. **误差归因**：把 FN（漏掉的真值地面点）按"概率门 / 轴线门"二分，
   把 FP（多留的地面点）按"离真值中轴线多远"分桶 —— 定位是横向漂移还是召回不足。
2. **单因素 Oracle**：分别用真值替换 概率门 / 轴线门，看 IoU 能涨多少，
   从而判断瓶颈在 2D 概率、轴线定位，还是数据本身（地面点缺失）。
3. **参数扫描**：在现有输出上扫 P 阈值 × 轴距，给出"当前输入可达的最大 IoU"。
   若扫描上限远低于 0.7，说明靠调参不可能达标。

用法：
    python tools/diag_iou_ceiling.py --site 西坝村 --truth new --truth-main
    python tools/diag_iou_ceiling.py --site 杨家坝 --truth old
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import LAS, RES, SHP_BASE, SHP_NEW  # noqa: E402
from eval_freeaxis import axis_field, fill_nodata, poly_mask_hr  # noqa: E402

CH = 4_000_000
GROUND = 0.3          # 公平口径的地面阈值（m）


def iou(tp, fp, fn):
    return tp / max(tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--axis", default="path", choices=("path", "skel"))
    ap.add_argument("--dist3", type=float, default=3.0)
    ap.add_argument("--agl", type=float, default=2.0)
    ap.add_argument("--truth", default="new", choices=("new", "old"))
    ap.add_argument("--truth-main", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    site = a.site
    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem = fill_nodata(d["dem_min"], valid)

    npz = a.npz or f"cold_start_{site}_ep.npz"
    if not os.path.isabs(npz):
        npz = os.path.join(RES, npz)
    prob = np.load(npz)["prob"].astype(np.float64)
    prob = np.where(valid, prob, 0.0)

    # 真值（与 eval_freeaxis 同口径）
    from skimage.morphology import skeletonize
    if a.truth == "old":
        gt = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
    else:
        from eval_new_shp import read_shp_any
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
    gt_skel = skeletonize(gt)
    # 真值中轴线的二维距离场（轴线 Oracle 用）
    d2_gt = distance_transform_edt(~gt_skel)

    d2, zax = axis_field(site, prob, valid, H, W, gsd, x0, y1, None, dem, a.axis, npz)
    print(f"[{site}] 真值 {int(gt.sum()):,} px，真值骨架 {int(gt_skel.sum()):,} px，"
          f"轴线 {a.axis}")

    # ---- 单次流式扫描：只保留候选点（P≥0.5）的逐点证据 ----
    import laspy
    A_agl, A_d3, A_dgt, A_prob, A_tr = [], [], [], [], []
    n_trg_all = n_trg_lowp = 0     # 真值地面点总数 / 其中像元 P<0.5（未进候选）
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
            pv_ = prob[r, c]
            agl_ = Z - dem[r, c]
            tg = gt[r, c] & (agl_ <= GROUND)
            n_trg_all += int(tg.sum())
            m = pv_ >= 0.5
            n_trg_lowp += int((tg & ~m).sum())
            if not m.any():
                continue
            X, Y, Z, c, r, pv = X[m], Y[m], Z[m], c[m], r[m], pv_[m]
            agl = Z - dem[r, c]
            d3 = np.hypot(d2[r, c], Z - zax[r, c])
            A_agl.append(agl.astype(np.float32))
            A_d3.append(d3.astype(np.float32))
            A_dgt.append(d2_gt[r, c].astype(np.float32))
            A_prob.append(pv.astype(np.float32))
            A_tr.append(gt[r, c])

    agl = np.concatenate(A_agl)
    d3 = np.concatenate(A_d3)
    dgt = np.concatenate(A_dgt)
    pv = np.concatenate(A_prob)
    tr = np.concatenate(A_tr)
    n = agl.size
    print(f"  候选点（P≥0.5）{n:,}")

    # ---- 公平口径：预测与真值都限制在 AGL ≤ 0.3 ----
    g = agl <= GROUND
    trg = tr & g

    def conf(keep):
        kg = keep & g
        tp = int((kg & tr).sum())
        fp = int((kg & ~tr).sum())
        # FN = 全部真值地面点中未交付的（含 P<0.5 未进候选的点）
        fn = int(n_trg_all - tp)
        return tp, fp, fn

    base_keep = (d3 <= a.dist3) & (agl <= a.agl)
    tp0, fp0, fn0 = conf(base_keep)
    iou0 = iou(tp0, fp0, fn0)
    print(f"  基线：TP {tp0:,} / FP {fp0:,} / FN {fn0:,} → 公平 IoU {iou0:.4f}")

    # ---- 误差归因 ----
    # FN 归因：真值地面点未被保留 → ① 像元 P<0.5（未进候选）② 轴线门 ③ 其它
    fn_total = n_trg_all - tp0
    fn_prob = n_trg_lowp
    fn_axis = int((trg & (d3 > a.dist3)).sum())
    fn_other = max(fn_total - fn_prob - fn_axis, 0)
    print(f"  FN 总计 {fn_total:,}：概率门（P<0.5，未进候选）{fn_prob:,}"
          f"（{fn_prob/max(fn_total,1):.1%}） / 轴线门（>±{a.dist3:g} m）{fn_axis:,}"
          f"（{fn_axis/max(fn_total,1):.1%}） / 其它 {fn_other:,}")

    # FP：多留的地面点 → 按离真值中轴线距离分桶（横向漂移证据）
    fpm = base_keep & g & ~tr
    bins = [0, 1, 2, 3, 5, 10, 1e9]
    fp_hist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = fpm & (dgt >= lo) & (dgt < hi)
        fp_hist.append({"range_m": f"{lo:g}-{'inf' if hi > 1e8 else f'{hi:g}'}",
                        "n": int(m.sum())})
    print("  FP 横向分布（离真值骨架距离）：",
          "  ".join(f"{h['range_m']}m:{h['n']:,}" for h in fp_hist))

    # ---- 单因素 Oracle ----
    orc = {}
    # B：概率完美（把"是否在真值内"当概率）
    keepB = tr & (d3 <= a.dist3) & (agl <= a.agl)
    tb, fb, nb_ = conf(keepB)
    orc["prob_perfect"] = {"IoU": round(iou(tb, fb, nb_), 4), "tp": tb, "fp": fb, "fn": nb_}
    # C：轴线完美（用真值骨架距离代替模型轴线）
    keepC = (dgt <= a.dist3) & (agl <= a.agl)
    tc, fc, nc_ = conf(keepC)
    orc["axis_perfect"] = {"IoU": round(iou(tc, fc, nc_), 4), "tp": tc, "fp": fc, "fn": nc_}
    # D：两者都完美（理论可达上限）
    keepD = tr & (dgt <= a.dist3)
    td, fd, nd = conf(keepD)
    orc["both_perfect"] = {"IoU": round(iou(td, fd, nd), 4), "tp": td, "fp": fd, "fn": nd}
    print(f"  Oracle 概率完美 {orc['prob_perfect']['IoU']:.4f} | "
          f"轴线完美 {orc['axis_perfect']['IoU']:.4f} | "
          f"双完美 {orc['both_perfect']['IoU']:.4f}")

    # ---- 参数扫描：当前输入下可达的最大 IoU ----
    sweep = []
    best = (0.0, None)
    for p in (0.5, 0.6, 0.7, 0.8, 0.9):
        for dd in (2.0, 2.5, 3.0, 4.0, 5.0):
            k = (pv >= p) & (d3 <= dd) & (agl <= a.agl)
            tp, fp, fn = conf(k)
            v = iou(tp, fp, fn)
            sweep.append({"p": p, "d3": dd, "IoU": round(v, 4)})
            if v > best[0]:
                best = (v, (p, dd))
    print(f"  扫描上限 IoU {best[0]:.4f}（P≥{best[1][0]}，±{best[1][1]:g} m）")

    # ---- 到 0.7 还差什么 ----
    # IoU=0.7 时，在 TP 不变的前提下允许的 (FP+FN) 上限
    allow = tp0 / 0.7 - tp0
    need_cut = (fp0 + fn0) - allow
    verdict = {
        "current_IoU": round(iou0, 4),
        "gap_to_0.7": round(0.7 - iou0, 4),
        "fp_plus_fn": fp0 + fn0,
        "allowed_fp_plus_fn_at_0.7": round(allow, 1),
        "must_reduce_errors_by": round(need_cut, 1),
        "must_reduce_ratio": round(need_cut / max(fp0 + fn0, 1), 4),
        "sweep_best_IoU": round(best[0], 4),
        "sweep_best_param": {"p": best[1][0], "d3": best[1][1]},
    }
    print(f"  到 0.7 需把 (FP+FN) 从 {fp0+fn0:,} 降到 {allow:,.0f}"
          f"（削减 {need_cut/max(fp0+fn0,1):.1%}）")

    out = {"site": site, "truth": a.truth, "truth_main": bool(a.truth_main),
           "axis": a.axis, "baseline": {"tp": tp0, "fp": fp0, "fn": fn0,
                                        "P": round(tp0 / max(tp0 + fp0, 1), 4),
                                        "R": round(tp0 / max(tp0 + fn0, 1), 4),
                                        "IoU": round(iou0, 4)},
           "fn_attribution": {"total": fn_total, "prob_gate": fn_prob,
                              "axis_gate": fn_axis, "other": fn_other,
                              "prob_share": round(fn_prob / max(fn_total, 1), 4)},
           "fp_lateral": fp_hist, "oracles": orc, "sweep": sweep, "verdict": verdict}

    tag = a.tag or f"{a.truth}{'_main' if a.truth_main else ''}_{a.axis}"
    p = os.path.join(RES, f"diag_iou_ceiling_{site}_{tag}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  写出 {p}")

    # ---- 结论提示（自动判断瓶颈） ----
    gb = orc["prob_perfect"]["IoU"] - iou0
    gc = orc["axis_perfect"]["IoU"] - iou0
    print("\n  瓶颈判断：")
    print(f"    概率门误差贡献 {gb:+.4f}，轴线误差贡献 {gc:+.4f}")
    if orc["both_perfect"]["IoU"] < 0.7:
        print("    → 即使两项都完美也 < 0.7：瓶颈在数据/真值（地面点缺失或标注范围）")
    elif best[0] < 0.7:
        print("    → 现有输入调参达不到 0.7：必须改进 2D 概率或轴线定位")
    else:
        print("    → 现有输入可达 0.7，当前参数不是最优")


if __name__ == "__main__":
    main()
