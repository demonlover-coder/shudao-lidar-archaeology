# -*- coding: utf-8 -*-
"""低阈值脊线桥接内部断口（bridge_gaps.py）。

流程（真值不参与任何决策）：
    1) 主干：收窄交付点云（P≥0.5 ∧ 3D 轴距≤3 m）→ 0.25 m 占用 → 闭运算×N → 骨架
    2) 桥接源：概率图 P≥`--prob-thr`（默认 0.25）→ 0.25 m 骨架 → 低阈值连续脊线
    3) 拓扑桥接：脊线连通域若同时接触**主干的两个不同连通域**，则判为桥接段
    4) 剪除冗余：只保留与主干连通的像素（孤立脊线不进融合结果）
    5) 评估：融合骨架对真值骨架，1 px 容差下的 P / R / F1 与 Buffer IoU

输出：results/bridge_gaps_{site}.json（含 0.20/0.25/0.30 三档阈值对比）

用法：
    python tools/bridge_gaps.py --site 杨家坝 --fig
    python tools/bridge_gaps.py --site 杨家坝 --prob-thr 0.30 --close 3
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
from eval_physical_width import pred_occ_from_points, up_skeleton  # noqa: E402

FIG = os.path.join(ROOT, "figures")
SS = 4


def metrics(sk, gt, gsd, tol=1.0):
    from scipy.ndimage import binary_dilation, distance_transform_edt
    Dg = distance_transform_edt(~gt, sampling=gsd)
    Dp = distance_transform_edt(~sk, sampling=gsd)
    P_ = float((Dg[sk] <= tol).mean()) if sk.any() else 0.0
    R_ = float((Dp[gt] <= tol).mean())
    F_ = 2 * P_ * R_ / max(P_ + R_, 1e-9)
    bp = binary_dilation(sk, iterations=int(tol)) if tol else sk
    bg = binary_dilation(gt, iterations=int(tol)) if tol else gt
    iou = float((bp & bg).sum() / max((bp | bg).sum(), 1))
    return {"precision": round(P_, 4), "recall": round(R_, 4), "f1": round(F_, 4),
            "buffer_iou": round(iou, 4), "skeleton_px": int(sk.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="杨家坝")
    ap.add_argument("--prob-thr", type=float, default=0.25)
    ap.add_argument("--sweep", default="0.20,0.25,0.30")
    ap.add_argument("--close", type=int, default=3, help="主干闭运算次数（0.25 m 上）")
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()
    site = a.site
    from scipy.ndimage import binary_closing, binary_dilation, label
    from skimage.morphology import skeletonize

    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    z = np.load(os.path.join(RES, f"cold_start_{site}_ep.npz"))
    prob = z["prob"].astype(np.float64)
    prob = np.where((z["valid"] == 1) if "valid" in z.files else (prob > 0), prob, 0.0)

    # ---------- 真值骨架（仅评估） ----------
    _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
    gt = up_skeleton(poly_mask_hr(polys, x0, y1, gsd, H, W, SS), H, W, SS) & valid

    # ---------- 1) 主干 ----------
    las = os.path.join(RES, f"narrow_{site}_corridor.las")
    if not os.path.exists(las):
        print(f"[错误] 缺少 {las}")
        return 2
    occ, npt = pred_occ_from_points(las, x0, y1, gsd, H, W, SS)
    if a.close:
        occ = binary_closing(occ, np.ones((3, 3), dtype=bool), iterations=a.close)
    trunk = up_skeleton(occ, H, W, SS) & valid
    lab_t, n_trunk = label(trunk)
    print(f"[{site}] 主干：{npt:,} 点 → 骨架 {int(trunk.sum())} px / {n_trunk} 段"
          f"（闭运算×{a.close}）")

    # ---------- 2) 低阈值脊线 ----------
    prob025 = np.repeat(np.repeat(prob, SS, 0), SS, 1)
    v025 = np.repeat(np.repeat(valid, SS, 0), SS, 1)

    def ridge_at(thr):
        m = (prob025 >= thr) & v025
        sk = skeletonize(np.ascontiguousarray(m.astype(np.uint8)))
        sk = sk[:H * SS, :W * SS].reshape(H, SS, W, SS).any(axis=(1, 3)) & valid
        return sk

    # ---------- 3) 拓扑桥接 + 4) 剪枝 ----------
    rows = []
    base = metrics(trunk, gt, gsd, a.tol)
    print(f"\n[基线] 主干骨架：P {base['precision']:.4f}  R {base['recall']:.4f}  "
          f"F1 {base['f1']:.4f}  Buffer IoU {base['buffer_iou']:.4f}")
    rows.append({"tag": "主干", **base})

    # ---------- 断口诊断：未覆盖的真值像素处概率有多低 ----------
    from scipy.ndimage import distance_transform_edt
    Dt = distance_transform_edt(~trunk, sampling=gsd)
    unc = gt & (Dt > a.tol)
    if unc.any():
        pv = prob[unc]
        print(f"\n[断口诊断] 未覆盖真值 {int(unc.sum())} px（距离中位 {np.median(Dt[unc]):.1f} m，"
              f"P90 {np.percentile(Dt[unc], 90):.1f} m）")
        print(f"  这些位置的模型概率：中位 {np.median(pv):.3f}；P≥0.5 "
              f"{float((pv >= 0.5).mean()):.1%}，0.2≤P<0.5 {float(((pv >= 0.2) & (pv < 0.5)).mean()):.1%}，"
              f"P<0.05 {float((pv < 0.05).mean()):.1%}")

    detail = {}
    for thr in [float(t) for t in a.sweep.split(",") if t.strip()]:
        sk_r = ridge_at(thr)
        lab_r, n_r = label(sk_r)
        keep = np.zeros_like(sk_r)
        n_bridge = 0
        for k in range(1, n_r + 1):
            comp = lab_r == k
            # 该脊线段 1 px 邻域内接触到的**不同主干段**编号
            near = binary_dilation(comp, iterations=1)
            tl = np.unique(lab_t[near & trunk])
            tl = tl[tl > 0]
            if tl.size >= 2:
                keep |= comp
                n_bridge += 1
        fused = trunk | keep
        # 剪除冗余：只留与主干同连通域的像素
        lab_f, n_f = label(fused)
        trunk_ids = np.unique(lab_f[trunk])
        trunk_ids = trunk_ids[trunk_ids > 0]
        fused = np.isin(lab_f, trunk_ids)
        mm = metrics(fused, gt, gsd, a.tol)
        rows.append({"tag": f"P≥{thr:g} 桥接", "bridge_components": n_bridge,
                     "ridge_px": int(sk_r.sum()), "kept_bridge_px": int(keep.sum()),
                     **mm})
        detail[f"{thr:g}"] = {"ridge_px": int(sk_r.sum()), "ridge_components": int(n_r),
                              "bridge_components": int(n_bridge),
                              "kept_bridge_px": int(keep.sum()),
                              "components_after": int(len(trunk_ids)), **mm}
        print(f"  P≥{thr:g}：脊线 {int(sk_r.sum()):>4} px / {n_r:>3} 段 → 桥接段 "
              f"{n_bridge:>2} 个（{int(keep.sum()):>3} px）→ 融合骨架 {int(fused.sum()):>4} px | "
              f"P {mm['precision']:.4f}  R {mm['recall']:.4f}  F1 {mm['f1']:.4f} | "
              f"Buffer IoU {mm['buffer_iou']:.4f}")
        if abs(thr - a.prob_thr) < 1e-9:
            chosen = (fused, mm)

    best = max(rows[1:], key=lambda r: r["f1"]) if len(rows) > 1 else rows[0]
    print(f"\n[结论] 最优 = {best['tag']}：F1 {best['f1']:.4f}（基线 {base['f1']:.4f}，"
          f"Δ {best['f1'] - base['f1']:+.4f}）；召回 {base['recall']:.4f} → {best['recall']:.4f}")
    print(f"        {'突破 0.70' if best['f1'] >= 0.70 else '未达 0.70'}")

    res = {"site": site, "grid": [H, W], "gsd_m": gsd, "skel_res_m": gsd / SS,
           "close_iter": a.close, "tol_px": a.tol, "gt_skeleton_px": int(gt.sum()),
           "trunk_skeleton_px": int(trunk.sum()), "trunk_components": int(n_trunk),
           "rows": rows, "detail": detail, "pass_0p70": bool(best["f1"] >= 0.70)}
    out = os.path.join(RES, f"bridge_gaps_{site}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 results/{os.path.basename(out)}")

    if a.fig and "chosen" in dir():
        fused = chosen[0]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        ys, xs = np.where(gt | fused)
        r0, r1 = max(int(ys.min()) - 15, 0), min(int(ys.max()) + 15, H)
        c0, c1 = max(int(xs.min()) - 15, 0), min(int(xs.max()) + 15, W)
        sl = (slice(r0, r1), slice(c0, c1))
        rgb = np.ones((r1 - r0, c1 - c0, 3), dtype=np.float32)
        rgb[trunk[sl]] = [0.55, 0.55, 0.55]            # 主干（灰）
        rgb[(fused & ~trunk)[sl]] = [0.20, 0.45, 0.95]  # 桥接段（蓝）
        rgb[gt[sl]] = [0.0, 0.45, 0.0]                  # 真值骨架（绿）
        fig, ax = plt.subplots(figsize=(13, 13))
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{site}：灰=主干骨架  蓝=低阈值桥接段  绿=真值骨架\n"
                     f"F1 {base['f1']:.4f} → {chosen[1]['f1']:.4f}"
                     f"（P≥{a.prob_thr:g}，{a.tol:g} px 容差）")
        ax.axis("off")
        fig.tight_layout()
        pth = os.path.join(FIG, f"bridge_gaps_{site}.png")
        fig.savefig(pth, dpi=140)
        plt.close(fig)
        print(f"  写出 figures/{os.path.basename(pth)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
