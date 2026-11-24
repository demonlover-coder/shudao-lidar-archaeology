# -*- coding: utf-8 -*-
"""西坝村重评（主板口径）：把未识别的第二支当作不存在。

真值裁剪规则（全自动，不手挑要素）：
    1) 各面要素单独栅格化 → 取最大连通域所在的分支为"主板"
    2) 只保留 ≥50% 像元落在主板内的面要素
    3) 线要素按 0.5 m 采样，≥50% 采样点落在主板内才保留

指标：
    A. 栅格：预测（model_guided_prob 的 P≥0.5）vs 主板面的 P/R/F1/IoU
    B. 线状：预测骨架 vs 主板中轴线 → 2.5 m Buffer IoU、Chamfer、≤2 m 覆盖率
    C. 点级双口径：主板面内全部点 / 主板面内 AGL≤0.3 m，并给"两边都限地面层"的公平口径

用法：
    python tools/eval_xbc_main.py --site 西坝村
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
from eval_new_shp import RES, ROOT, SHP_BASE, bilinear, parts_of, poly_mask, read_shp_any  # noqa: E402
from line_metrics_narrow import prf  # noqa: E402

LOG = os.path.join(RES, "eval_xbc_main.log")
REAL = os.path.join(
    os.environ.get("SHUDAO_DATA_ROOT", os.path.join(ROOT, "data")), "data_real"
)


def log(s):
    print(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="西坝村")
    ap.add_argument("--prob", default=None)
    ap.add_argument("--chunk", type=int, default=3_000_000)
    ap.add_argument("--agl", type=float, default=0.3)
    ap.add_argument("--tol-px", type=int, default=1)
    ap.add_argument("--gt-dilate", type=int, default=2)
    a = ap.parse_args()
    site = a.site
    log(f"\n{'='*76}\n[eval_xbc_main] {site}  {time.strftime('%Y-%m-%d %H:%M:%S')}")

    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem_min = d["dem_min"].astype(np.float64)
    pf = a.prob or os.path.join(RES, f"model_guided_prob_{site}.npz")
    z = np.load(pf)
    prob = z["prob"].astype(np.float64)
    pv = (z["valid"] == 1) if "valid" in z.files else (prob > 0)
    prob = np.where(pv, prob, 0.0)
    pred = (prob >= 0.5) & valid
    log(f"  网格 {H}×{W} @ {gsd} m；预测 P≥0.5 = {int(pred.sum()):,} px"
        f"（{os.path.basename(pf)}）")

    # ---------- 真值裁剪：主板 ----------
    from scipy.ndimage import binary_dilation, distance_transform_edt, label

    _, polys = read_shp_any(os.path.join(SHP_BASE, r"西坝村\XBC_orig_clean.shp"))
    masks = [poly_mask([sh], x0, y1, gsd, H, W) & valid for sh in polys]
    U = np.zeros((H, W), dtype=bool)
    for m in masks:
        U |= m
    lab, ncc = label(U)
    sizes = [(int((lab == k).sum()), k) for k in range(1, ncc + 1)]
    sizes.sort(reverse=True)
    main_k = sizes[0][1]
    core = lab == main_k
    log(f"  真值并集 {int(U.sum()):,} px → {ncc} 个连通域，"
        f"主板 {sizes[0][0]:,} px，次支 {sizes[1][0]:,} px（按用户要求视为不存在）")

    keep_feat = [i for i, m in enumerate(masks) if (m & core).sum() / max(int(m.sum()), 1) >= 0.5]
    drop_feat = [i for i in range(len(masks)) if i not in keep_feat]
    poly_main = np.zeros((H, W), dtype=bool)
    for i in keep_feat:
        poly_main |= masks[i]
    log(f"  面要素：保留 {[i + 1 for i in keep_feat]}，剔除 {[i + 1 for i in drop_feat]}"
        f" → 主板面 {int(poly_main.sum()):,} px（原 {int(U.sum()):,}，-"
        f"{(1 - poly_main.sum() / max(int(U.sum()), 1)):.1%}）")

    # ---------- A. 栅格指标 ----------
    raw = prf(int((pred & poly_main).sum()), int((pred & ~poly_main).sum()),
              int((~pred & poly_main).sum()))
    bufp = binary_dilation(pred, iterations=a.tol_px)
    buft = binary_dilation(poly_main, iterations=a.tol_px)
    bufpoly = prf(int((bufp & buft).sum()), int((bufp & ~buft).sum()), int((~bufp & buft).sum()))
    log(f"\n[A] 栅格（1 m）：P {raw['precision']:.4f}  R {raw['recall']:.4f}  "
        f"F1 {raw['f1']:.4f}  IoU {raw['iou']:.4f}")
    log(f"    含 {a.tol_px} px 容差：IoU {bufpoly['iou']:.4f}"
        f"（P {bufpoly['precision']:.4f} / R {bufpoly['recall']:.4f}）")

    # ---------- B. 线状指标 ----------
    from skimage.morphology import skeletonize
    _, lines = read_shp_any(os.path.join(SHP_BASE, r"西坝村\XBC_zhanda_line.shp"))
    segs = []          # [(samples Nx2, length_m)]
    for sh in lines:
        for xy in parts_of(sh):
            for k in range(len(xy) - 1):
                p0, p1 = xy[k], xy[k + 1]
                L = float(np.hypot(*(p1 - p0)))
                n = max(int(L / 0.5), 1)
                t = np.linspace(0, 1, n, endpoint=False)[:, None]
                segs.append((p0 + t * (p1 - p0), L))
    keep_seg, drop_seg = [], []
    for i, (s, L) in enumerate(segs):
        c = np.clip(((s[:, 0] - x0) / gsd).astype(int), 0, W - 1)
        r = np.clip(((y1 - s[:, 1]) / gsd).astype(int), 0, H - 1)
        (keep_seg if core[r, c].mean() >= 0.5 else drop_seg).append(i)
    gl = np.zeros((H, W), dtype=bool)
    for i in keep_seg:
        s = segs[i][0]
        c = np.clip(((s[:, 0] - x0) / gsd).astype(int), 0, W - 1)
        r = np.clip(((y1 - s[:, 1]) / gsd).astype(int), 0, H - 1)
        gl[r, c] = True
    gl = binary_dilation(gl, iterations=a.gt_dilate) if a.gt_dilate else gl
    ps = skeletonize(np.ascontiguousarray(pred.astype(np.uint8)))
    psb = binary_dilation(ps, iterations=a.tol_px)
    glb = binary_dilation(gl, iterations=a.tol_px)
    line_iou = float((psb & glb).sum() / max((psb | glb).sum(), 1))
    Dgl = distance_transform_edt(~gl, sampling=gsd)
    Dps = distance_transform_edt(~ps, sampling=gsd)
    ch = {"gt_to_pred_p50": round(float(np.percentile(Dps[gl], 50)), 2),
          "gt_to_pred_mean": round(float(Dps[gl].mean()), 2),
          "pred_to_gt_p50": round(float(np.percentile(Dgl[ps], 50)), 2),
          "pred_to_gt_mean": round(float(Dgl[ps].mean()), 2)}
    log(f"\n[B] 线状：保留线要素 {[i + 1 for i in keep_seg]}，剔除 {[i + 1 for i in drop_seg]}"
        f"（总长 {sum(segs[i][1] for i in keep_seg):.1f} m）")
    log(f"    2.5 m Buffer IoU {line_iou:.4f}（容差 {a.tol_px} px）")
    log(f"    Chamfer gt→pred 中位 {ch['gt_to_pred_p50']} m / 均值 {ch['gt_to_pred_mean']} m；"
        f"pred→gt 中位 {ch['pred_to_gt_p50']} m / 均值 {ch['pred_to_gt_mean']} m")
    log(f"    ≤2 m 覆盖：预测骨架 {float((Dgl[ps] <= 2).mean()):.1%}；"
        f"真值线 {float((Dps[gl] <= 2).mean()):.1%}")

    # ---------- C. 点级双口径 ----------
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    from matplotlib.path import Path
    paths = [Path(xy) for i in keep_feat for xy in parts_of(polys[i])]
    bbs = [p_.get_extents() for p_ in paths]
    st = {"all": [0, 0, 0], "gnd": [0, 0, 0], "fair": [0, 0, 0]}
    n_ta = n_tg = 0
    kept = 0
    n = xyz.shape[0]
    t0 = time.time()
    for s in range(0, n, a.chunk):
        e = min(s + a.chunk, n)
        p = np.asarray(xyz[s:e], dtype=np.float64) + o[None, :]
        X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
        c = (X - x0) / gsd - 0.5
        r = (y1 - Y) / gsd - 0.5
        ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
        pb = bilinear(prob, r, c)
        m05 = ins & (pb >= 0.5)
        cr = np.clip(np.rint(r), 0, H - 1).astype(np.int64)
        cl = np.clip(np.rint(c), 0, W - 1).astype(np.int64)
        agl = np.where(valid[cr, cl], Z - dem_min[cr, cl], np.nan)
        ta = np.zeros(e - s, dtype=bool)
        for p_, b in zip(paths, bbs):
            sel = ((X >= b.x0 - 2) & (X <= b.x1 + 2) & (Y >= b.y0 - 2) & (Y <= b.y1 + 2))
            kk = np.flatnonzero(sel)
            if kk.size:
                ta[kk] |= p_.contains_points(np.column_stack([X[kk], Y[kk]]))
        ta &= ins
        tg = ta & (agl <= a.agl)
        n_ta += int(ta.sum())
        n_tg += int(tg.sum())
        kept += int(m05.sum())
        st["all"][0] += int((m05 & ta).sum())
        st["all"][1] += int((m05 & ~ta).sum())
        st["all"][2] += int((~m05 & ta).sum())
        st["gnd"][0] += int((m05 & tg).sum())
        st["gnd"][1] += int((m05 & ~tg).sum())
        st["gnd"][2] += int((~m05 & tg).sum())
        mg = m05 & (agl <= a.agl)
        st["fair"][0] += int((mg & tg).sum())
        st["fair"][1] += int((mg & ~tg).sum())
        st["fair"][2] += int((~mg & tg).sum())
    log(f"\n[C] 点云 {n:,} 点遍历完成（{time.time()-t0:.0f}s）；真值 全部 "
        f"{n_ta:,} / 地面 {n_tg:,}；预测 {kept:,}")
    pt = {"truth_all": prf(*st["all"]), "truth_ground": prf(*st["gnd"]),
          "fair_ground_only": prf(*st["fair"]), "keep": kept}
    log(f"    面内全部点：P {pt['truth_all']['precision']:.4f}  R {pt['truth_all']['recall']:.4f}  "
        f"F1 {pt['truth_all']['f1']:.4f}  IoU {pt['truth_all']['iou']:.4f}")
    log(f"    面内 AGL≤{a.agl:g} m：P {pt['truth_ground']['precision']:.4f}  "
        f"R {pt['truth_ground']['recall']:.4f}  F1 {pt['truth_ground']['f1']:.4f}  "
        f"IoU {pt['truth_ground']['iou']:.4f}")
    log(f"    公平口径（两边限地面）：P {pt['fair_ground_only']['precision']:.4f}  "
        f"R {pt['fair_ground_only']['recall']:.4f}  F1 {pt['fair_ground_only']['f1']:.4f}  "
        f"IoU {pt['fair_ground_only']['iou']:.4f}")

    res = {"site": site, "truth_scope": "主板 only（第二支视为不存在）",
           "grid": [H, W], "gsd_m": gsd,
           "truth": {"union_px": int(U.sum()), "main_px": int(poly_main.sum()),
                     "main_component_px": int(core.sum()),
                     "components_px": [s_[0] for s_ in sizes],
                     "features_kept": [i + 1 for i in keep_feat],
                     "features_dropped": [i + 1 for i in drop_feat],
                     "line_segments_kept": [i + 1 for i in keep_seg],
                     "line_segments_dropped": [i + 1 for i in drop_seg],
                     "line_len_m": round(sum(segs[i][1] for i in keep_seg), 1),
                     "truth_all_pts": int(n_ta), "truth_ground_pts": int(n_tg)},
           "pixel": {"raw": raw, "tol_px": a.tol_px, "tolerant": bufpoly,
                     "pred_px": int(pred.sum())},
           "line": {"buffer_iou_2p5m": round(line_iou, 4), "chamfer": ch,
                    "coverage": {"pred_within2m": round(float((Dgl[ps] <= 2).mean()), 4),
                                 "gt_covered_2m": round(float((Dps[gl] <= 2).mean()), 4)}},
           "point": pt}
    out = os.path.join(RES, f"eval_xbc_main.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"\n  写出 results/{os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
