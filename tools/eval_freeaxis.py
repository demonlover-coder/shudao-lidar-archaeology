# -*- coding: utf-8 -*-
"""无 SHP 轴线（模型自产）下的收窄交付评估。

轴线来源（二选一，均不读 SHP 几何）：
    path : cold_start_{site}_lc6.npz 里的 Dijkstra 路径（openness + h1 相位推出）
    skel : 概率图 P>=0.5 掩膜的骨架

流程：候选(P>=0.5) → 到轴线 3D 距离<=d → AGL<=2.0 m（低植被层上限，文献口径）→ 交付点，
再算三套口径：点级（全部/公平）、像素 Buffer IoU、骨架 F1。

SHP 仅用于最后评分（真值面 → 1m 掩膜 → 5m 等宽带 / 骨架）。

用法：
    python tools/eval_freeaxis.py --site 杨家坝 --axis path --dist3 3.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, maximum_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import LAS, RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402
from eval_physical_width import up_skeleton  # noqa: E402

SS = 4          # 高分辨率倍率（0.25 m）
CH = 3_000_000


def fill_nodata(a, valid):
    if valid.all():
        return a.astype(np.float64)
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return a[tuple(idx)].astype(np.float64)


def axis_field(site, prob, valid, H, W, gsd, x0, y1, s, dem, mode, npz):
    """返回 (二维轴距场 d2, 轴线高程场 zax)。"""
    if mode == "skel":
        from skimage.morphology import skeletonize
        core = (prob >= 0.5) & valid
        ax = skeletonize(binary_closing(core, np.ones((3, 3), bool)))
    else:
        z = np.load(npz)
        p = z["path"]
        ax = np.zeros((H, W), bool)
        if p.ndim == 2 and p.shape[1] == 2:
            rr = np.round(p[:, 0]).astype(int)
            cc = np.round(p[:, 1]).astype(int)
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            ax[rr[ok], cc[ok]] = True
        else:
            ax = p.astype(bool)
    if not ax.any():
        raise RuntimeError("轴线为空")
    d2, idx = distance_transform_edt(~ax, return_indices=True)
    zax = dem[tuple(idx)]
    print(f"  轴线[{mode}] {int(ax.sum()):,} px，长度约 {ax.sum() * gsd:.0f} m")
    return d2, zax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="杨家坝")
    ap.add_argument("--axis", default="path", choices=["path", "skel"])
    ap.add_argument("--dist3", type=float, default=3.5)
    ap.add_argument("--agl", type=float, default=2.0,
                    help="AGL 上限（米）：取低植被层上限，文献口径")
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--out", default="eval_freeaxis")
    ap.add_argument("--npz", default="",
                    help="概率图 npz（默认 cold_start_{site}_lc6.npz，即主结果所用模型）")
    ap.add_argument("--skel", default="1m", choices=("1m", "hr"),
                    help="骨架口径：1m=预测与真值都在 1 m 掩膜上取骨架（默认，可比）；"
                         "hr=预测用 0.25 m 稀疏占用（毛刺多，仅历史对照）")
    ap.add_argument("--truth", default="new", choices=("new", "old"),
                    help="评分真值：new=新 SHP 面；old=旧 SHP 栅格 mask_1m_{site}.npy")
    ap.add_argument("--truth-main", action="store_true",
                    help="真值只取最大连通域（西坝村剔除第二支的主板口径）")
    a = ap.parse_args()

    site = a.site
    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem = fill_nodata(d["dem_min"], valid)
    npz = a.npz or f"cold_start_{site}_lc6.npz"
    if not os.path.isabs(npz):
        npz = os.path.join(RES, npz)
    prob = np.load(npz)["prob"].astype(np.float64)
    prob = np.where(valid, prob, 0.0)
    print(f"  概率图 {os.path.basename(npz)}")

    d2, zax = axis_field(site, prob, valid, H, W, gsd, x0, y1, None, dem, a.axis, npz)

    # 真值（仅评分）：truth=new → 新 SHP 面；truth=old → 旧 SHP 栅格（mask_1m）
    mask_hr = None
    if a.truth == "old":
        gt = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
        print(f"  真值=旧 SHP 栅格 mask_1m_{site}.npy：{int(gt.sum()):,} px")
    else:
        _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        mask_hr = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
        gt = mask_hr.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5
        gt &= valid
        print(f"  真值=新 SHP {SHP_NEW[site]['poly']}：{int(gt.sum()):,} px")
    if a.truth_main:
        # 只保留最大连通域（西坝村第二支"当不存在"时的主板口径）
        from scipy.ndimage import label as cc_lab
        lb, ncc = cc_lab(gt)
        if ncc > 1:
            sz = np.bincount(lb.ravel())
            sz[0] = 0
            gt = lb == sz.argmax()
            print(f"  真值取主板：{ncc} 个连通域 → 保留 {int(gt.sum()):,} px")
    from skimage.morphology import skeletonize
    if a.skel == "hr" and mask_hr is not None:
        gt_skel = up_skeleton(mask_hr, H, W, SS) & valid & gt
    else:
        gt_skel = skeletonize(gt)

    # 流式读点云
    tp = fp = fn = 0
    tp_f = fp_f = fn_f = 0
    n_keep = n_cand = 0
    n_tr_all = n_tr_g = 0
    XY = []
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
            agl_all = Z - dem[r, c]
            tr_all = gt[r, c]
            # FN 分母：全部真值点 / 其中地面点（含 P<0.5 未进候选的点）
            n_tr_all += int(tr_all.sum())
            n_tr_g += int((tr_all & (agl_all <= 0.3)).sum())
            cand = prob[r, c] >= 0.5
            n_cand += int(cand.sum())
            if not cand.any():
                continue
            X, Y, Z, c, r = X[cand], Y[cand], Z[cand], c[cand], r[cand]
            agl = agl_all[cand]
            d3 = np.hypot(d2[r, c], Z - zax[r, c])
            keep = (d3 <= a.dist3) & (agl <= a.agl)
            tr = gt[r, c]
            ground = agl <= 0.3
            tp += int((keep & tr).sum())
            fp += int((keep & ~tr).sum())
            kg = keep & ground
            tp_f += int((kg & tr).sum())
            fp_f += int((kg & ~tr).sum())
            n_keep += int(keep.sum())
            if keep.any():
                XY.append(np.stack([X[keep], Y[keep]], 1))

    XY = np.concatenate(XY) if XY else np.zeros((0, 2))
    # FN = 全部真值点（地面口径则为全部真值地面点）中未交付的数量，
    # 含像元 P<0.5 因而从未进入候选的那些点
    fn = n_tr_all - tp
    fn_f = n_tr_g - tp_f
    occ = np.zeros((H, W), bool)
    if XY.size:
        cc = np.floor((XY[:, 0] - x0) / gsd).astype(int)
        rr = np.floor((y1 - XY[:, 1]) / gsd).astype(int)
        occ[rr, cc] = True

    # 骨架口径：1m = 交付栅格（闭运算 1 次）→ skeletonize，与真值同构；
    #           hr = 0.25 m 稀疏占用 + 闭运算×3（历史口径，毛刺多）
    from skimage.morphology import skeletonize
    if a.skel == "hr":
        hi = np.zeros((H * SS, W * SS), bool)
        if XY.size:
            ch = np.floor((XY[:, 0] - x0) / (gsd / SS)).astype(int)
            rh = np.floor((y1 - XY[:, 1]) / (gsd / SS)).astype(int)
            ok = (ch >= 0) & (ch < W * SS) & (rh >= 0) & (rh < H * SS)
            hi[rh[ok], ch[ok]] = True
        sk = skeletonize(binary_closing(hi, np.ones((3, 3), bool), iterations=3))
        sk1 = np.zeros((H, W), bool)
        hh, ww = H * SS, W * SS
        sk1[:] = sk[:hh - hh % SS, :ww - ww % SS].reshape(H, SS, W, SS).any(axis=(1, 3))
    else:
        sk1 = skeletonize(binary_closing(occ, np.ones((3, 3), bool)))

    def f1(pred, gt_, t):
        bp = binary_dilation(pred, iterations=t) if t else pred
        bg = binary_dilation(gt_, iterations=t) if t else gt_
        tp_ = (bp & bg).sum()
        return (tp_ / max(bp.sum(), 1), tp_ / max(bg.sum(), 1),
                tp_ / max(bp.sum() + bg.sum() - tp_, 1))

    band = binary_dilation(gt_skel, iterations=2)
    biou = {}
    for t in (0, 1, 2):
        bp = binary_dilation(occ, iterations=t) if t else occ
        bg = binary_dilation(band, iterations=t) if t else band
        biou[t] = float((bp & bg).sum() / max((bp | bg).sum(), 1))

    def prf(tp_, fp_, fn_):
        p = tp_ / max(tp_ + fp_, 1)
        r = tp_ / max(tp_ + fn_, 1)
        return {"P": round(p, 4), "R": round(r, 4),
                "F1": round(2 * p * r / max(p + r, 1e-9), 4),
                "IoU": round(tp_ / max(tp_ + fp_ + fn_, 1), 4)}

    res = {"site": site, "axis": a.axis, "dist3": a.dist3, "agl": a.agl,
           "n_cand": n_cand, "n_keep": n_keep, "px": int(occ.sum()),
           "skel_px": int(sk1.sum()), "gt_skel_px": int(gt_skel.sum()),
           "point_all": prf(tp, fp, fn), "point_fair": prf(tp_f, fp_f, fn_f),
           "buffer_iou_5m_band": {str(k): round(v, 4) for k, v in biou.items()},
           "skel_f1": {}}
    for t in (1, 2):
        p, r, f_ = f1(sk1, gt_skel, t)
        res["skel_f1"][str(t)] = {"P": round(p, 4), "R": round(r, 4), "F1": round(f_, 4)}
    print(f"  候选 {n_cand:,} → 交付 {n_keep:,} 点（栅格 {occ.sum():,} px，骨架 {sk1.sum()} px）")
    print(f"  点级全点  {res['point_all']}")
    print(f"  点级公平  {res['point_fair']}")
    print(f"  骨架 F1   {res['skel_f1']}")
    print(f"  Buffer IoU {res['buffer_iou_5m_band']}")
    out = os.path.join(RES, f"{a.out}_{site}_{a.axis}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 {out}")


if __name__ == "__main__":
    main()
