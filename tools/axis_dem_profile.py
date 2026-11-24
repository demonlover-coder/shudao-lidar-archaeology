# -*- coding: utf-8 -*-
"""检查模型轴"拉直"段是否横跨冲沟/山脊（只看地形，不参与流程）。

做法：
1. 把模型轴与真值骨架都排成有序折线，按 1 m 采样；
2. 找模型轴上|到真值距离|最大且连续 >3 m 的段落（即"抄近路/拉直"段）；
3. 在该段提取：
   · 沿程高程（模型轴 vs 真值轴）
   · 沿程纵坡
   · 段中点的法向横断面（±25 m），看模型轴与真值之间是否有沟/脊
4. 判定：若两线之间的横断面出现"下降再上升"超过 1.5 m → 跨沟；
          出现"上升再下降" → 跨脊；否则两线位于同一坡面/台地。

用法：
    python tools/axis_dem_profile.py --site 杨家坝 --truth old
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from fig_pred_vs_truth import order_path  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
GULLY = 1.5      # 沟/脊判定幅度（m）


def resample(path, step_m=1.0, gsd=1.0):
    """有序折线 → 按弧长 1 m 重采样（返回行、列、累计距离）。"""
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))] * gsd
    s = np.arange(0, d[-1], step_m)
    r = np.interp(s, d, path[:, 0])
    c = np.interp(s, d, path[:, 1])
    return r, c, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="杨家坝")
    ap.add_argument("--npz", default="")
    ap.add_argument("--npz2", default="", help="对照 npz（同一图上再画一条轴，如加高差惩罚后的版本）")
    ap.add_argument("--truth", default="old", choices=("new", "old"))
    ap.add_argument("--truth-main", action="store_true")
    ap.add_argument("--dev", type=float, default=3.0, help="判定偏离的阈值（m）")
    a = ap.parse_args()

    site = a.site
    x0, y1, wm, hm, gsd, dd = P.dem_window(site)
    H, W = dd["dem_mean"].shape
    valid = dd["valid"] == 1
    dem = fill_nodata(dd["dem_mean"], valid)       # 地形剖面用 dem_mean（平滑）
    dem_min = fill_nodata(dd["dem_min"], valid)

    npz = a.npz or f"cold_start_{site}_ep.npz"
    if not os.path.isabs(npz):
        npz = os.path.join(RES, npz)
    prob = np.where(valid, np.load(npz)["prob"].astype(float), 0.0)

    from skimage.morphology import skeletonize
    if a.truth == "old":
        gt = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
    else:
        from eval_new_shp import read_shp_any, SHP_BASE, SHP_NEW
        from eval_freeaxis import poly_mask_hr
        SS = 4
        _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        mh = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
        gt = mh.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5
        gt &= valid
    if a.truth_main:
        from scipy.ndimage import label as ccl
        lb, n = ccl(gt)
        if n > 1:
            sz = np.bincount(lb.ravel()); sz[0] = 0
            gt = lb == sz.argmax()
    gt_skel = skeletonize(gt)

    axis = np.load(npz)["path"].astype(bool)
    pm = order_path(axis)
    pt = order_path(gt_skel)
    if pm is None or pt is None:
        print("轴线或真值骨架过短，退出")
        return

    # 模型轴按 1 m 采样；到真值骨架的距离
    rm, cm, sm = resample(pm)
    ri = np.clip(np.round(rm).astype(int), 0, H - 1)
    ci = np.clip(np.round(cm).astype(int), 0, W - 1)
    d_gt = distance_transform_edt(~gt_skel)
    dev = d_gt[ri, ci]                      # 每米一个偏离量（m）

    # 找连续偏离段，取最长（且平均偏离最大）
    m = dev > a.dev
    segs, i = [], 0
    while i < len(m):
        if not m[i]:
            i += 1
            continue
        j = i
        while j < len(m) and m[j]:
            j += 1
        if j - i >= 20:                     # 至少 20 m
            segs.append((i, j))
        i = j
    if not segs:
        print(f"没有连续偏离 >{a.dev:g} m 的 20 m 以上段落")
        return
    segs.sort(key=lambda s: -dev[s[0]:s[1]].mean())
    i0, i1 = segs[0]
    print(f"最长/最偏段落：沿程 {sm[i0]:.0f}–{sm[i1]:.0f} m（{i1-i0} m），"
          f"平均偏离 {dev[i0:i1].mean():.1f} m，最大 {dev[i0:i1].max():.1f} m")

    # 段内沿程高程（模型轴 vs 真值最近点）
    rt, ct, st = resample(pt)
    st_i = np.clip(np.round(rt).astype(int), 0, H - 1)
    ct_i = np.clip(np.round(ct).astype(int), 0, W - 1)
    # 真值点投影到模型弧长（最近邻）
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[rm, cm])
    dist_t, idx_t = tree.query(np.c_[rt, ct])
    keep = dist_t < 25
    s_proj = sm[idx_t[keep]]
    z_gt_proj = dem[st_i[keep], ct_i[keep]]

    s_seg = sm[i0:i1]
    z_mod = dem[ri[i0:i1], ci[i0:i1]]
    z_mod_min = dem_min[ri[i0:i1], ci[i0:i1]]
    z_gt_on_seg = np.full(len(s_seg), np.nan)
    order = np.argsort(s_proj)
    z_gt_on_seg = np.interp(s_seg, s_proj[order], z_gt_proj[order],
                            left=np.nan, right=np.nan)

    dz_m = np.gradient(gaussian_filter1d(z_mod, 2.0))
    rough_m = np.abs(np.diff(gaussian_filter1d(z_mod, 1.0))).mean()
    ok = np.isfinite(z_gt_on_seg)
    rough_g = (np.abs(np.diff(gaussian_filter1d(z_gt_on_seg[ok], 1.0))).mean()
               if ok.sum() > 5 else float("nan"))
    print(f"  模型轴：高差 {z_mod.max()-z_mod.min():.1f} m，"
          f"最大纵坡 {np.abs(dz_m).max()*100:.0f}%，起伏 {rough_m:.2f} m/m")
    print(f"  真值轴：高差 {np.nanmax(z_gt_on_seg)-np.nanmin(z_gt_on_seg):.1f} m，"
          f"起伏 {rough_g:.2f} m/m")

    # 对照轴（可选）：同一图上再画一条（例如加了沿程高差惩罚的版本）
    r2 = c2 = s2 = z2 = None
    if a.npz2:
        npz2 = a.npz2 if os.path.isabs(a.npz2) else os.path.join(RES, a.npz2)
        p2_ = order_path(np.load(npz2)["path"].astype(bool))
        if p2_ is not None:
            r2, c2, s2 = resample(p2_)
            r2i = np.clip(np.round(r2).astype(int), 0, H - 1)
            c2i = np.clip(np.round(c2).astype(int), 0, W - 1)
            z2 = dem[r2i, c2i]
            dv2 = d_gt[r2i, c2i]
            print(f"  对照轴 {os.path.basename(npz2)}：{len(s2)} m，高差 "
                  f"{z2.max()-z2.min():.1f} m，最大纵坡 "
                  f"{np.abs(np.gradient(gaussian_filter1d(z2, 2.0))).max()*100:.0f}%，"
                  f"到真值平均 {dv2.mean():.2f} m")

    # 段中点法向横断面
    k = (i0 + i1) // 2
    dr, dc = rm[k + 2] - rm[k - 2], cm[k + 2] - cm[k - 2]
    L = np.hypot(dr, dc) or 1.0
    nr, nc = -dc / L, dr / L
    TR = np.arange(-25, 26)
    rr = np.clip(np.round(rm[k] + nr * TR).astype(int), 0, H - 1)
    cc = np.clip(np.round(cm[k] + nc * TR).astype(int), 0, W - 1)
    zs = gaussian_filter1d(dem[rr, cc], 1.5)
    t_truth = float(np.mean(TR[gt_skel[rr, cc]])) if gt_skel[rr, cc].any() else np.nan
    # 两线之间（0 与 t_truth）的极值
    if np.isfinite(t_truth):
        lo_, hi_ = sorted((0.0, t_truth))
        sel = (TR > lo_ + 1) & (TR < hi_ - 1)
        z0 = np.interp(0, TR, zs)
        zt = np.interp(t_truth, TR, zs)
        if sel.any():
            zmid_lo, zmid_hi = zs[sel].min(), zs[sel].max()
            dip = min(z0, zt) - zmid_lo
            bump = zmid_hi - max(z0, zt)
            verdict = ("横跨冲沟（两线间下凹 %.1f m）" % dip if dip > GULLY else
                       "横跨山脊（两线间上凸 %.1f m）" % bump if bump > GULLY else
                       "同一坡面/台地（无沟脊，起伏 %.1f/%.1f m）" % (dip, bump))
        else:
            verdict = "两线几乎重合"
    else:
        verdict = "断面上未见真值骨架"
    print(f"  段中点法向偏移 {t_truth:+.1f} m → 判定：{verdict}")

    # ---- 出图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 4)

    pad = 40
    r0, r1 = int(max(rm[i0] - pad, 0)), int(min(rm[i1] + pad, H))
    c0, c1 = int(max(min(cm[i0], cm[i1]) - pad, 0)), int(min(max(cm[i0], cm[i1]) + pad, W))
    ext = [x0 + c0 * gsd, x0 + c1 * gsd, y1 - r1 * gsd, y1 - r0 * gsd]
    im = ax1.imshow(np.ma.masked_invalid(dem[r0:r1, c0:c1]), cmap="terrain", extent=ext)
    plt.colorbar(im, ax=ax1, fraction=0.04, label="高程 (m, dem_mean)")
    ax1.plot(x0 + (cm + 0.5) * gsd, y1 - (rm + 0.5) * gsd, "-", color="#00b7cc",
             lw=1.8, label="模型轴")
    ax1.plot(x0 + (ct + 0.5) * gsd, y1 - (rt + 0.5) * gsd, "-", color="#00cc44",
             lw=1.6, label="真值骨架")
    if r2 is not None:
        ax1.plot(x0 + (c2 + 0.5) * gsd, y1 - (r2 + 0.5) * gsd, "-",
                 color="#d400ff", lw=1.4, label="对照轴")
    ax1.plot(x0 + (cm[i0:i1] + 0.5) * gsd, y1 - (rm[i0:i1] + 0.5) * gsd, "-",
             color="#e3001b", lw=3.0, alpha=0.55, label="拉直段")
    ax1.set_title(f"{site}：拉直段 {sm[i0]:.0f}–{sm[i1]:.0f} m（平均偏离 {dev[i0:i1].mean():.1f} m）")
    ax1.legend(fontsize=8)
    ax1.axis("off")

    ax2.plot(s_seg, z_mod, "-", color="#00b7cc", lw=1.6, label="模型轴高程")
    ax2.plot(s_seg, z_gt_on_seg, "-", color="#00cc44", lw=1.6, label="真值轴高程")
    ax2.plot(s_seg, z_mod_min, ":", color="#888888", lw=1.0, label="模型轴 dem_min")
    if z2 is not None:
        ax2.plot(s2, z2, "-", color="#d400ff", lw=1.4, label="对照轴高程")
    ax2.set_xlabel("沿程距离 (m)")
    ax2.set_ylabel("高程 (m)")
    ax2.set_title("沿程高程对比")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    ax3.plot(TR, zs, "-", color="#8a6d3b", lw=1.6, label="断面高程")
    ax3.axvline(0, color="#00b7cc", lw=1.4, ls="--", label="模型轴")
    if np.isfinite(t_truth):
        ax3.axvline(t_truth, color="#00cc44", lw=1.4, ls="--", label="真值骨架")
    ax3.set_xlabel("法向距离 (m)")
    ax3.set_ylabel("高程 (m)")
    ax3.set_title(f"段中点横断面：{verdict}")
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8)

    fig.suptitle(f"{site}：模型轴拉直段的地形剖面（dem_mean）", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG, f"axis_dem_profile_{site}_{a.truth}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  写出 {out}")


if __name__ == "__main__":
    main()
