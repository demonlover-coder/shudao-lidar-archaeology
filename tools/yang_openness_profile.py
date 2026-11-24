# -*- coding: utf-8 -*-
"""openness 沿"真值骨架法线"的剖面：骨架处是不是 openness 的局部最小？

用途
----
杨家坝 openness 的路径偏差诊断是"**整条均匀偏 ~15 px**"（results/path_offset_diag.json：
P90/P50 = 1.4、偏段占比 0%、四象限极差 1.6×）。两种可能：
  (a) openness 在杨站真值骨架附近**本来就不是局部最小** → 代价图本身不可分 → 修算法没用
  (b) openness 在骨架处确是局部最小，但 Dijkstra 被端点对/连通性带偏 → 算法问题

做法
----
对每个真值骨架像元估计局部切向（邻域 PCA），取法向 n；沿 ±n 采样 openness，
距离 d = 0..DMAX px，统计各 d 的均值 → **法向剖面**。
另给"各向同性同心环"剖面（round(距离变换)==d 的均值）作稳健对照。

判据
----
  d=0（骨架）处 openness 最低            → (b) 算法问题 → 与西站一样做多路径
  最小值出现在 d≥8~15（骨架外更低）       → (a) 代价图不可分 → 换信息源（DOM RGB）
  其余                                    → 中间，需看图

对照
----
西坝村 openness 像素级有效 AUC 0.845（强）、杨家坝 0.591（弱）→ 两站互为对照，
预期西坝村呈 (b)、杨家坝呈 (a)。

输出
----
  results/openness_profile_{site}.json / .log
  figures/openness_profile_{site}.png

用法
----
    python tools/yang_openness_profile.py --sites 杨家坝 西坝村
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
SITES = ("杨家坝", "西坝村")
DMAX = 30


def openness_grid(site):
    """openness = feature_matrix_{site}.npy 的 openness 列（已填 nodata），摊回 [H,W]（valid 外 NaN）。"""
    meta = json.load(open(os.path.join(DATA, f"feature_matrix_{site}.meta.json"),
                          encoding="utf-8"))
    k = meta["factors"].index("openness")
    X = np.load(os.path.join(DATA, f"feature_matrix_{site}.npy"))
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    g = np.full(valid.size, np.nan, dtype=np.float64)
    g[np.flatnonzero(valid.ravel())] = X[:, k]
    return g.reshape(valid.shape), valid


def local_normals(pts, r=5.0):
    """对骨架点集做邻域 PCA，返回每个点的法向单位向量（切向的垂直方向）。"""
    P = pts.astype(np.float64)
    N = len(P)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    normals = np.zeros((N, 2), dtype=np.float64)
    for i in range(N):
        nb = P[d2[i] <= r * r]
        if len(nb) < 3:
            normals[i] = (0.0, 1.0)
            continue
        c = nb - nb.mean(0)
        cov = c.T @ c
        w, v = np.linalg.eigh(cov)
        t = v[:, int(np.argmax(w))]                 # 切向 = 最大特征向量
        normals[i] = (-t[1], t[0])                  # 法向 = 切向旋转 90°
    return normals


def profile(site, log):
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    g, valid = openness_grid(site)
    gt_skel = skeletonize(label)
    ys, xs = np.where(gt_skel)
    pts = np.stack([ys, xs], axis=1)
    if len(pts) < 20:
        log(f"[{site}] 骨架点太少，跳过")
        return None

    # 走廊局部半宽（骨架到最近背景的距离）→ 给"15px 落在哪"提供参照
    half = distance_transform_edt(label)[gt_skel]
    hw = float(np.median(half))

    # 法向剖面：d = 0..DMAX，两侧各采样一次
    Nr = local_normals(pts, r=5.0)
    H, W = label.shape
    cnt = np.zeros(DMAX + 1)
    acc = np.zeros(DMAX + 1)
    acc_p = np.zeros(DMAX + 1); cnt_p = np.zeros(DMAX + 1)   # 加号侧
    acc_m = np.zeros(DMAX + 1); cnt_m = np.zeros(DMAX + 1)   # 减号侧
    for d in range(DMAX + 1):
        for sgn in (1.0, -1.0):
            q = pts + sgn * d * Nr
            yy = np.round(q[:, 0]).astype(int)
            xx = np.round(q[:, 1]).astype(int)
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            ok &= valid[yy.clip(0, H - 1), xx.clip(0, W - 1)]
            vals = g[yy[ok], xx[ok]]
            ok2 = np.isfinite(vals)
            v = vals[ok2]
            acc[d] += v.sum(); cnt[d] += v.size
            if sgn > 0:
                acc_p[d] += v.sum(); cnt_p[d] += v.size
            else:
                acc_m[d] += v.sum(); cnt_m[d] += v.size

    prof = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    prof_p = np.where(cnt_p > 0, acc_p / np.maximum(cnt_p, 1), np.nan)
    prof_m = np.where(cnt_m > 0, acc_m / np.maximum(cnt_m, 1), np.nan)

    # 各向同性同心环剖面（对照）
    dt_skel = distance_transform_edt(~gt_skel)
    iso = np.full(DMAX + 1, np.nan)
    for d in range(DMAX + 1):
        m = (np.round(dt_skel) == d) & valid & np.isfinite(g)
        if int(m.sum()) > 0:
            iso[d] = float(g[m].mean())

    # 判据：只看采样数足够的 d
    okd = cnt >= max(20, 0.1 * len(pts))
    dd = np.arange(DMAX + 1)[okd]
    pv = prof[okd]
    dmin = int(dd[int(np.nanargmin(pv))])
    p0 = float(prof[0]) if cnt[0] > 0 else float("nan")
    pmin = float(np.nanmin(pv))
    # 判据：骨架(d=0)是不是 openness 的局部最小。
    # 只在与骨架距离**超出走廊半宽**（d ≥ D_FAR）处仍更深，才算真正的"谷底偏移"(a)；
    # 走廊内部(d≈半宽内)的平坦谷底不算（那是正常展宽）。分 +/− 两侧，任一更深即判 (a)。
    SIDE_HI = 20
    D_FAR = int(max(5, np.ceil(hw) + 3))
    seg_far = slice(D_FAR, SIDE_HI + 1)
    pv_fp, pv_fm = prof_p[seg_far], prof_m[seg_far]
    ok_p, ok_m = np.isfinite(pv_fp).any(), np.isfinite(pv_fm).any()
    min_p = float(np.nanmin(pv_fp)) if ok_p else float("nan")
    min_m = float(np.nanmin(pv_fm)) if ok_m else float("nan")
    dmin_p = (int(np.nanargmin(pv_fp)) + D_FAR) if ok_p else -1
    dmin_m = (int(np.nanargmin(pv_fm)) + D_FAR) if ok_m else -1
    drop_p, drop_m = p0 - min_p, p0 - min_m       # >0 表示该侧在走廊外更深
    drop = float(np.nanmax([drop_p, drop_m]))
    if not np.isfinite(drop):
        vd = "样本不足，无法判定"
    elif drop <= 1.5:
        vd = ("(b) 骨架处即 openness 局部最小（走廊外两侧剖面均未更深）"
              " → 算法问题 → 与西站一样做多路径")
    else:
        dstar_side = dmin_p if drop_p >= drop_m else dmin_m
        side = "+" if drop_p >= drop_m else "−"
        vd = (f"(a) {side} 侧走廊外更深（最深 {drop:.1f} @d≈{dstar_side}px，"
              f"已超半宽 {hw:.1f}px）→ 代价图与真值不对齐 → 换信息源（DOM RGB）")

    log(f"\n[{site}]  骨架 {len(pts)}px  走廊局部半宽(中位) {hw:.1f}px")
    log(f"  法向剖面 openness（d=0..{DMAX}，两侧合并）：")
    log("    d:  " + " ".join(f"{d:>5d}" for d in (0, 5, 10, 15, 20, 25, 30)))
    log("    val:" + " ".join(f"{prof[d]:>5.1f}" if np.isfinite(prof[d]) else "  nan" for d in (0, 5, 10, 15, 20, 25, 30)))
    log("    加号侧:" + " ".join(f"{prof_p[d]:>5.1f}" if np.isfinite(prof_p[d]) else "  nan" for d in (0, 5, 10, 15, 20, 25, 30)))
    log("    减号侧:" + " ".join(f"{prof_m[d]:>5.1f}" if np.isfinite(prof_m[d]) else "  nan" for d in (0, 5, 10, 15, 20, 25, 30)))
    log("    同心环:" + " ".join(f"{iso[d]:>5.1f}" if np.isfinite(iso[d]) else "  nan" for d in (0, 5, 10, 15, 20, 25, 30)))
    log(f"  走廊外分侧最低（d∈[{D_FAR},{SIDE_HI}]）：+侧 {min_p:.1f}(@d={dmin_p})"
        f"   −侧 {min_m:.1f}(@d={dmin_m})   → 较深一侧低于骨架 {drop:+.1f}")
    log(f"  骨架 d=0: {p0:.1f}  合并剖面最小 {pmin:.1f}(@d={dmin})  差 {p0 - pmin:+.1f}"
        f"  → {vd}")

    return dict(site=site, n_skel=int(len(pts)), half_width_med=hw, dmax=DMAX,
                profile=[None if not np.isfinite(v) else float(v) for v in prof],
                profile_plus=[None if not np.isfinite(v) else float(v) for v in prof_p],
                profile_minus=[None if not np.isfinite(v) else float(v) for v in prof_m],
                profile_iso=[None if not np.isfinite(v) else float(v) for v in iso],
                d0=p0, d_min=pmin, d_star=dmin, delta=p0 - pmin, verdict=vd,
                side_min_p=min_p, side_min_m=min_m, side_min_d=(dmin_p, dmin_m),
                d_far=D_FAR, side_hi=SIDE_HI,
                drop_below_skel=drop, drop_plus=drop_p, drop_minus=drop_m,
                normal_plus=[None if not np.isfinite(v) else float(v) for v in prof_p],
                normal_minus=[None if not np.isfinite(v) else float(v) for v in prof_m],
                count=[int(c) for c in cnt])


def figure(row, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    d = np.arange(DMAX + 1)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=140)
    ax.plot(d, row["profile"], "-o", ms=3, color="k", label="法向剖面（两侧合并）")
    ax.plot(d, row["profile_plus"], "--", color="#d62728", label="法向 +侧")
    ax.plot(d, row["profile_minus"], "--", color="#1f77b4", label="法向 −侧")
    ax.plot(d, row["profile_iso"], ":", color="0.5", label="同心环（对照）")
    ax.axvline(row["d_star"], color="g", lw=1, ls="-.", label=f"剖面最小 d*={row['d_star']}px")
    ax.axvline(row["half_width_med"], color="m", lw=1, ls="-.", label=f"走廊半宽中位 {row['half_width_med']:.1f}px")
    ax.set_xlabel("距真值骨架的距离 d（px = m）")
    ax.set_ylabel("openness 均值")
    dr = row["drop_below_skel"]
    short = "(a) 代价图不对齐" if dr > 1.0 else "(b) 算法问题"
    ax.set_title(f"{row['site']}：openness 沿骨架法向剖面\n"
                 f"d=0 = {row['d0']:.1f}；较深一侧低于骨架 {dr:+.1f}"
                 f"（+{row['drop_plus']:+.1f} / −{row['drop_minus']:+.1f}）  →  {short}",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    os.makedirs(FIG, exist_ok=True)
    fp = os.path.join(FIG, f"openness_profile_{row['site']}.png")
    fig.tight_layout(); fig.savefig(fp); plt.close(fig)
    log(f"  图: {os.path.relpath(fp, ROOT)}")
    return fp


def main():
    ap = argparse.ArgumentParser(description="openness 沿真值骨架法向的剖面诊断")
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--fig", action="store_true", default=True)
    ap.add_argument("--no-fig", dest="fig", action="store_false")
    a = ap.parse_args()

    lf = open(os.path.join(RES, "yang_openness_profile.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()

    log("openness 沿真值骨架法向的剖面（判据：骨架处最低=(b)算法问题；骨架外更低=(a)代价图不可分）")
    rows = []
    for site in a.sites:
        r = profile(site, log)
        if r is None:
            continue
        rows.append(r)
        if a.fig:
            r["figure"] = figure(r, log)

    jp = os.path.join(RES, "openness_profile.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    log(f"\n保存: {jp}")
    lf.close()


if __name__ == "__main__":
    main()
