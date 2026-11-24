# -*- coding: utf-8 -*-
"""走廊引导滤波 vs 主流形态学地面滤波：蜀道点保留率 / 植被去除率对照。

问题背景
--------
主流地面滤波的目标是"保留地面点、去除非地面点"，在林地陡坡上会把蜀道一起删掉。
本脚本用**同一批 LAS 点**做逐点判定，量化这一损坏，并验证"走廊内放宽地面判据"
能否在保持激进去植被的同时把蜀道路面点保下来。

关键机制（对照组 morph_r*）
--------------------------
主流形态学滤波（grid-based morphological filter）用 1 m 格网参考面 gmorph 判地面：
`Z - gmorph <= 0.3 m` 才算地面。gmorph 由 `grey_opening` 再 `grey_closing` 得到
（`disk(r)` 结构元）。**开运算对窄于 2r 的凹陷会被邻域高值"填平"**：蜀道在陡坡上是
一条宽约 2–3 m、深 0.5–1.5 m 的窄槽，开运算把槽底抬到两侧坡面的高度，于是路面点的
`Z - gmorph` 远大于 0.3 m，被判成植被删掉。坡度越陡、槽越窄、r 越大，误删越严重——
这就是"主流滤波损坏蜀道"的几何机制，也正是本脚本要量化的东西。

方案（全部逐点判定，同一批点）
------------------------------
    global_agl03               AGL <= 0.3 m             最简单的"只留地面"
    morph_r1/r2/r3             Z - gmorph(r) <= 0.3 m  主流形态学滤波代理
    corridor_guided  (本文)    走廊内 AGL<=0.5；走廊外 Z-gmorph(r2)<=0.3
    corridor_guided_no_agl_cap  走廊内不设 AGL 上限，仅按 3D 走廊保留（消融）

走廊 = 概率图 P>=0.5 掩膜闭运算 3×3 → skeletonize 轴线 → 逐点 3D 距离 d3；d3 <= d3_site。
AGL 基准面为 1 m `dem_min`（valid 外最近邻填洞）。

指标（分母定义同时写进 JSON 的 definitions）
----------------------------------------------
    road_preservation      保留的 "蜀道路面点" / 全部蜀道路面点（真值内且 AGL<=0.5）
    veg_removal            1 - 保留植被点 / 全部植被点（AGL>=1.0，全场）
    veg_removal_truth      同上，但只统计真值掩膜内的植被点
    keep_ratio             保留点数 / 窗口内总点数
    road_fp                保留但不在真值掩膜内的点数（"保蜀道"的误保代价）
    corridor_veg_residual  走廊内(AGL>=1.0)植被点的保留比例（越小越好）

SHP 只用于最后算指标，不参与任何筛选决策。

用法：
    python tools/corridor_filter.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, label as cc_lab
from skimage.morphology import disk, skeletonize
# skimage 0.26 起 grey_opening/grey_closing 更名为 opening/closing（同一灰度形态学实现）
from skimage.morphology import closing as grey_closing
from skimage.morphology import opening as grey_opening

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import p2_dem as P  # noqa: E402
from eval_new_shp import LAS, RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from p2_adaptive_gsd import FIG  # noqa: E402

CH = 3_000_000
SS = 4
SITES = ("杨家坝", "西坝村")
CFG = {
    # truth=old  : 旧 SHP 栅格 results/mask_1m_{site}.npy
    # truth=new_main: 新 SHP 主板（poly_mask_hr → 1 m → 最大连通域）
    "杨家坝": {"prob": "cold_start_杨家坝_lc6.npz", "d3": 3.5, "truth": "old"},
    "西坝村": {"prob": "cold_start_西坝村_lc6.npz", "d3": 3.5, "truth": "new_main"},
}
SCHEMES = ("global_agl03", "morph_r1", "morph_r2", "morph_r3",
           "corridor_guided", "corridor_guided_no_agl_cap")
# 路面点按其所在像元的局部坡度分档，用于检验"决定保留率的是路面所在坡度、而非站点平均坡度"
SLOPE_BINS = (0.0, 10.0, 20.0, 30.0, 45.0, 90.0)
SLOPE_KEY = ("global_agl03", "morph_r2", "corridor_guided")   # 分档表只列三套代表方案

DEFINITIONS = {
    "n_total": "窗口内 LAS 点数（落在 1 m 网格内），keep_ratio 的分母",
    "road_point": "真值掩膜内且 AGL<=0.5 m 的点",
    "veg_point": "AGL>=1.0 m 的点（全场）",
    "veg_point_truth": "真值掩膜内且 AGL>=1.0 m 的点",
    "road_preservation": "保留的蜀道路面点 / 全部蜀道路面点",
    "veg_removal": "1 - 保留植被点 / 全部植被点（全场）",
    "veg_removal_truth": "同上，但只统计真值掩膜内的植被点",
    "keep_ratio": "保留点数 / 窗口内总点数",
    "road_fp": "保留但不在真值掩膜内的点数（绝对数）",
    "road_fp_rate": "road_fp / 保留点数",
    "corridor_veg_residual": "走廊内(d3<=d3_site)且 AGL>=1.0 的点被保留的比例",
    "slope_bin": "路面点所在 1 m 像元的局部坡度分档（0-10/10-20/20-30/30-45/45-90°）",
    "road_preservation_by_bin": "该坡度档内保留的路面点 / 该档全部路面点",
}


# ---------------------------------------------------------------- 先验/真值
def load_prior(site):
    """读 1 m 网格、概率图、骨架轴线，返回逐点判定需要的全部场。"""
    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem = fill_nodata(d["dem_min"], valid)          # AGL 基准面
    npz = os.path.join(RES, CFG[site]["prob"])
    prob = np.load(npz)["prob"].astype(np.float64)
    prob = np.where(valid, prob, 0.0)

    core = (prob >= 0.5) & valid
    ax = skeletonize(binary_closing(core, np.ones((3, 3), bool)))
    if not ax.any():
        raise RuntimeError(f"{site} 轴线为空")
    d2, idx = distance_transform_edt(~ax, return_indices=True)
    zax = dem[tuple(idx)]                            # 轴线像元的基准高程
    return {"site": site, "x0": x0, "y1": y1, "gsd": gsd, "H": H, "W": W,
            "valid": valid, "dem": dem, "dm": d["dem_min"], "prob": prob,
            "ax": ax, "d2": d2, "zax": zax, "d3_site": CFG[site]["d3"],
            "slope_deg": d["slope_deg"]}


def truth_mask(site, pr):
    """真值掩膜（仅用于评分）。杨家坝旧栅格；西坝村新 SHP 主板。"""
    x0, y1, gsd, H, W, valid = pr["x0"], pr["y1"], pr["gsd"], pr["H"], pr["W"], pr["valid"]
    if CFG[site]["truth"] == "old":
        gt = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
    else:
        _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        mh = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
        gt = (mh.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5) & valid
        lb, n = cc_lab(gt)
        if n > 1:
            sz = np.bincount(lb.ravel())
            sz[0] = 0
            gt = lb == sz.argmax()
    return gt


def gmorph_ref(site, dem):
    """1 m 形态学参考地面（r=1/2/3），按站点缓存 results/gmorph_{site}.npy。

    grey_opening(disk(r)) 会把窄于 2r 的凹陷填平（蜀道窄槽因此被"抹掉"），
    grey_closing(disk(r)) 再补平小坑。返回 (3,H,W) float32，索引 0/1/2 ↔ r=1/2/3。
    """
    cache = os.path.join(RES, f"gmorph_{site}.npy")
    if os.path.exists(cache):
        a = np.load(cache)
        if a.shape == (3,) + dem.shape:
            return a
    out = np.empty((3,) + dem.shape, np.float32)
    for i, r in enumerate((1, 2, 3)):
        o = grey_opening(dem, footprint=disk(r))
        out[i] = grey_closing(o, footprint=disk(r))
    np.save(cache, out)
    return out


# ---------------------------------------------------------------- 逐点流式
def stream_schemes(pr, gt, gmr):
    """流式读 LAS，逐点判定 6 套方案并累加计数。返回 (counts, n_points)。"""
    x0, y1, gsd, H, W = pr["x0"], pr["y1"], pr["gsd"], pr["H"], pr["W"]
    dem, d2, zax, d3s = pr["dem"], pr["d2"], pr["zax"], pr["d3_site"]
    g1, g2, g3 = gmr[0], gmr[1], gmr[2]
    keys = SCHEMES
    c = {k: {"keep": 0, "road": 0, "veg": 0, "veg_t": 0, "fp": 0, "cor_veg": 0}
         for k in keys}
    n_total = n_las = 0
    road_tot = veg_tot = veg_t_tot = cor_veg_tot = 0
    nb = len(SLOPE_BINS) - 1
    road_bin_tot = np.zeros(nb, np.int64)
    road_bin_keep = {k: np.zeros(nb, np.int64) for k in keys}

    import laspy
    t0 = time.time()
    with laspy.open(LAS[pr["site"]]) as f:
        for ci, pts in enumerate(f.chunk_iterator(CH), start=1):
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            Z = np.asarray(pts.z, np.float64)
            n_las += X.size
            col = np.floor((X - x0) / gsd).astype(np.int64)
            row = np.floor((y1 - Y) / gsd).astype(np.int64)
            ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
            if not ok.any():
                continue
            X, Y, Z = X[ok], Y[ok], Z[ok]
            row, col = row[ok], col[ok]

            agl = Z - dem[row, col]
            d3 = np.hypot(d2[row, col], Z - zax[row, col])
            tr = gt[row, col]
            veg = agl >= 1.0
            road = tr & (agl <= 0.5)
            cor = d3 <= d3s
            cor_veg = cor & veg
            sl = pr["slope_deg"][row, col]
            ib = np.digitize(sl, SLOPE_BINS) - 1
            rb = road & (ib >= 0) & (ib < nb)

            n_total += Z.size
            road_tot += int(road.sum())
            if rb.any():
                np.add.at(road_bin_tot, ib[rb], 1)
            veg_tot += int(veg.sum())
            veg_t_tot += int((veg & tr).sum())
            cor_veg_tot += int(cor_veg.sum())

            keeps = {
                "global_agl03": agl <= 0.3,
                "morph_r1": Z - g1[row, col] <= 0.3,
                "morph_r2": Z - g2[row, col] <= 0.3,
                "morph_r3": Z - g3[row, col] <= 0.3,
                "corridor_guided": np.where(cor, agl <= 0.5, Z - g2[row, col] <= 0.3),
                "corridor_guided_no_agl_cap": cor | (Z - g2[row, col] <= 0.3),
            }
            for k, keep in keeps.items():
                d = c[k]
                d["keep"] += int(keep.sum())
                d["road"] += int((keep & road).sum())
                d["veg"] += int((keep & veg).sum())
                d["veg_t"] += int((keep & veg & tr).sum())
                d["fp"] += int((keep & ~tr).sum())
                d["cor_veg"] += int((keep & cor_veg).sum())
                mb = rb & keep
                if mb.any():
                    np.add.at(road_bin_keep[k], ib[mb], 1)
            if ci % 5 == 0:
                print(f"    {pr['site']} 块 {ci}：{n_total:,} 点，{time.time()-t0:.0f}s",
                      flush=True)

    denom = {"n_total": n_total, "road_total": road_tot, "veg_total": veg_tot,
             "veg_truth_total": veg_t_tot, "corridor_veg_total": cor_veg_tot,
             "road_bin_total": road_bin_tot}
    return c, denom, n_las, road_bin_keep


def finish_metrics(c, denom, n_las, site, road_bin_keep=None):
    schemes = {}
    for k, d in c.items():
        road_p = d["road"] / max(denom["road_total"], 1)
        veg_r = 1.0 - d["veg"] / max(denom["veg_total"], 1)
        veg_rt = 1.0 - d["veg_t"] / max(denom["veg_truth_total"], 1)
        keep_r = d["keep"] / max(denom["n_total"], 1)
        schemes[k] = {
            "n_keep": d["keep"],
            "keep_ratio": round(keep_r, 4),
            "road_preservation": round(road_p, 4),
            "veg_removal": round(veg_r, 4),
            "veg_removal_truth": round(veg_rt, 4),
            "road_fp": d["fp"],
            "road_fp_rate": round(d["fp"] / max(d["keep"], 1), 4),
            "corridor_veg_residual": round(
                d["cor_veg"] / max(denom["corridor_veg_total"], 1), 4),
        }
    out = {
        "site": site,
        "n_las_points": n_las,
        "n_window_points": denom["n_total"],
        "n_road_points": denom["road_total"],
        "n_veg_points": denom["veg_total"],
        "n_veg_points_truth": denom["veg_truth_total"],
        "n_corridor_veg_points": denom["corridor_veg_total"],
        "d3_site_m": CFG[site]["d3"],
        "prob_file": CFG[site]["prob"],
        "truth_source": ("results/mask_1m_%s.npy" % site if CFG[site]["truth"] == "old"
                         else "SHP %s → poly_mask_hr → 1 m 最大连通域"
                              % SHP_NEW[site]["poly"]),
        "gmorph_cache": "results/gmorph_%s.npy (3,H,W) r=1,2,3" % site,
        "definitions": DEFINITIONS,
        "schemes": schemes,
    }
    if road_bin_keep is not None:
        tot = denom["road_bin_total"]
        out["slope_bins"] = {
            "edges_deg": list(SLOPE_BINS),
            "labels": [f"{SLOPE_BINS[i]:.0f}-{SLOPE_BINS[i+1]:.0f}°"
                       for i in range(len(SLOPE_BINS) - 1)],
            "road_points": [int(x) for x in tot],
            "road_preservation_by_bin": {
                k: [round(float(road_bin_keep[k][i]) / max(int(tot[i]), 1), 4)
                    for i in range(len(tot))]
                for k in SCHEMES},
        }
    return out


# ---------------------------------------------------------------- 出图
def make_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    short = ["AGL≤0.3", "morph r1", "morph r2", "morph r3",
             "走廊引导(本文)", "无AGL上限(消融)"]
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    for j, site in enumerate(SITES):
        ax = axes[0, j]
        r = results[site]
        x = np.arange(len(SCHEMES))
        w = 0.26
        for i, (key, nm) in enumerate([
                ("road_preservation", "蜀道保留率"),
                ("veg_removal", "植被去除率(全场)"),
                ("keep_ratio", "总保留率")]):
            ax.bar(x + (i - 1) * w, [r["schemes"][k][key] for k in SCHEMES],
                   w, label=nm, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=22, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.set_ylabel("比例", fontsize=9)
        ax.set_title(f"{site}：走廊引导 vs 主流形态学滤波（中位坡度 "
                     f"{r['slope_median_deg']:.1f}°）", fontsize=11.5)
        ax.legend(fontsize=8, loc="upper right")

    for j, (key, nm, ylab) in enumerate([
            ("veg_removal_truth", "真值走廊内植被去除率（越高越好）", "比例"),
            ("road_fp_rate", "误保率 road_fp / 保留点数", "比例")]):
        ax = axes[1, j]
        x = np.arange(len(SCHEMES))
        w = 0.36
        for i, site in enumerate(SITES):
            ax.bar(x + (i - 0.5) * w, [results[site]["schemes"][k][key] for k in SCHEMES],
                   w, label=site, color=("#d62728", "#1f77b4")[i])
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=22, ha="right", fontsize=8.5)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(nm, fontsize=11.5)
        ax.legend(fontsize=8)

    fig.suptitle("走廊引导滤波：在保护蜀道路面点的同时保持激进去植被（两站逐点统计）",
                 fontsize=14, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG, "corridor_filter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"写出 {out}")


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    results, rows = {}, []
    for site in SITES:
        t0 = time.time()
        print(f"\n[{site}] 读取网格/概率/轴线 ...", flush=True)
        pr = load_prior(site)
        gt = truth_mask(site, pr)
        print(f"  网格 {pr['H']}×{pr['W']}，轴线 {int(pr['ax'].sum())} px，"
              f"真值 {int(gt.sum()):,} px，d3={pr['d3_site']} m", flush=True)
        t1 = time.time()
        gmr = gmorph_ref(site, pr["dem"])
        print(f"  形态学参考面 gmorph r=1/2/3 就绪（{time.time()-t1:.1f}s）", flush=True)

        c, denom, n_las, rbk = stream_schemes(pr, gt, gmr)
        res = finish_metrics(c, denom, n_las, site, rbk)
        res["slope_median_deg"] = round(float(np.median(pr["slope_deg"][pr["valid"]])), 1)
        res["elapsed_s"] = round(time.time() - t0, 1)
        results[site] = res
        out = os.path.join(RES, f"corridor_filter_{site}.json")
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  写出 {out}（{res['elapsed_s']}s，LAS {n_las:,} 点，"
              f"窗口内 {denom['n_total']:,} 点）", flush=True)
        for k in SCHEMES:
            m = res["schemes"][k]
            rows.append({"site": site, "scheme": k, **m})

    csv_path = os.path.join(RES, "corridor_filter.csv")
    cols = ["site", "scheme", "n_keep", "keep_ratio", "road_preservation",
            "veg_removal", "veg_removal_truth", "road_fp", "road_fp_rate",
            "corridor_veg_residual"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in cols})
    print(f"\n写出 {csv_path}")

    # 路面点保留率按局部坡度分档（投稿材料 表 S3）
    bin_path = os.path.join(RES, "corridor_filter_slope_bins.csv")
    with open(bin_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["site", "slope_bin_deg", "n_road_points"]
                    + [f"{k}_road_preservation" for k in SLOPE_KEY])
        for site in SITES:
            sb = results[site]["slope_bins"]
            for i, lab in enumerate(sb["labels"]):
                wr.writerow([site, lab, sb["road_points"][i]]
                            + [sb["road_preservation_by_bin"][k][i] for k in SLOPE_KEY])
    print(f"写出 {bin_path}")

    make_figure(results)

    # markdown 汇总
    print("\n### 走廊引导滤波 vs 主流形态学滤波（逐点统计）\n")
    print("| 站点 | 方案 | 蜀道保留率 | 植被去除率(全场) | 植被去除率(真值内) "
          "| 总保留率 | road_fp | 走廊内植被残留 |")
    print("|---|---|---|---|---|---|---|---|")
    for site in SITES:
        for k in SCHEMES:
            m = results[site]["schemes"][k]
            print(f"| {site} | `{k}` | {m['road_preservation']:.4f} | "
                  f"{m['veg_removal']:.4f} | {m['veg_removal_truth']:.4f} | "
                  f"{m['keep_ratio']:.4f} | {m['road_fp']:,} | "
                  f"{m['corridor_veg_residual']:.4f} |")
    print("\n分母：")
    for site in SITES:
        r = results[site]
        print(f"- {site}：窗口内 {r['n_window_points']:,} 点，"
              f"蜀道路面点 {r['n_road_points']:,}，植被点 {r['n_veg_points']:,}"
              f"（真值内 {r['n_veg_points_truth']:,}），"
              f"走廊内植被点 {r['n_corridor_veg_points']:,}")

    print("\n### 路面点保留率按所在坡度分档（表 S3）\n")
    print("| 站点 | 坡度档 | 路面点数 | AGL≤0.3 | morph r2 | 走廊引导 |")
    print("|---|---|---|---|---|---|")
    for site in SITES:
        sb = results[site]["slope_bins"]
        for i, lab in enumerate(sb["labels"]):
            print(f"| {site} | {lab} | {sb['road_points'][i]:,} | "
                  + " | ".join(f"{sb['road_preservation_by_bin'][k][i]:.4f}"
                               for k in SLOPE_KEY) + " |")


if __name__ == "__main__":
    main()
