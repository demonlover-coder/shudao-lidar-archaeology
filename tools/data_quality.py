# -*- coding: utf-8 -*-
"""两站数据质量统计：点云 / 栅格 / 真值 / 成果四组指标。

全部现场计算（点云流式扫描，不读 SHP 几何，只读已栅格化的真值掩膜与统计 json），
输出：
    results/data_quality.json
    results/data_quality.csv
    figures/data_quality.png

用法：
    python tools/data_quality.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import p2_dem as P  # noqa: E402
from eval_new_shp import RES  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from p2_adaptive_gsd import FIG  # noqa: E402

REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")
SITES = ("杨家坝", "西坝村")
CH = 4_000_000


def pct(a, ps):
    return [float(np.percentile(a, p)) for p in ps]


def scan_points(site, x0, y1, gsd, H, W, valid, dem_min):
    """流式扫描点云：密度、回波、强度、AGL 分层。"""
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    ft = np.load(os.path.join(REAL, f"{site}_lasefeat.npy"), mmap_mode="r")
    o = np.load(os.path.join(REAL, f"{site}_origin.npy")).astype(np.float64)
    n = xyz.shape[0]
    n_in = 0
    single = first = last = 0
    agl_cnt = np.zeros(4, np.int64)          # ≤0.3 / ≤2 / 2~6.45 / >6.45
    agl_s, inten_s = [], []
    for s in range(0, n, CH):
        e = min(s + CH, n)
        p = np.asarray(xyz[s:e], dtype=np.float64) + o[None, :]
        f = np.asarray(ft[s:e])
        c = np.round((p[:, 0] - x0) / gsd).astype(np.int32)
        r = np.round((y1 - p[:, 1]) / gsd).astype(np.int32)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        rc, rr_ = np.clip(c, 0, W - 1), np.clip(r, 0, H - 1)
        ok &= valid[rr_, rc]
        n_in += int(ok.sum())
        rn, nr = f[:, 0], f[:, 1]
        single += int(((rn == 1) & (nr == 1)).sum())
        first += int((rn == 1).sum())
        last += int((rn == nr).sum())
        if ok.any():
            agl = p[ok, 2] - dem_min[rr_[ok], rc[ok]]
            agl_cnt += [int((agl <= 0.3).sum()), int(((agl > 0.3) & (agl <= 2)).sum()),
                        int(((agl > 2) & (agl <= 6.45)).sum()), int((agl > 6.45).sum())]
            if len(agl_s) < 3_000_000:
                agl_s.append(agl[::20].astype(np.float32))
                inten_s.append(f[ok, 2][::20].astype(np.float32))
    agl_s = np.concatenate(agl_s) if agl_s else np.zeros(1)
    inten_s = np.concatenate(inten_s) if inten_s else np.zeros(1)
    return dict(n=n, n_in_valid=n_in,
                single=np.float64(single) / n, first=np.float64(first) / n,
                last=np.float64(last) / n,
                agl_frac=[round(v / max(n_in, 1), 4) for v in agl_cnt],
                agl_p50=round(float(np.percentile(agl_s, 50)), 2),
                agl_p90=round(float(np.percentile(agl_s, 90)), 2),
                inten_p=[round(v, 1) for v in pct(inten_s, (5, 50, 95))])


def grid_stats(site, valid, dem_mean, dem_min):
    H, W = valid.shape
    area = int(valid.sum())
    v = dem_mean[valid]
    gy, gx = np.gradient(np.where(valid, dem_mean, np.nanmedian(v)), 1.0, 1.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))[valid]
    resid = (np.where(valid, dem_mean, 0) - gaussian_filter(
        np.where(valid, dem_mean, np.nanmedian(v)), 3.0))
    rough = np.abs(resid[valid])
    return dict(valid_px=area, valid_frac=round(area / (H * W), 4),
                void_frac=round(1 - area / (H * W), 4),
                valid_area_km2=round(area * 1.0 / 1e6, 4),
                elev=[round(float(v.min()), 1), round(float(v.max()), 1)],
                relief=round(float(v.max() - v.min()), 1),
                slope_med=round(float(np.median(slope)), 1),
                slope_p95=round(float(np.percentile(slope, 95)), 1),
                rough_p50=round(float(np.median(rough)), 2),
                rough_p90=round(float(np.percentile(rough, 90)), 2))


def truth_stats(site, x0, y1, gsd, H, W, valid):
    d = json.load(open(os.path.join(RES, f"diag_truth_core_{site}.json"), encoding="utf-8"))
    s = json.load(open(os.path.join(RES, f"diag_shp_truth_{site}.json"), encoding="utf-8"))
    meta = json.load(open(os.path.join(REAL, f"{site}_meta.json"), encoding="utf-8"))
    cg, rg, og = (s["agl"][k]["frac_le0p3"] for k in ("corridor2.5", "ring5_15", "outside15"))
    from fig_recognition import load_truth
    gt = load_truth(site, valid, x0, y1, gsd, H, W)     # 与评分/出图同一口径
    return dict(line_len_m=s["line_len_m"], truth_px=int(gt.sum()),
                width_med=d["width_median"], width_mean=d["width_mean"],
                width_p90=d["width_p90"], area_per_len=d["area_per_len_m"],
                road_pts_ratio=round(meta["road_ratio"], 4),
                road_pts_density=round(meta["road_density_pts_m2"], 1),
                ground_frac_corridor=cg, ground_frac_ring5_15=rg,
                ground_frac_outside15=og,
                old_vs_new_iou=s["shift_old_vs_new"]["iou"] if site == "杨家坝" else 1.0)


def result_stats(site):
    e = json.load(open(os.path.join(RES, f"eval_lc6_skel_{site}_skel.json"),
                       encoding="utf-8"))
    pf, sk = e["point_fair"], e["skel_f1"]["2"]
    return dict(P=pf["P"], R=pf["R"], IoU=pf["IoU"], F1=pf["F1"],
                skelF1=sk["F1"], bufIoU=e["buffer_iou_5m_band"]["1"])


def main():
    q = {}
    for site in SITES:
        x0, y1, wm, hm, gsd, d = P.dem_window(site)
        H, W = d["dem_mean"].shape
        valid = d["valid"] == 1
        dem_min = fill_nodata(d["dem_min"], valid)
        g = grid_stats(site, valid, d["dem_mean"], d["dem_min"])
        s = scan_points(site, x0, y1, gsd, H, W, valid, dem_min)
        s["density_valid"] = round(s["n_in_valid"] / g["valid_area_km2"] / 1e6, 1)
        pm = json.load(open(os.path.join(ROOT, "data", f"p2_{site}_points_meta.json"),
                            encoding="utf-8"))
        s["extent"] = [round(float(v), 1) for v in pm["extent_m"][:2]]
        q[site] = dict(points=s, grid=g,
                       truth=truth_stats(site, x0, y1, gsd, H, W, valid),
                       result=result_stats(site))
        print(f"[{site}] 完成")

    json.dump(q, open(os.path.join(RES, "data_quality.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(RES, "data_quality.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["组", "指标", "杨家坝", "西坝村", "单位"])
        for grp, rows in GROUPS:
            for key, label, unit in rows:
                w.writerow([grp, label,
                            q["杨家坝"][grp][key] if not isinstance(q["杨家坝"][grp][key], list)
                            else " / ".join(str(v) for v in q["杨家坝"][grp][key]),
                            q["西坝村"][grp][key] if not isinstance(q["西坝村"][grp][key], list)
                            else " / ".join(str(v) for v in q["西坝村"][grp][key]), unit])
    print("写出 results/data_quality.json / .csv")

    # ---------------- 表格 ----------------
    print()
    for grp, rows in GROUPS:
        print(f"【{grp}】")
        for key, label, unit in rows:
            vals = []
            for site in SITES:
                v = q[site][grp][key]
                vals.append(" / ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x)
                                       for x in v) if isinstance(v, list) else
                            (f"{v:g}" if isinstance(v, (int, float)) else str(v)))
            print(f"  {label:<24s}{vals[0]:>26s}{vals[1]:>26s}  {unit}")

    # ---------------- 图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    panels = [("点云密度（有效区）", "points", "density_valid", "pts/m²", (0, 420)),
              ("有效覆盖率", "grid", "valid_frac", "比例", (0, 1.05)),
              ("走廊内地面点占比", "truth", "ground_frac_corridor", "比例", (0, 0.32)),
              ("成果公平 IoU", "result", "IoU", "IoU", (0, 1.0))]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    cols = ["#1f4fd8", "#e3001b"]
    for ax, (title, grp, key, unit, ylim) in zip(axes, panels):
        vals = [q[s][grp][key] for s in SITES]
        bars = ax.bar(SITES, vals, color=cols, width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center",
                    va="bottom", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("两站数据质量关键指标", fontsize=13)
    out = os.path.join(FIG, "data_quality.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"写出 {out}")


GROUPS = [
    ("points", [
        ("n", "原始点数", "点"),
        ("extent", "点云范围 X×Y", "m"),
        ("density_valid", "有效区点密度", "pts/m²"),
        ("single", "单次回波占比", "比例"),
        ("first", "首回波占比", "比例"),
        ("last", "末回波占比", "比例"),
        ("agl_frac", "AGL ≤0.3 / ≤2 / 2–6.45 / >6.45", "比例"),
        ("agl_p50", "AGL 中位数", "m"),
        ("agl_p90", "AGL P90", "m"),
        ("inten_p", "强度 P5/P50/P95", "原始值"),
    ]),
    ("grid", [
        ("valid_px", "有效像元", "px"),
        ("valid_frac", "有效覆盖率", "比例"),
        ("void_frac", "空区率（测区外）", "比例"),
        ("valid_area_km2", "有效面积", "km²"),
        ("elev", "高程 min/max", "m"),
        ("relief", "高差", "m"),
        ("slope_med", "坡度中位数", "°"),
        ("slope_p95", "坡度 P95", "°"),
        ("rough_p50", "DEM 微起伏 P50", "m"),
        ("rough_p90", "DEM 微起伏 P90", "m"),
    ]),
    ("truth", [
        ("line_len_m", "走廊长度", "m"),
        ("truth_px", "真值像元（评分口径）", "px"),
        ("width_med", "走廊宽度中位数", "m"),
        ("width_mean", "走廊宽度均值", "m"),
        ("width_p90", "走廊宽度 P90", "m"),
        ("area_per_len", "单位长度面积", "m²/m"),
        ("road_pts_ratio", "真值点云占比", "比例"),
        ("road_pts_density", "走廊内点密度", "pts/m²"),
        ("ground_frac_corridor", "走廊内地面点占比", "比例"),
        ("ground_frac_outside15", "走廊外地面点占比", "比例"),
        ("old_vs_new_iou", "新旧 SHP 一致性 IoU", "IoU"),
    ]),
    ("result", [
        ("P", "精度 P", "比例"),
        ("R", "召回 R", "比例"),
        ("IoU", "公平 IoU", "IoU"),
        ("skelF1", "骨架 F1@2px", "F1"),
        ("bufIoU", "Buffer IoU@1px", "IoU"),
    ]),
]


if __name__ == "__main__":
    main()
