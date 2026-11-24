# -*- coding: utf-8 -*-
"""端点扰动敏感性（杨家坝 + 西坝村）：路径级鲁棒性检验。

端点由实地给定（results/cold_start_{site}*_ep.json 的 CLI 坐标），本脚本**不读 SHP
决定任何东西**——SHP 派生的 mask_1m_{site}.npy 只在最后一步算 IoU。

每站 5 个变体：
    1. base    ：给定端点
    2. out5    ：两端沿首/末段切线方向各外移 5 m
    3. out10   ：两端各外移 10 m
    4. start10 ：仅起点外移 10 m
    5. perp5   ：两端各沿路径法向偏移 5 m（同侧）

每个变体重跑宏观寻路（同一 λc=6、gate 按站点），路径膨胀后与 mask_1m 求**路径级
IoU**（不重训模型，明确是路径级口径）。外移后越出有效区的端点用
path_extraction.snap_to_domain 吸附。

输出：
    results/endpoint_sensitivity.json
    results/endpoint_sensitivity.csv
    figures/endpoint_sensitivity.png（两站分组柱状图）

用法：
    python tools/endpoint_sensitivity.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from argparse import Namespace

import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cold_start_transfer as C          # noqa: E402
import path_extraction as P              # noqa: E402
from fig_pred_vs_truth import order_path  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
VARIANTS = ("base", "out5", "out10", "start10", "perp5")
STATIONS = {"杨家坝": dict(gate=3.0, ep_json="cold_start_杨家坝_ep.json"),
            "西坝村": dict(gate=5.0, ep_json="cold_start_西坝村_lc6.json")}


def path_iou(path, truth, valid, gate):
    band = (distance_transform_edt(~path) <= gate) & valid
    inter = int((band & truth & valid).sum())
    union = int(((band | truth) & valid).sum())
    return inter / max(union, 1), int(band.sum())


def orient_and_tangents(path, ep0, ep1, k=10):
    """把有机路径排成 ep0→ep1，返回 (起点切向, 末点切向)，均指向行进方向。"""
    p = order_path(np.asarray(path, dtype=bool))
    if p is None or len(p) < 4:
        raise RuntimeError("基准路径过短，无法取切线")
    if np.linalg.norm(p[-1] - ep0) < np.linalg.norm(p[0] - ep0):
        p = p[::-1]
    kk = min(k, len(p) - 1)
    t0 = p[kk] - p[0]
    t1 = p[-1] - p[-1 - kk]
    t0 = t0 / max(np.linalg.norm(t0), 1e-9)
    t1 = t1 / max(np.linalg.norm(t1), 1e-9)
    return t0, t1


def variant_endpoints(ep0, ep1, t0, t1, gsd):
    """按行进方向构造 5 个变体端点（像素坐标）。"""
    step = 1.0 / gsd                       # 米 → 像素
    n0 = np.array([t0[1], -t0[0]])         # 左法向（同侧）
    n1 = np.array([t1[1], -t1[0]])
    return {
        "base": (ep0.copy(), ep1.copy()),
        "out5": (ep0 - 5.0 * step * t0, ep1 + 5.0 * step * t1),
        "out10": (ep0 - 10.0 * step * t0, ep1 + 10.0 * step * t1),
        "start10": (ep0 - 10.0 * step * t0, ep1.copy()),
        "perp5": (ep0 + 5.0 * step * n0, ep1 + 5.0 * step * n1),
    }


def run_site(site, log=print):
    gate = STATIONS[site]["gate"]
    log(f"\n[{site}] 端点扰动（λc={C.LAMBDA_C:g}，gate={gate:g}px，仅路径级）")
    dem_f, valid = C.load_dem(site, "mean")
    op, n_row, n_col, conf, _ = C.macro_fields(dem_f, valid)
    meta = json.load(open(os.path.join(ROOT, "data", f"meta_{site}.json"), encoding="utf-8"))
    gsd = float(meta.get("gsd_m", 1.0))
    epj = json.load(open(os.path.join(RES, STATIONS[site]["ep_json"]), encoding="utf-8"))
    ep0 = np.asarray(epj["endpoints"]["ep0"], dtype=np.float64)
    ep1 = np.asarray(epj["endpoints"]["ep1"], dtype=np.float64)
    ep0 = np.asarray(P.snap_to_domain(ep0, valid), dtype=np.float64)
    ep1 = np.asarray(P.snap_to_domain(ep1, valid), dtype=np.float64)
    if site == "西坝村":
        # 与 eval_freeaxis --truth new --truth-main 完全同口径：新 SHP → 1 m → 最大连通域
        import p2_dem as PD
        from eval_new_shp import SHP_BASE, SHP_NEW, read_shp_any
        from eval_line_iou import poly_mask_hr
        from scipy.ndimage import label as _lab
        x0, y1, _wm, _hm, gsd_s, dd = PD.dem_window(site)
        H, W = dd["dem_mean"].shape
        _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        mh = poly_mask_hr(polys, x0, y1, gsd_s, H, W, 4)
        truth = (mh.reshape(H, 4, W, 4).mean(axis=(1, 3)) >= 0.5) & valid
        lb, ncc = _lab(truth)
        if ncc > 1:
            sz = np.bincount(lb.ravel())
            sz[0] = 0
            truth = lb == sz.argmax()
        log(f"  真值=新 SHP 主板：{int(truth.sum()):,} px（只用于最后算 IoU）")
    else:
        truth = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
        log(f"  真值 mask_1m_{site}.npy：{int(truth.sum()):,} px（只用于最后算 IoU）")
    log(f"  给定端点 ({ep0[0]:.0f},{ep0[1]:.0f}) ↔ ({ep1[0]:.0f},{ep1[1]:.0f})，gsd={gsd:g} m")

    a = Namespace(z_lambda=C.LAMBDA_Z, lam_c=C.LAMBDA_C, slope_lambda=0.0,
                  dir_lambda=1.5, turn_lambda=0.2, turn_min=1,
                  max_grade=0.0, grade_lag=5)

    # 基准路径 → 切线
    t0b = time.time()
    path0, _, _, _ = C.macro_path(op, valid, [ep0, ep1], n_row, n_col, conf, a,
                                  log=log, dem=dem_f)
    t0, t1 = orient_and_tangents(path0, ep0, ep1)
    log(f"  基准路径 {int(path0.sum())}px（{time.time() - t0b:.0f}s）；"
        f"起点切向 ({t0[0]:+.2f},{t0[1]:+.2f})，末点切向 ({t1[0]:+.2f},{t1[1]:+.2f})")

    epps = variant_endpoints(ep0, ep1, t0, t1, gsd)
    out = {}
    for name in VARIANTS:
        e0, e1 = epps[name]
        s0 = np.asarray(P.snap_to_domain(e0, valid), dtype=np.float64)
        s1 = np.asarray(P.snap_to_domain(e1, valid), dtype=np.float64)
        snapped = ",".join(k for k, raw, sn in (("ep0", e0, s0), ("ep1", e1, s1))
                           if not np.allclose(raw, sn))
        rec = dict(ep0=[round(float(s0[0]), 2), round(float(s0[1]), 2)],
                   ep1=[round(float(s1[0]), 2), round(float(s1[1]), 2)],
                   snapped=snapped or "none")
        try:
            t = time.time()
            path, _, _, _ = C.macro_path(op, valid, [s0, s1], n_row, n_col, conf, a,
                                         log=lambda *_: None, dem=dem_f)
            iou, band_px = path_iou(path, truth, valid, gate)
            rec.update(iou=round(iou, 4), path_px=int(path.sum()), band_px=band_px,
                       seconds=round(time.time() - t, 1), error=None)
            log(f"  {name:<7s} 端点 ({s0[0]:.0f},{s0[1]:.0f})↔({s1[0]:.0f},{s1[1]:.0f})"
                f"  吸附={rec['snapped']:<7s} IoU={iou:.4f} 路径 {int(path.sum())}px")
        except Exception as exc:                      # noqa: BLE001
            rec.update(iou=None, path_px=0, band_px=0, error=str(exc), seconds=0)
            log(f"  {name:<7s} **失败**：{exc}（端点 "
                f"({s0[0]:.0f},{s0[1]:.0f})↔({s1[0]:.0f},{s1[1]:.0f})）")
        out[name] = rec

    return dict(gate_px=gate, gsd_m=gsd, lam_c=C.LAMBDA_C,
                endpoints_source=STATIONS[site]["ep_json"],
                endpoints_base=dict(ep0=ep0.tolist(), ep1=ep1.tolist()),
                tangents=dict(start=t0.tolist(), end=t1.tolist()),
                variants=out)


def make_figure(stations, out_png, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    x = np.arange(len(VARIANTS))
    w = 0.36
    colors = {"杨家坝": "#1f4fd8", "西坝村": "#e3001b"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (site, st) in enumerate(stations.items()):
        vals = [st["variants"][v]["iou"] if st["variants"][v]["iou"] is not None else 0.0
                for v in VARIANTS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=f"{site}（gate={st['gate_px']:g}px）",
                      color=colors.get(site))
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS)
    ax.set_xlabel("端点扰动变体（base / 外移 5 m / 外移 10 m / 仅起点 10 m / 法向 5 m）")
    ax.set_ylabel("路径级 IoU（膨胀 gate px vs mask_1m）")
    ax.set_title(f"端点扰动敏感性：路径级 IoU（λc={C.LAMBDA_C:g}，不重训模型）", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    log(f"写出 {out_png}")


def print_tables(data, log=print):
    log("\n### 端点扰动（两站，路径级 IoU）")
    log("| 站点 | gate | " + " | ".join(VARIANTS) + " |")
    log("|---|---:|" + "---:|" * len(VARIANTS))
    for site, st in data["stations"].items():
        cells = []
        for v in VARIANTS:
            iou = st["variants"][v]["iou"]
            cells.append("失败" if iou is None else f"{iou:.4f}")
        log(f"| {site} | {st['gate_px']:g} | " + " | ".join(cells) + " |")


def main():
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    stations = {}
    for site in STATIONS:
        stations[site] = run_site(site)
    data = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                note="端点扰动鲁棒性检验；路径级 IoU = 宏观路径膨胀 gate px vs "
                     "results/mask_1m_{site}.npy；不重训模型；SHP 仅用于最后评分",
                lambda_c=C.LAMBDA_C, variants=list(VARIANTS), stations=stations,
                seconds=round(time.time() - t0, 1))
    jp = os.path.join(RES, "endpoint_sensitivity.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"\n写出 {jp}")

    cp = os.path.join(RES, "endpoint_sensitivity.csv")
    with open(cp, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["site", "variant", "gate_px", "ep0_r", "ep0_c", "ep1_r", "ep1_c",
                     "snapped", "iou", "path_px"])
        for site, st in stations.items():
            for v in VARIANTS:
                r = st["variants"][v]
                wr.writerow([site, v, st["gate_px"], r["ep0"][0], r["ep0"][1],
                             r["ep1"][0], r["ep1"][1], r["snapped"],
                             "" if r["iou"] is None else f"{r['iou']:.4f}", r["path_px"]])
    print(f"写出 {cp}")

    make_figure(stations, os.path.join(FIG, "endpoint_sensitivity.png"))
    print_tables(data)

    # 若任务 1 已产出 λc 表，一并打印，控制台最后给出两张 markdown 表
    lp = os.path.join(RES, "lambda_c_sensitivity_杨家坝.json")
    if os.path.exists(lp):
        lj = json.load(open(lp, encoding="utf-8"))
        rows = lj["stations"]["杨家坝"]["rows"]
        print(f"\n### 杨家坝 λc 敏感性（路径级 IoU，gate="
              f"{lj['stations']['杨家坝']['gate_px']:g}px）")
        print("| λc | IoU | 纵坡 P95 (%) | 路径 px |")
        print("|---:|---:|---:|---:|")
        for r in rows:
            print(f"| {r['lam_c']} | {r['iou']:.4f} | {r['grade_p95_pct']:.4f} | {r['path_px']} |")
    print(f"\n总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
