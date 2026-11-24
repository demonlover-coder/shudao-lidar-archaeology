# -*- coding: utf-8 -*-
"""λc 敏感性（杨家坝）：λc=1..10 的路径级 IoU 与 10 m 窗纵坡 P95。

与西坝村 `fig_lambda_c_sensitivity.py` 同版式、同 λc 档位（1..10），但口径按本研究
"路径级"定义**现场重算**：
    · 端点固定（results/cold_start_杨家坝_ep.json 的 CLI 起止点，不读 SHP）
    · 每个 λc 只重跑宏观寻路（openness + h1 方向先验 + 方向余弦惩罚），不重训模型
    · 路径膨胀 gate=3 px（欧氏距离 ≤ gate）→ 与旧版真值 results/mask_1m_杨家坝.npy
      求路径级 IoU
    · 纵坡 P95 用 diag_path_grade 同款算法现场计算（只读 DEM 与路径）
同时把西坝村**同口径路径级 IoU**（由已存 cold_start_西坝村_lc*.npz 的 path 现场
计算，gate=5 px）一并写入 JSON，便于两站合并出图；西坝村原有的点级公平 IoU
（results/eval_lc*_西坝村_skel.json）作为 reference 字段保留，不覆盖任何旧文件。

用法：
    python tools/lambda_c_sens_yang.py
"""
from __future__ import annotations

import glob
import json
import os
import re
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
LCS = list(range(1, 11))
STATIONS = {"杨家坝": dict(gate=3.0, ep_json="cold_start_杨家坝_ep.json"),
            "西坝村": dict(gate=5.0, ep_json="cold_start_西坝村_lc6.json")}


def path_iou(path, truth, valid, gate):
    """路径膨胀（欧氏距离 ≤ gate）后与真值掩膜的 IoU（只统计 valid 像元）。"""
    band = (distance_transform_edt(~path) <= gate) & valid
    inter = int((band & truth & valid).sum())
    union = int(((band | truth) & valid).sum())
    return inter / max(union, 1), int(band.sum())


def grade_p95(dem, path):
    """10 m 窗纵坡 P95（%，|z(i+10)−z(i)|/10 m）；与 diag_path_grade 同算法。"""
    pl = order_path(np.asarray(path, dtype=bool))
    if pl is None or len(pl) < 12:
        return float("nan")
    p = np.asarray(pl, np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    si = np.arange(0.0, s[-1], 1.0)
    rr = np.clip(np.round(np.interp(si, s, p[:, 0])).astype(int), 0, dem.shape[0] - 1)
    cc = np.clip(np.round(np.interp(si, s, p[:, 1])).astype(int), 0, dem.shape[1] - 1)
    zz = dem[rr, cc]
    g = np.abs(zz[10:] - zz[:-10]) / 10.0
    return float(np.percentile(g, 95) * 100)


def path_args(lam_c):
    """与 cold_start_transfer 默认一致，只改 lam_c。"""
    return Namespace(z_lambda=C.LAMBDA_Z, lam_c=float(lam_c), slope_lambda=0.0,
                     dir_lambda=1.5, turn_lambda=0.2, turn_min=1,
                     max_grade=0.0, grade_lag=5)


def xiba_npz(lc):
    for name in (f"cold_start_西坝村_lcs{lc}.npz", f"cold_start_西坝村_lc{lc}.npz"):
        p = os.path.join(RES, name)
        if os.path.exists(p):
            return p, name
    return None, None


def xiba_point_fair_ref():
    """西坝村已有点级公平 IoU（旧图口径），仅作 reference 保留。"""
    ref = {}
    for f in glob.glob(os.path.join(RES, "eval_lc*_西坝村_skel.json")):
        tag = os.path.basename(f)[len("eval_"):].split("_西坝村")[0].replace("_skel", "")
        m = re.fullmatch(r"lcs?(\d+(?:\.\d+)?)", tag)
        if m:
            ref[int(float(m.group(1)))] = json.load(
                open(f, encoding="utf-8"))["point_fair"]["IoU"]
    return {str(k): round(v, 4) for k, v in sorted(ref.items())}


def run_yang(log=print):
    site = "杨家坝"
    gate = STATIONS[site]["gate"]
    log(f"\n[{site}] 现场重算 λc=1..10 宏观路径（gate={gate:g}px）")
    dem_f, valid = C.load_dem(site, "mean")
    dem_min, _ = C.load_dem(site, "min")
    op, n_row, n_col, conf, minfo = C.macro_fields(dem_f, valid)
    epj = json.load(open(os.path.join(RES, STATIONS[site]["ep_json"]), encoding="utf-8"))
    ep0 = np.asarray(epj["endpoints"]["ep0"], dtype=np.float64)
    ep1 = np.asarray(epj["endpoints"]["ep1"], dtype=np.float64)
    ep = [np.asarray(P.snap_to_domain(p, valid), dtype=np.float64) for p in (ep0, ep1)]
    if site == "西坝村":
        # 与 eval_freeaxis --truth new --truth-main 同口径；旧栅格含"错误支线"，不再使用
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
        log(f"  真值（新 SHP 主板）：{int(truth.sum()):,} px；"
            f"端点 ({ep[0][0]:.0f},{ep[0][1]:.0f}) ↔ ({ep[1][0]:.0f},{ep[1][1]:.0f})")
    else:
        truth = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
        log(f"  真值（旧 SHP 栅格）mask_1m_{site}.npy：{int(truth.sum()):,} px；"
            f"端点 ({ep[0][0]:.0f},{ep[0][1]:.0f}) ↔ ({ep[1][0]:.0f},{ep[1][1]:.0f})")

    rows = []
    for lc in LCS:
        t0 = time.time()
        path, dist, _, _ = C.macro_path(op, valid, ep, n_row, n_col, conf,
                                        path_args(lc), log=log, dem=dem_f)
        iou, band_px = path_iou(path, truth, valid, gate)
        gr = grade_p95(dem_min, path)
        rows.append(dict(lam_c=lc, iou=round(iou, 4), grade_p95_pct=round(gr, 4),
                         path_px=int(path.sum()), band_px=band_px,
                         seconds=round(time.time() - t0, 1)))
        log(f"  λc={lc:>2d}  IoU={iou:.4f}  P95={gr:5.1f}%  "
            f"路径 {int(path.sum())}px / 带宽 {band_px}px")
    return dict(gate_px=gate, endpoints=dict(source="cold_start_杨家坝_ep.json",
                                            ep0=ep0.tolist(), ep1=ep1.tolist()),
                macro=minfo, rows=rows)


def run_xiba(log=print):
    """西坝村：直接读已存 path（同一 λc 网格），现场算同口径路径级 IoU 与纵坡 P95。"""
    site = "西坝村"
    gate = STATIONS[site]["gate"]
    log(f"\n[{site}] 读取已存 cold_start_西坝村_lc*.npz 的 path，按同一口径复算（gate={gate:g}px）")
    _, valid = C.load_dem(site, "mean")
    dem_min, _ = C.load_dem(site, "min")
    truth = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
    epj = json.load(open(os.path.join(RES, STATIONS[site]["ep_json"]), encoding="utf-8"))
    rows, missing = [], []
    for lc in LCS:
        npz, name = xiba_npz(lc)
        if npz is None:
            missing.append(lc)
            continue
        path = np.load(npz)["path"].astype(bool)
        iou, band_px = path_iou(path, truth, valid, gate)
        gr = grade_p95(dem_min, path)
        rows.append(dict(lam_c=lc, iou=round(iou, 4), grade_p95_pct=round(gr, 4),
                         path_px=int(path.sum()), band_px=band_px, source_npz=name))
        log(f"  λc={lc:>2d}  IoU={iou:.4f}  P95={gr:5.1f}%  "
            f"路径 {int(path.sum())}px（{name}）")
    if missing:
        log(f"  **缺失 λc={missing} 的 npz，已跳过（JSON 中记录）")
    return dict(gate_px=gate,
                endpoints=dict(source=STATIONS[site]["ep_json"],
                               ep0=epj["endpoints"]["ep0"], ep1=epj["endpoints"]["ep1"]),
                rows=rows, missing_lam_c=missing)


def plateau(rows, tol=0.01):
    """IoU ≥ max−tol 的连续 λc 区间（返回 (lo, hi, iou_min, iou_max)）。"""
    d = {r["lam_c"]: r["iou"] for r in rows}
    if not d:
        return None
    mx = max(d.values())
    good = sorted(l for l, v in d.items() if v >= mx - tol)
    if not good:
        return None
    return min(good), max(good), min(d[l] for l in good), mx


def make_figure(rows, out_png, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    xs = [r["lam_c"] for r in rows]
    iou = [r["iou"] for r in rows]
    gr = [r["grade_p95_pct"] for r in rows]
    pl = plateau(rows)
    lo, hi, pmin, pmax = pl if pl else (None, None, None, None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, iou, "o-", color="#1f4fd8", lw=2, ms=6,
            label="路径级 IoU（膨胀 gate=3px）")
    for x, y in zip(xs, iou):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="#1f4fd8")
    if lo is not None and lo < hi:
        ax.axvspan(lo, hi, color="#9ad0f5", alpha=0.25, label=f"平台区 {lo}–{hi}")
    ax.axvline(6, color="#e3001b", ls="--", lw=1, label="锁定值 λc=6")
    ax.set_xlabel("λc（方向余弦惩罚权重）")
    ax.set_ylabel("路径级 IoU", color="#1f4fd8")
    ax.tick_params(axis="y", labelcolor="#1f4fd8")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(xs, gr, "s--", color="#2ca02c", lw=1.4, ms=5, label="路径纵坡 P95")
    ax2.axhline(20, color="#2ca02c", ls=":", lw=1, alpha=0.7)
    ax2.set_ylabel("10 m 窗纵坡 P95 (%)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.margins(y=0.14)
    ax2.margins(y=0.14)
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right", framealpha=0.95)
    if lo is not None and lo < hi:
        ax.set_title(f"杨家坝 λc 敏感性：平台区 {lo}–{hi}（IoU {pmin:.3f}–{pmax:.3f}）",
                     fontsize=11)
    else:
        ax.set_title("杨家坝 λc 敏感性", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    log(f"写出 {out_png}")


def print_table(rows, gate, log=print):
    log(f"\n| λc | 路径级 IoU（gate={gate:g}px） | 纵坡 P95 (%) | 路径 px |")
    log("|---:|---:|---:|---:|")
    for r in rows:
        log(f"| {r['lam_c']} | {r['iou']:.4f} | {r['grade_p95_pct']:.4f} | {r['path_px']} |")


def main():
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    yang = run_yang()
    xiba = run_xiba()

    # 失效点 / 相变点：相邻 λc 的最大 IoU 落差
    d = {r["lam_c"]: r["iou"] for r in yang["rows"]}
    drop = max(((d[b] - d[a], a, b) for a, b in zip(sorted(d)[:-1], sorted(d)[1:])),
               key=lambda t: abs(t[0]))
    pl = plateau(yang["rows"])

    out = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metric": "path-level IoU：宏观路径膨胀（欧氏距离 ≤ gate_px）vs results/mask_1m_{site}.npy（只统计 valid 像元）",
        "grade_metric": "10 m 窗纵坡 |z(i+10)-z(i)|/10 m 的 P95（%）",
        "lam_c_locked": C.LAMBDA_C,
        "stations": {"杨家坝": yang, "西坝村": xiba},
        "yang_plateau": dict(lam_c=[pl[0], pl[1]], iou=[round(pl[2], 4), round(pl[3], 4)])
        if pl else None,
        "yang_largest_drop": dict(from_lam_c=drop[1], to_lam_c=drop[2],
                                 delta_iou=round(drop[0], 4)),
        "xiba_point_fair_reference": xiba_point_fair_ref(),
        "seconds": round(time.time() - t0, 1),
    }
    jp = os.path.join(RES, "lambda_c_sensitivity_杨家坝.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n写出 {jp}")
    make_figure(yang["rows"], os.path.join(FIG, "lambda_c_sensitivity_杨家坝.png"))
    print_table(yang["rows"], yang["gate_px"])
    print(f"\n平台区 λc={pl[0]}–{pl[1]}，IoU {pl[2]:.4f}–{pl[3]:.4f}；"
          f"最大落差在 λc={drop[1]}→{drop[2]}（ΔIoU {drop[0]:+.4f}）")
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
