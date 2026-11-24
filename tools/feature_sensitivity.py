# -*- coding: utf-8 -*-
"""特征参数敏感性（路径级，不重训）：一次只改一个宏观特征参数。

只改宏观场/门控参数，其余保持 `cold_start_*_lc6` 的现行设置：
    cost = P5~P95 归一化 openness + 0.05
    方向先验 λdir=1.5、转角 λturn=0.2、方向余弦 λc=6.0（替代 λz）
    端点 = 现行 lc6 路径（npz）的骨架直径两端，固定不变

配置（单因子变化）：
    base    openness R=30，h1 门控 P60   ← 现行
    R20     openness R=20
    R40     openness R=40
    gate50  h1 门控 P50
    gate70  h1 门控 P70

评价（路径级）：
    宏观路径 → 膨胀 gate（杨家坝 3 px / 西坝村 5 px）→ 1 m 掩膜 IoU（对真值）
    10 m 窗纵坡 P95（沿路径 1 m 重采样，|z[i+10]−z[i]|/10 的 P95，单位 %）

**这是路径级敏感性，不是最终模型指标**：不重训、不改网络，只回答"宏观特征参数
是否决定路径位置"。SHP 只用于最后评分。

用法：
    python tools/feature_sensitivity.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_lab
from skimage.morphology import disk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import cold_start_transfer as C  # noqa: E402
import p2_dem as P  # noqa: E402
from eval_new_shp import RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402
from fig_pred_vs_truth import order_path  # noqa: E402
from path_extraction import find_endpoints, merge_and_pick_pair  # noqa: E402

SITES = ("杨家坝", "西坝村")
GATE = {"杨家坝": 3, "西坝村": 5}
SS = 4
CONFIGS = [
    ("base",   dict(R=30, gate_pct=60.0, note="现行：openness R=30，h1 门控 P60")),
    ("R20",    dict(R=20, gate_pct=60.0, note="只改 openness 半径 R=20")),
    ("R40",    dict(R=40, gate_pct=60.0, note="只改 openness 半径 R=40")),
    ("gate50", dict(R=30, gate_pct=50.0, note="只改 h1 门控 P50")),
    ("gate70", dict(R=30, gate_pct=70.0, note="只改 h1 门控 P70")),
]
PATH_ARGS = dict(dir_lambda=1.5, turn_lambda=0.2, turn_min=1, slope_lambda=0.0,
                 max_grade=0.0, grade_lag=5, lam_c=6.0, z_lambda=2.0)


def load_truth(site, valid, x0, y1, gsd, H, W):
    """真值（仅评分）：杨家坝旧栅格；西坝村新 SHP 主板。"""
    if site == "杨家坝":
        gt = np.load(os.path.join(RES, "mask_1m_杨家坝.npy")).astype(bool) & valid
    else:
        _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        hr = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
        gt = (hr.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5) & valid
    lb, n = cc_lab(gt)
    if n > 1:
        sz = np.bincount(lb.ravel())
        sz[0] = 0
        gt = lb == sz.argmax()
    return gt


def iou(mask, gt):
    u = int((mask | gt).sum())
    return float((mask & gt).sum() / u) if u else 0.0


def grade_p95(path, dem, H, W, win=10):
    """沿路径 1 m 重采样后的 10 m 窗纵坡 P95（%）。"""
    pl = order_path(path)
    if pl is None:
        return float("nan")
    p = np.asarray(pl, np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    si = np.arange(0.0, s[-1], 1.0)
    rr = np.interp(si, s, p[:, 0])
    cc = np.interp(si, s, p[:, 1])
    zz = dem[np.clip(np.round(rr).astype(int), 0, H - 1),
             np.clip(np.round(cc).astype(int), 0, W - 1)]
    g = np.abs(zz[win:] - zz[:-win]) / float(win)
    return float(np.percentile(g, 95) * 100.0)


def current_endpoints(site):
    """现行端点：从 lc6 路径掩膜取骨架端点 → 合并近邻 → 选最远一对。"""
    pat = np.load(os.path.join(RES, f"cold_start_{site}_lc6.npz"))["path"] > 0
    ends = find_endpoints(pat)
    p, q = merge_and_pick_pair(np.argwhere(ends))
    return (float(p[0]), float(p[1])), (float(q[0]), float(q[1]))


def make_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    labels = [c[0] for c in CONFIGS]
    colors = ("#4C78A8", "#F58518")
    x = np.arange(len(labels))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    for i, site in enumerate(SITES):
        vals = [results[site]["configs"][c[0]]["iou"] for c in CONFIGS]
        axes[0].bar(x + (i - 0.5) * w, vals, w, label=site, color=colors[i])
        for xi, v in zip(x + (i - 0.5) * w, vals):
            axes[0].text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5)
        g = [results[site]["configs"][c[0]]["grade_p95"] for c in CONFIGS]
        axes[1].plot(x, g, "o-", color=colors[i], lw=1.6, ms=5, label=site)
        for xi, v in zip(x, g):
            axes[1].annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                             xytext=(0, 7), ha="center", fontsize=7.5, color=colors[i])
    axes[0].axhline(results[SITES[0]]["configs"]["base"]["iou"], color=colors[0],
                    ls=":", lw=1, alpha=0.8)
    axes[0].axhline(results[SITES[1]]["configs"]["base"]["iou"], color=colors[1],
                    ls=":", lw=1, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("1 m 掩膜 IoU（路径膨胀后）")
    axes[0].set_title("路径级 IoU：一次只改一个特征参数", fontsize=10.5)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(axis="y", ls=":", alpha=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("10 m 窗纵坡 P95 (%)")
    axes[1].set_title("路径纵坡 P95（物理合理性检查）", fontsize=10.5)
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(ls=":", alpha=0.4)
    fig.suptitle("宏观特征参数敏感性（两站，路径级、不重训；SHP 仅评分）", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(ROOT, "figures", "feature_sensitivity.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"写出 {out}")


def main():
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    results, rows = {}, []
    for site in SITES:
        print(f"\n=== {site}", flush=True)
        dem_f, valid = C.load_dem(site, "mean", log=print)   # 特征基准 = dem_mean（现行）
        x0, y1, wm, hm, gsd, d = P.dem_window(site)
        H, W = d["dem_mean"].shape
        dem_agl = fill_nodata(d["dem_min"], valid)            # 纵坡统计的 AGL 基准面
        gt = load_truth(site, valid, x0, y1, gsd, H, W)
        ep = current_endpoints(site)
        print(f"  端点 {ep[0]} ↔ {ep[1]}（固定，取自 lc6 路径）；"
              f"真值 {int(gt.sum()):,} px；gate={GATE[site]} px")

        results[site] = dict(
            site=site, endpoints=[list(ep[0]), list(ep[1])], gate_px=GATE[site],
            truth_source=("results/mask_1m_%s.npy" % site if site == "杨家坝" else
                          "SHP %s → poly_mask_hr → 1 m 最大连通域" % SHP_NEW[site]["poly"]),
            note=("路径级敏感性：不重训、不改网络；IoU = 路径膨胀后 1 m 掩膜 vs 真值。"
                  "仅反映宏观特征参数对路径位置的影响，不是最终模型指标。"),
            configs={})

        for name, cfg in CONFIGS:
            print(f"  -- {name}（R={cfg['R']}, gate P{cfg['gate_pct']:g}）", flush=True)
            op, n_row, n_col, conf, minfo = C.macro_fields(
                dem_f, valid, gate_pct=cfg["gate_pct"], radii=[cfg["R"]])
            a = SimpleNamespace(**PATH_ARGS)
            path, dist, cost, crange = C.macro_path(
                op, valid, ep, n_row, n_col, conf, a, dem=dem_f, log=print)
            pred = binary_dilation(path, disk(GATE[site])) & valid
            v = iou(pred, gt)
            g95 = grade_p95(path, dem_agl, H, W)
            results[site]["configs"][name] = dict(
                R=cfg["R"], gate_pct=cfg["gate_pct"], note=cfg["note"],
                path_px=int(path.sum()), pred_px=int(pred.sum()),
                iou=round(v, 4), grade_p95=round(g95, 4),
                h1_gate_deg=minfo["h1_gate_deg"], conf_px=minfo["conf_px"])
            rows.append({"site": site, "config": name, "R": cfg["R"],
                         "gate_pct": cfg["gate_pct"], "gate_px": GATE[site],
                         "path_px": int(path.sum()), "pred_px": int(pred.sum()),
                         "iou": round(v, 4), "grade_p95": round(g95, 4),
                         "h1_gate_deg": minfo["h1_gate_deg"]})
            print(f"     IoU={v:.4f}  纵坡 P95={g95:.1f}%  路径 {int(path.sum())} px")

    out = os.path.join(RES, "feature_sensitivity.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写出 {out}")

    csv_path = os.path.join(RES, "feature_sensitivity.csv")
    cols = ["site", "config", "R", "gate_pct", "gate_px", "path_px", "pred_px",
            "iou", "grade_p95", "h1_gate_deg"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"写出 {csv_path}")

    make_figure(results)

    print("\n### 特征参数敏感性（路径级 IoU / 纵坡 P95）\n")
    print("| 配置 | R | h1 门控 | 杨家坝 IoU | 杨 纵坡P95 | 西坝村 IoU | 西 纵坡P95 |")
    print("|---|---|---|---|---|---|---|")
    for name, cfg in CONFIGS:
        y = results["杨家坝"]["configs"][name]
        x = results["西坝村"]["configs"][name]
        print(f"| {name} | {cfg['R']} | P{cfg['gate_pct']:g} | {y['iou']:.4f} | "
              f"{y['grade_p95']:.1f}% | {x['iou']:.4f} | {x['grade_p95']:.1f}% |")


if __name__ == "__main__":
    main()
