# -*- coding: utf-8 -*-
"""路径偏差诊断：中心线 ~12px 偏差是"整条均匀偏"还是"分段偏"。

动机
----
迭代 3（results/eval_path_adaptive.json）定出：即便消除宽度惩罚，openness 的中心线距离
仍为 西坝村 12.0 px / 杨家坝 13.2 px（1 m 网格 ⇒ 12~13 m），**大于走廊自身宽度**
（2.75~9.73 m，中位 6.95 m）→ 这解释了 mask IoU 只有 0.126。

但"偏 12 m"有两种来源，决定候选 2（多路径）有没有救：
  · 均匀偏移：整条系统性偏一侧 → ❌ 多路径救不了（是代价图整体偏）
  · 分段偏移：只有部分段偏（抄近道 / 跳支路）→ ✅ 能救（补支路）
选 2 之前必须先分清是哪种。

方法
----
  gt_skel   = skeletonize(真值走廊 mask)
  pred_skel = skeletonize(预测路径)      # 原始 1px 路径，即中心线
  d_pred = 每个**预测骨架**像元到最近真值骨架的欧氏距离（px，1px = 1 m）
  d_gt   = 每个**真值骨架**像元到最近预测骨架的距离（反向）

判据（用户）
----
  P90/P50(d_pred) < 2 且四象限距离接近   → 整条均匀偏 → ❌ 改代价图
  P90/P50 > 3 或某象限距离特别大          → 分段偏     → ✅ 做候选 2
  四象限距离差 > 3 倍                     → 某一段明显跑偏（多路径有救）

路径约定
----
  预测路径取 `results/path_raw_{tag}_{site}.npy`（tag = cost 的 "+"→"P"），由
  `tools/path_extraction.extract_path` 落盘；真值取 `results/mask_1m_{site}.npy`。

输出
----
  results/path_offset_diag.json / .log
  figures/path_offset_diag_{tag}_{site}.png（预测骨架按 d_pred 着色，直观看均匀/分段）

用法
----
    python tools/path_offset_diag.py                       # 默认 open
    python tools/path_offset_diag.py --costs open tpi31    # 对照
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
QUADS = ((0, 0, "左上"), (0, 1, "右上"), (1, 0, "左下"), (1, 1, "右下"))


def _raw_path(site, cost):
    tag = cost.replace("+", "P")
    p = os.path.join(RES, f"path_raw_{tag}_{site}.npy")
    return p if os.path.exists(p) else None


def verdict_of(ratio, quad_p50):
    """按用户判据表给动作。quad_p50: 各象限 d_pred 中位（px）。"""
    vals = [v for v in quad_p50 if v is not None]
    spread = (max(vals) / max(min(vals), 0.1)) if len(vals) >= 2 else 1.0
    if ratio < 2.0 and spread <= 3.0:
        return "整条均匀偏 → ❌ 多路径救不了，改代价图", spread
    if ratio > 3.0 or spread > 3.0:
        return "分段偏（局部抄近道/跳支路）→ ✅ 做候选 2（多路径）", spread
    return "中间地带（P90/P50 2~3）→ 需看图确认", spread


def diagnose(site, cost, log):
    pp = _raw_path(site, cost)
    if pp is None:
        log(f"[{site}/{cost}] 缺 path_raw 产物，跳过")
        return None
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    pred = np.load(pp).astype(bool)
    gt_skel = skeletonize(label)
    pred_skel = skeletonize(pred)
    if not pred_skel.any() or not gt_skel.any():
        log(f"[{site}/{cost}] 骨架为空，跳过")
        return None

    dt_gt = distance_transform_edt(~gt_skel)      # 每点到最近真值骨架的距离
    dt_pred = distance_transform_edt(~pred_skel)  # 每点到最近预测骨架的距离
    d_pred = dt_gt[pred_skel]                     # 预测→真值（px = m）
    d_gt = dt_pred[gt_skel]                       # 真值→预测

    p50, p90 = float(np.median(d_pred)), float(np.percentile(d_pred, 90))
    p50g, p90g = float(np.median(d_gt)), float(np.percentile(d_gt, 90))
    ratio = p90 / max(p50, 0.1)
    frac_off = float((d_pred > 2.0 * max(p50, 0.1)).mean())   # 偏段占比（> 2×中位）

    H, W = label.shape
    ys, xs = np.where(pred_skel)
    quads, quad_p50 = {}, []
    for qy, qx, qname in QUADS:
        m = ((ys < H // 2) == bool(qy)) & ((xs < W // 2) == bool(qx))
        if int(m.sum()) >= 10:
            q50 = float(np.median(d_pred[m]))
            q90 = float(np.percentile(d_pred[m], 90))
            quads[qname] = dict(n=int(m.sum()), p50=q50, p90=q90)
            quad_p50.append(q50)

    vd, spread = verdict_of(ratio, quad_p50)
    log(f"\n[{site}/{cost}]  预测骨架 {int(pred_skel.sum())}px  真值骨架 {int(gt_skel.sum())}px")
    log(f"  预测→真值：中位 {p50:.1f}  P90 {p90:.1f}  max {d_pred.max():.1f}  "
        f"P90/P50 = {ratio:.1f}   偏段占比(>2×中位) = {frac_off:.1%}")
    log(f"  真值→预测：中位 {p50g:.1f}  P90 {p90g:.1f}")
    for qname, q in quads.items():
        log(f"    {qname}: n={q['n']:5d}  中位 {q['p50']:5.1f}  P90 {q['p90']:5.1f}")
    log(f"  四象限中位极差 = {spread:.1f}×   判据 → {vd}")

    return dict(site=site, cost=cost,
                pred_skel_px=int(pred_skel.sum()), gt_skel_px=int(gt_skel.sum()),
                d_pred_med=p50, d_pred_p90=p90, d_pred_max=float(d_pred.max()),
                d_gt_med=p50g, d_gt_p90=p90g,
                ratio_p90_p50=ratio, frac_off=frac_off,
                quads=quads, quad_spread=spread, verdict=vd)


def figure(site, cost, log, row):
    """预测骨架按 d_pred 着色（turbo）：整条一个色 → 均匀偏；忽蓝忽红 → 分段偏。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    tag = cost.replace("+", "P")
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    pred = np.load(_raw_path(site, cost)).astype(bool)
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    gt_skel = skeletonize(label)
    pred_skel = skeletonize(pred)
    dt_gt = distance_transform_edt(~gt_skel)

    off = np.full(label.shape, np.nan)
    off[pred_skel] = dt_gt[pred_skel]
    ys, xs = np.where(pred_skel)
    cvals = off[ys, xs]

    fig, ax = plt.subplots(figsize=(7.2, 7), dpi=140)
    ax.imshow(np.where(valid, 0.85, 1.0), cmap="gray", vmin=0, vmax=1)
    ax.contour(gt_skel.astype(float), levels=[0.5], colors="k", linewidths=0.7)
    ax.contour(label.astype(float), levels=[0.5], colors="0.6", linewidths=0.6)
    sc = ax.scatter(xs, ys, c=cvals, s=7, cmap="turbo",
                    vmin=0, vmax=max(np.percentile(cvals, 95), 1.0))
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("预测骨架 → 真值骨架 距离（px = m）", fontsize=9)
    vd_plain = row["verdict"].replace("❌", "").replace("✅", "").strip()
    ax.set_title(f"{site} · {cost}：中心线偏差诊断\n"
                 f"d_pred 中位 {row['d_pred_med']:.1f} P90 {row['d_pred_p90']:.1f}  "
                 f"P90/P50 = {row['ratio_p90_p50']:.1f}（四象限极差 {row['quad_spread']:.1f}×）\n"
                 f"{vd_plain}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    os.makedirs(FIG, exist_ok=True)
    fp = os.path.join(FIG, f"path_offset_diag_{tag}_{site}.png")
    fig.tight_layout(); fig.savefig(fp); plt.close(fig)
    log(f"  图: {os.path.relpath(fp, ROOT)}")
    return fp


def main():
    ap = argparse.ArgumentParser(description="路径中心线偏差：均匀 vs 分段 诊断")
    ap.add_argument("--costs", nargs="*", default=["open"])
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--fig", action="store_true", default=True)
    ap.add_argument("--no-fig", dest="fig", action="store_false")
    a = ap.parse_args()

    lf = open(os.path.join(RES, "path_offset_diag.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()

    log("路径中心线偏差诊断（均匀偏 vs 分段偏）")
    log("判据：P90/P50<2 且四象限接近 → 均匀偏(改代价图)；>3 或某象限特别大 → 分段偏(候选 2)")

    rows = []
    for cost in a.costs:
        for site in a.sites:
            r = diagnose(site, cost, log)
            if r is None:
                continue
            rows.append(r)
            if a.fig:
                r["figure"] = figure(site, cost, log, r)

    log(f"\n{'=' * 96}\n汇总")
    log(f"{'cost':>7s} {'站点':>6s} {'d_pred中位':>10s} {'d_predP90':>10s} {'P90/P50':>8s} "
        f"{'偏段占比':>8s} {'象限极差':>8s}  判据")
    log("-" * 116)
    for r in rows:
        log(f"{r['cost']:>7s} {r['site']:>6s} {r['d_pred_med']:>10.1f} {r['d_pred_p90']:>10.1f} "
            f"{r['ratio_p90_p50']:>8.1f} {r['frac_off']:>8.1%} {r['quad_spread']:>8.1f}  {r['verdict']}")

    jp = os.path.join(RES, "path_offset_diag.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    log(f"\n保存: {jp}")
    lf.close()


if __name__ == "__main__":
    main()
