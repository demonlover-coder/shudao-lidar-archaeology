# -*- coding: utf-8 -*-
"""路径提取改进评估：骨架 IoU + 自适应膨胀 IoU（第 3 步 · 迭代 3）。

背景
----
迭代 2（results/path_extraction_open_d3.json）定出 openness 单通道为两站通用最优代价：
固定 3px 膨胀下 西坝村 IoU 0.126 / 杨家坝 0.062。但固定宽度膨胀把"落点误差"与
"宽度不匹配"混在同一个 IoU 里 → 无法判断"中心线本身准不准"。本脚本因此做两件事：

A. 自适应膨胀（用真值半宽）
   半宽场 half_w = distance_transform_edt(gt_label)（每个前景像元到最近背景的欧氏距离）；
   预测路径骨架每点按 r = min(half_w[y, x], max_width) 膨胀 → 宽度惩罚被消除，
   剩下的 IoU 只反映"落点"。预期相对固定宽度 IoU/F1 上升。

B. 骨架化评估（只在中心线上比）
   预测路径与真值都骨架化，IoU 只在中心线上算 → 诊断"如果只看中心线，模型多准"。
   ⚠ 两条 1px 骨架"精确重合"要求过苛，exact IoU 几乎必然≈0，是退化指标；
   故同时给出 tol=2px 的**缓冲骨架 IoU**（两条骨架各膨胀 2px 再求 IoU），后者才可作判据。

C. 判据（按 tol 缓冲骨架 IoU）
   >0.3    → 中心线准、宽度没调好 → 自适应膨胀即为最终评估
   0.15~0.3 → 中心线有信号、宽度不敏感
   <0.15   → 中心线本身就偏 → 直接走候选 2（多路径）

输出
----
results/eval_path_adaptive.json / .txt —— 对每个 `--cost` 预设 × 两站，给出三种指标：
固定宽度 mask IoU（D10 口径）/ 骨架 IoU（exact + buf2）/ 自适应膨胀 IoU。

用法
----
    python tools/eval_path_adaptive.py                       # 全部 8 个 cost 预设 × 两站
    python tools/eval_path_adaptive.py --costs open cliff
    python tools/eval_path_adaptive.py --max-width 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from eval_linear_metrics import linear_metrics, window_mask           # noqa: E402
from path_extraction import COST_PRESETS, SITES, RES, extract_path    # noqa: E402

FIG = os.path.join(ROOT, "figures")
DATA_1M = os.path.join(ROOT, "data")
TOL = 2.0


# ------------------------------------------------------------ 三种指标 ----
def _iou(a, b, win):
    inter = int((a & b & win).sum())
    union = int(((a | b) & win).sum())
    return inter / union if union else 0.0, inter, union


def mask_iou_fixed(pred_path, dom, label, win, radius):
    """固定宽度膨胀 + D10 口径 IoU（现有指标，作对照）。"""
    pred = binary_dilation(pred_path, iterations=int(radius)) & dom
    iou, inter, union = _iou(pred, label, win)
    return iou, inter, union, int(pred.sum())


def skel_exact_iou(pred_path, label, win):
    """精确骨架 IoU：两条 1px 骨架精确重合 → 退化指标（预期≈0）。"""
    ps = skeletonize(pred_path)
    gs = skeletonize(label)
    return _iou(ps, gs, win)


def skel_buf_iou(pred_path, label, win, tol=TOL):
    """缓冲骨架 IoU：两条骨架各膨胀 tol px 再求 IoU（可作判据，容忍 1~2px 落点误差）。"""
    ps = binary_dilation(skeletonize(pred_path), iterations=int(tol))
    gs = binary_dilation(skeletonize(label), iterations=int(tol))
    return _iou(ps, gs, win)


def adaptive_dilate(pred_path, label, max_width=6):
    """自适应膨胀：预测骨架每点按真值局部半宽 min(half_w, max_width) 膨胀。

    half_w[y, x] = 该点到最近背景的欧氏距离 = 走廊局部半宽；
    预测路径落在真值走廊外时 half_w≈0 → 只记该像元本身（不虚增 union）。
    """
    half_w = distance_transform_edt(label)
    sk = skeletonize(pred_path)
    H, W = label.shape
    out = np.zeros_like(label, dtype=bool)
    ys, xs = np.where(sk)
    for y, x in zip(ys.tolist(), xs.tolist()):
        r = int(min(float(half_w[y, x]), max_width))
        out[max(0, y - r):min(H, y + r + 1),
            max(0, x - r):min(W, x + r + 1)] = True
    return out


def verdict_of(skel_buf):
    """按用户判据表给动作（用 tol 缓冲骨架 IoU，因 exact 退化）。"""
    if skel_buf > 0.3:
        return "中心线准、宽度没调好 → 自适应膨胀作最终评估"
    if skel_buf >= 0.15:
        return "中心线有信号、宽度不敏感"
    return "中心线本身就偏 → 走候选 2（多路径）"


# ------------------------------------------------------------ 取原始路径 ----
def get_raw_path(site, cost, eps, norm, roi, merge, log):
    """优先读缓存 path_raw_{tag}_{site}.npy，缺失则调 path_extraction.extract_path 现算。"""
    tag = cost.replace("+", "P")
    cache = os.path.join(RES, f"path_raw_{tag}_{site}.npy")
    if os.path.exists(cache):
        p = np.load(cache)
        label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
        npz = np.load(os.path.join(DATA_1M, f"{site}_1m.npz"))
        valid = npz["valid"] == 1
        return p, label, valid, None
    r = extract_path(site, cost, eps, norm, roi, merge, log)
    if r["path"] is not None:
        np.save(cache, r["path"])
    return r["path"], r["label"], r["valid"], r


# ------------------------------------------------------------ 主 ----
def main():
    ap = argparse.ArgumentParser(description="路径提取改进评估：骨架 IoU + 自适应膨胀")
    ap.add_argument("--costs", nargs="*", default=list(COST_PRESETS), choices=list(COST_PRESETS))
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--max-width", type=int, default=6, help="自适应膨胀半径上限（px）")
    ap.add_argument("--fixed-radius", type=int, default=3, help="固定宽度对照的膨胀半径（px）")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--norm", type=str, default="pct", choices=["pct", "minmax"])
    ap.add_argument("--merge-px", type=float, default=10.0)
    a = ap.parse_args()

    lf = open(os.path.join(RES, "eval_path_adaptive.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()

    log(f"路径提取改进评估（骨架 IoU + 自适应膨胀） 固定半径={a.fixed_radius}px "
        f"自适应上限={a.max_width}px eps={a.eps} norm={a.norm}")

    out = []
    for cost in a.costs:
        for site in a.sites:
            raw, label, valid, ctx = get_raw_path(site, cost, a.eps, a.norm, 0, a.merge_px, log)
            if raw is None:
                log(f"[{site}] cost={cost}: 无可用路径，跳过")
                continue
            dom = ctx["dom"] if ctx else (valid.copy())
            win = window_mask(valid)

            iou_fix, i_f, u_f, px_f = mask_iou_fixed(raw, dom, label, win, a.fixed_radius)
            iou_ex, i_e, u_e = skel_exact_iou(raw, label, win)
            iou_buf, i_b, u_b = skel_buf_iou(raw, label, win, TOL)
            adapt = adaptive_dilate(raw, label, a.max_width)
            iou_ad, i_a, u_a = _iou(adapt, label, win)
            lm = linear_metrics(raw, label, win, TOL)     # 原始 1px 路径的中心线指标

            row = dict(cost=cost, site=site,
                       path_px=int(raw.sum()),
                       mask_iou_fixed=iou_fix, fixed_radius=a.fixed_radius,
                       mask_inter_fixed=i_f, mask_union_fixed=u_f,
                       skel_iou_exact=iou_ex, skel_inter_exact=i_e, skel_union_exact=u_e,
                       skel_iou_buf2=iou_buf, skel_inter_buf2=i_b, skel_union_buf2=u_b,
                       adaptive_iou=iou_ad, adaptive_px=int(adapt.sum()),
                       adaptive_inter=i_a, adaptive_union=u_a, max_width=a.max_width,
                       raw_lr=float(lm["lr"]), raw_lp=float(lm["lp"]),
                       raw_centerline_f1=float(lm["f1"]),
                       raw_centerline_px=float(lm["centerline_px"]),
                       verdict=verdict_of(iou_buf))
            out.append(row)
            log(f"[{site:>6s}] {cost:>11s}  路径{int(raw.sum()):>4d}px  "
                f"maskIoU@{a.fixed_radius}({iou_fix:.4f})  "
                f"骨架exact({iou_ex:.4f})  骨架buf2({iou_buf:.4f})  "
                f"自适应({iou_ad:.4f})  中心线F1({lm['f1']:.4f})={lm['centerline_px']:.1f}px")

    # 汇总（按 cost 汇总两站）
    log(f"\n{'=' * 100}\n汇总（判据用 tol=2px 缓冲骨架 IoU：>0.3 走自适应 / 0.15~0.3 中心线有信号 / <0.15 走候选 2）")
    log(f"{'cost':>11s} {'站点':>6s} {'maskIoU':>8s} {'骨架exact':>9s} {'骨架buf2':>9s} "
        f"{'自适应IoU':>9s} {'中心线F1':>8s} {'中心线px':>8s}  判据")
    log("-" * 116)
    for r in out:
        log(f"{r['cost']:>11s} {r['site']:>6s} {r['mask_iou_fixed']:>8.4f} "
            f"{r['skel_iou_exact']:>9.4f} {r['skel_iou_buf2']:>9.4f} "
            f"{r['adaptive_iou']:>9.4f} {r['raw_centerline_f1']:>8.4f} "
            f"{r['raw_centerline_px']:>8.2f}  {r['verdict']}")

    # 按 cost 选两站平均自适应 IoU 最优者（诊断用）
    agg = {}
    for r in out:
        agg.setdefault(r["cost"], {"ad": [], "buf": [], "fix": []})
        agg[r["cost"]]["ad"].append(r["adaptive_iou"])
        agg[r["cost"]]["buf"].append(r["skel_iou_buf2"])
        agg[r["cost"]]["fix"].append(r["mask_iou_fixed"])
    log("\n各 cost 两站均值（自适应 / 骨架buf2 / 固定 mask）：")
    for c, d in sorted(agg.items(), key=lambda kv: -float(np.mean(kv[1]["ad"]))):
        log(f"  {c:>11s}  自适应={np.mean(d['ad']):.4f}  骨架buf2={np.mean(d['buf']):.4f}  "
            f"固定mask={np.mean(d['fix']):.4f}")

    log("\n对照：固定 3px 下 openness 西 0.1262 / 杨 0.0615；v6 跨站 IoU≈0.021；几何基线 0.011~0.028")

    jp = os.path.join(RES, "eval_path_adaptive.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(fixed_radius=a.fixed_radius, max_width=a.max_width, tol=TOL,
                       eps=a.eps, norm=a.norm, rows=out), f, ensure_ascii=False, indent=2)
    log(f"\n保存: {jp}")
    lf.close()


if __name__ == "__main__":
    main()
