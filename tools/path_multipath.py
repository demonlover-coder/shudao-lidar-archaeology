# -*- coding: utf-8 -*-
"""候选 2 · 方案 A：每域"端点对穷举"，取代"只取最远端点对"。

动机
----
西坝村 openness 的中心线偏差诊断为**分段偏**（results/path_offset_diag.json：
P90/P50 = 2.4、偏段占比 20.6%、四象限极差 3.2×；诊断图中段一个橙红"鼓包"跳到
15~20 px 外）。当前 `path_extraction` 每域只取"合并近邻后欧氏最远的一对端点"跑一条
Dijkstra——若该域真值走廊含支路，最远端点对会跳过支路。方案 A 把该域所有端点两两
配对、各跑一条 Dijkstra，在候选里选一条。

⚠ 两种选择口径（必须分开报告，否则会把 oracle 上界当成模型性能）
----
  --select oracle ：按**与真值标签的 IoU** 选最优 → **用真值选路径，不可部署**，
      仅作"端点对选择的天花板"诊断（回答"就算端点对选对了，能到多少 IoU"）。
  --select cost   ：按 Dijkstra **路径代价最小** 选（完全不看真值）→ 可部署口径。
本脚本两种都算，并在输出里明确标注口径。

输出
----
  results/path_multipath_{tag}.json / .log
  figures/path_multipath_{tag}_{site}.png（真值轮廓 + oracle 路径 + 可部署(cost)路径）

用法
----
    python tools/path_multipath.py --sites 西坝村 --cost open
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from eval_linear_metrics import linear_metrics, window_mask            # noqa: E402
from path_extraction import (COST_PRESETS, SITES, RES, DATA, S8,      # noqa: E402
                             build_cost, build_graph, find_endpoints,
                             merge_and_pick_pair, snap_to_domain)

FIG = os.path.join(ROOT, "figures")
TOL = 2.0
DILATE = 3


def _iou(a, b, win):
    inter = int((a & b & win).sum())
    union = int(((a | b) & win).sum())
    return (inter / union if union else 0.0), inter, union


def _trace(pred_row, src, dst, shape):
    """沿 pred_row（源为 src）从 dst 回溯出 1px 路径。"""
    H, W = shape
    path = np.zeros(H * W, dtype=bool)
    cur = dst
    while cur != src and cur >= 0:
        path[cur] = True
        cur = int(pred_row[cur])
    path[src] = True
    return path.reshape(H, W)


def solve_site(site, cost, eps, norm, log):
    """返回两套路径（oracle / 可部署 cost）与其评估，外加几何基线与候选统计。"""
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    dom = valid.copy()
    cost_map, _ = build_cost(site, tpi31, dom, eps, norm, cost)
    G = build_graph(cost_map, dom)
    win = window_mask(valid)
    H, W = dom.shape

    cc, n_cc = cc_label(label, structure=S8)
    or_paths, co_paths, base_paths, comps = [], [], [], []
    for cid in range(1, n_cc + 1):
        m = cc == cid
        pts = np.argwhere(find_endpoints(skeletonize(m)))
        if len(pts) < 2:
            continue
        nodes = []
        for p in pts:
            nd = snap_to_domain(p, dom)
            if nd not in nodes:
                nodes.append(nd)
        if len(nodes) < 2:
            continue

        # 基线：合并近邻后取最远一对（= 现行 path_extraction 行为，用于复现）
        s_r, e_r = merge_and_pick_pair(pts, 10.0)
        s_b, e_b = snap_to_domain(s_r, dom), snap_to_domain(e_r, dom)
        pj = binary_dilation(_trace(
            dijkstra(G, directed=False, indices=int(s_b[0] * W + s_b[1]),
                     return_predecessors=True)[1],
            int(s_b[0] * W + s_b[1]), int(e_b[0] * W + e_b[1]), dom.shape),
            iterations=DILATE) & dom
        base_iou = _iou(pj, label, win)[0]
        base_paths.append(_trace(
            dijkstra(G, directed=False, indices=int(s_b[0] * W + s_b[1]),
                     return_predecessors=True)[1],
            int(s_b[0] * W + s_b[1]), int(e_b[0] * W + e_b[1]), dom.shape))

        # 端点对穷举：多源 Dijkstra 一次算完所有端点
        idxs = [int(y * W + x) for y, x in nodes]
        dist, pred = dijkstra(G, directed=False, indices=idxs, return_predecessors=True)
        cands = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if not np.isfinite(dist[i][idxs[j]]):
                    continue
                pth = _trace(pred[i], idxs[i], idxs[j], dom.shape)
                pj = binary_dilation(pth, iterations=DILATE) & dom
                iou = _iou(pj, label, win)[0]
                cands.append(dict(i=i, j=j, path=pth, cost=float(dist[i][idxs[j]]),
                                  iou=float(iou), path_px=int(pth.sum())))
        if not cands:
            continue
        b_iou = max(cands, key=lambda c: c["iou"])
        b_cost = min(cands, key=lambda c: c["cost"])
        or_paths.append(b_iou["path"])
        co_paths.append(b_cost["path"])
        comps.append(dict(cid=cid, n_endpoints=int(len(pts)), n_nodes=len(nodes),
                          n_candidates=len(cands),
                          baseline_iou=round(float(base_iou), 4),
                          best_iou=round(float(b_iou["iou"]), 4),
                          iou_at_min_cost=round(float(b_cost["iou"]), 4),
                          min_cost=round(float(b_cost["cost"]), 2),
                          cost_of_best_iou=round(float(b_iou["cost"]), 2)))
        log(f"  域{cid}: 端点 {len(pts)}→去重 {len(nodes)}  候选 {len(cands)} 对  "
            f"基线IoU {base_iou:.4f} | 最好IoU {b_iou['iou']:.4f}(代价 {b_iou['cost']:.1f}) | "
            f"最小代价IoU {b_cost['iou']:.4f}(代价 {b_cost['cost']:.1f})")

    def _merge(lst):
        return np.logical_or.reduce(lst) if lst else None

    return dict(label=label, valid=valid, dom=dom, win=win,
                base=_merge(base_paths), oracle=_merge(or_paths), cost=_merge(co_paths),
                comps=comps, n_cc=int(n_cc))


def _report(name, path, r, log):
    if path is None:
        log(f"  {name:<10s} 无路径")
        return None
    pred = binary_dilation(path, iterations=DILATE) & r["dom"]
    iou, inter, union = _iou(pred, r["label"], r["win"])
    lm = linear_metrics(pred, r["label"], r["win"], TOL)
    log(f"  {name:<10s} 路径 {int(path.sum()):>4d}px  IoU@0.5={iou:.4f}  "
        f"中心线F1={lm['f1']:.4f}  LR={lm['lr']:.4f} LP={lm['lp']:.4f}  "
        f"中心线={lm['centerline_px']:.2f}px  拓扑R/P={lm['topo_recall']:.2f}/{lm['topo_precision']:.2f}")
    return dict(name=name, path_px=int(path.sum()), iou=float(iou), inter=inter, union=union,
                centerline_f1=float(lm["f1"]), lr=float(lm["lr"]), lp=float(lm["lp"]),
                centerline_px=float(lm["centerline_px"]),
                topo_recall=float(lm["topo_recall"]), topo_precision=float(lm["topo_precision"]))


def figure(site, cost, r, res, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    tag = cost.replace("+", "P")
    fig, ax = plt.subplots(figsize=(7.2, 7), dpi=140)
    ax.imshow(np.where(r["valid"], 0.85, 1.0), cmap="gray", vmin=0, vmax=1)
    ax.contour(r["label"].astype(float), levels=[0.5], colors="0.6", linewidths=0.8)
    for nm, pth, col, ls in (("基线(现行)", r["base"], "0.25", ":"),
                             (f"oracle(用真值选)", r["oracle"], "red", "--"),
                             ("可部署(最小代价)", r["cost"], "blue", "-")):
        if pth is not None:
            ax.contour(pth.astype(float), levels=[0.5], colors=col, linewidths=1.6, linestyles=ls)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="0.6", lw=2, label="真值走廊轮廓"),
                       Line2D([], [], color="0.25", lw=2, ls=":", label="基线(现行最远端点对)"),
                       Line2D([], [], color="red", lw=2, ls="--", label="oracle(用真值选,不可部署)"),
                       Line2D([], [], color="blue", lw=2, ls="-", label="可部署(最小代价选)")],
              loc="upper right", fontsize=8)
    ax.set_title(f"{site} · {cost}：端点对穷举（候选 2 · 方案 A）", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    os.makedirs(FIG, exist_ok=True)
    fp = os.path.join(FIG, f"path_multipath_{tag}_{site}.png")
    fig.tight_layout(); fig.savefig(fp); plt.close(fig)
    log(f"  图: {os.path.relpath(fp, ROOT)}")
    return fp


def main():
    ap = argparse.ArgumentParser(description="候选 2 · 方案 A：端点对穷举")
    ap.add_argument("--sites", nargs="*", default=["西坝村"])
    ap.add_argument("--cost", default="open", choices=list(COST_PRESETS))
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--norm", type=str, default="pct", choices=["pct", "minmax"])
    ap.add_argument("--fig", action="store_true", default=True)
    ap.add_argument("--no-fig", dest="fig", action="store_false")
    a = ap.parse_args()
    tag = a.cost.replace("+", "P")

    lf = open(os.path.join(RES, f"path_multipath_{tag}.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()

    log(f"候选 2 · 方案 A（每域端点对穷举）  cost={a.cost} eps={a.eps} norm={a.norm} 膨胀={DILATE}px")
    out = []
    for site in a.sites:
        r = solve_site(site, a.cost, a.eps, a.norm, log)
        log(f"\n[{site}]  连通域 {r['n_cc']}")
        res = {}
        for nm, p in (("基线", r["base"]), ("oracle", r["oracle"]), ("cost", r["cost"])):
            res[nm] = _report(nm, p, r, log)
        row = dict(site=site, n_cc=r["n_cc"], comps=r["comps"],
                   baseline=res["基线"], oracle=res["oracle"], cost=res["cost"])
        out.append(row)
        if a.fig:
            row["figure"] = figure(site, a.cost, r, res, log)

    log(f"\n{'=' * 92}\n汇总（oracle = 用真值选，**不可部署上界**；cost = 不看真值，可部署）")
    log(f"{'站点':>6s} {'候选数':>6s} {'基线IoU':>9s} {'oracleIoU':>10s} {'costIoU':>9s}  "
        f"{'oracle中心线F1':>13s} {'cost中心线F1':>12s}")
    log("-" * 96)
    for r in out:
        ncand = sum(c["n_candidates"] for c in r["comps"])
        b = r["baseline"]["iou"] if r["baseline"] else float("nan")
        o = r["oracle"]["iou"] if r["oracle"] else float("nan")
        c = r["cost"]["iou"] if r["cost"] else float("nan")
        of = r["oracle"]["centerline_f1"] if r["oracle"] else float("nan")
        cf = r["cost"]["centerline_f1"] if r["cost"] else float("nan")
        log(f"{r['site']:>6s} {ncand:>6d} {b:>9.4f} {o:>10.4f} {c:>9.4f} {of:>13.4f} {cf:>12.4f}")
    log("\n对照：现行 openness 固定 3px 膨胀 IoU 西 0.1262 / 杨 0.0615；"
        "v6 跨站 IoU≈0.021；几何基线 0.011~0.028")

    jp = os.path.join(RES, f"path_multipath_{tag}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(cost=a.cost, eps=a.eps, norm=a.norm, dilate=DILATE,
                       note="oracle 用真值选=不可部署上界；cost 不看真值=可部署",
                       sites=out), f, ensure_ascii=False, indent=2)
    log(f"\n保存: {jp}")
    lf.close()


if __name__ == "__main__":
    main()
