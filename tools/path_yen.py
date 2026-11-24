# -*- coding: utf-8 -*-
"""候选 2 · 方案 C：每域取 K 条最短路径（Yen's K-shortest paths），在候选里选。

动机
----
西坝村 openness 中心线偏差诊断为**分段偏**（results/path_offset_diag.json：
P90/P50 = 2.4、偏段占比 20.6%、四象限极差 3.2×；中段一个"鼓包"跳到 15~20px 外）。
且方案 A（端点对穷举）实测增益仅 +0.0016（results/path_multipath_open.json）——
说明"错的仍是当前图上的最短路"。方案 C 保留**前 K 条最短路**作候选：
若真值走廊恰是第 2/3 短路（被第 1 短路抄近道跳过），K 短路就能把它捞回来。

⚠ 选择口径（必须分开报告）
----
  oracle   ：在 K 条候选里按**与真值的 IoU** 选最优 → **不可部署上界**。
             回答"真值路径是否在 top-K 里"——若 oracle@K ≫ 基线，则问题在"选路规则"，
             若 oracle@K ≈ 基线，则 top-K 根本不含真值，代价模型本身错了。
  deploy  ：按**路径代价最小**选 → 恒等于第 1 短路（基线），故**无增益**（仅作对照说明）。

实现
----
Yen's，图 = 与 `path_extraction.build_graph` 同口径的 8 邻域（对角线 √2）栅格图，
本脚本另建**对称** COO 以便按边"禁用"；每轮把已接受路径在 root 之后的那条边权重剔除，
再跑一次 Dijkstra 生成候选。spur 节点按 `--stride` 抽稀以控时。

输出：results/path_yen_{tag}.json / .log、figures/path_yen_{tag}_{site}.png

用法
----
    python tools/path_yen.py --sites 西坝村 --cost open --k 3 --stride 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from eval_linear_metrics import linear_metrics, window_mask                       # noqa: E402
from path_extraction import (COST_PRESETS, SITES, RES, DATA, SQ2, S8,            # noqa: E402
                             build_cost, find_endpoints, merge_and_pick_pair,
                             snap_to_domain)

FIG = os.path.join(ROOT, "figures")
TOL = 2.0
DILATE = 3


def _iou(a, b, win):
    inter = int((a & b & win).sum())
    union = int(((a | b) & win).sum())
    return (inter / union if union else 0.0), inter, union


def build_sym_coo(cost, dom):
    """对称 8 邻域图（对角线 √2），边权 = 步长 × 两端点代价均值。返回 (rows, cols, data)。"""
    H, W = dom.shape
    idx = np.arange(H * W, dtype=np.int32).reshape(H, W)
    cf = cost.ravel()
    dr, dc, dd = [], [], []
    for dy, dx, L in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQ2), (1, -1, SQ2)):
        if dx >= 0:
            A, B = idx[:H - dy, :W - dx], idx[dy:, dx:]
        else:
            A, B = idx[:H - dy, -dx:], idx[dy:, :W + dx]
        sa, sb = A.ravel(), B.ravel()
        ok = dom.ravel()[sa] & dom.ravel()[sb]
        sa, sb = sa[ok], sb[ok]
        d = (L * 0.5 * (cf[sa] + cf[sb])).astype(np.float64)
        for u, v in ((sa, sb), (sb, sa)):                       # 两个方向都存
            dr.append(u); dc.append(v); dd.append(d)
    return (np.concatenate(dr), np.concatenate(dc), np.concatenate(dd))


class Grid:
    """对称 COO 图 + 边键索引，支持按"边集合"临时禁用后跑 Dijkstra。"""

    def __init__(self, rows, cols, data, shape):
        self.rows, self.cols, self.data = rows, cols, data
        self.N = shape[0] * shape[1]
        self.H, self.W = shape
        k = np.minimum(rows, cols).astype(np.int64) * self.N + np.maximum(rows, cols)
        self.order = np.argsort(k, kind="stable")
        self.ks = k[self.order]                       # 排序后的键（含重复）
        self.M = len(data)

    def _key(self, u, v):
        return int(min(u, v)) * self.N + int(max(u, v))

    def edge_idx(self, u, v):
        key = self._key(u, v)
        lo = int(np.searchsorted(self.ks, key, side="left"))
        hi = int(np.searchsorted(self.ks, key, side="right"))
        if lo >= hi:
            return None
        return self.order[lo:hi]

    def incident(self, node):
        """node 的全部关联边索引（8 邻域）。"""
        y, x = divmod(node, self.W)
        out = []
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.H and 0 <= nx < self.W:
                e = self.edge_idx(node, ny * self.W + nx)
                if e is not None:
                    out.append(e)
        return np.concatenate(out) if out else np.array([], dtype=np.int64)

    def dijkstra(self, src, dst, banned=()):
        mask = np.ones(self.M, bool)
        if len(banned):
            mask[np.concatenate(list(banned))] = False
        G = coo_matrix((self.data[mask], (self.rows[mask], self.cols[mask])),
                       shape=(self.N, self.N)).tocsr()
        dist, pred = dijkstra(G, directed=False, indices=int(src),
                              return_predecessors=True)
        if not np.isfinite(dist[dst]):
            return None, np.inf
        nodes = []
        cur = int(dst)
        while cur != int(src) and cur >= 0:
            nodes.append(cur)
            cur = int(pred[cur])
            if cur < 0:
                return None, np.inf
        nodes.append(int(src))
        nodes.reverse()
        cost = sum(self.weight(u, v) for u, v in zip(nodes[:-1], nodes[1:]))
        return nodes, float(cost)

    def weight(self, u, v):
        key = self._key(u, v)
        p = int(np.searchsorted(self.ks, key))
        return float(self.data[self.order[p]]) if p < len(self.ks) and self.ks[p] == key else 0.0


def _nodes_to_path(nodes, shape):
    p = np.zeros(shape[0] * shape[1], dtype=bool)
    p[np.asarray(nodes, dtype=np.int64)] = True
    return p.reshape(shape)


def _cpl(a, b):
    """最长公共前缀长度。"""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def yen(grid, src, dst, K, stride, log=None):
    """返回 [(nodes, cost), ...]，第 1 条为最短路。"""
    p1, c1 = grid.dijkstra(src, dst)
    if p1 is None:
        return []
    A = [(p1, c1)]
    for _ in range(1, K):
        prev = A[-1][0]
        if len(prev) < 4:
            break
        cand = []
        spurs = range(1, len(prev) - 1, max(1, stride))
        for t in spurs:
            spur = prev[t]
            root = prev[:t + 1]                    # 含 spur
            banned = []
            for ap, _c in A:                       # 禁用 A 中各路径在 root 之后的那条边
                if _cpl(ap, prev) >= t + 1 and t + 1 < len(ap):
                    e = grid.edge_idx(ap[t], ap[t + 1])
                    if e is not None:
                        banned.append(e)
            if not banned:
                continue
            sp, sc = grid.dijkstra(spur, dst, banned=banned)
            if sp is None:
                continue
            total = root[:-1] + sp               # root 末位=spur，sp 首位=spur → 去重
            cost = sum(grid.weight(u, v) for u, v in zip(total[:-1], total[1:]))
            cand.append((total, float(cost)))
        if not cand:
            break
        cand.sort(key=lambda x: x[1])
        A.append(cand[0])
    return A


def solve_site(site, cost, eps, norm, K, stride, log):
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    dom = valid.copy()
    cost_map, _ = build_cost(site, tpi31, dom, eps, norm, cost)
    rows, cols, data = build_sym_coo(cost_map, dom)
    grid = Grid(rows, cols, data, dom.shape)
    win = window_mask(valid)
    H, W = dom.shape

    cc, n_cc = cc_label(label, structure=S8)
    base_paths, or_paths, comps = [], [], []
    for cid in range(1, n_cc + 1):
        m = cc == cid
        pts = np.argwhere(find_endpoints(skeletonize(m)))
        if len(pts) < 2:
            continue
        s_r, e_r = merge_and_pick_pair(pts, 10.0)
        src = int(snap_to_domain(s_r, dom)[0] * W + snap_to_domain(s_r, dom)[1])
        dst = int(snap_to_domain(e_r, dom)[0] * W + snap_to_domain(e_r, dom)[1])
        t0 = time.perf_counter()
        A = yen(grid, src, dst, K, stride, log)
        if not A:
            continue
        cands = []
        for nodes, cst in A:
            pth = _nodes_to_path(nodes, dom.shape)
            pj = binary_dilation(pth, iterations=DILATE) & dom
            iou = _iou(pj, label, win)[0]
            cands.append(dict(path=pth, cost=float(cst), iou=float(iou),
                              path_px=int(pth.sum()), n_nodes=len(nodes)))
        base = cands[0]
        oracle = max(cands, key=lambda c: c["iou"])
        base_paths.append(base["path"])
        or_paths.append(oracle["path"])
        comps.append(dict(cid=int(cid), n_k=len(cands), sec=round(time.perf_counter() - t0, 1),
                          baseline_iou=round(base["iou"], 4),
                          oracle_iou=round(oracle["iou"], 4),
                          oracle_k=int(np.argmax([c["iou"] for c in cands]) + 1),
                          cands=[dict(k=i + 1, iou=round(c["iou"], 4),
                                      cost=round(c["cost"], 2), px=c["path_px"])
                                 for i, c in enumerate(cands)]))
        log(f"  域{cid}: K={len(cands)}  基线(第1短路)IoU {base['iou']:.4f} | "
            f"oracle IoU {oracle['iou']:.4f}(第{comps[-1]['oracle_k']}短路)  "
            f"[{comps[-1]['sec']}s]")

    def _merge(lst):
        return np.logical_or.reduce(lst) if lst else None

    return dict(label=label, valid=valid, dom=dom, win=win,
                base=_merge(base_paths), oracle=_merge(or_paths), comps=comps)


def _report(name, path, r, log):
    if path is None:
        log(f"  {name:<12s} 无路径")
        return None
    pred = binary_dilation(path, iterations=DILATE) & r["dom"]
    iou, inter, union = _iou(pred, r["label"], r["win"])
    lm = linear_metrics(pred, r["label"], r["win"], TOL)
    log(f"  {name:<12s} 路径 {int(path.sum()):>4d}px  IoU@0.5={iou:.4f}  "
        f"中心线F1={lm['f1']:.4f}  中心线={lm['centerline_px']:.2f}px  "
        f"拓扑R/P={lm['topo_recall']:.2f}/{lm['topo_precision']:.2f}")
    return dict(name=name, path_px=int(path.sum()), iou=float(iou), inter=inter, union=union,
                centerline_f1=float(lm["f1"]), centerline_px=float(lm["centerline_px"]),
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
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="0.6", lw=2, label="真值走廊轮廓")]
    if r["base"] is not None:
        ax.contour(r["base"].astype(float), levels=[0.5], colors="blue", linewidths=1.5)
        handles.append(Line2D([], [], color="blue", lw=2, label="第1短路(=基线)"))
    if r["oracle"] is not None and (r["base"] is None or not np.array_equal(r["oracle"], r["base"])):
        ax.contour(r["oracle"].astype(float), levels=[0.5], colors="red",
                   linewidths=1.6, linestyles="--")
    handles.append(Line2D([], [], color="red", lw=2, ls="--",
                          label=f"top-K 最优(oracle,第{res['oracle_k']}短路)"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(f"{site} · {cost}：Yen's K={res['k']} 最短路（候选 2 · 方案 C）", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    os.makedirs(FIG, exist_ok=True)
    fp = os.path.join(FIG, f"path_yen_{tag}_{site}.png")
    fig.tight_layout(); fig.savefig(fp); plt.close(fig)
    log(f"  图: {os.path.relpath(fp, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="候选 2 · 方案 C：Yen's K 最短路")
    ap.add_argument("--sites", nargs="*", default=["西坝村"])
    ap.add_argument("--cost", default="open", choices=list(COST_PRESETS))
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--norm", type=str, default="pct", choices=["pct", "minmax"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--fig", action="store_true", default=True)
    ap.add_argument("--no-fig", dest="fig", action="store_false")
    a = ap.parse_args()
    tag = a.cost.replace("+", "P")

    lf = open(os.path.join(RES, f"path_yen_{tag}.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()

    log(f"候选 2 · 方案 C（Yen's K={a.k}，spur 抽稀 stride={a.stride}）  cost={a.cost} "
        f"eps={a.eps} norm={a.norm} 膨胀={DILATE}px")
    report = {}
    for site in a.sites:
        r = solve_site(site, a.cost, a.eps, a.norm, a.k, a.stride, log)
        rows = [_report(f"基线(第1短路)", r["base"], r, log),
                _report(f"oracle(top-{a.k})", r["oracle"], r, log)]
        best = max(r["comps"], key=lambda c: c["oracle_iou"] - c["baseline_iou"])
        res = dict(site=site, k=a.k, stride=a.stride, n_cc=len(r["comps"]),
                   oracle_k=int(max(best["oracle_k"], 1)), comps=r["comps"],
                   metrics=[x for x in rows if x])
        log(f"  → 最大增益域 {best['cid']}: 基线 {best['baseline_iou']} → "
            f"oracle {best['oracle_iou']}（第{best['oracle_k']}短路）")
        if a.fig:
            figure(site, a.cost, r, res, log)
        report[site] = res

    log("\n" + "=" * 92)
    log("汇总（oracle = top-K 内按真值选，**不可部署上界**；基线 = 第 1 短路 = 可部署且恒无增益）")
    log(f"{'站点':>6s}{'域数':>6s}{'基线IoU':>10s}{'oracleIoU':>11s}{'oracle第K':>10s}{'基线中心线F1':>13s}{'oracle中心线F1':>14s}")
    for site, res in report.items():
        bm = next(x for x in res["metrics"] if x["name"].startswith("基线"))
        om = next(x for x in res["metrics"] if x["name"].startswith("oracle"))
        log(f"{site:>6s}{res['n_cc']:>6d}{bm['iou']:>10.4f}{om['iou']:>11.4f}"
            f"{res['oracle_k']:>10d}{bm['centerline_f1']:>13.4f}{om['centerline_f1']:>14.4f}")

    out = os.path.join(RES, f"path_yen_{tag}.json")
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"\n保存: {os.path.relpath(out, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
