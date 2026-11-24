# -*- coding: utf-8 -*-
"""全自动路径提取的**端点精度敏感性**：第三方起止点误差 → IoU 上限。

背景：路线级第三方端点（褒谷口/斜谷口）落到站点尺度后 IoU=0.0000/0.0065，
说明瓶颈不在代价场而在端点位置。本脚本量化"第三方若能给出站点尺度端点，
允许多大误差"：对理想端点施加 σ 像素（= σ 米）高斯扰动，统计 IoU。

不改变任何既有结果文件，只输出 results/path_endpoint_sensitivity.json。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import path_extraction as P                                    # noqa: E402
from eval_linear_metrics import window_mask                    # noqa: E402

SIGMAS = [0, 5, 10, 20, 30, 50, 100]
REPS = 5
SEED = 42


def base_endpoints(site, dom):
    """理想（第三方无误差）端点：真值骨架端点，仅作为本敏感性实验的基准。"""
    label = np.load(os.path.join(P.RES, f"mask_1m_{site}.npy")).astype(bool)
    cc, n = cc_label(label, structure=P.S8)
    best = None
    for cid in range(1, n + 1):
        m = cc == cid
        pts = np.argwhere(P.find_endpoints(skeletonize(m)))
        if len(pts) < 2:
            continue
        s, e = P.merge_and_pick_pair(pts, 10.0)
        if best is None or np.hypot(*(s - e)) > np.hypot(*(best[0] - best[1])):
            best = (s, e)
    return P.snap_to_domain(best[0], dom), P.snap_to_domain(best[1], dom)


def dijkstra_on(rows, cols, data, N, s, e):
    """在已建好的状态图上换端点重跑（超源/超汇 16 条边重建）。"""
    ss, ee = 8 * N, 8 * N + 1
    d = np.arange(8, dtype=np.int64) * N
    r = np.concatenate([rows, np.full(8, ss, dtype=np.int64), d + int(e)])
    c = np.concatenate([cols, d + int(s), np.full(8, ee, dtype=np.int64)])
    w = np.concatenate([data, np.zeros(16)])
    G = csr_matrix((w, (r, c)), shape=(8 * N + 2, 8 * N + 2))
    dist, pred = dijkstra(G, directed=True, indices=ss, return_predecessors=True)
    if not np.isfinite(dist[ee]):
        return None
    path = np.zeros(N, dtype=bool)
    cur = ee
    while cur >= 0:
        if cur < 8 * N:
            path[cur % N] = True
        if cur == ss:
            break
        cur = int(pred[cur])
    return path


def clamp_snap(pt, dom):
    r = min(max(float(pt[0]), 0.0), dom.shape[0] - 1.0)
    c = min(max(float(pt[1]), 0.0), dom.shape[1] - 1.0)
    return P.snap_to_domain((r, c), dom)


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for site in P.SITES:
        label = np.load(os.path.join(P.RES, f"mask_1m_{site}.npy")).astype(bool)
        valid = np.load(os.path.join(P.DATA, f"{site}_1m.npz"))["valid"] == 1
        dom = valid.copy()
        tpi31 = np.load(os.path.join(P.DATA, f"p2_terrain_{site}.npy"))[7]
        cost, _ = P.build_cost(site, tpi31, dom, 0.05, "pct", "open")
        nr, nc, conf, _ = P.direction_field(site, P.DFS_GATE, smooth=5)
        (rows, cols, data), N = P.build_state_graph(cost, dom, nr, nc, conf,
                                                   lam_dir=1.5, lam_turn=0.2, turn_min=1)
        s0, e0 = base_endpoints(site, dom)
        H, W = dom.shape
        win = window_mask(valid)
        print(f"\n[{site}] 基准端点 {s0} / {e0}  域 {int(dom.sum()):,}px", flush=True)

        res = {}
        for sg in SIGMAS:
            ious = []
            for _ in range(REPS):
                if sg == 0:
                    s, e = s0, e0
                else:
                    ds = rng.normal(0, sg, 2)
                    de = rng.normal(0, sg, 2)
                    s = clamp_snap((s0[0] + ds[0], s0[1] + ds[1]), dom)
                    e = clamp_snap((e0[0] + de[0], e0[1] + de[1]), dom)
                path = dijkstra_on(rows, cols, data, N,
                                   int(s[0] * W + s[1]), int(e[0] * W + e[1]))
                if path is None:
                    ious.append(0.0)
                    continue
                pred = binary_dilation(path.reshape(H, W), iterations=3) & dom
                inter = int((pred & label & win).sum())
                union = int(((pred | label) & win).sum())
                ious.append(inter / union if union else 0.0)
            res[sg] = dict(mean=float(np.mean(ious)), std=float(np.std(ious)),
                           min=float(np.min(ious)), max=float(np.max(ious)))
            print(f"  σ={sg:4d}m  IoU={res[sg]['mean']:.4f} ± {res[sg]['std']:.4f}"
                  f"  [{res[sg]['min']:.4f}, {res[sg]['max']:.4f}]", flush=True)
        out[site] = dict(base_start=[float(v) for v in s0],
                         base_end=[float(v) for v in e0],
                         reps=REPS, by_sigma={str(k): v for k, v in res.items()})

    fp = os.path.join(P.RES, "path_endpoint_sensitivity.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(dict(cost="open", dir_lambda=1.5, turn_lambda=0.2, turn_min=1,
                       dilate_radius=3, seed=SEED, sites=out), f,
                  ensure_ascii=False, indent=2)
    print(f"\n保存: {fp}")


if __name__ == "__main__":
    main()
