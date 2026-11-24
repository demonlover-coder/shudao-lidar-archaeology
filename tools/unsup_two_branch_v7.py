# -*- coding: utf-8 -*-
"""无监督单支 v7：CSSC 加权骨架最长路径端点 → 3px 窄门控 → 固定 0.5 二值化。

相对 v6 的三条修正（每条都对应 v6 的实测证据）
--------------------------------------------
1. **彻底移除 h1_phase / conf 的逐像元空间过滤**。v6 实测：该过滤在杨家坝把
   39,686px 的连通域打碎成 66 块（最大仅 19,918px，测地直径 516→397px），在
   西坝村是反相关的——真值密度在 |Δ|=30~90° 区间为 7.99%/6.27%/6.37%，而
   10~30° 只有 2.30%/2.04%；且 `conf` 像元的真值密度仅 0.68%，非 conf 达 9.62%
   （杨家坝，14 倍差）。v7 只把 h1_phase 用于 Dijkstra 的**边权方向惩罚**。
2. **保留 3px 线性衰减窄门控**：`W_topo = clip(1 − dist/3, 0, 1)`。v6 消融显示
   杨家坝 IoU 0.1181→0.1594（+35%），是唯一稳定正收益的改动。
3. **重构端点：CSSC 加权的最长路径**（替代外接测地直径）。v6 的直径取法在两站
   都偏心（示意图上的"斜对角"而非走廊走向）。v7 在骨架上建图，做两次 Dijkstra
   扫描取直径，再从第二扫描距离场里取 K 个远端节点各跑一次，得到候选路径集合；
   在长度 ≥ `--len-tol`×最长 的候选里，选**路径平均 CSSC 最高**的一条，取其两端。

铁律
----
`mask_1m_{site}.npy`（SHP 真值）只在最后算 IoU/F1/P/R 与诊断时读取。
端点、代价、方向、门控、二值化全链路零标签、零训练。

流程
----
CSSC>P95 → binary_closing(3×3) → dilate(disk 5) → 最大连通域 → skeletonize
  → 骨架图 → CSSC 加权最长路径 → 两端 (start, end)
  → Dijkstra(openness 代价 + h1 方向惩罚 λ=1.5 + 转角惩罚)
  → dist_map → W_topo = clip(1−dist/3,0,1) → mask = W_topo > 0.5 → IoU/F1

用法
----
    python tools/unsup_two_branch_v7.py
    python tools/unsup_two_branch_v7.py --thr 0.5 --gate-width 3
    python tools/unsup_two_branch_v7.py --branch-k 12 --len-tol 0.9
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, distance_transform_edt,
                           label as cc_label)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import disk, skeletonize
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import path_extraction as P                                      # noqa: E402
from eval_linear_metrics import window_mask                       # noqa: E402
from fix_tpi31_auc import fill_nodata                             # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
SITES = ("杨家坝", "西坝村")
NB8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# ------------------------------------------------------------ 输入 ----
def load_terrain(site):
    """返回 (dem 已填洞, Hillshade(az=315°,alt=45°), valid)。"""
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    valid = d["valid"] == 1
    dem = fill_nodata(d["dem_mean"].astype(np.float64), valid)

    dzdx = np.gradient(dem, 1.0, axis=1)
    dzdy = -np.gradient(dem, 1.0, axis=0)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az, alt = np.radians(315.0), np.radians(45.0)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = (np.clip(hs, -1.0, 1.0) + 1.0) / 2.0
    return dem, np.where(valid, hs, 0.0), valid


# ------------------------------------------------------------ 骨架图 ----
def skel_graph(skel):
    """骨架像素 → 8 邻域图。返回 (G, pts, ids)。边权 = 1 或 √2。"""
    pts = np.argwhere(skel)
    H, W = skel.shape
    ids = np.full((H, W), -1, dtype=np.int64)
    ids[skel] = np.arange(len(pts))
    rows, cols = [], []
    for dy, dx in NB8:
        r0, r1 = max(0, -dy), min(H, H - dy)
        c0, c1 = max(0, -dx), min(W, W - dx)
        a = ids[r0:r1, c0:c1]
        b = ids[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
        m = (a >= 0) & (b >= 0)
        if not m.any():
            continue
        rows.append(a[m]); cols.append(b[m])
    r = np.concatenate(rows); c = np.concatenate(cols)
    w = np.hypot(pts[r, 0] - pts[c, 0], pts[r, 1] - pts[c, 1])
    return csr_matrix((w, (r, c)), shape=(len(pts), len(pts))), pts, ids


def _walk(pred, src, dst):
    """按 predecessor 回溯出 src→dst 的节点索引序列（含两端）。"""
    out, cur = [], int(dst)
    while cur != -9999 and cur >= 0:
        out.append(cur)
        if cur == src:
            break
        cur = int(pred[cur])
    return out[::-1]


def longest_paths(G, pts, k, len_tol, cssc_vals):
    """骨架上的候选最长路径：先两次扫描定直径，再从远端取 K 个起点补候选。

    返回 (best_path 节点索引序列, 诊断列表)。best = 长度 ≥ len_tol×max 的候选里
    平均 CSSC 最高者。
    """
    n = G.shape[0]
    d0 = dijkstra(G, indices=0)
    d0 = np.where(np.isfinite(d0), d0, -1.0)
    a = int(np.argmax(d0))

    d1, pred1 = dijkstra(G, indices=a, return_predecessors=True)
    d1 = np.where(np.isfinite(d1), d1, -1.0)
    b = int(np.argmax(d1))
    cands = [(a, b, d1[b], _walk(pred1, a, b))]

    # 从第二扫描距离场里取 K 个远端节点，各自再扫一次，补出"分支"候选
    far = np.argsort(d1)[::-1]
    picked = []
    for i in far:
        if d1[i] <= 0:
            break
        if all(abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1]) > 20 for j in picked):
            picked.append(int(i))
        if len(picked) >= k:
            break
    for c in picked:
        dd, pp = dijkstra(G, indices=c, return_predecessors=True)
        dd = np.where(np.isfinite(dd), dd, -1.0)
        e = int(np.argmax(dd))
        cands.append((c, e, dd[e], _walk(pp, c, e)))

    maxlen = max(x[2] for x in cands)
    rows = []
    for (s, e, L, idx) in cands:
        if L < len_tol * maxlen:
            continue
        v = cssc_vals[idx]
        rows.append(dict(len=float(L), mean_cssc=float(np.mean(v)),
                         ep0=[float(pts[s][0]), float(pts[s][1])],
                         ep1=[float(pts[e][0]), float(pts[e][1])], idx=idx))
    best = max(rows, key=lambda r: (r["mean_cssc"], r["len"]))
    return best, rows, float(maxlen)


def endpoints_from_cssc(site, valid, pct, r, k, len_tol, log):
    """CSSC>P{pct} → close → dilate(r) → 最大连通域 → 骨架 → CSSC 加权最长路径。"""
    cs = np.load(os.path.join(RES, f"cs_fields_{site}.npz"))
    cssc = cs["cssc"].astype(np.float64)
    v = cssc[valid]
    v = v[np.isfinite(v)]
    thr = float(np.percentile(v, pct))
    raw_b = (cssc >= thr) & valid & np.isfinite(cssc)
    if not raw_b.any():
        log(f"  [FAIL] CSSC ≥ P{pct} 为空")
        return None, dict(thr=thr)

    closed = binary_closing(raw_b, np.ones((3, 3), dtype=bool)) & valid
    b = binary_dilation(closed, disk(r)) & valid
    lab, ncc = cc_label(b)
    comp = lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)
    skel = skeletonize(comp)
    if skel.sum() < 10:
        log("  [FAIL] 骨架像元过少")
        return None, dict(thr=thr)

    G, pts, _ = skel_graph(skel)
    best, rows, maxlen = longest_paths(G, pts, k, len_tol, cssc[skel])
    p = np.array(best["ep0"]); q = np.array(best["ep1"])
    glen = best["len"]

    info = dict(thr=thr, dilate_r=r, branch_k=k, len_tol=len_tol,
                raw_px=int(raw_b.sum()), raw_cc=int(cc_label(raw_b)[1]),
                closed_px=int(closed.sum()), dil_px=int(b.sum()), dil_cc=int(ncc),
                comp_px=int(comp.sum()), skel_px=int(skel.sum()),
                max_len=round(maxlen, 1), n_candidates=len(rows),
                geo_len_px=round(glen, 1),
                euc_len_px=round(float(np.hypot(p[0] - q[0], p[1] - q[1])), 1),
                path_mean_cssc=round(best["mean_cssc"], 3),
                ep0=[float(p[0]), float(p[1])], ep1=[float(q[0]), float(q[1])],
                candidates=[{kk: vv for kk, vv in r_.items() if kk != "idx"} for r_ in rows])
    log(f"  CSSC≥P{pct}={thr:.0f}: {info['raw_px']:,}px/{info['raw_cc']} 连通域 → "
        f"close → dilate(r={r}) → {info['dil_px']:,}px/{ncc} 连通域，"
        f"最大 {comp.sum():,}px，骨架 {skel.sum():,}px")
    log(f"  骨架候选路径 {len(rows)} 条（阈值 {len_tol:g}×最长 {maxlen:.0f}px）：")
    for r_ in sorted(rows, key=lambda x: -x["mean_cssc"])[:5]:
        log(f"    长度 {r_['len']:7.1f}px  平均CSSC {r_['mean_cssc']:6.3f}  "
            f"端点 ({r_['ep0'][0]:.0f},{r_['ep0'][1]:.0f})↔"
            f"({r_['ep1'][0]:.0f},{r_['ep1'][1]:.0f})")
    log(f"  选定：长度 {glen:.1f}px（直线 {info['euc_len_px']:.1f}px）、"
        f"平均 CSSC {best['mean_cssc']:.3f}；端点 ({p[0]:.0f},{p[1]:.0f}) ↔ "
        f"({q[0]:.0f},{q[1]:.0f})")
    return (p, q), info


# ------------------------------------------------------------ 寻路 ----
def dijkstra_on(rows, cols, data, N, s, e):
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


def branch_topo(site, valid, ep, gate_width=3.0, dir_lambda=1.5, dir_gate=None,
                turn_lambda=0.2, turn_min=1, log=print):
    """openness 代价 + h1 方向惩罚 → 1px 路径 → 线性窄门控 W_topo。"""
    dom = valid.copy()
    H, W = dom.shape
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    cost, info = P.build_cost(site, tpi31, dom, 0.05, "pct", "open")
    nr, nc, conf, gate_thr = P.direction_field(
        site, P.DFS_GATE if dir_gate is None else dir_gate)
    log("  分支1 openness " +
        "  ".join(f"{k}[{lo:.3f},{hi:.3f}]" for k, (lo, hi) in info.items()) +
        f"；λ_dir={dir_lambda}（仅边权；conf 覆盖 valid "
        f"{conf[valid].mean() * 100:.1f}%，h1_deg≥{gate_thr:.1f}°）")

    (rows, cols, data), N = P.build_state_graph(cost, dom, nr, nc, conf,
                                                lam_dir=dir_lambda, lam_turn=turn_lambda,
                                                turn_min=turn_min)
    s = int(round(ep[0][0]) * W + round(ep[0][1]))
    e = int(round(ep[1][0]) * W + round(ep[1][1]))
    path = dijkstra_on(rows, cols, data, N, s, e)
    if path is None:
        log("  [FAIL] 分支1 无连通路径")
        return None, None
    path = path.reshape(H, W)

    dist_map = distance_transform_edt(~path)
    wtopo = np.clip(1.0 - dist_map / float(gate_width), 0.0, 1.0)
    wtopo = np.where(dom, wtopo, 0.0)
    log(f"  分支1 路径 {int(path.sum())}px；W_topo=clip(1−d/{gate_width:g})，"
        f"带内 {(wtopo > 0).sum():,}px")
    return wtopo, path


# ------------------------------------------------------------ 主 ----
def run_site(site, a, log):
    t0 = time.time()
    dem, hs, valid = load_terrain(site)
    log(f"\n[{site}] 网格 {hs.shape}  valid {int(valid.sum()):,}px")

    ep, ep_info = endpoints_from_cssc(site, valid, a.pct, a.r, a.branch_k,
                                     a.len_tol, log)
    if ep is None:
        return None

    r_topo = branch_topo(site, valid, ep, gate_width=a.gate_width,
                         dir_lambda=a.dir_lambda, dir_gate=a.dir_gate,
                         turn_lambda=a.turn_lambda, turn_min=a.turn_min, log=log)
    if r_topo[0] is None:
        return None
    wtopo, path = r_topo

    mask = (wtopo > a.thr) & valid                     # 固定阈值，不走 Otsu
    log(f"  掩膜 W_topo>{a.thr:g} → {int(mask.sum()):,}px"
        f"（等效 dist < {(1.0 - a.thr) * a.gate_width:.2f}px）")

    # ---- 以下才读真值，只评分 ----
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    win = window_mask(valid)

    def sc(m, sel):
        inter = int((m & label & sel).sum())
        union = int(((m | label) & sel).sum())
        tp, fp = inter, int((m & ~label & sel).sum())
        fn = int((~m & label & sel).sum())
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return dict(iou=float(inter / union) if union else 0.0,
                    f1=float(2 * pr * rc / max(pr + rc, 1e-9)),
                    precision=float(pr), recall=float(rc), tp=tp, fp=fp, fn=fn,
                    pred_px=int((m & sel).sum()), truth_px=int((label & sel).sum()))

    raw, d10 = sc(mask, valid), sc(mask, valid & win)

    sk = skeletonize(label)
    dt = distance_transform_edt(~label)
    neg = (dt >= 45.0) & (dt <= 55.0) & valid

    def auc(x):
        p = x[sk & valid]; q = x[neg]
        p, q = p[np.isfinite(p)], q[np.isfinite(q)]
        if p.size < 10 or q.size < 10:
            return float("nan")
        return float(roc_auc_score(np.r_[np.ones(p.size), np.zeros(q.size)], np.r_[p, q]))

    a_top = auc(wtopo)
    dpath = dt[path]
    log(f"  全域 IoU={raw['iou']:.4f} F1={raw['f1']:.4f} P={raw['precision']:.4f} "
        f"R={raw['recall']:.4f} (TP {raw['tp']:,}/FP {raw['fp']:,}/FN {raw['fn']:,})")
    log(f"  D10  IoU={d10['iou']:.4f} F1={d10['f1']:.4f}")
    log(f"  诊断 路径→真值 中位 {np.median(dpath):.1f}px、≤3px 占比 "
        f"{(dpath <= 3).mean() * 100:.1f}%；AUC(W_topo) {a_top:.3f}")

    os.makedirs(FIG, exist_ok=True)
    np.savez_compressed(os.path.join(RES, f"unsup_v7_fields_{site}.npz"),
                        wtopo=wtopo.astype(np.float32), mask=mask.astype(np.uint8),
                        path=path.astype(np.uint8))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 5.6), dpi=140)
    ax[0].imshow(np.where(valid, wtopo, np.nan), cmap="viridis")
    ax[0].set_title(f"W_topo（clip(1−d/{a.gate_width:g})）", fontsize=10)
    ax[1].imshow(np.where(valid, mask.astype(float), np.nan), cmap="gray")
    ax[1].contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.2)
    ax[1].set_title(f"掩膜 W_topo>{a.thr:g}（IoU={raw['iou']:.4f}）", fontsize=10)
    ax[2].imshow(np.where(valid, hs, np.nan), cmap="gray")
    ax[2].contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.2)
    ax[2].set_title("Hillshade + SHP 真值", fontsize=10)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{site} · v7 CSSC 加权最长路径端点 + {a.gate_width:g}px 窄门控", fontsize=11)
    fig.tight_layout()
    fp = os.path.join(FIG, f"unsup_v7_{site}.png")
    fig.savefig(fp); plt.close(fig)

    return dict(site=site, pct=a.pct, r=a.r, branch_k=a.branch_k, len_tol=a.len_tol,
                gate_width=a.gate_width, thr=a.thr, dir_lambda=a.dir_lambda,
                turn_lambda=a.turn_lambda, turn_min=a.turn_min,
                endpoints=ep_info, path_px=int(path.sum()),
                path_med_to_truth_px=round(float(np.median(dpath)), 1),
                path_hit3_frac=round(float((dpath <= 3).mean()), 3),
                auc=a_top, raw=raw, d10=d10,
                seconds=round(time.time() - t0, 1), figure=os.path.relpath(fp, ROOT))


def main():
    p = argparse.ArgumentParser(description="无监督 v7（CSSC 加权最长路径端点 + 窄门控 + 固定阈值）")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--pct", type=float, default=95.0)
    p.add_argument("--r", type=int, default=5, help="CSSC 膨胀半径 px")
    p.add_argument("--branch-k", type=int, default=8, help="补候选路径的远端起点数")
    p.add_argument("--len-tol", type=float, default=0.85, help="候选长度下限（×最长）")
    p.add_argument("--gate-width", type=float, default=3.0, help="门控带宽 px（1px=1m）")
    p.add_argument("--thr", type=float, default=0.5, help="W_topo 二值化阈值")
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--dir-gate", type=float, default=None)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "unsup_two_branch_v7.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"无监督 v7 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"端点 CSSC≥P{a.pct:g} → close → dilate(r={a.r}) → 骨架 → CSSC 加权最长路径"
        f"（k={a.branch_k}, len_tol={a.len_tol:g}）")
    log(f"门控 W=clip(1−d/{a.gate_width:g})，掩膜 W>{a.thr:g}；"
        f"λ_dir={a.dir_lambda} λ_turn={a.turn_lambda} m={a.turn_min}")
    log("无方向空间过滤、无 conf 筛选、无训练；SHP 真值仅在最后算指标")

    out = []
    for s in a.sites:
        r = run_site(s, a, log)
        if r:
            out.append(r)

    log("\n" + "=" * 78)
    log(f"{'站点':>8s} {'路径px':>7s} {'平均CSSC':>9s} {'中位距':>7s} {'预测px':>8s} "
        f"{'TP':>6s} {'FP':>7s} {'IoU':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    log("-" * 90)
    for r in out:
        d = r["raw"]
        log(f"{r['site']:>8s} {r['path_px']:>7,d} {r['endpoints']['path_mean_cssc']:>9.3f} "
            f"{r['path_med_to_truth_px']:>7.1f} {d['pred_px']:>8,d} {d['tp']:>6,d} "
            f"{d['fp']:>7,d} {d['iou']:>7.4f} {d['f1']:>7.4f} "
            f"{d['precision']:>7.4f} {d['recall']:>7.4f}")

    jp = os.path.join(RES, f"unsup_two_branch_v7_p{a.pct:g}_r{a.r}"
                          f"_w{a.gate_width:g}_t{a.thr:g}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="无监督 v7：CSSC 加权骨架最长路径端点 + 3px 线性窄门控 + 固定阈值",
                       pct=a.pct, r=a.r, branch_k=a.branch_k, len_tol=a.len_tol,
                       gate_width=a.gate_width, thr=a.thr, dir_lambda=a.dir_lambda,
                       turn_lambda=a.turn_lambda, turn_min=a.turn_min, sites=out), f,
                  ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
