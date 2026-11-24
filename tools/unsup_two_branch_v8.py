# -*- coding: utf-8 -*-
"""无监督 v8（最终版）：面积自适应 CSSC 阈值 + 3px 窄门控 + 0.33 二值化。

v7 的核心否证
------------
百分位阈值不可迁移：同一个 P99 在杨家坝把连通域压到 6,853 px（骨架 575 px，太短，
IoU 0.0063），在西坝村却压到 13,791 px（接近真值 5,859 px，IoU 0.0296 翻倍）。
两站真值都只占 valid 的 2.0~2.3%，所以真正稳定的量是**连通域面积**，不是百分位。

v8 的修正
--------
1. **面积自适应阈值**：从 P95 起按 `--step` 递增到 P99.9，每次 `CSSC ≥ p → dilate(disk r)
   → 最大连通域`，取第一个满足 `cc_area ≤ valid_area × --area-frac`（默认 5%）的档位。
   这样阈值由几何量决定，再交给站点自己去落位。
2. **端点 = 骨架双次扫描最远两点**（v7 已证明 CSSC 加权选路是空操作：9 条候选的
   平均 CSSC 差 1%，端点全落在同一处）。
3. **门控阈值 0.33**（等效 2 px 带宽，v6/v7 实测的最优平衡点：0.5 阈值的 1.5 px
   带宽精度最高但召回偏低，Otsu 等效 2.27 px 在杨家坝给出 0.1594 的最高 IoU）。

铁律
----
`mask_1m_{site}.npy`（SHP 真值）只在最后算 IoU/F1/P/R 与诊断时读取；
端点、代价、方向、门控、二值化全链路零标签、零训练。h1_phase 只做 Dijkstra 边权
方向惩罚，不做任何逐像元空间过滤（v6 实测该过滤会断裂连通域且在西坝村反相关）。

用法
----
    python tools/unsup_two_branch_v8.py
    python tools/unsup_two_branch_v8.py --area-frac 0.04 --step 0.5
    python tools/unsup_two_branch_v8.py --thr 0.33 --gate-width 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import (binary_dilation, distance_transform_edt, label as cc_label)
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


# ------------------------------------------------------------ 端点 ----
def skel_graph(skel):
    """骨架像素 → 8 邻域图（边权 1 或 √2）。返回 (G, pts)。"""
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
    return csr_matrix((w, (r, c)), shape=(len(pts), len(pts))), pts


def diameter_pair(skel):
    """骨架双次扫描（等价 BFS 找树直径）：首扫取最远点，再扫取对径点。"""
    G, pts = skel_graph(skel)
    d0 = dijkstra(G, indices=0)
    d0 = np.where(np.isfinite(d0), d0, -1.0)
    i = int(np.argmax(d0))
    d1 = dijkstra(G, indices=i)
    d1 = np.where(np.isfinite(d1), d1, -1.0)
    j = int(np.argmax(d1))
    return pts[i], pts[j], float(d1[j])


def adaptive_endpoints(site, valid, r, area_frac, step, log):
    """CSSC 百分位自适应：从 P95 递增，直到最大连通域面积 ≤ valid×area_frac。"""
    cssc = np.load(os.path.join(RES, f"cs_fields_{site}.npz"))["cssc"].astype(np.float64)
    v = cssc[valid]
    v = v[np.isfinite(v)]
    area_valid = int(valid.sum())
    target_px = area_frac * area_valid

    pcts = list(np.arange(95.0, 99.9 + 1e-9, step))
    if not any(abs(p - 99.9) < 1e-6 for p in pcts):
        pcts.append(99.9)

    sweep, chosen = [], None
    for pct in pcts:
        thr = float(np.percentile(v, pct))
        raw_b = (cssc >= thr) & valid & np.isfinite(cssc)
        if not raw_b.any():
            continue
        b = binary_dilation(raw_b, disk(r)) & valid
        lab, ncc = cc_label(b)
        comp = lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)
        area = int(comp.sum())
        sweep.append(dict(pct=float(pct), thr=thr, raw_px=int(raw_b.sum()),
                          cc_px=area, cc_frac=round(area / area_valid, 4)))
        if chosen is None and area <= target_px:
            chosen = (pct, thr, raw_b, comp, area)

    log(f"  面积自适应扫描（目标 ≤ {area_frac * 100:.1f}% × valid {area_valid:,} = "
        f"{target_px:,.0f}px）：")
    for s in sweep:
        flag = "  ← 选定" if chosen and abs(s["pct"] - chosen[0]) < 1e-9 else ""
        log(f"    P{s['pct']:4.1f} thr={s['thr']:.0f}  高分区 {s['raw_px']:6,d}px  "
            f"最大连通域 {s['cc_px']:6,d}px（{s['cc_frac'] * 100:5.1f}% of valid）{flag}")
    if chosen is None:
        pct, thr, raw_b, comp, area = sweep[-1]["pct"], sweep[-1]["thr"], None, None, \
            sweep[-1]["cc_px"]
        # 用最后一个档位重算连通域
        raw_b = (cssc >= thr) & valid & np.isfinite(cssc)
        b = binary_dilation(raw_b, disk(r)) & valid
        lab, _ = cc_label(b)
        comp = lab == (int(np.argmax(np.bincount(lab.ravel(), minlength=lab.max() + 1)[1:])) + 1)
        area = int(comp.sum())
        log(f"  [WARN] 所有档位均超目标，取最严档 P{pct:g}（{area:,}px）")
    else:
        pct, thr, raw_b, comp, area = chosen
        log(f"  [选定] P{pct:g}（thr={thr:.0f}）：连通域 {area:,}px ≤ 目标 {target_px:,.0f}px")

    skel = skeletonize(comp)
    if skel.sum() < 10:
        log("  [FAIL] 骨架像元过少")
        return None, dict(pct=float(pct), thr=thr, cc_px=area, sweep=sweep)
    p, q, glen = diameter_pair(skel)

    info = dict(pct=float(pct), thr=thr, dilate_r=r, area_frac=area_frac, step=step,
                valid_px=area_valid, cc_px=area,
                cc_frac=round(area / area_valid, 4), skel_px=int(skel.sum()),
                geo_len_px=round(glen, 1),
                euc_len_px=round(float(np.hypot(p[0] - q[0], p[1] - q[1])), 1),
                ep0=[float(p[0]), float(p[1])], ep1=[float(q[0]), float(q[1])],
                sweep=sweep)
    log(f"  选定 P{pct:g}：连通域 {area:,}px（{info['cc_frac'] * 100:.1f}% of valid），"
        f"骨架 {skel.sum():,}px，测地长度 {glen:.1f}px（直线 {info['euc_len_px']:.1f}px）")
    log(f"  端点 ({p[0]:.0f},{p[1]:.0f}) ↔ ({q[0]:.0f},{q[1]:.0f})")
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
    """openness 代价 + h1 方向惩罚（仅边权）→ 1px 路径 → 线性窄门控 W_topo。"""
    dom = valid.copy()
    H, W = dom.shape
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    cost, info = P.build_cost(site, tpi31, dom, 0.05, "pct", "open")
    nr, nc, conf, gate_thr = P.direction_field(
        site, P.DFS_GATE if dir_gate is None else dir_gate)
    log("  Dijkstra openness " +
        "  ".join(f"{k}[{lo:.3f},{hi:.3f}]" for k, (lo, hi) in info.items()) +
        f"；λ_dir={dir_lambda} 仅作边权（conf 覆盖 valid "
        f"{conf[valid].mean() * 100:.1f}%，h1_deg≥{gate_thr:.1f}°）")

    (rows, cols, data), N = P.build_state_graph(cost, dom, nr, nc, conf,
                                                lam_dir=dir_lambda, lam_turn=turn_lambda,
                                                turn_min=turn_min)
    s = int(round(ep[0][0]) * W + round(ep[0][1]))
    e = int(round(ep[1][0]) * W + round(ep[1][1]))
    path = dijkstra_on(rows, cols, data, N, s, e)
    if path is None:
        log("  [FAIL] 无连通路径")
        return None, None
    path = path.reshape(H, W)

    dist_map = distance_transform_edt(~path)
    wtopo = np.clip(1.0 - dist_map / float(gate_width), 0.0, 1.0)
    wtopo = np.where(dom, wtopo, 0.0)
    log(f"  路径 {int(path.sum())}px；W_topo=clip(1−d/{gate_width:g})，"
        f"带内 {(wtopo > 0).sum():,}px")
    return wtopo, path


# ------------------------------------------------------------ 主 ----
def run_site(site, a, log):
    t0 = time.time()
    dem, hs, valid = load_terrain(site)
    log(f"\n[{site}] 网格 {hs.shape}  valid {int(valid.sum()):,}px")

    ep, ep_info = adaptive_endpoints(site, valid, a.r, a.area_frac, a.step, log)
    if ep is None:
        return None

    r_topo = branch_topo(site, valid, ep, gate_width=a.gate_width, dir_lambda=a.dir_lambda,
                         dir_gate=a.dir_gate, turn_lambda=a.turn_lambda,
                         turn_min=a.turn_min, log=log)
    if r_topo[0] is None:
        return None
    wtopo, path = r_topo

    mask = (wtopo > a.thr) & valid
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
    np.savez_compressed(os.path.join(RES, f"unsup_v8_fields_{site}.npz"),
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
    ax[1].set_title(f"掩膜 W>{a.thr:g}（IoU={raw['iou']:.4f}）", fontsize=10)
    ax[2].imshow(np.where(valid, hs, np.nan), cmap="gray")
    ax[2].contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.2)
    ax[2].set_title("Hillshade + SHP 真值", fontsize=10)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{site} · v8 面积自适应 P{ep_info['pct']:g} 端点 + 3px 门控 + 阈值 {a.thr:g}",
                 fontsize=11)
    fig.tight_layout()
    fp = os.path.join(FIG, f"unsup_v8_{site}.png")
    fig.savefig(fp); plt.close(fig)

    return dict(site=site, area_frac=a.area_frac, step=a.step, r=a.r,
                gate_width=a.gate_width, thr=a.thr, dir_lambda=a.dir_lambda,
                turn_lambda=a.turn_lambda, turn_min=a.turn_min,
                endpoints=ep_info, path_px=int(path.sum()),
                path_med_to_truth_px=round(float(np.median(dpath)), 1),
                path_hit3_frac=round(float((dpath <= 3).mean()), 3),
                auc=a_top, raw=raw, d10=d10,
                seconds=round(time.time() - t0, 1), figure=os.path.relpath(fp, ROOT))


def main():
    p = argparse.ArgumentParser(description="无监督 v8（面积自适应端点 + 3px 门控 + 0.33）")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--area-frac", type=float, default=0.05, help="连通域面积上限（× valid）")
    p.add_argument("--step", type=float, default=1.0, help="百分位搜索步长")
    p.add_argument("--r", type=int, default=5, help="CSSC 膨胀半径 px")
    p.add_argument("--gate-width", type=float, default=3.0, help="门控带宽 px（1px=1m）")
    p.add_argument("--thr", type=float, default=0.33, help="W_topo 二值化阈值（0.33≈2px）")
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--dir-gate", type=float, default=None)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "unsup_two_branch_v8.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"无监督 v8 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"端点 面积自适应：P95→P99.9 step {a.step:g}，停于 cc ≤ {a.area_frac * 100:g}% valid；"
        f"dilate(r={a.r})；骨架双次扫描")
    log(f"门控 W=clip(1−d/{a.gate_width:g})，掩膜 W>{a.thr:g}"
        f"（≈{(1 - a.thr) * a.gate_width:.2f}px）；λ_dir={a.dir_lambda} "
        f"λ_turn={a.turn_lambda} m={a.turn_min}")
    log("无方向空间过滤、无 conf 筛选、无训练；SHP 真值仅在最后算指标")

    out = []
    for s in a.sites:
        r = run_site(s, a, log)
        if r:
            out.append(r)

    log("\n" + "=" * 78)
    log(f"{'站点':>8s} {'P档':>6s} {'连通域px':>9s} {'路径px':>7s} {'中位距':>7s} "
        f"{'预测px':>8s} {'TP':>6s} {'FP':>7s} {'IoU':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    log("-" * 100)
    for r in out:
        d = r["raw"]
        log(f"{r['site']:>8s} {r['endpoints']['pct']:>6.1f} {r['endpoints']['cc_px']:>9,d} "
            f"{r['path_px']:>7,d} {r['path_med_to_truth_px']:>7.1f} {d['pred_px']:>8,d} "
            f"{d['tp']:>6,d} {d['fp']:>7,d} {d['iou']:>7.4f} {d['f1']:>7.4f} "
            f"{d['precision']:>7.4f} {d['recall']:>7.4f}")

    jp = os.path.join(RES, f"unsup_two_branch_v8_a{a.area_frac:g}_s{a.step:g}"
                          f"_w{a.gate_width:g}_t{a.thr:g}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="无监督 v8：面积自适应 CSSC 端点 + 3px 线性窄门控 + 0.33 二值化",
                       area_frac=a.area_frac, step=a.step, r=a.r,
                       gate_width=a.gate_width, thr=a.thr, dir_lambda=a.dir_lambda,
                       turn_lambda=a.turn_lambda, turn_min=a.turn_min, sites=out), f,
                  ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
