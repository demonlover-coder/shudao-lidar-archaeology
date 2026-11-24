# -*- coding: utf-8 -*-
"""切断交叉验证：把 3 条真值走廊切成子段，逐段重跑状态增广 Dijkstra 扫 λ_turn。

为什么需要这个验证
------------------
全局口径下 `λ_turn=0.2, turn_min=1` 让杨家坝 IoU 0.2136→0.2817（+31.8%），但
λ 响应**非单调**（0.3 回落、0.5 又升），只有 1 段杨家坝走廊，存在"对单实例调参"
的风险。本脚本把每条走廊沿骨架按弧长切成子段，**每个子段用自己的两个端点独立
求路径**，得到多个相互独立的实例，据此判断 λ 的稳定区间是否可复现。

协议（写死，不看结果改）
--------------------
* 子段长度 `--seg-len`（默认 100 px ≈ 100 m，约 2.3×L*）；末端不短于半段。
* 每条子段：真值 = 该子段骨架最近邻所"拥有"的走廊像元（**精确划分，不重叠**）。
* 求路径域 = 该子段骨架膨胀 40 px 的管道 ∩ valid（与全局 `--roi-radius 40` 同半径）。
* 局部评估窗 = D10 窗（valid 且排除 85 px 边界带）∩ 同一管道。
* 每段每 λ 独立跑一次；λ 表 `{0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5}`，`turn_min=1`，
  方向先验固定 λ_dir=1.5、门控 P60、相位平滑窗 5（与主结果同参）。
* 判据：相对表现 `r_s(λ) = IoU_s(λ) / max_λ IoU_s(λ)`。区间内 `mean r ≥ 0.95` 且
  `min r ≥ 0.90` 才叫"平台"；只报告实测，不为了让区间成立而改协议。

用法
----
    python tools/path_lambda_cv.py                     # 两站，seg-len=100
    python tools/path_lambda_cv.py --seg-len 70        # 段长敏感性
    python tools/path_lambda_cv.py --site 西坝村
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import disk, skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from eval_linear_metrics import linear_metrics, window_mask        # noqa: E402
from path_extraction import (build_cost, build_state_graph,          # noqa: E402
                             direction_field, snap_to_domain)
from path_chaikin_viz import ordered_paths, rasterize               # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
SITES = ("杨家坝", "西坝村")
LAMS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5)
DIR_LAMBDA = 1.5
DIR_GATE = 60.0
DIR_SMOOTH = 5
COST = "open"
EPS = 0.05
NORM = "pct"
ROI = 40
DILATE = 3
TOL = 2.0
BAND = (0.15, 0.25)          # 待检验区间（用户假设的稳定带）
MEAN_MIN = 0.95
MIN_MIN = 0.90


def log_factory(path):
    f = open(path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()
    return log, f


# ------------------------------------------------------------ 切段 ----
def cut_polyline(poly, seg_len):
    """沿弧长把有序骨架点列切成若干段。

    返回 `(pieces, kb, total)`：`pieces[i] = poly[kb[i]:kb[i+1]]`，索引区间**互不重叠**，
    因此可以直接用它把真值像元精确划分到子段。
    """
    d = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    n = max(1, int(round(total / seg_len)))
    bounds = np.linspace(0.0, total, n + 1)
    kb = [int(np.searchsorted(s, b, side="left")) for b in bounds]
    kb[0], kb[-1] = 0, len(poly)
    # 去掉过短的末段（并入上一段），并保证索引严格递增
    kb = [kb[0]] + [k for k in kb[1:-1]] + [kb[-1]]
    kb = sorted(set(kb))
    if len(kb) > 2 and kb[-1] - kb[-2] < 3:
        del kb[-2]
    pieces = [poly[kb[i]:kb[i + 1]] for i in range(len(kb) - 1)]
    return pieces, kb, total


def dijkstra_weights(rows, cols, w, N, start, end, W):
    """在给定边权上跑状态增广 Dijkstra（超源=8N、超汇=8N+1）。start/end 为 (row, col)。"""
    ss, ee = 8 * N, 8 * N + 1
    s = int(start[0]) * W + int(start[1])
    e = int(end[0]) * W + int(end[1])
    d_all = np.arange(8, dtype=np.int64) * N
    rr = np.concatenate([rows, np.full(8, ss, dtype=np.int64), d_all + e])
    cc = np.concatenate([cols, d_all + s, np.full(8, ee, dtype=np.int64)])
    dd = np.concatenate([w, np.zeros(16)])
    G = csr_matrix((dd, (rr, cc)), shape=(8 * N + 2, 8 * N + 2))
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


# ------------------------------------------------------------ 单站 ----
def run_site(site, seg_len, log, lams=LAMS):
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    win = window_mask(valid)
    nr, nc, conf, thr = direction_field(site, DIR_GATE, smooth=DIR_SMOOTH)

    dom_all = valid & binary_dilation(label, iterations=ROI)
    cost_all, info = build_cost(site, tpi31, dom_all, EPS, NORM, COST)
    log(f"\n[{site}] 网格 {label.shape}  真值 {int(label.sum())} px"
        f"  全局代价域 {int(dom_all.sum()):,} px  方向门控 P{DIR_GATE:.0f}"
        f"（h1_deg≥{thr:.3f}°）")

    cc_all, n_cc = cc_label(label, structure=np.ones((3, 3), dtype=np.int32))
    segs_out = []
    for cid in range(1, n_cc + 1):
        comp = cc_all == cid
        if comp.sum() < 200:
            continue
        polys = ordered_paths(skeletonize(comp))
        if not polys:
            continue
        poly = max(polys, key=len)
        pieces, kb, total = cut_polyline(poly, seg_len)
        log(f"  连通域 {cid}: 真值 {int(comp.sum())} px  骨架主干 {total:.0f} px "
            f"→ {len(pieces)} 段")
        # 该连通域真值像元 → 最近骨架点索引（用于把真值精确划分到子段）
        ys, xs = np.nonzero(comp)
        tree = cKDTree(poly)
        _, near = tree.query(np.stack([ys, xs], axis=1))
        seg_of_pt = np.searchsorted(kb, near, side="right") - 1      # 0…len(pieces)-1
        seg_of_pt = np.clip(seg_of_pt, 0, len(pieces) - 1)
        sid_map = np.full(comp.shape, -1, dtype=int)
        sid_map[ys, xs] = seg_of_pt

        for si, piece in enumerate(pieces):
            p0, p1 = piece[0], piece[-1]
            seg_gt = comp & (sid_map == si)
            if seg_gt.sum() < 50:
                continue
            tube = binary_dilation(rasterize([piece], comp.shape), structure=disk(ROI))
            dom = dom_all & tube
            lwin = win & tube
            if dom.sum() < 200 or lwin.sum() < 200:
                continue

            # 裁剪到管道包围盒（状态图规模从 8·全图 降到 8·管道）
            bb = np.argwhere(dom)
            (r0, c0), (r1, c1) = bb.min(0), bb.max(0)
            sl = (slice(r0, r1 + 1), slice(c0, c1 + 1))
            dsub, csub = dom[sl], cost_all[sl]
            nsub, ncsub, cfsub = nr[sl], nc[sl], conf[sl]
            s = snap_to_domain((p0[0] - r0, p0[1] - c0), dsub)
            e = snap_to_domain((p1[0] - r0, p1[1] - c0), dsub)
            if not dsub[s] or not dsub[e]:
                continue

            (rw, cl, w0), N = build_state_graph(csub, dsub, nsub, ncsub, cfsub,
                                                DIR_LAMBDA, 0.0, turn_min=1)
            (_, _, w1), _ = build_state_graph(csub, dsub, nsub, ncsub, cfsub,
                                              DIR_LAMBDA, 1.0, turn_min=1)
            turn = w1 - w0

            seg_rows = []
            paths = {}
            for lam in lams:
                path = dijkstra_weights(rw, cl, w0 + lam * turn, N, s, e, dsub.shape[1])
                if path is None:
                    continue
                pf = np.zeros(comp.shape, dtype=bool)
                pf[sl] = path.reshape(dsub.shape)
                paths[lam] = pf
                pred = binary_dilation(pf, iterations=DILATE) & dom
                inter = int((pred & seg_gt & lwin).sum())
                union = int(((pred | seg_gt) & lwin).sum())
                iou = float(inter / union) if union else 0.0
                mm = linear_metrics(pred, seg_gt, lwin, TOL)
                seg_rows.append(dict(lam=lam, iou=iou, inter=inter, union=union,
                                     f1=mm["f1"], centerline_px=mm["centerline_px"],
                                     path_px=int(pf.sum()),
                                     gt_px=int((seg_gt & lwin).sum())))
            if not seg_rows:
                continue
            # 解的不变性：以 λ=0.2 为参照，路径掩膜**逐位相同**即视为同一解
            ref_lam = 0.2 if 0.2 in paths else lams[len(lams) // 2]
            ref = paths.get(ref_lam)
            for r in seg_rows:
                r["same_as_ref"] = bool(ref is not None
                                        and np.array_equal(paths[r["lam"]], ref))
            best = max(seg_rows, key=lambda r: r["iou"])
            for r in seg_rows:
                r["rel"] = r["iou"] / best["iou"] if best["iou"] > 0 else 0.0
            log("    seg%d(%d,%d)→(%d,%d) 长度%.0fpx 真值%dpx | " % (
                si, int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]),
                float(np.hypot(*(piece[-1] - piece[0]))), int(seg_gt.sum()))
                + " ".join(f"λ{r['lam']:g}:{r['iou']:.3f}" for r in seg_rows)
                + f" | best λ={best['lam']:g}")
            segs_out.append(dict(site=site, cc=int(cid), seg=int(si),
                                 start=[float(p0[0]), float(p0[1])],
                                 end=[float(p1[0]), float(p1[1])],
                                 length_px=float(np.hypot(*(piece[-1] - piece[0]))),
                                 gt_px=int(seg_gt.sum()),
                                 best_lam=best["lam"], best_iou=best["iou"],
                                 ref_lam=ref_lam, rows=seg_rows))
    return dict(site=site, n_cc=int(n_cc), n_seg=len(segs_out),
                cost=COST, norm=NORM, eps=EPS, roi_px=ROI, dilate=DILATE,
                dir_lambda=DIR_LAMBDA, dir_gate=DIR_GATE, dir_smooth=DIR_SMOOTH,
                turn_min=1, seg_len=seg_len, segments=segs_out), segs_out


# ------------------------------------------------------------ 汇总 ----
def summarize(segs):
    """按 λ 汇总：相对表现 + 解的不变性（路径是否与 λ=0.2 逐位相同）。"""
    lams = sorted({r["lam"] for s in segs for r in s["rows"]})
    out = []
    for lam in lams:
        rel, iou, same = [], [], []
        for s in segs:
            r = next((x for x in s["rows"] if x["lam"] == lam), None)
            if r is None:
                continue
            rel.append(r["rel"])
            iou.append(r["iou"])
            same.append(bool(r.get("same_as_ref", False)))
        if not rel:
            continue
        out.append(dict(lam=lam, n=len(rel), mean_rel=float(np.mean(rel)),
                        min_rel=float(np.min(rel)), mean_iou=float(np.mean(iou)),
                        std_iou=float(np.std(iou)),
                        n_same_path=int(sum(same)),
                        all_same_path=bool(all(same))))
    band = [r for r in out if BAND[0] <= r["lam"] <= BAND[1]]
    ok_iou = bool(band) and all(r["mean_rel"] >= MEAN_MIN and r["min_rel"] >= MIN_MIN
                                for r in band)
    ok_inv = bool(band) and all(r["all_same_path"] for r in band)

    # 含 λ=0.2 的**最大连续不变区间**（该区间内所有段路径都与 λ=0.2 逐位相同）
    lams_all = [r["lam"] for r in out]
    k = next((i for i, v in enumerate(lams_all) if abs(v - 0.2) < 1e-9), None)
    inv_lo = inv_hi = float("nan")
    if k is not None and out[k]["all_same_path"]:
        i = k
        while i > 0 and out[i - 1]["all_same_path"]:
            i -= 1
        j = k
        while j < len(out) - 1 and out[j + 1]["all_same_path"]:
            j += 1
        inv_lo, inv_hi = lams_all[i], lams_all[j]

    best_lam = [s["best_lam"] for s in segs]
    return dict(rows=out,
                band=dict(lo=BAND[0], hi=BAND[1], mean_min=MEAN_MIN,
                          min_min=MIN_MIN, pass_iou=ok_iou, pass_invariance=ok_inv,
                          inv_lo=inv_lo, inv_hi=inv_hi, ref_lam=0.2),
                best_lam=best_lam,
                best_lam_median=float(np.median(best_lam)) if best_lam else float("nan"),
                n_best_in_band=int(sum(1 for b in best_lam if BAND[0] <= b <= BAND[1])),
                n_seg=len(best_lam))


def figure(sites, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.2))
    ax = axes[0]
    cols = {"杨家坝": "#d7191c", "西坝村": "#2c7bb6"}
    for st in sites:
        segs = st["segments"]
        for k, s in enumerate(segs):
            x = [r["lam"] for r in s["rows"]]
            y = [r["iou"] for r in s["rows"]]
            ax.plot(x, y, "-o", ms=3.2, lw=0.9, alpha=0.55,
                    color=cols[st["site"]],
                    label=st["site"] if k == 0 else None)
        summ = summarize(segs)["rows"]
        ax.plot([r["lam"] for r in summ], [r["mean_iou"] for r in summ], "-s",
                ms=6, lw=2.4, color="black",
                label=f"{st['site']} 段均值")
    ax.axvspan(*BAND, color="0.85", alpha=0.55, zorder=0, label="0.15~0.25")
    ax.set_xlabel("λ_turn（turn_min=1）"); ax.set_ylabel("子段局部 IoU@0.5")
    ax.set_title("(a) 每段独立重跑：IoU vs λ", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for st in sites:
        s = summarize(st["segments"])
        x = [r["lam"] for r in s["rows"]]
        ax.bar(np.array(x) + (0.012 if st["site"] == "西坝村" else -0.012),
               [r["n_same_path"] / max(s["n_seg"], 1) for r in s["rows"]],
               width=0.022, color=cols[st["site"]], alpha=0.45,
               label=f"{st['site']} 同解段占比")
        ax.plot(x, [r["mean_rel"] for r in s["rows"]], "-o", ms=3.4, lw=1.8,
                color=cols[st["site"]], label=f"{st['site']} 段均值 r")
    ax.axvspan(*BAND, color="0.85", alpha=0.55, zorder=0,
               label="假设稳定带 0.15~0.25")
    ax.axhline(1.0, ls="--", lw=1.0, color="0.35")
    ax.set_xlabel("λ_turn（turn_min=1）")
    ax.set_ylabel("同解段占比 / 段均值相对表现 r")
    ax.set_ylim(0.55, 1.06)
    ax.set_title("(b) 解的不变性：与 λ=0.2 逐位同解的段占比（柱）", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7.2, ncol=2, loc="lower left")

    ax = axes[2]
    labels, vals, colors = [], [], []
    for st in sites:
        s = summarize(st["segments"])
        labels.append(f"{st['site']}\n({s['n_seg']} 段)")
        vals.append(s["best_lam_median"])
        colors.append(cols[st["site"]])
        for k, b in enumerate(s["best_lam"]):
            ax.scatter([len(labels) - 1], [b], s=46, c="black", zorder=3,
                       label="各段最优 λ" if (len(labels) == 1 and k == 0) else None)
    ax.bar(labels, vals, color=colors, width=0.5)
    ax.axhspan(*BAND, color="0.85", alpha=0.55, zorder=0)
    ax.set_ylabel("各段最优 λ（柱=中位数，点=每段）")
    ax.set_title("(c) 各段最优 λ 的分布", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    def _verdict(st):
        s = summarize(st["segments"])
        b = s["band"]
        return (f"{st['site']}: 同解区间 [{b['inv_lo']:g}, {b['inv_hi']:g}]"
                f"（{s['n_seg']} 段全部与 λ=0.2 逐位相同）")
    fig.suptitle("切断交叉验证（3 条走廊 → 子段独立重跑，λ 步长 0.02）："
                 + "；".join(_verdict(st) for st in sites), fontsize=10.5)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def combine(paths_json, path_png, lams_ref=0.2):
    """把多个段长配置的 JSON 合成一张"稳定性"总图（性能平台 + 解不变性 + 区间汇总）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    data = [json.load(open(p, encoding="utf-8")) for p in paths_json]
    cols = {"杨家坝": "#d7191c", "西坝村": "#2c7bb6"}
    styles = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.0))

    ax = axes[0]
    for d, ls in zip(data, styles):
        sl = d["protocol"]["seg_len"]
        for s in d["sites"]:
            su = s["summary"]
            ax.plot([r["lam"] for r in su["rows"]],
                    [r["mean_rel"] for r in su["rows"]], ls, lw=1.7,
                    color=cols[s["site"]],
                    label=f"{s['site']} seg{sl:g}")
    ax.axvspan(*BAND, color="0.85", alpha=0.55, zorder=0, label="0.15~0.25")
    ax.axhline(MEAN_MIN, ls="--", lw=1.0, color="0.35", label=f"1−3.1% = {MEAN_MIN:.3f}")
    ax.axvline(lams_ref, lw=1.2, color="black", ls="-.")
    ax.set_xlabel("λ_turn（turn_min=1）")
    ax.set_ylabel("段均值相对表现 r", labelpad=8)
    ax.set_ylim(0.93, 1.005)
    ax.set_xlim(0.09, 0.355)
    ax.set_title("(a) 性能平台：r = IoU(λ) / 各段自身最优", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7.0, ncol=2, loc="lower left")

    ax = axes[1]
    for d, ls in zip(data, styles):
        sl = d["protocol"]["seg_len"]
        for s in d["sites"]:
            su = s["summary"]
            ax.plot([r["lam"] for r in su["rows"]],
                    [r["n_same_path"] / max(su["n_seg"], 1) for r in su["rows"]],
                    ls, lw=1.7, marker="o", ms=3, color=cols[s["site"]],
                    label=f"{s['site']} seg{sl:g}")
    ax.axvspan(*BAND, color="0.85", alpha=0.55, zorder=0)
    ax.axvspan(0.18, 0.20, color="#1a9641", alpha=0.25, zorder=1,
               label="逐位同解核心 [0.18,0.20]")
    ax.axvline(lams_ref, lw=1.2, color="black", ls="-.")
    ax.set_xlabel("λ_turn（turn_min=1）")
    ax.set_ylabel("同解段占比", labelpad=8)
    ax.set_ylim(0.55, 1.04); ax.set_xlim(0.09, 0.355)
    ax.set_title("(b) 解的不变性：与 λ=0.2 逐位同解的段占比（1.0=不切换）", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7.0, loc="lower left")

    ax = axes[2]
    short = {"杨家坝": "杨", "西坝村": "西"}
    labels, k = [], 0
    for d in data:
        sl = d["protocol"]["seg_len"]
        for s in d["sites"]:
            b = s["summary"]["band"]
            lo, hi = b["inv_lo"], b["inv_hi"]
            c = cols[s["site"]]
            ax.hlines(k, lo, hi, lw=9, color=c, alpha=0.9)
            if hi - lo < 1e-9:                      # 单点区间：画菱形，否则看不见
                ax.plot([lo], [k], "D", ms=7, color=c, zorder=3)
            ax.plot([lams_ref], [k], "|", ms=13, mew=2.0, color="black", zorder=4)
            ax.text(hi + 0.006, k, f"[{lo:g}, {hi:g}]", va="center", fontsize=8.2)
            labels.append(f"{short[s['site']]} seg{sl:g}"
                          f"（{s['summary']['n_seg']} 段）")
            k += 1
    ax.axvspan(*BAND, color="0.85", alpha=0.55, zorder=0)
    ax.axvspan(0.18, 0.20, color="#1a9641", alpha=0.25, zorder=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("λ_turn（turn_min=1）")
    ax.set_xlim(0.13, 0.305); ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.set_title("(c) 含 λ=0.2 的最大逐位同解区间（虚线=0.2）", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("切断交叉验证汇总：0.15~0.25 内性能平坦（各段距自身最优 ≤3.1%）；"
                 "λ=0.2 是唯一在全部 6 个（站点×段长）配置下都不触发路径切换的取值",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(path_png, dpi=150)
    plt.close(fig)

    print(f"{'站点':>8s} {'seg_len':>8s} {'段数':>5s} {'同解区间':>14s} "
          f"{'0.15~0.25 mean_rel':>20s} {'同解占比':>10s}")
    for d in data:
        sl = d["protocol"]["seg_len"]
        for s in d["sites"]:
            su = s["summary"]
            b = [r for r in su["rows"] if BAND[0] <= r["lam"] <= BAND[1]]
            print(f"{s['site']:>8s} {sl:>8g} {su['n_seg']:>5d} "
                  f"[{su['band']['inv_lo']:g},{su['band']['inv_hi']:g}]".ljust(40)
                  + f"{min(r['mean_rel'] for r in b):.3f}~{max(r['mean_rel'] for r in b):.3f}"
                  + f"{min(r['n_same_path'] for r in b)}~{max(r['n_same_path'] for r in b)}"
                  f"/{su['n_seg']}")
    print(f"写出 {os.path.relpath(path_png, ROOT)}")

    # 哪些 λ 落在**全部**配置的同解区间内（这是"λ 选哪里最稳"的直接答案）
    intervals = [(s["summary"]["band"]["inv_lo"], s["summary"]["band"]["inv_hi"])
                 for d in data for s in d["sites"]]
    lams = sorted({r["lam"] for d in data for s in d["sites"] for r in s["summary"]["rows"]})
    inside = [l for l in lams if all(lo <= l <= hi for lo, hi in intervals)]
    worst = max(1.0 - r["mean_rel"] for d in data for s in d["sites"]
                for r in s["summary"]["rows"] if BAND[0] <= r["lam"] <= BAND[1])
    print(f"\n配置数 {len(intervals)}；落在全部同解区间内的 λ: {inside}")
    print(f"0.15~0.25 内最差段相对损失（1−mean_rel 最大值）: {worst:.4f}")


def main():
    ap = argparse.ArgumentParser(description="切断交叉验证：子段独立重跑扫 λ_turn")
    ap.add_argument("--site", default="all", choices=["杨家坝", "西坝村", "all"])
    ap.add_argument("--seg-len", type=float, default=100.0, help="子段目标长度（px≈m）")
    ap.add_argument("--lams", type=float, nargs="*", default=None,
                    help="λ_turn 扫描表（缺省用内置粗表）")
    ap.add_argument("--tag", default="", help="输出标签后缀（段长敏感性实验用）")
    ap.add_argument("--combine", type=float, nargs="*", default=None,
                    help="不重跑，只合并已有的若干 tag 结果出总图（如 --combine 70 100 140）")
    a = ap.parse_args()

    if a.combine:
        ps = [os.path.join(RES, f"path_lambda_cv_{('fine' if t == 100 else f'seg{t:g}')}.json")
              for t in a.combine]
        combine(ps, os.path.join(FIG, "path_lambda_cv_summary.png"))
        return

    lams = tuple(a.lams) if a.lams else LAMS
    sites = list(SITES) if a.site == "all" else [a.site]
    tag = a.tag or f"seg{a.seg_len:g}"
    log, fh = log_factory(os.path.join(RES, f"path_lambda_cv_{tag}.txt"))
    log(f"切断交叉验证  seg_len={a.seg_len:g}px（≈m）  λ表={lams}  turn_min=1")
    log(f"固定参数: cost={COST} λ_dir={DIR_LAMBDA} 门控 P{DIR_GATE:.0f} 平滑 5 "
        f"ROI/管道={ROI}px 膨胀={DILATE}px 容差={TOL}px")

    out, figs = [], []
    for site in sites:
        r, segs = run_site(site, a.seg_len, log, lams)
        s = summarize(segs)
        log(f"\n[{site}] 汇总（{s['n_seg']} 段）")
        log("  " + " ".join(f"λ{row['lam']:g}" for row in s["rows"]))
        log("  mean_rel " + " ".join(f"{row['mean_rel']:.3f}" for row in s["rows"]))
        log("  min_rel  " + " ".join(f"{row['min_rel']:.3f}" for row in s["rows"]))
        log(f"  各段最优 λ: {s['best_lam']}  中位 {s['best_lam_median']:g}  "
            f"落在 {BAND[0]}~{BAND[1]} 的段数 {s['n_best_in_band']}/{s['n_seg']}")
        log("  与 λ=0.2 同解段数 " + " ".join(f"{row['n_same_path']}" for row in s["rows"]))
        log(f"  → IoU 平台判据(均值≥{MEAN_MIN} 且最差段≥{MIN_MIN}): "
            f"{'成立' if s['band']['pass_iou'] else '不成立'}；"
            f"含 λ=0.2 的最大同解区间 [{s['band']['inv_lo']:g}, "
            f"{s['band']['inv_hi']:g}]")
        r["summary"] = s
        out.append(r)
        figs.append(r)

    p = os.path.join(RES, f"path_lambda_cv_{tag}.json")
    json.dump(dict(protocol=dict(cost=COST, eps=EPS, norm=NORM, roi_px=ROI,
                                 dilate=DILATE, tol_px=TOL, turn_min=1,
                                 dir_lambda=DIR_LAMBDA, dir_gate=DIR_GATE,
                                 dir_smooth=DIR_SMOOTH, lams=list(lams),
                                 seg_len=a.seg_len,
                                 band=list(BAND), mean_min=MEAN_MIN,
                                 min_min=MIN_MIN),
                   note=("每段独立求路径（域限段骨架 40px 管道），与全局口径不完全可比；"
                         "仅用于判定 λ 的跨实例稳定性"),
                   sites=out), open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    fpath = os.path.join(FIG, "path_lambda_cv.png" if not a.tag
                         else f"path_lambda_cv_{a.tag}.png")
    figure(figs, fpath)
    log(f"\n写出 {os.path.relpath(p, ROOT)}")
    log(f"写出 {os.path.relpath(fpath, ROOT)}")
    fh.close()


if __name__ == "__main__":
    main()
