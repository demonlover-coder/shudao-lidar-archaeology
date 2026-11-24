# -*- coding: utf-8 -*-
"""无监督双分支融合 v6：方向过滤端点 + 3px 严格门控带（放弃 AE 融合）。

针对 v5 的两条归因
------------------
1. **西坝村端点偏移**：`CSSC>P95 → dilate(5) → 最大连通域` 在膨胀后过度生长
   （100,932 px），测地直径从 (508,763) 拉到 (140,348)，路径中位偏移 16.9 px。
   v6 在膨胀与取径之间插入**方向过滤**：用 h1_phase 求该连通域的主走向，
   只保留局部走向与之夹角 < 30° 的像素，再骨架化取测地直径。
2. **门控带宽过粗**：σ=10 m 的高斯带半宽 14 px，把 IoU 上限压在 ~0.17
   （v5 杨家坝门控带 11,799 px 覆盖真值 71.4%，但带宽是真值的 5.8 倍）。
   v6 改为**线性衰减硬带** `W_topo = clip(1 − dist/3, 0, 1)`（`--gate-mode hard`
   可用 `dist ≤ 3` 的二值带）。
3. **放弃 AE**：v5 实测 AE 特征两站方向不一致（AUC 0.386 / 0.737），
   乘法门控在两站都拉低 IoU（0.0439<0.1181、0.0728<0.0822）。v6 只输出 W_topo
   过 Otsu 的掩膜，不再引入任何学习参数——全流程零训练、零标签。

方向过滤的实现细节
------------------
`direction_field` 返回的是**走廊法向** n̂ = (−cosθ, sinθ)，θ 为 h1 相位。法向是
轴向量（θ 与 θ+π 等价），故主走向用**倍角圆周均值**求：

    θ̄ = ½·angle( Σ w·exp(i·2θ) )，w = conf（仅置信像元参与，背景 φ 是噪声）

像元与主走向的夹角取倍角差的一半：`Δ = ½·|angle(exp(i·2(θ−θ̄))))| ∈ [0°, 90°]`。
比较法向与法向、走向与走向的夹角是同一件事，故这里直接用相位差。

严格约束
--------
`mask_1m_{site}.npy`（SHP 真值）只在最后算 IoU/F1/P/R 与诊断时读取；
端点、代价、方向、门控、阈值全链路零标签、零训练。

用法
----
    python tools/unsup_two_branch_v6.py
    python tools/unsup_two_branch_v6.py --gate-mode hard --gate-width 3
    python tools/unsup_two_branch_v6.py --angle-mode conf --angle-max 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation,
                           distance_transform_edt, gaussian_filter, label as cc_label)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.filters import threshold_otsu
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
def geodesic_diameter(skel):
    """骨架上测地最远两点（两次 Dijkstra 扫描）。返回 ((r,c),(r,c), 测地长度)。"""
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
    if not rows:
        return pts[0], pts[0], 0.0
    r = np.concatenate(rows); c = np.concatenate(cols)
    step = np.hypot(pts[r, 0] - pts[c, 0], pts[r, 1] - pts[c, 1])
    G = csr_matrix((step, (r, c)), shape=(len(pts), len(pts)))
    d0 = dijkstra(G, indices=0)
    i = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    d1 = dijkstra(G, indices=i)
    j = int(np.argmax(np.where(np.isfinite(d1), d1, -1)))
    return pts[i], pts[j], float(d1[j])


def main_axis(theta, conf, sel):
    """法向轴向量的倍角圆周均值（仅置信像元参与）。返回 (θ̄, 参与像元数)。"""
    ok = sel & np.isfinite(theta)
    w = ok & conf
    if int(w.sum()) < 30:
        w = ok
    z = np.exp(1j * 2.0 * theta[w])
    th0 = 0.5 * float(np.angle(z.mean()))
    return th0, int(w.sum())


def axis_diff(theta, th0):
    """与主走向的夹角（轴向，°）。Δ = ½·|angle(exp(i·2(θ−θ̄)))| ∈ [0,90]。"""
    d = np.angle(np.exp(1j * 2.0 * (theta - th0)))
    return np.degrees(np.abs(d) / 2.0)


def endpoints_from_cssc(site, valid, pct, r, angle_max, angle_mode, log):
    """CSSC≥P{pct} → dilate(r) → 方向过滤(<angle_max°) → 最大连通域 → 骨架测地直径。"""
    cssc = np.load(os.path.join(RES, f"cs_fields_{site}.npz"))["cssc"].astype(np.float64)
    v = cssc[valid]
    v = v[np.isfinite(v)]
    thr = float(np.percentile(v, pct))
    raw_b = (cssc >= thr) & valid & np.isfinite(cssc)
    if not raw_b.any():
        log(f"  [FAIL] CSSC ≥ P{pct} 为空")
        return None, dict(thr=thr)

    b = binary_dilation(raw_b, disk(r)) & valid
    lab, ncc = cc_label(b)
    sizes = np.bincount(lab.ravel())[1:]
    comp = lab == (int(np.argmax(sizes)) + 1)

    # ---- 方向过滤：主走向 vs 局部 h1 相位 ----
    nr, nc, conf, gate_thr = P.direction_field(site, P.DFS_GATE)
    theta = np.arctan2(nc, -nr)                    # 复原平滑后的 h1 相位
    th0, n_w = main_axis(theta, conf, comp)
    da = axis_diff(theta, th0)
    if angle_mode == "strict":
        keep = comp & (da < angle_max) & conf
    else:
        keep = comp & ((da < angle_max) | ~conf)
    keep &= valid
    if not keep.any():
        log("  [FAIL] 方向过滤后为空")
        return None, dict(thr=thr)

    lab2, ncc2 = cc_label(keep)
    sizes2 = np.bincount(lab2.ravel())[1:]
    comp2 = lab2 == (int(np.argmax(sizes2)) + 1)
    skel = skeletonize(comp2)
    p, q, glen = geodesic_diameter(skel)

    info = dict(thr=thr, dilate_r=r, angle_max=angle_max, angle_mode=angle_mode,
                h1_gate_thr=round(gate_thr, 1), main_phase_deg=round(np.degrees(th0), 1),
                main_n=(round(-float(np.cos(th0)), 3), round(float(np.sin(th0)), 3)),
                main_t=(round(float(np.sin(th0)), 3), round(float(np.cos(th0)), 3)),
                axis_samples=n_w,
                raw_px=int(raw_b.sum()), raw_cc=int(cc_label(raw_b)[1]),
                dil_px=int(b.sum()), dil_cc=int(ncc), dil_max_cc=int(comp.sum()),
                keeppx=int(keep.sum()), keep_frac=round(float(keep.sum() / max(comp.sum(), 1)), 3),
                keep_cc=int(ncc2), final_cc_px=int(comp2.sum()),
                skel_px=int(skel.sum()), geo_len_px=round(glen, 1),
                euc_len_px=round(float(np.hypot(p[0] - q[0], p[1] - q[1])), 1),
                ep0=[float(p[0]), float(p[1])], ep1=[float(q[0]), float(q[1])],
                ang_med=round(float(np.median(da[comp & conf])), 1))

    log(f"  CSSC≥P{pct}={thr:.0f}: {info['raw_px']:,}px/{info['raw_cc']} 连通域 → "
        f"dilate(r={r}) → {info['dil_px']:,}px/{ncc} 连通域，最大 {comp.sum():,}px")
    log(f"    主走向相位 {info['main_phase_deg']:+.1f}°（法向 {info['main_n']}，"
        f"{n_w:,} 个置信像元参与）")
    log(f"    方向过滤 |Δ|<{angle_max:g}°（{angle_mode}，域内夹角中位 "
        f"{info['ang_med']:.1f}°）→ 保留 {keep.sum():,}px"
        f"（{info['keep_frac'] * 100:.1f}%）/{ncc2} 连通域，最大 {comp2.sum():,}px")
    log(f"    骨架 {skel.sum():,}px，测地长度 {glen:.1f}px（直线 {info['euc_len_px']:.1f}px）")
    log(f"    端点 ({p[0]:.0f},{p[1]:.0f}) ↔ ({q[0]:.0f},{q[1]:.0f})")
    return (p, q), info


# ------------------------------------------------------------ 分支 1 ----
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


def branch_topo(site, valid, ep, gate_mode="linear", gate_width=3.0,
                dir_lambda=1.5, dir_gate=None, turn_lambda=0.2, turn_min=1, log=print):
    """openness 代价 + h1 方向惩罚 → 1px 路径 → 截断/线性门控带 W_topo。"""
    dom = valid.copy()
    H, W = dom.shape
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    cost, info = P.build_cost(site, tpi31, dom, 0.05, "pct", "open")
    nr, nc, conf, gate_thr = P.direction_field(
        site, P.DFS_GATE if dir_gate is None else dir_gate)
    log("  分支1 代价 openness " +
        "  ".join(f"{k}[{lo:.3f},{hi:.3f}]" for k, (lo, hi) in info.items()) +
        f"；λ_dir={dir_lambda} 方向门控 h1_deg≥{gate_thr:.1f}° "
        f"覆盖 valid 的 {conf[valid].mean() * 100:.1f}%")

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
    if gate_mode == "hard":
        wtopo = (dist_map <= gate_width).astype(np.float64)
    else:
        wtopo = np.clip(1.0 - dist_map / float(gate_width), 0.0, 1.0)
    wtopo = np.where(dom, wtopo, 0.0)
    log(f"  分支1 路径 {int(path.sum())}px；门控 {gate_mode}(w={gate_width:g}px) → "
        f"带内 {(wtopo > 0).sum():,}px，均值 {wtopo[dom].mean():.4f}")
    return wtopo, path


# ------------------------------------------------------------ 主 ----
def run_site(site, a, log):
    t0 = time.time()
    dem, hs, valid = load_terrain(site)
    log(f"\n[{site}] 网格 {hs.shape}  valid {int(valid.sum()):,}px")

    ep, ep_info = endpoints_from_cssc(site, valid, a.pct, a.r, a.angle_max,
                                     a.angle_mode, log)
    if ep is None:
        return None

    r_topo = branch_topo(site, valid, ep, gate_mode=a.gate_mode, gate_width=a.gate_width,
                         dir_lambda=a.dir_lambda, dir_gate=a.dir_gate,
                         turn_lambda=a.turn_lambda, turn_min=a.turn_min, log=log)
    if r_topo[0] is None:
        return None
    wtopo, path = r_topo

    # 融合后高斯平滑（保留 v2 起的 smoothing 约定，仅为抑制骨架锯齿）
    sm = gaussian_filter(wtopo, sigma=a.sigma) if a.sigma > 0 else wtopo
    sm = np.where(valid, sm, 0.0)

    thr = float(threshold_otsu(sm[valid]))
    mask = (sm >= thr) & valid
    if a.close > 0:
        mask = binary_closing(mask, iterations=a.close) & valid
    eff = "dist ≤ 3px（硬带）" if a.gate_mode == "hard" else \
        f"dist ≤ {(1.0 - thr) * a.gate_width:.2f}px（由 Otsu 反推）"
    log(f"  Otsu {thr:.4f}（valid 内）→ 掩膜 {int(mask.sum()):,}px；等效截断 {eff}")

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

    a_mask, a_top = auc(sm), auc(wtopo.astype(np.float64))
    dpath = dt[path]
    log(f"  全域 IoU={raw['iou']:.4f} F1={raw['f1']:.4f} P={raw['precision']:.4f} "
        f"R={raw['recall']:.4f} (TP {raw['tp']:,}/FP {raw['fp']:,}/FN {raw['fn']:,})")
    log(f"  D10  IoU={d10['iou']:.4f} F1={d10['f1']:.4f}")
    log(f"  诊断 路径→真值 中位 {np.median(dpath):.1f}px、"
        f"≤3px 占比 {(dpath <= 3).mean() * 100:.1f}%；AUC(门控) {a_top:.3f} / "
        f"AUC(平滑后) {a_mask:.3f}")

    os.makedirs(FIG, exist_ok=True)
    np.savez_compressed(os.path.join(RES, f"unsup_v6_fields_{site}.npz"),
                        wtopo=wtopo.astype(np.float32), smoothed=sm.astype(np.float32),
                        mask=mask.astype(np.uint8), path=path.astype(np.uint8))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 5.6), dpi=140)
    ax[0].imshow(np.where(valid, wtopo, np.nan), cmap="viridis")
    ax[0].set_title(f"W_topo（{a.gate_mode}, w={a.gate_width:g}px）", fontsize=10)
    ax[1].imshow(np.where(valid, mask.astype(float), np.nan), cmap="gray")
    ax[1].contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.2)
    ax[1].set_title(f"Otsu 掩膜（IoU={raw['iou']:.4f}）", fontsize=10)
    ax[2].imshow(np.where(valid, hs, np.nan), cmap="gray")
    ax[2].contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.2)
    ax[2].set_title("Hillshade + SHP 真值", fontsize=10)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{site} · v6 方向过滤端点 + {a.gate_width:g}px 严格门控", fontsize=11)
    fig.tight_layout()
    fp = os.path.join(FIG, f"unsup_v6_{site}.png")
    fig.savefig(fp); plt.close(fig)

    return dict(site=site, pct=a.pct, r=a.r, angle_max=a.angle_max,
                angle_mode=a.angle_mode, gate_mode=a.gate_mode,
                gate_width=a.gate_width, sigma=a.sigma, close=a.close,
                dir_lambda=a.dir_lambda, turn_lambda=a.turn_lambda, turn_min=a.turn_min,
                endpoints=ep_info, path_px=int(path.sum()), otsu=thr,
                path_med_to_truth_px=round(float(np.median(dpath)), 1),
                path_hit3_frac=round(float((dpath <= 3).mean()), 3),
                auc=dict(gate=a_top, smoothed=a_mask),
                raw=raw, d10=d10, seconds=round(time.time() - t0, 1),
                figure=os.path.relpath(fp, ROOT))


def main():
    p = argparse.ArgumentParser(description="无监督双分支 v6（方向过滤端点 + 严格窄门控，SHP 仅评分）")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--pct", type=float, default=95.0)
    p.add_argument("--r", type=int, default=5, help="CSSC 膨胀半径 px")
    p.add_argument("--angle-max", type=float, default=30.0, help="主走向夹角上限（°）")
    p.add_argument("--angle-mode", type=str, default="strict",
                   choices=["strict", "conf"], help="strict=仅置信像元参与判定")
    p.add_argument("--gate-mode", type=str, default="linear", choices=["linear", "hard"])
    p.add_argument("--gate-width", type=float, default=3.0, help="门控带宽（px，1px=1m）")
    p.add_argument("--sigma", type=float, default=1.0, help="Otsu 前高斯 σ（0=关）")
    p.add_argument("--close", type=int, default=0, help="掩膜闭运算次数（0=关）")
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--dir-gate", type=float, default=None)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "unsup_two_branch_v6.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"无监督双分支 v6 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"端点 CSSC≥P{a.pct:g} → dilate(r={a.r}) → 方向过滤 |Δ|<{a.angle_max:g}°"
        f"（{a.angle_mode}）→ 骨架测地直径")
    log(f"门控 {a.gate_mode}(w={a.gate_width:g}px)  σ={a.sigma:g}  close={a.close}  "
        f"λ_dir={a.dir_lambda} λ_turn={a.turn_lambda} m={a.turn_min}")
    log("无训练、无 AE；SHP 真值仅在最后算 IoU/F1/AUC 时读取")

    out = []
    for s in a.sites:
        r = run_site(s, a, log)
        if r:
            out.append(r)

    log("\n" + "=" * 78)
    log(f"{'站点':>8s} {'端点跨度':>9s} {'路径px':>7s} {'中位距':>7s} {'Otsu':>7s} "
        f"{'预测px':>9s} {'TP':>6s} {'FP':>8s} {'IoU':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    log("-" * 96)
    for r in out:
        d = r["raw"]
        log(f"{r['site']:>8s} {r['endpoints']['euc_len_px']:>9.1f} {r['path_px']:>7,d} "
            f"{r['path_med_to_truth_px']:>7.1f} {r['otsu']:>7.4f} {d['pred_px']:>9,d} "
            f"{d['tp']:>6,d} {d['fp']:>8,d} {d['iou']:>7.4f} {d['f1']:>7.4f} "
            f"{d['precision']:>7.4f} {d['recall']:>7.4f}")

    # 文件名带配置签名，避免不同配置互相覆盖
    jp = os.path.join(RES, f"unsup_two_branch_v6_{a.angle_mode}{a.angle_max:g}"
                          f"_{a.gate_mode}{a.gate_width:g}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="无监督双分支 v6：方向过滤端点 + 严格窄门控，无 AE 融合",
                       pct=a.pct, r=a.r, angle_max=a.angle_max, angle_mode=a.angle_mode,
                       gate_mode=a.gate_mode, gate_width=a.gate_width, sigma=a.sigma,
                       close=a.close, dir_lambda=a.dir_lambda,
                       turn_lambda=a.turn_lambda, turn_min=a.turn_min, sites=out), f,
                  ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
