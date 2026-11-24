# -*- coding: utf-8 -*-
"""无监督双分支融合 v2：CSASC 端点 + 管状/不对称视觉分支 → 乘法门控融合 → Otsu。

与 v1 的差别（按上一轮否证结论逐条修正）
----------------------------------------
v1 失败的三条归因：① 两分支量级失衡 10~18×，相加退化为"只用视觉分支"；
② z-score 后更差（拓扑尖峰主导 Otsu）；③ 两分支都无走廊选择性——分支 1 的上限
被"路线级外部端点"锁死在 IoU 0.0000/0.0065，分支 2 的 ResNet18 在 1 m Hillshade
上只是低频亮度统计（ImageNet 分布外）。v2 对应改三点：

1. **端点不再来自外部文件**（SHP 与第三方起止点都不参与）。改从 CSASC 高分连通域
   自取：`CSASC > 0.85` → 最大连通域 → 最远两点 → Dijkstra 起终点。
2. **视觉分支放弃 ResNet18**。改为 CSASC（横断面不对称特征）
   或 `--visual frangi` 的多尺度 Frangi 管状响应。
3. **融合改为乘法门控**：`F_fused = F_topo · (1 + F_visual)`，融合后高斯平滑。
   稀疏的拓扑先验充当"闸门"，视觉分支只能放大闸门内的响应，无法在闸门外凭空产生
   正响应——正是压假阳性的机制。
4. 融合前两分支在 valid 内严格 min-max 到 [0,1]（并记录原始范围）。

CSASC 定义（本脚本自算，`cs_fields_*.npz` 中无此字段）
------------------------------------------------------
沿走廊法向 `n̂`（`path_extraction.direction_field` 的 h1 相位，与寻路同源）以 1 m
步长取 ±15.5 m 剖面，去均值后求两侧平均坡度的**幅值不对称**：

    CSASC = | mean|∂z/∂s|_{s<−2m} − mean|∂z/∂s|_{s>+2m} |
            / ( 同两项之和 + ε )                       ∈ [0,1]

均匀斜坡两侧坡度幅值相等 → 0；人工切坡（一侧陡坎、一侧平缓平台）→ 高。
实测走廊选择性（骨架 vs 45~55 m 带 AUC）杨家坝 0.600 / 西坝村 0.654，
弱于已有 CSSC（0.721/0.889），但满足"横断面不对称、无量纲、落在 [0,1]"。

严格约束
--------
* `mask_1m_{site}.npy`（SHP 真值）**只在最后算 IoU/F1 时读取**；端点、代价、融合、
  阈值全链路零标签。
* 无任何训练：Dijkstra 是纯图搜索；Frangi / CSASC 都是确定性算子。

用法
----
    python tools/unsup_two_branch_v2.py
    python tools/unsup_two_branch_v2.py --sites 杨家坝 --visual frangi
    python tools/unsup_two_branch_v2.py --csasc-thr 0.80        # 敏感性对照
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, distance_transform_edt,
                           gaussian_filter, label as cc_label, map_coordinates)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from skimage.filters import threshold_otsu

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

HALF, STEP = 15.5, 1.0        # 剖面半长 / 步长（与 build_cs_fields 同口径）
NEAR = 2.0                    # 中心 ±2 m 留空，避开路面本身
BLOCK = 96


# ------------------------------------------------------------ 输入 ----
def load_terrain(site):
    """返回 (dem 已填洞, hillshade, valid)。"""
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    valid = d["valid"] == 1
    dem = fill_nodata(d["dem_mean"].astype(np.float64), valid)

    gsd = 1.0
    dzdx = np.gradient(dem, gsd, axis=1)          # 东为正
    dzdy = -np.gradient(dem, gsd, axis=0)         # 北为正（行向下）
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)              # ESRI 约定，自北顺时针
    az, alt = np.radians(315.0), np.radians(45.0)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = (np.clip(hs, -1.0, 1.0) + 1.0) / 2.0
    return dem, np.where(valid, hs, 0.0), valid


# ------------------------------------------------------------ CSASC ----
def csasc_field(site, dem, valid):
    """横断面两侧坡度幅值不对称度，[0,1]，valid 外 NaN。"""
    n_row, n_col, _, _ = P.direction_field(site)
    H, W = dem.shape
    lags = np.arange(-HALF, HALF + STEP * 0.5, STEP)
    L, R = lags < -NEAR, lags > NEAR
    out = np.full((H, W), np.nan, dtype=np.float32)

    for r0 in range(0, H, BLOCK):
        r1 = min(r0 + BLOCK, H)
        rr0, cc0 = np.nonzero(valid[r0:r1])
        if rr0.size == 0:
            continue
        rr, cc = rr0 + r0, cc0
        z = np.empty((rr.size, lags.size), dtype=np.float64)
        for k, s in enumerate(lags):
            z[:, k] = map_coordinates(dem, [rr + s * n_row[rr, cc],
                                            cc + s * n_col[rr, cc]],
                                      order=1, mode="nearest")
        z -= z.mean(axis=1, keepdims=True)                        # 去本底高差
        d1 = np.abs(np.gradient(z, STEP, axis=1))
        aL, aR = d1[:, L].mean(axis=1), d1[:, R].mean(axis=1)
        out[rr, cc] = (np.abs(aL - aR) / (aL + aR + 1e-6)).astype(np.float32)

    out[~valid] = np.nan
    return out


# ------------------------------------------------------------ 端点 ----
def farthest_pair(pts):
    """连通域内最远两点。>4000 px 时先取凸包顶点再做两两距离（等价直径）。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 1:
        return pts[0], pts[0], 0.0
    if len(pts) > 4000:
        pts = pts[ConvexHull(pts).vertices]
    d = squareform(pdist(pts))
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return pts[i], pts[j], float(d[i, j])


def endpoints_from_csasc(csasc, valid, thr, log):
    """CSASC > thr 的最大连通域 → 最远两点。返回 ((r0,c0),(r1,c1)) 与诊断字典。"""
    b = (csasc > thr) & valid
    n = int(b.sum())
    if n == 0:
        log(f"  [FAIL] CSASC>{thr:.2f} 为空")
        return None, dict(px=0)
    lab, ncc = cc_label(b)
    sizes = np.bincount(lab.ravel())[1:]
    k = int(np.argmax(sizes)) + 1
    pts = np.argwhere(lab == k)
    p, q, span = farthest_pair(pts)
    info = dict(px=n, n_cc=int(ncc), max_cc_px=int(len(pts)),
                max_cc_frac=float(len(pts) / n), span_px=round(span, 1),
                ep0=[float(p[0]), float(p[1])], ep1=[float(q[0]), float(q[1])])
    log(f"  CSASC>{thr:.2f}: {n:,}px / {ncc} 连通域；最大 {len(pts):,}px"
        f"（占 {info['max_cc_frac'] * 100:.1f}%），两端跨度 {span:.1f}px")
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


def branch_topo(hs, valid, ep, dilate=6, turn_lambda=0.2, turn_min=1, log=print):
    """阴影最小代价路径（CSASC 端点）→ 膨胀 → 高斯 → [0,1]。"""
    dom = valid.copy()
    H, W = dom.shape
    cost = np.where(dom, 1.0 - hs, 1.0).astype(np.float64) + 0.05
    z = np.zeros(dom.shape)
    zc = np.zeros(dom.shape, dtype=bool)                    # 方向先验关闭
    (rows, cols, data), N = P.build_state_graph(cost, dom, z, z, zc,
                                                lam_dir=0.0, lam_turn=turn_lambda,
                                                turn_min=turn_min)
    s = int(round(ep[0][0]) * W + round(ep[0][1]))
    e = int(round(ep[1][0]) * W + round(ep[1][1]))
    path = dijkstra_on(rows, cols, data, N, s, e)
    if path is None:
        log("  [FAIL] 分支1 无连通路径")
        return None
    band = binary_dilation(path.reshape(H, W), iterations=dilate) & dom
    f = gaussian_filter(band.astype(np.float64), sigma=3.0)
    log(f"  分支1 路径 {int(path.sum())}px → 膨胀带 {int(band.sum()):,}px")
    return np.where(dom, f, 0.0), int(path.sum())


# ------------------------------------------------------------ 分支 2 ----
def branch_visual(kind, csasc, dem, valid, log=print):
    """F_visual：CSASC（横断面不对称）或多尺度 Frangi（管状地形）。"""
    if kind == "csasc":
        g = np.where(valid, csasc, np.nan)
        tag = "CSASC"
    else:
        from skimage.filters import frangi
        hp = dem - gaussian_filter(dem, 3.0)               # 高通，去掉坡面趋势
        fr = frangi(hp, sigmas=np.arange(1, 5), black_ridges=False)
        g = np.where(valid, fr, np.nan)
        tag = "Frangi σ=1~4"
    v = g[valid]
    lo, hi = float(np.nanpercentile(v, 1)), float(np.nanpercentile(v, 99))
    log(f"  分支2 {tag}: valid P1~P99=[{lo:.4f},{hi:.4f}]")
    return g, tag


# ------------------------------------------------------------ 归一化 ----
def minmax01(x, valid, tag, log):
    """valid 内严格 min-max 到 [0,1]（域外 = 0）。返回 (f, (lo, hi))。"""
    v = x[valid]
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max())
    y = np.zeros_like(x, dtype=np.float64)
    y[valid] = (np.nan_to_num(x[valid], nan=lo) - lo) / max(hi - lo, 1e-12)
    y = np.clip(y, 0.0, 1.0)
    log(f"  {tag} 原始范围 [{lo:.4f},{hi:.4f}] → 归一化 [0,1]，"
        f"均值 {y[valid].mean():.4f}")
    return y, (lo, hi)


# ------------------------------------------------------------ 主 ----
def run_site(site, a, log):
    t0 = time.time()
    dem, hs, valid = load_terrain(site)
    log(f"\n[{site}] 网格 {hs.shape}  valid {int(valid.sum()):,}px")

    csasc = csasc_field(site, dem, valid)
    v = csasc[valid]
    q = {f"p{p}": round(float(np.percentile(v, p)), 4) for p in (50, 90, 95, 99)}
    log(f"  CSASC 范围 [{v.min():.4f},{v.max():.4f}]  分位 {q}")

    ep, ep_info = endpoints_from_csasc(csasc, valid, a.csasc_thr, log)
    if ep is None:
        return None

    r_topo = branch_topo(hs, valid, ep, dilate=a.dilate,
                         turn_lambda=a.turn_lambda, turn_min=a.turn_min, log=log)
    if r_topo is None:
        return None
    ft, path_px = r_topo

    raw_vis, vis_tag = branch_visual(a.visual, csasc, dem, valid, log=log)

    # 融合前严格归一化
    ft, ft_rng = minmax01(ft, valid, "F_topo", log)
    fv, fv_rng = minmax01(raw_vis, valid, "F_visual", log)

    # 乘法门控：稀疏拓扑先验当闸门，视觉项只能放大闸门内响应
    fused = ft * (1.0 + fv)
    fused = gaussian_filter(fused, sigma=a.sigma)
    fused = np.where(valid, fused, 0.0)
    fused, fu_rng = minmax01(fused, valid, "F_fused（平滑后）", log)

    thr = float(threshold_otsu(fused[valid]))
    mask = (fused >= thr) & valid
    if a.close > 0:
        mask = binary_closing(mask, iterations=a.close) & valid
    log(f"  Otsu 阈值 {thr:.4f}（valid 内），掩膜 {int(mask.sum()):,}px")

    # ---- 以下才读真值：只评分，不回流 ----
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    win = window_mask(valid)

    def sc(sel):
        inter = int((mask & label & sel).sum())
        union = int(((mask | label) & sel).sum())
        tp, fp = inter, int((mask & ~label & sel).sum())
        fn = int((~mask & label & sel).sum())
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return dict(iou=float(inter / union) if union else 0.0,
                    f1=float(2 * pr * rc / max(pr + rc, 1e-9)),
                    precision=float(pr), recall=float(rc),
                    tp=tp, fp=fp, fn=fn, pred_px=int((mask & sel).sum()),
                    truth_px=int((label & sel).sum()))

    raw = sc(valid)
    d10 = sc(valid & win)
    log(f"  全域  IoU={raw['iou']:.4f} F1={raw['f1']:.4f} "
        f"P={raw['precision']:.4f} R={raw['recall']:.4f} "
        f"(TP {raw['tp']:,} / FP {raw['fp']:,} / FN {raw['fn']:,})")
    log(f"  D10   IoU={d10['iou']:.4f} F1={d10['f1']:.4f} "
        f"P={d10['precision']:.4f} R={d10['recall']:.4f}")

    # ---- 三张图 ----
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    def save(arr, cmap, title, fn, contour_truth=False):
        fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)
        ax.imshow(np.where(valid, arr, np.nan), cmap=cmap)
        if contour_truth:
            ax.contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.3)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        p = os.path.join(FIG, fn)
        fig.savefig(p); plt.close(fig)
        return os.path.relpath(p, ROOT)

    suffix = f"_{a.visual}" if a.visual != "csasc" else ""
    f1 = save(ft, "viridis", f"{site} · F_topo（CSASC 端点 Dijkstra）",
              f"unsup_v2_topo_{site}{suffix}.png")
    f2 = save(fv, "viridis", f"{site} · F_visual（{vis_tag}）",
              f"unsup_v2_visual_{site}{suffix}.png")
    f3 = save(mask.astype(float), "gray",
              f"{site} · F_fused Otsu 掩膜（IoU={raw['iou']:.4f}）",
              f"unsup_v2_mask_{site}{suffix}.png", contour_truth=True)

    return dict(site=site, visual=a.visual, csasc_thr=a.csasc_thr,
                csasc_quantile=q, csasc_range=[float(v.min()), float(v.max())],
                endpoints=ep_info, path_px=path_px,
                ft_range=ft_rng, fv_range=fv_rng, fused_range=fu_rng,
                otsu=thr, dilate=a.dilate, sigma=a.sigma, close=a.close,
                turn_lambda=a.turn_lambda, turn_min=a.turn_min,
                raw=raw, d10=d10, seconds=round(time.time() - t0, 1),
                figures=[f1, f2, f3])


def main():
    p = argparse.ArgumentParser(description="无监督双分支融合 v2（乘法门控，SHP 仅评分）")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--visual", type=str, default="csasc", choices=["csasc", "frangi"],
                   help="视觉分支：csasc=横断面不对称；frangi=多尺度管状响应")
    p.add_argument("--csasc-thr", type=float, default=0.85, help="CSASC 高分阈值")
    p.add_argument("--dilate", type=int, default=6, help="拓扑路径膨胀半径 px")
    p.add_argument("--sigma", type=float, default=2.0, help="融合后高斯 σ")
    p.add_argument("--close", type=int, default=1, help="掩膜闭运算次数（0=关）")
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lp = os.path.join(RES, "unsup_two_branch_v2.log")
    lf = open(lp, "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"无监督双分支融合 v2 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"视觉分支={a.visual}  CSASC>{a.csasc_thr}  dilate={a.dilate}px  σ={a.sigma}  "
        f"close={a.close}  λ_turn={a.turn_lambda} m={a.turn_min}")
    log("SHP 真值仅在最后算 IoU/F1 时读取；端点自 CSASC 高分域生成，无外部起止点")

    out = []
    for s in a.sites:
        r = run_site(s, a, log)
        if r:
            out.append(r)

    log("\n" + "=" * 78)
    log(f"{'站点':>8s} {'Otsu':>7s} {'预测px':>9s} {'TP':>7s} {'FP':>8s} "
        f"{'IoU':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    log("-" * 74)
    for r in out:
        d = r["raw"]
        log(f"{r['site']:>8s} {r['otsu']:>7.4f} {d['pred_px']:>9,d} {d['tp']:>7,d} "
            f"{d['fp']:>8,d} {d['iou']:>7.4f} {d['f1']:>7.4f} "
            f"{d['precision']:>7.4f} {d['recall']:>7.4f}")

    jp = os.path.join(RES, "unsup_two_branch_v2.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="无监督双分支融合 v2：CSASC 端点拓扑门 × (1+视觉分支)，Otsu",
                       visual=a.visual, csasc_thr=a.csasc_thr, dilate=a.dilate,
                       sigma=a.sigma, close=a.close, turn_lambda=a.turn_lambda,
                       turn_min=a.turn_min, sites=out), f,
                  ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
