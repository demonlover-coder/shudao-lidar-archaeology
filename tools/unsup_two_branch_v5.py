# -*- coding: utf-8 -*-
"""无监督双分支融合 v5：CSSC 形态学连线端点 × 距离场软门控 × 分站自编码器特征。

相对 v4 的两处结构性修正（针对 v4 的实测归因）
--------------------------------------------
v4 归因：① CSSC≥P95 的连通域是"一堆斑点"，最大连通域外接直径两点跨度只有 38.6/122.7 px，
端点必然脱靶；② 乘法门控 `F_topo×(1+F_visual)` 把 F_fused 的支撑域锁死在 1px 路径的
膨胀带上，于是西坝村 AUC 0.737 的自编码器特征在融合后 AUC 掉到 0.500。

v5 对应改两点：
1. **端点先连线再取径**：CSSC>P95 → 形态学膨胀 `disk(5px)`（填平 ≤10px 的斑点间隙）
   → 最大连通域 → `skeletonize` → 骨架上的 **测地直径**（两次 Dijkstra 扫描，
   首扫取最远点、再扫取对径点）作为起终点。这样端点落在同一条连通骨架上，
   不会再出现"两个斑点各取一点"。
2. **软门控**：Dijkstra 出 1px 路径后，计算全图到路径的欧氏距离 `dist_map`，
   `W_topo = exp(−dist_map²/(2σ²))`，σ=10 px（=10 m），给出 10~20 m 的容差空间。
   融合改成 `F_fused = W_topo × F_visual`（不再有 `1+`）。

铁律
----
`mask_1m_{site}.npy`（SHP 真值）只在最后算 IoU/F1 与诊断 AUC 时读取；端点、代价、
方向、自编码器训练、融合、阈值全链路零标签。自编码器只重构输入自身（MSE），
**不用分支 1 当标签**，不是蒸馏；两站**各自独立训练**，不共享权重。

流程
----
1. 分支 1：端点（上述形态学连线 + 测地直径）+ openness 代价 + h1_phase 方向惩罚
   （λ_dir=1.5）+ 转角惩罚 → 1px 路径 → 欧氏距离场 → `W_topo ∈ (0,1]`。
2. 分支 2：64×64 Hillshade 块无监督训练卷积自编码器（4 层 stride-2，瓶颈 4×4×128），
   丢解码器、瓶颈层上采样、128 通道均值 → valid 内 min-max → `F_visual ∈ [0,1]`。
3. 融合：`F_fused = W_topo × F_visual` → 高斯 σ 平滑 → valid 内 Otsu。
4. 评价：IoU / F1 / P / R；并给出 **F_visual 单独 Otsu**、**W_topo 单独 Otsu** 的对照。

用法
----
    python tools/unsup_two_branch_v5.py
    python tools/unsup_two_branch_v5.py --sites 杨家坝 --sigma 20
    python tools/unsup_two_branch_v5.py --fill close --r 5
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
from scipy.spatial.distance import pdist, squareform
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

PATCH = 64
STRIDE_DOWN = 16
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
        L = float(np.hypot(dy, dx))
        rows.append(a[m]); cols.append(b[m])
    r = np.concatenate(rows); c = np.concatenate(cols)
    step = np.hypot(pts[r, 0] - pts[c, 0], pts[r, 1] - pts[c, 1])   # 1 或 √2
    G = csr_matrix((step, (r, c)), shape=(len(pts), len(pts)))
    d0 = dijkstra(G, indices=0)
    i = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    d1 = dijkstra(G, indices=i)
    j = int(np.argmax(np.where(np.isfinite(d1), d1, -1)))
    return pts[i], pts[j], float(d1[j])


def endpoints_from_cssc(site, valid, pct, fill, r, log):
    """CSSC ≥ P{pct} → 形态学填隙 → 最大连通域 → 骨架测地直径两端。"""
    cssc = np.load(os.path.join(RES, f"cs_fields_{site}.npz"))["cssc"].astype(np.float64)
    v = cssc[valid]
    v = v[np.isfinite(v)]
    thr = float(np.percentile(v, pct))
    raw_b = (cssc >= thr) & valid & np.isfinite(cssc)
    if not raw_b.any():
        log(f"  [FAIL] CSSC ≥ P{pct} 为空")
        return None, dict(thr=thr)

    st = disk(r) if fill == "dilate" else disk(r)
    b = binary_dilation(raw_b, st) if fill == "dilate" else binary_closing(raw_b, st)
    b &= valid
    lab, ncc = cc_label(b)
    sizes = np.bincount(lab.ravel())[1:]
    k = int(np.argmax(sizes)) + 1
    comp = lab == k
    skel = skeletonize(comp)
    p, q, glen = geodesic_diameter(skel)

    info = dict(thr=thr, fill=fill, r=r, raw_px=int(raw_b.sum()),
                raw_cc=int(cc_label(raw_b)[1]), filled_px=int(b.sum()),
                filled_cc=int(ncc), max_cc_px=int(comp.sum()),
                skel_px=int(skel.sum()), geo_len_px=round(glen, 1),
                euc_len_px=round(float(np.hypot(p[0] - q[0], p[1] - q[1])), 1),
                ep0=[float(p[0]), float(p[1])], ep1=[float(q[0]), float(q[1])])
    log(f"  CSSC≥P{pct}={thr:.0f}: {info['raw_px']:,}px/{info['raw_cc']} 连通域 → "
        f"{fill}(r={r}) → {int(b.sum()):,}px/{ncc} 连通域")
    log(f"    最大连通域 {comp.sum():,}px，骨架 {skel.sum():,}px，"
        f"测地长度 {glen:.1f}px（直线 {info['euc_len_px']:.1f}px）")
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


def branch_topo(site, valid, ep, sigma_gate=10.0, dir_lambda=1.5, dir_gate=None,
                turn_lambda=0.2, turn_min=1, log=print):
    """openness 代价 + h1 方向惩罚 → 1px 路径 → 欧氏距离场高斯软门控 W_topo。"""
    dom = valid.copy()
    H, W = dom.shape
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    cost, info = P.build_cost(site, tpi31, dom, 0.05, "pct", "open")
    nr, nc, conf, gate_thr = P.direction_field(
        site, P.DFS_GATE if dir_gate is None else dir_gate)
    log("  分支1 代价 openness " +
        "  ".join(f"{k}[{lo:.3f},{hi:.3f}]" for k, (lo, hi) in info.items()) +
        f"；方向门控 h1_deg≥{gate_thr:.1f}° 覆盖 valid 的 {conf[valid].mean() * 100:.1f}%")

    (rows, cols, data), N = P.build_state_graph(cost, dom, nr, nc, conf,
                                                lam_dir=dir_lambda, lam_turn=turn_lambda,
                                                turn_min=turn_min)
    s = int(round(ep[0][0]) * W + round(ep[0][1]))
    e = int(round(ep[1][0]) * W + round(ep[1][1]))
    path = dijkstra_on(rows, cols, data, N, s, e)
    if path is None:
        log("  [FAIL] 分支1 无连通路径")
        return None
    path = path.reshape(H, W)
    dist_map = distance_transform_edt(~path)
    wtopo = np.exp(-(dist_map ** 2) / (2.0 * sigma_gate ** 2))
    wtopo = np.where(dom, wtopo, 0.0)
    log(f"  分支1 路径 {int(path.sum())}px；距离场 max {dist_map.max():.0f}px → "
        f"W_topo(σ={sigma_gate:g}) 均值 {wtopo[dom].mean():.4f}，"
        f"≥0.5 覆盖 {int(((wtopo >= 0.5) & dom).sum()):,}px")
    return wtopo, int(path.sum())


# ------------------------------------------------------------ 分支 2 ----
def _ae():
    """4 层卷积自编码器：64×64×1 → 4×4×128 瓶颈 → 重构。"""
    import torch.nn as nn

    def blk(i, o):
        return [nn.Conv2d(i, o, 3, 2, 1, bias=False),
                nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=False)]

    enc = nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 64), *blk(64, 128))
    dec = nn.Sequential(
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=False),
        nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2, inplace=False),
        nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False), nn.BatchNorm2d(16),
        nn.LeakyReLU(0.2, inplace=False),
        nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid())
    return enc, dec


def _patches(hs, valid, n, rng, min_valid=0.95):
    """按 valid 像元随机取 n 个 64×64 块（块内 valid 占比 ≥ min_valid）。"""
    H, W = hs.shape
    m = PATCH // 2
    idx = np.argwhere(valid[m:H - m, m:W - m]) + m
    sel = rng.choice(len(idx), size=min(n * 4, len(idx)), replace=False)
    out = []
    for r, c in idx[sel]:
        r0, c0 = int(r) - m, int(c) - m
        if valid[r0:r0 + PATCH, c0:c0 + PATCH].mean() < min_valid:
            continue
        out.append(hs[r0:r0 + PATCH, c0:c0 + PATCH])
        if len(out) >= n:
            break
    return np.asarray(out, dtype=np.float32)


def branch_visual(hs, valid, epochs=20, n_patch=1500, batch=64, lr=1e-3,
                  seed=0, device="cpu", log=print):
    """分站独立训练自编码器 → 瓶颈通道均值 → F_visual ∈ [0,1]。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    dev = torch.device(device)

    rng = np.random.default_rng(seed)
    X = _patches(hs, valid, n_patch, rng)
    log(f"  分支2 本站独立训练：{len(X)} 个 {PATCH}×{PATCH} 块，epochs={epochs} "
        f"batch={batch} lr={lr} device={dev}（不跨站共享权重）")

    enc, dec = _ae()
    net = nn.Sequential(enc, dec).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    log(f"  分支2 参数量 {sum(p.numel() for p in net.parameters()) / 1e3:.1f}k；"
        f"瓶颈 {PATCH // STRIDE_DOWN}×{PATCH // STRIDE_DOWN}×128")

    ds = torch.from_numpy(X[:, None])
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(ds))
        tot = 0.0
        for i in range(0, len(ds), batch):
            xb = ds[perm[i:i + batch]].to(dev)
            loss = lossf(net(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.detach().item() * len(xb)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            log(f"    epoch {ep:3d}/{epochs}  MSE {tot / len(ds):.6f}  "
                f"({time.time() - t0:.0f}s)")

    H, W = hs.shape
    ph, pw = (-H) % STRIDE_DOWN, (-W) % STRIDE_DOWN
    img = np.pad(hs, ((0, ph), (0, pw)), mode="reflect").astype(np.float32)
    enc.eval()
    with torch.no_grad():
        b = enc(torch.from_numpy(img[None, None]).to(dev))
        f = Fn.interpolate(b, size=(H + ph, W + pw), mode="bilinear", align_corners=False)
        f = f.mean(dim=1)[0, :H, :W].cpu().numpy().astype(np.float64)
    log(f"  分支2 瓶颈 {tuple(b.shape)} → 通道均值 → {f.shape}")

    g = np.where(valid, f, np.nan)
    vv = g[valid]
    lo, hi = float(np.nanpercentile(vv, 1)), float(np.nanpercentile(vv, 99))
    log(f"  分支2 特征 valid P1~P99=[{lo:.5f},{hi:.5f}]")
    return g, (lo, hi), len(X)


# ------------------------------------------------------------ 工具 ----
def minmax01(x, valid, tag, log):
    v = x[valid]
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max())
    y = np.zeros_like(x, dtype=np.float64)
    y[valid] = (np.nan_to_num(x[valid], nan=lo) - lo) / max(hi - lo, 1e-12)
    y = np.clip(y, 0.0, 1.0)
    log(f"  {tag} 原始 [{lo:.4f},{hi:.4f}] → [0,1]，均值 {y[valid].mean():.4f}")
    return y, (lo, hi)


# ------------------------------------------------------------ 主 ----
def run_site(site, a, log):
    t0 = time.time()
    dem, hs, valid = load_terrain(site)
    log(f"\n[{site}] 网格 {hs.shape}  valid {int(valid.sum()):,}px")

    ep, ep_info = endpoints_from_cssc(site, valid, a.pct, a.fill, a.r, log)
    if ep is None:
        return None

    r_topo = branch_topo(site, valid, ep, sigma_gate=a.sigma_gate, dir_lambda=a.dir_lambda,
                         dir_gate=a.dir_gate, turn_lambda=a.turn_lambda,
                         turn_min=a.turn_min, log=log)
    if r_topo is None:
        return None
    wtopo, path_px = r_topo

    raw_vis, _, n_patch = branch_visual(
        hs, valid, epochs=a.epochs, n_patch=a.patches, batch=a.batch, lr=a.lr,
        seed=a.seed, device=a.device, log=log)

    fv, fv_rng = minmax01(raw_vis, valid, "F_visual", log)
    ft, ft_rng = minmax01(wtopo, valid, "F_topo(=W_topo)", log)

    fused = ft * fv
    fused = gaussian_filter(fused, sigma=a.sigma)
    fused = np.where(valid, fused, 0.0)
    log(f"  F_fused = W_topo × F_visual，高斯 σ={a.sigma:g}；"
        f"valid 内 max {fused[valid].max():.4f}")

    def otsu_mask(f):
        thr = float(threshold_otsu(f[valid]))
        m = (f >= thr) & valid
        if a.close > 0:
            m = binary_closing(m, iterations=a.close) & valid
        return m, thr

    mask, thr = otsu_mask(fused)
    mask_v, thr_v = otsu_mask(fv)                 # 对照 1：F_visual 直接 Otsu
    mask_t, thr_t = otsu_mask(ft)                 # 对照 2：W_topo 直接 Otsu
    log(f"  Otsu：融合 {thr:.4f}（{int(mask.sum()):,}px）| "
        f"仅视觉 {thr_v:.4f}（{int(mask_v.sum()):,}px）| "
        f"仅门控 {thr_t:.4f}（{int(mask_t.sum()):,}px）")

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

    res = {k: (sc(m, valid), sc(m, valid & win))
           for k, m in (("fused", mask), ("visual", mask_v), ("topo", mask_t))}

    sk = skeletonize(label)
    dist = distance_transform_edt(~label)
    neg = (dist >= 45.0) & (dist <= 55.0) & valid

    def auc(x):
        p = x[sk & valid]; q = x[neg]
        p, q = p[np.isfinite(p)], q[np.isfinite(q)]
        if p.size < 10 or q.size < 10:
            return float("nan")
        return float(roc_auc_score(np.r_[np.ones(p.size), np.zeros(q.size)], np.r_[p, q]))

    aucs = {k: auc(x) for k, x in (("fused", fused), ("visual", fv), ("topo", ft))}
    for k, lab_ in (("fused", "融合"), ("visual", "仅 F_visual"), ("topo", "仅 W_topo")):
        d = res[k][0]
        log(f"  [{lab_}] IoU={d['iou']:.4f} F1={d['f1']:.4f} P={d['precision']:.4f} "
            f"R={d['recall']:.4f} (TP {d['tp']:,}/FP {d['fp']:,}/FN {d['fn']:,})  "
            f"AUC={aucs[k]:.3f}")

    # ---- 图 ----
    os.makedirs(FIG, exist_ok=True)
    np.savez_compressed(os.path.join(RES, f"unsup_v5_fields_{site}.npz"),
                        wtopo=ft.astype(np.float32), fvis=fv.astype(np.float32),
                        fused=fused.astype(np.float32), mask=mask.astype(np.uint8),
                        mask_visual=mask_v.astype(np.uint8), mask_topo=mask_t.astype(np.uint8))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    def save(arr, cmap, title, fn, truth=False):
        fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)
        ax.imshow(np.where(valid, arr, np.nan), cmap=cmap)
        if truth:
            ax.contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.3)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        p = os.path.join(FIG, fn)
        fig.savefig(p); plt.close(fig)
        return os.path.relpath(p, ROOT)

    f1 = save(ft, "viridis", f"{site} · W_topo（路径距离场 σ={a.sigma_gate:g}m）",
              f"unsup_v5_topo_{site}.png")
    f2 = save(fv, "viridis", f"{site} · F_visual（分站自编码器瓶颈）",
              f"unsup_v5_visual_{site}.png")
    f3 = save(fused, "magma", f"{site} · F_fused = W_topo × F_visual",
              f"unsup_v5_fused_{site}.png")

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 10.6), dpi=140)
    for k, (m, name) in enumerate(((mask, "融合"), (mask_v, "仅 F_visual"),
                                   (mask_t, "仅 W_topo"))):
        a_ = ax.flat[k]
        a_.imshow(np.where(valid, m.astype(float), np.nan), cmap="gray")
        a_.contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.2)
        a_.set_title(f"{name} Otsu（IoU={res[['fused','visual','topo'][k]][0]['iou']:.4f}）",
                     fontsize=10)
        a_.set_xticks([]); a_.set_yticks([])
    a_ = ax.flat[3]
    a_.imshow(np.where(valid, hs, np.nan), cmap="gray")
    a_.contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.2)
    a_.set_title("Hillshade + SHP 真值", fontsize=10)
    a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{site} · v5 软门控融合 vs 单分支对照", fontsize=11)
    fig.tight_layout()
    f4 = os.path.join(FIG, f"unsup_v5_compare_{site}.png")
    fig.savefig(f4); plt.close(fig)

    return dict(site=site, pct=a.pct, fill=a.fill, r=a.r,
                sigma_gate=a.sigma_gate, sigma=a.sigma, dilate=None,
                dir_lambda=a.dir_lambda, turn_lambda=a.turn_lambda, turn_min=a.turn_min,
                epochs=a.epochs, patches=n_patch, lr=a.lr, batch=a.batch,
                endpoints=ep_info, path_px=path_px,
                ft_range=ft_rng, fv_range=fv_rng,
                otsu=dict(fused=thr, visual=thr_v, topo=thr_t),
                results={k: dict(raw=v[0], d10=v[1]) for k, v in res.items()},
                auc=aucs, seconds=round(time.time() - t0, 1),
                figures=[f1, f2, f3, os.path.relpath(f4, ROOT)])


def main():
    p = argparse.ArgumentParser(description="无监督双分支融合 v5（CSSC 连线端点 + 距离场软门控）")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--pct", type=float, default=95.0)
    p.add_argument("--fill", type=str, default="dilate", choices=["dilate", "close"],
                   help="斑点连线方式：dilate=膨胀填隙；close=闭运算")
    p.add_argument("--r", type=int, default=5, help="形态学半径 px")
    p.add_argument("--sigma-gate", type=float, default=10.0, help="W_topo 高斯 σ（米）")
    p.add_argument("--sigma", type=float, default=2.0, help="融合后高斯 σ")
    p.add_argument("--close", type=int, default=1, help="掩膜闭运算次数（0=关）")
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--dir-gate", type=float, default=None)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patches", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "unsup_two_branch_v5.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"无监督双分支融合 v5 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"端点 CSSC≥P{a.pct:g} → {a.fill}(r={a.r}px) → 骨架测地直径；"
        f"W_topo σ={a.sigma_gate:g}m；融合 σ={a.sigma:g}")
    log(f"自编码器 分站独立 epochs={a.epochs} patches={a.patches} batch={a.batch} "
        f"lr={a.lr} device={a.device} seed={a.seed}")
    log("SHP 真值仅在最后算 IoU/F1/AUC 时读取；分支2 只重构输入自身，无蒸馏、无标签")

    out = []
    for s in a.sites:
        r = run_site(s, a, log)
        if r:
            out.append(r)

    log("\n" + "=" * 78)
    log(f"{'站点':>8s} {'方案':>10s} {'Otsu':>7s} {'预测px':>9s} {'TP':>6s} {'FP':>8s} "
        f"{'IoU':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    log("-" * 78)
    for r in out:
        for k, name in (("fused", "融合"), ("visual", "仅视觉"), ("topo", "仅门控")):
            d = r["results"][k]["raw"]
            log(f"{r['site']:>8s} {name:>10s} {r['otsu'][k]:>7.4f} {d['pred_px']:>9,d} "
                f"{d['tp']:>6,d} {d['fp']:>8,d} {d['iou']:>7.4f} {d['f1']:>7.4f} "
                f"{d['precision']:>7.4f} {d['recall']:>7.4f}")

    jp = os.path.join(RES, "unsup_two_branch_v5.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="无监督双分支融合 v5：CSSC 形态学连线端点 + 距离场软门控 × 分站自编码器",
                       pct=a.pct, fill=a.fill, r=a.r, sigma_gate=a.sigma_gate,
                       sigma=a.sigma, close=a.close, dir_lambda=a.dir_lambda,
                       turn_lambda=a.turn_lambda, turn_min=a.turn_min,
                       epochs=a.epochs, patches=a.patches, lr=a.lr, batch=a.batch,
                       seed=a.seed, sites=out), f, ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
