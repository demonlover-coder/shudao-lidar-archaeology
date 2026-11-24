# -*- coding: utf-8 -*-
"""无监督双分支融合 v4：CSSC 端点图论门 × (1 + 卷积自编码器深度特征) → Otsu。

铁律
----
* `mask_1m_{site}.npy`（SHP 真值）**只在最后算 IoU/F1 时读取**；端点、代价、方向、
  自编码器训练、融合、阈值全链路零标签。
* 分支 2 的训练目标只有"重构输入自身"（MSE），**不用分支 1 的输出当软标签**，
  因此不是蒸馏，是纯无监督表征学习。

流程
----
1. 分支 1（宏观图论）
   * 端点：`cs_fields_{site}.npz` 的 CSSC 场取 valid 内 **P95 以上** → 二值 →
     最大连通域 → 域内最远两点（凸包直径，>4000px 时用凸包顶点避免 O(N²)）。
   * 代价：`openness`（低=更像走廊；P5~P95 线性归一化）+ eps。
   * 方向：`h1_phase` 法向先验（`direction_field`，置信门控 P60），`λ_dir=1.5`；
     叠加转角惩罚 `λ_turn=0.2, turn_min=1`（只罚 >45° 折返）。
   * 输出：路径膨胀 `--dilate 3` px → 高斯 σ=2 → valid 内 min-max → `F_topo ∈ [0,1]`。

2. 分支 2（无监督深度特征）
   * 数据：从 Hillshade 上按 valid 像元随机取 64×64 块（valid 占比 ≥0.95）。
   * 网络：4 层 stride-2 卷积编码器（16/32/64/128 通道，瓶颈 4×4×128）
     + 对称转置卷积解码器；参数量约 0.16 M。
   * 训练：Adam，loss = 重构 MSE(输入, 输出)，`--epochs 20`，`--patches 1500`。
     无标签、无分支 1 参与。
   * 特征：丢掉 Decoder，整幅 Hillshade 过 Encoder，瓶颈层双线性上采样回原尺寸，
     **128 通道取均值**，valid 内 min-max → `F_visual ∈ [0,1]`。

3. 融合：`F_fused = F_topo · (1 + F_visual)`，高斯 σ 平滑，valid 内 Otsu 二值化。

4. 评价：读 SHP 算全域与 D10 口径的 IoU / F1 / P / R。

设计上的已知边界（实测，不是推测）
---------------------------------
CSSC 的逐像元走廊选择性不弱（骨架 vs 45~55 m 带 AUC 0.721/0.889），但它的**高分
尾部是碎片化的**：P95 以上在杨家坝是 5,111px/1,387 个连通域、最大 138px（跨度 39px），
在西坝村是 32,525px/3,172 个连通域、最大 2,296px（跨度 123px），两站最大连通域的
两个最远点都落在真值走廊之外。故"最大连通域最远两点"给出的端点只是局部小斑点，
分支 1 的路径必然很短。v4 的输出会把这一点量化出来。

用法
----
    python tools/unsup_two_branch_v4.py
    python tools/unsup_two_branch_v4.py --sites 杨家坝 --epochs 30 --patches 2000
    python tools/unsup_two_branch_v4.py --pct 99            # 端点阈值敏感性
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, distance_transform_edt,
                           gaussian_filter, label as cc_label)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
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
STRIDE_DOWN = 16          # 4 层 stride-2 → 1/16


# ------------------------------------------------------------ 输入 ----
def load_terrain(site):
    """返回 (dem 已填洞, Hillshade(az=315°,alt=45°), valid)。"""
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    valid = d["valid"] == 1
    dem = fill_nodata(d["dem_mean"].astype(np.float64), valid)

    dzdx = np.gradient(dem, 1.0, axis=1)          # 东为正
    dzdy = -np.gradient(dem, 1.0, axis=0)         # 北为正（行向下）
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)              # ESRI 约定，自北顺时针
    az, alt = np.radians(315.0), np.radians(45.0)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = (np.clip(hs, -1.0, 1.0) + 1.0) / 2.0
    return dem, np.where(valid, hs, 0.0), valid


# ------------------------------------------------------------ 端点 ----
def farthest_pair(pts):
    """连通域内最远两点；>4000 px 时先取凸包顶点再做两两距离（等价直径）。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 1:
        return pts[0], pts[0], 0.0
    if len(pts) > 4000:
        pts = pts[ConvexHull(pts).vertices]
    d = squareform(pdist(pts))
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return pts[i], pts[j], float(d[i, j])


def endpoints_from_cssc(site, valid, pct, log):
    """CSSC ≥ valid 内 P{pct} → 最大连通域 → 最远两点。返回 (ep, 诊断)。"""
    cssc = np.load(os.path.join(RES, f"cs_fields_{site}.npz"))["cssc"].astype(np.float64)
    v = cssc[valid]
    v = v[np.isfinite(v)]
    thr = float(np.percentile(v, pct))
    b = (cssc >= thr) & valid & np.isfinite(cssc)
    if not b.any():
        log(f"  [FAIL] CSSC ≥ P{pct} 为空")
        return None, dict(thr=thr)
    lab, ncc = cc_label(b)
    sizes = np.bincount(lab.ravel())[1:]
    k = int(np.argmax(sizes)) + 1
    pts = np.argwhere(lab == k)
    p, q, span = farthest_pair(pts)
    info = dict(thr=thr, px=int(b.sum()), n_cc=int(ncc), max_cc_px=int(len(pts)),
                max_cc_frac=float(len(pts) / b.sum()), span_px=round(span, 1),
                ep0=[float(p[0]), float(p[1])], ep1=[float(q[0]), float(q[1])])
    log(f"  CSSC ≥ P{pct}={thr:.0f}: {int(b.sum()):,}px / {ncc} 连通域；"
        f"最大 {len(pts):,}px（{info['max_cc_frac'] * 100:.1f}%），跨度 {span:.1f}px")
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


def branch_topo(site, valid, ep, dilate=3, dir_lambda=1.5, dir_gate=None,
                turn_lambda=0.2, turn_min=1, log=print):
    """openness 代价 + h1_phase 方向惩罚的状态增广 Dijkstra → 膨胀 → 高斯 → [0,1]。"""
    dom = valid.copy()
    H, W = dom.shape
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    cost, info = P.build_cost(site, tpi31, dom, 0.05, "pct", "open")
    nr, nc, conf, gate_thr = P.direction_field(site, P.DFS_GATE if dir_gate is None else dir_gate)
    log("  分支1 代价 openness 范围 " +
        "  ".join(f"{k}[{lo:.3f},{hi:.3f}]" for k, (lo, hi) in info.items()) +
        f"；方向门控 h1_deg≥{gate_thr:.1f}° 覆盖 valid 的 "
        f"{conf[valid].mean() * 100:.1f}%")

    (rows, cols, data), N = P.build_state_graph(cost, dom, nr, nc, conf,
                                                lam_dir=dir_lambda, lam_turn=turn_lambda,
                                                turn_min=turn_min)
    s = int(round(ep[0][0]) * W + round(ep[0][1]))
    e = int(round(ep[1][0]) * W + round(ep[1][1]))
    path = dijkstra_on(rows, cols, data, N, s, e)
    if path is None:
        log("  [FAIL] 分支1 无连通路径")
        return None
    band = binary_dilation(path.reshape(H, W), iterations=dilate) & dom
    f = gaussian_filter(band.astype(np.float64), sigma=2.0)
    log(f"  分支1 路径 {int(path.sum())}px → 膨胀带 {int(band.sum()):,}px")
    return np.where(dom, f, 0.0), int(path.sum())


# ------------------------------------------------------------ 分支 2 ----
def _ae():
    """4 层卷积自编码器：64×64×1 → 4×4×128 瓶颈 → 重构。"""
    import torch.nn as nn

    def blk(i, o, stride):
        return [nn.Conv2d(i, o, 3, stride, 1, bias=False),
                nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=False)]

    enc = nn.Sequential(*blk(1, 16, 2), *blk(16, 32, 2),
                        *blk(32, 64, 2), *blk(64, 128, 2))
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
    idx = np.argwhere(valid[(m):H - m, m:W - m]) + m
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
    """无监督训练卷积自编码器 → 瓶颈特征通道均值 → F_visual ∈ [0,1]。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    dev = torch.device(device)

    rng = np.random.default_rng(seed)
    X = _patches(hs, valid, n_patch, rng)
    if len(X) < 32:
        log(f"  [WARN] 可用 64×64 块仅 {len(X)} 个")
    log(f"  分支2 训练块 {len(X)} 个（{PATCH}×{PATCH}，valid 占比≥0.95），"
        f"epochs={epochs} batch={batch} lr={lr} device={dev}")

    enc, dec = _ae()
    net = nn.Sequential(enc, dec).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    npar = sum(p.numel() for p in net.parameters())
    log(f"  分支2 参数量 {npar / 1e3:.1f}k；瓶颈 {PATCH // STRIDE_DOWN}×"
        f"{PATCH // STRIDE_DOWN}×128")

    ds = torch.from_numpy(X[:, None])
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(ds))
        tot = 0.0
        for i in range(0, len(ds), batch):
            xb = ds[perm[i:i + batch]].to(dev)
            out = net(xb)
            loss = lossf(out, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(xb)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            log(f"    epoch {ep:3d}/{epochs}  MSE {tot / len(ds):.6f}  "
                f"({time.time() - t0:.0f}s)")

    # ---- 特征：整幅图过 Encoder，瓶颈上采样，通道均值 ----
    H, W = hs.shape
    ph, pw = (-H) % STRIDE_DOWN, (-W) % STRIDE_DOWN
    img = np.pad(hs, ((0, ph), (0, pw)), mode="reflect").astype(np.float32)
    x = torch.from_numpy(img[None, None]).to(dev)
    enc.eval()
    with torch.no_grad():
        b = enc(x)                                    # (1,128,H/16,W/16)
        f = Fn.interpolate(b, size=(H + ph, W + pw),
                           mode="bilinear", align_corners=False)
        f = f.mean(dim=1)[0, :H, :W].cpu().numpy().astype(np.float64)
    log(f"  分支2 瓶颈 {tuple(b.shape)} → 通道均值 → {f.shape}")

    g = np.where(valid, f, np.nan)
    v = g[valid]
    lo, hi = float(np.nanpercentile(v, 1)), float(np.nanpercentile(v, 99))
    log(f"  分支2 特征 valid P1~P99=[{lo:.5f},{hi:.5f}]")
    return g, (lo, hi), len(X)


# ------------------------------------------------------------ 归一化 ----
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

    ep, ep_info = endpoints_from_cssc(site, valid, a.pct, log)
    if ep is None:
        return None

    r_topo = branch_topo(site, valid, ep, dilate=a.dilate, dir_lambda=a.dir_lambda,
                         dir_gate=a.dir_gate, turn_lambda=a.turn_lambda,
                         turn_min=a.turn_min, log=log)
    if r_topo is None:
        return None
    ft, path_px = r_topo

    raw_vis, vis_rng, n_patch = branch_visual(
        hs, valid, epochs=a.epochs, n_patch=a.patches, batch=a.batch, lr=a.lr,
        seed=a.seed, device=a.device, log=log)

    ft, ft_rng = minmax01(ft, valid, "F_topo", log)
    fv, fv_rng = minmax01(raw_vis, valid, "F_visual", log)

    fused = ft * (1.0 + fv)
    fused = gaussian_filter(fused, sigma=a.sigma)
    fused = np.where(valid, fused, 0.0)
    fused, fu_rng = minmax01(fused, valid, "F_fused（平滑后）", log)

    thr = float(threshold_otsu(fused[valid]))
    mask = (fused >= thr) & valid
    if a.close > 0:
        mask = binary_closing(mask, iterations=a.close) & valid
    log(f"  Otsu {thr:.4f}（valid 内）→ 掩膜 {int(mask.sum()):,}px")

    # ---- 以下才读真值，只评分 ----
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
                    precision=float(pr), recall=float(rc), tp=tp, fp=fp, fn=fn,
                    pred_px=int((mask & sel).sum()), truth_px=int((label & sel).sum()))

    raw, d10 = sc(valid), sc(valid & win)
    log(f"  全域 IoU={raw['iou']:.4f} F1={raw['f1']:.4f} "
        f"P={raw['precision']:.4f} R={raw['recall']:.4f} "
        f"(TP {raw['tp']:,} / FP {raw['fp']:,} / FN {raw['fn']:,})")

    # 附加诊断（同样是"只评分"）：各分支逐像元的走廊选择性 AUC
    sk = skeletonize(label)
    dist = distance_transform_edt(~label)
    neg = (dist >= 45.0) & (dist <= 55.0) & valid

    def auc(a):
        p = a[sk & valid]
        q = a[neg]
        p, q = p[np.isfinite(p)], q[np.isfinite(q)]
        if p.size < 10 or q.size < 10:
            return float("nan")
        return float(roc_auc_score(np.r_[np.ones(p.size), np.zeros(q.size)], np.r_[p, q]))

    a_topo, a_vis, a_fus = auc(ft), auc(fv), auc(fused)
    log(f"  诊断 AUC（骨架 vs 45~55m 带）F_topo {a_topo:.3f} | "
        f"F_visual {a_vis:.3f} | F_fused {a_fus:.3f}")
    np.savez_compressed(os.path.join(RES, f"unsup_v4_fields_{site}.npz"),
                        ft=ft.astype(np.float32), fv=fv.astype(np.float32),
                        fused=fused.astype(np.float32), mask=mask.astype(np.uint8))

    os.makedirs(FIG, exist_ok=True)
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

    f1 = save(ft, "viridis", f"{site} · F_topo（CSSC 端点 + openness/h1 方向）",
              f"unsup_v4_topo_{site}.png")
    f2 = save(fv, "viridis", f"{site} · F_visual（卷积自编码器瓶颈）",
              f"unsup_v4_visual_{site}.png")
    f3 = save(mask.astype(float), "gray",
              f"{site} · F_fused Otsu 掩膜（IoU={raw['iou']:.4f}）",
              f"unsup_v4_mask_{site}.png", truth=True)

    return dict(site=site, pct=a.pct, endpoints=ep_info, path_px=path_px,
                ft_range=ft_rng, fv_range=fv_rng, fused_range=fu_rng,
                otsu=thr, dilate=a.dilate, sigma=a.sigma, close=a.close,
                dir_lambda=a.dir_lambda, turn_lambda=a.turn_lambda,
                turn_min=a.turn_min, epochs=a.epochs, patches=n_patch,
                lr=a.lr, batch=a.batch, raw=raw, d10=d10,
                auc=dict(topo=a_topo, visual=a_vis, fused=a_fus),
                seconds=round(time.time() - t0, 1), figures=[f1, f2, f3])


def main():
    p = argparse.ArgumentParser(description="无监督双分支融合 v4（CSSC 端点 × 自编码器，SHP 仅评分）")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--pct", type=float, default=95.0, help="CSSC 端点百分位阈值")
    p.add_argument("--dilate", type=int, default=3, help="路径膨胀半径 px")
    p.add_argument("--sigma", type=float, default=2.0, help="融合后高斯 σ")
    p.add_argument("--close", type=int, default=1, help="掩膜闭运算次数（0=关）")
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--dir-gate", type=float, default=None, help="h1_deg 置信分位（默认 P60）")
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patches", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", help="cpu / cuda / npu 设备串")
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "unsup_two_branch_v4.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"无监督双分支融合 v4 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"端点 CSSC≥P{a.pct:g}  dilate={a.dilate}px σ={a.sigma} close={a.close} "
        f"λ_dir={a.dir_lambda} λ_turn={a.turn_lambda} m={a.turn_min}")
    log(f"自编码器 epochs={a.epochs} patches={a.patches} batch={a.batch} "
        f"lr={a.lr} device={a.device} seed={a.seed}")
    log("SHP 真值仅在最后算 IoU/F1 时读取；分支2 只重构输入自身，无蒸馏、无标签")

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

    jp = os.path.join(RES, "unsup_two_branch_v4.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="无监督双分支融合 v4：CSSC 端点拓扑门 × (1+卷积自编码器特征)，Otsu",
                       pct=a.pct, dilate=a.dilate, sigma=a.sigma, close=a.close,
                       dir_lambda=a.dir_lambda, turn_lambda=a.turn_lambda,
                       turn_min=a.turn_min, epochs=a.epochs, patches=a.patches,
                       lr=a.lr, batch=a.batch, seed=a.seed, sites=out), f,
                  ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
