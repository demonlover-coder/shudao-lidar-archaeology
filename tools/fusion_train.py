# -*- coding: utf-8 -*-
"""双分支特征融合与自监督训练：F1(图论 2ch) ⊕ F2(自编码器 4ch + 阴影 + CSSC 6ch) → CNN → 概率图。

输入
----
F1（宏观图论，2 通道）
  ch0 = 路径距离场 `clip(1 − d/3, 0, 1)`（1px Dijkstra 路径的欧氏距离衰减）
  ch1 = openness 的走廊侧响应 `1 − norm(openness)`（openness 低=更像走廊，故取反）
  路径由起止点跑状态增广 Dijkstra 得到：代价 openness + h1_phase 方向惩罚 + 转角惩罚。

F2（微观视觉，6 通道）
  ch0~3 = 无监督卷积自编码器瓶颈层（4×4×128）上采样后经 **PCA 降到 4 维**
  ch4 = Hillshade
  ch5 = CSSC（横断面二阶导符号交替次数，整数场）
  自编码器每站独立训练，目标只有重构 MSE；权重缓存在 `results/ae_{site}.pt`，
  若已存在则直接复用（"之前训练好的"）。

模型与训练
----------
Concat(2 + 6 = 8ch) → Conv3×3(8→32) + BN + ReLU → Conv3×3(32→1) → logit。参数量 2.7k。
训练用**自监督伪标签**：F1 路径膨胀 3px 作为正类，BCEWithLogits（pos_weight 平衡），
在 96×96 块上采样，直到 loss 平台。**SHP 全程不参与训练**，只在最后评分。

必须明确的一点（结构上限，不是调参问题）
--------------------------------------
伪标签 = F1 的路径膨胀 3px，而 F1 的 ch0 就是这条路径的距离场。模型最容易学到的解是
"对 ch0 做阈值"复刻伪标签，BCE 会持续把任何偏离推向该解。因此本脚本的**性能上限就是
分支 1 自身的掩膜**（实测 v7：杨家坝 0.1448、西坝村 0.0161），F2 只能在此约束下做
边缘微调，不可能"发现"伪标签里没有的走廊。脚本会同时输出：
  · F1 伪标签掩膜 的 IoU（上限）
  · 训练后模型 的 IoU
  · 模型 vs 伪标签 的 IoU（拟合度，越接近 1 说明越彻底地复刻了分支 1）

用法
----
    python tools/fusion_train.py
    python tools/fusion_train.py --sites 杨家坝 --epochs 40
    python tools/fusion_train.py --ep "119,344;460,186"      # 手动指定端点（row,col）
    python tools/fusion_train.py --retrain-ae                # 重训自编码器
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import disk, skeletonize
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import path_extraction as P                                       # noqa: E402
from eval_linear_metrics import window_mask                        # noqa: E402
from fix_tpi31_auc import fill_nodata                              # noqa: E402
from unsup_two_branch_v5 import _ae, _patches, PATCH, STRIDE_DOWN   # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
SITES = ("杨家坝", "西坝村")
NB8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# ------------------------------------------------------------ 输入 ----
def load_terrain(site):
    """(dem 填洞, Hillshade(315°/45°), valid)。"""
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
    G, pts = skel_graph(skel)
    d0 = dijkstra(G, indices=0)
    d0 = np.where(np.isfinite(d0), d0, -1.0)
    i = int(np.argmax(d0))
    d1 = dijkstra(G, indices=i)
    d1 = np.where(np.isfinite(d1), d1, -1.0)
    j = int(np.argmax(d1))
    return pts[i], pts[j], float(d1[j])


def auto_endpoints(site, valid, pct, r, log):
    """CSSC≥P{pct} → close → dilate(r) → 最大连通域 → 骨架双次扫描直径。"""
    cssc = np.load(os.path.join(RES, f"cs_fields_{site}.npz"))["cssc"].astype(np.float64)
    v = cssc[valid]; v = v[np.isfinite(v)]
    thr = float(np.percentile(v, pct))
    raw_b = (cssc >= thr) & valid & np.isfinite(cssc)
    from scipy.ndimage import binary_closing
    b = binary_dilation(binary_closing(raw_b, np.ones((3, 3), bool)), disk(r)) & valid
    lab, ncc = cc_label(b)
    comp = lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)
    skel = skeletonize(comp)
    p, q, gl = diameter_pair(skel)
    log(f"  端点（CSSC≥P{pct:g}={thr:.0f}，连通域 {comp.sum():,}px，骨架 {skel.sum():,}px，"
        f"测地 {gl:.0f}px）→ ({p[0]:.0f},{p[1]:.0f}) ↔ ({q[0]:.0f},{q[1]:.0f})")
    return (p, q), dict(thr=thr, cc_px=int(comp.sum()), skel_px=int(skel.sum()),
                        geo_len_px=round(gl, 1), n_cc=int(ncc))


def shp_endpoints(site, valid, merge_px=10.0, log=print):
    """半监督设定：SHP 真值 mask 骨架的**最远两点**，仅取坐标作为起止点。

    取法遵循项目既有约定（path_extraction 的骨架端点合并法）：先取骨架端点
    （8 邻域仅 1 个邻居），10px 内合并近邻，再取欧氏距离最大的一对。这样在真值面
    被分成多块时（西坝村 3,711+2,148 px）也能取到跨块的走廊两端；单块时与测地
    直径一致（杨家坝两种算法给出同一对）。

    注意：本模式把 SHP 用作了坐标输入（不是训练标签、不是融合特征），因此结果
    属于"给定起止点"的上界参考，不能再当作全自动无 SHP 的结果汇报。
    """
    from path_extraction import find_endpoints, merge_and_pick_pair

    mask = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    lab, ncc = cc_label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    skel = skeletonize(mask)
    eps = np.argwhere(find_endpoints(skel))
    a_, b_ = merge_and_pick_pair(eps, merge_px=merge_px)
    p, q, gl = diameter_pair(skel)
    log(f"  真值面 {int(mask.sum()):,}px / {ncc} 块（最大 {int(sizes.max()):,}px）")
    log(f"  端点（SHP 骨架最远两点，仅坐标）：({a_[0]:.0f},{a_[1]:.0f}) ↔ "
        f"({b_[0]:.0f},{b_[1]:.0f})，欧氏 {np.hypot(a_[0]-b_[0], a_[1]-b_[1]):.0f}px；"
        f"骨架测地直径 ({p[0]:.0f},{p[1]:.0f})↔({q[0]:.0f},{q[1]:.0f}) {gl:.0f}px")
    return (a_, b_), dict(source="shp_skeleton_far_pair", n_cc=int(ncc),
                          truth_px=int(mask.sum()), skel_px=int(skel.sum()),
                          geo_diam_px=round(gl, 1),
                          euc_len_px=round(float(np.hypot(a_[0] - b_[0], a_[1] - b_[1])), 1),
                          ep0=[float(a_[0]), float(a_[1])],
                          ep1=[float(b_[0]), float(b_[1])],
                          geo_ep0=[float(p[0]), float(p[1])],
                          geo_ep1=[float(q[0]), float(q[1])])


# ------------------------------------------------------------ F1 ----
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


def build_f1(site, valid, ep, gate=3.0, dir_lambda=1.5, turn_lambda=0.2, turn_min=1,
             log=print):
    """F1 = (距离场门控, openness 走廊响应)。同时返回 1px 路径与伪标签。"""
    dom = valid.copy()
    H, W = dom.shape
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    cost, _ = P.build_cost(site, tpi31, dom, 0.05, "pct", "open")
    open_feat = np.clip(1.0 - (cost - 0.05), 0.0, 1.0)          # 高=更像走廊

    nr, nc, conf, gate_thr = P.direction_field(site, P.DFS_GATE)
    (rows, cols, data), N = P.build_state_graph(cost, dom, nr, nc, conf,
                                                lam_dir=dir_lambda, lam_turn=turn_lambda,
                                                turn_min=turn_min)
    s = int(round(ep[0][0]) * W + round(ep[0][1]))
    e = int(round(ep[1][0]) * W + round(ep[1][1]))
    path = dijkstra_on(rows, cols, data, N, s, e)
    if path is None:
        log("  [FAIL] 无连通路径")
        return None
    path = path.reshape(H, W)

    dist = distance_transform_edt(~path)
    ch0 = np.where(dom, np.clip(1.0 - dist / gate, 0.0, 1.0), 0.0)
    ch1 = np.where(dom, open_feat, 0.0)
    f1 = np.stack([ch0, ch1], axis=0)
    log(f"  F1：路径 {int(path.sum())}px；ch0 带内 {(ch0 > 0).sum():,}px；"
        f"ch1∈[{ch1[dom].min():.3f},{ch1[dom].max():.3f}]（λ_dir={dir_lambda}，"
        f"h1 门控 {gate_thr:.1f}°）")
    return dict(f1=f1, path=path, dist=dist, cost=cost)


# ------------------------------------------------------------ F2 ----
def load_or_train_ae(site, hs, valid, epochs, patches, seed, device, retrain, log):
    """载入 results/ae_{site}.pt；不存在或 --retrain-ae 则用无监督 MSE 训练并保存。"""
    import torch
    import torch.nn as nn

    ck = os.path.join(RES, f"ae_{site}.pt")
    enc, dec = _ae()
    if os.path.exists(ck) and not retrain:
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        enc.load_state_dict(sd["enc"]); dec.load_state_dict(sd["dec"])
        log(f"  F2：载入已训练自编码器 {os.path.relpath(ck, ROOT)}"
            f"（当时 MSE {sd.get('mse', float('nan')):.6f}，epochs {sd.get('epochs', '?')}）")
        return enc, float(sd.get("mse", float("nan")))

    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    X = _patches(hs, valid, patches, rng)
    net = nn.Sequential(enc, dec).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    ds = torch.from_numpy(X[:, None])
    log(f"  F2：无监督训练自编码器（{len(X)} 块 {PATCH}×{PATCH}，{epochs} epoch，device={dev}）")
    mse = float("nan")
    t0 = time.time()
    for epn in range(1, epochs + 1):
        perm = torch.randperm(len(ds)); tot = 0.0
        for i in range(0, len(ds), 64):
            xb = ds[perm[i:i + 64]].to(dev)
            loss = lossf(net(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.detach().item() * len(xb)
        mse = tot / len(ds)
        if epn == 1 or epn % 5 == 0 or epn == epochs:
            log(f"    epoch {epn:3d}/{epochs}  MSE {mse:.6f}  ({time.time() - t0:.0f}s)")
    torch.save(dict(enc=enc.state_dict(), dec=dec.state_dict(),
                    mse=mse, epochs=epochs, patches=len(X), seed=seed), ck)
    log(f"    保存 {os.path.relpath(ck, ROOT)}")
    return enc, mse


def bottleneck_map(enc, hs, device="cpu"):
    """整幅图过 Encoder → 瓶颈 (128, H/16, W/16) → 双线性上采样回 (128, H, W)。"""
    import torch
    import torch.nn.functional as Fn

    H, W = hs.shape
    ph, pw = (-H) % STRIDE_DOWN, (-W) % STRIDE_DOWN
    img = np.pad(hs, ((0, ph), (0, pw)), mode="reflect").astype(np.float32)
    enc.eval()
    with torch.no_grad():
        b = enc(torch.from_numpy(img[None, None]).to(device))
        f = Fn.interpolate(b, size=(H + ph, W + pw), mode="bilinear", align_corners=False)
        f = f[0, :, :H, :W].cpu().numpy().astype(np.float64)
    return f, tuple(b.shape)


def pca4(bt, valid, seed=0, n=20000):
    """瓶颈 128 通道 → 4 维主成分（在 valid 像元上拟合，无标签，确定性）。"""
    C, H, W = bt.shape
    idx = np.flatnonzero(valid.ravel())
    if len(idx) > n:
        idx = np.random.default_rng(seed).choice(idx, size=n, replace=False)
    X = bt.reshape(C, -1)[:, idx].T                       # (n, C)
    mu = X.mean(0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:4]                                          # (4, C)
    flat = (bt.reshape(C, -1) - mu[:, None]).T @ comp.T     # (H*W, 4)
    out = flat.T.reshape(4, H, W)
    return out, mu, comp


def build_f2(site, hs, valid, enc, device, log):
    """F2 = 4 通道 AE 主成分 + Hillshade + CSSC。每通道 valid 内归一化到 [0,1]。"""
    bt, bshape = bottleneck_map(enc, hs, device)
    pc, _, _ = pca4(bt, valid)
    cssc = np.load(os.path.join(RES, f"cs_fields_{site}.npz"))["cssc"].astype(np.float64)

    chans, rngs = [], []
    for k in range(4):
        g = np.where(valid, pc[k], 0.0)
        v = g[valid]
        lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
        g = np.clip((g - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        chans.append(np.where(valid, g, 0.0)); rngs.append((lo, hi))
    chans.append(np.where(valid, hs, 0.0))
    cmax = float(np.nanmax(cssc[valid]))
    chans.append(np.where(valid, np.clip(cssc / max(cmax, 1e-9), 0.0, 1.0), 0.0))
    f2 = np.stack(chans, axis=0)
    log(f"  F2：瓶颈 {bshape} → PCA 4 维（各维 P1~P99 "
        + " / ".join(f"[{lo:.3f},{hi:.3f}]" for lo, hi in rngs) + "）"
        f" + Hillshade + CSSC（÷{cmax:.0f}）→ {f2.shape}")
    return f2


# ------------------------------------------------------------ 模型 ----
def build_net(in_ch=8, width=32):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_ch, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width),
        nn.ReLU(inplace=False),
        nn.Conv2d(width, 1, 3, 1, 1, bias=True))


def train_net(X, Y, valid, epochs, batch, lr, patch, seed, device, log):
    """在 patch 上采样训练：X (8,H,W)、Y 伪标签、valid。返回 (net, 曲线)。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    dev = torch.device(device)

    net = build_net(X.shape[0]).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    # 软标签下用"硬正类"计数，避免 pos_weight 被高斯尾稀释后爆炸（二值标签结果不变）
    npos, nneg = int((Y[valid] >= 0.5).sum()), int((Y[valid] < 0.5).sum())
    pw = torch.tensor([nneg / max(npos, 1)], dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    log(f"  模型：8→32→1，参数量 {npar / 1e3:.1f}k；伪标签 正 {npos:,} / 负 {nneg:,}px，"
        f"pos_weight={float(pw):.1f}；loc={patch}px batch={batch} lr={lr}")

    H, W = Y.shape
    m = patch // 2
    cand = np.argwhere(valid[m:H - m, m:W - m]) + m
    Xt = torch.from_numpy(X.astype(np.float32))
    Yt = torch.from_numpy(Y.astype(np.float32))
    rng = np.random.default_rng(seed)

    hist, best, patience = [], float("inf"), 0
    t0 = time.time()
    for epn in range(1, epochs + 1):
        pick = rng.choice(len(cand), size=min(batch * 4, len(cand)), replace=False)
        tot, nb = 0.0, 0
        for i in range(0, len(pick), batch):
            rows = cand[pick[i:i + batch]]
            xb = torch.stack([Xt[:, r - m:r + m, c - m:c + m] for r, c in rows]).to(dev)
            yb = torch.stack([Yt[r - m:r + m, c - m:c + m] for r, c in rows]).to(dev)
            vb = torch.stack([torch.from_numpy(valid[r - m:r + m, c - m:c + m])
                              for r, c in rows]).to(dev)
            logits = net(xb)[:, 0]
            loss = lossf(logits[vb], yb[vb])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.detach().item(); nb += 1
        hist.append(tot / max(nb, 1))
        if epn == 1 or epn % 5 == 0 or epn == epochs:
            log(f"    epoch {epn:3d}/{epochs}  BCE {hist[-1]:.5f}  ({time.time() - t0:.0f}s)")
        if hist[-1] < best - 1e-4:
            best, patience = hist[-1], 0
        else:
            patience += 1
            if patience >= 5 and epn >= 10:
                log(f"    收敛：连续 5 epoch 无改善（best {best:.5f}），停在 {epn}")
                break
    return net, hist


def infer(net, X, valid, device, tile=384, log=print):
    """分块推理（避免整幅 32 通道激活占满内存），返回 sigmoid 概率图。"""
    import torch
    import torch.nn.functional as Fn

    H, W = valid.shape
    net.eval()
    prob = np.zeros((H, W), dtype=np.float64)
    Xt = torch.from_numpy(X.astype(np.float32))[None]
    ov = 32
    with torch.no_grad():
        for r0 in range(0, H, tile):
            for c0 in range(0, W, tile):
                r1, c1 = min(r0 + tile, H), min(c0 + tile, W)
                a0, b0 = max(0, r0 - ov), max(0, c0 - ov)
                a1, b1 = min(H, r1 + ov), min(W, c1 + ov)
                p = torch.sigmoid(net(Xt[:, :, a0:a1, b0:b1].to(device)))[0, 0]
                prob[r0:r1, c0:c1] = p[r0 - a0:r1 - a0, c0 - b0:c1 - b0].cpu().numpy()
    log(f"  推理完成（tile {tile}px + {ov}px 重叠），prob∈"
        f"[{prob[valid].min():.4f},{prob[valid].max():.4f}]")
    return prob


# ------------------------------------------------------------ 主 ----
def run_site(site, a, log):
    t0 = time.time()
    dem, hs, valid = load_terrain(site)
    log(f"\n[{site}] 网格 {hs.shape}  valid {int(valid.sum()):,}px")

    if a.ep:
        p, q = [np.array([float(x) for x in s.split(",")]) for s in a.ep.split(";")]
        ep, ep_info = (p, q), dict(source="cli", ep0=p.tolist(), ep1=q.tolist())
        log(f"  端点（命令行）→ ({p[0]:.0f},{p[1]:.0f}) ↔ ({q[0]:.0f},{q[1]:.0f})")
    elif a.ep_shp:
        ep, ep_info = shp_endpoints(site, valid, log=log)
    else:
        ep, ep_info = auto_endpoints(site, valid, a.pct, a.r, log)
        ep_info["source"] = "cs_sc_auto"

    r1 = build_f1(site, valid, ep, gate=a.gate, dir_lambda=a.dir_lambda,
                  turn_lambda=a.turn_lambda, turn_min=a.turn_min, log=log)
    if r1 is None:
        return None
    f1, path, dist = r1["f1"], r1["path"], r1["dist"]
    pseudo = (dist <= 3.0) & valid                       # 伪标签：路径膨胀 3px

    enc, ae_mse = load_or_train_ae(site, hs, valid, a.ae_epochs, a.ae_patches,
                                   a.seed, a.device, a.retrain_ae, log)
    f2 = build_f2(site, hs, valid, enc, a.device, log)

    X = np.concatenate([f1, f2], axis=0)                  # (8, H, W)
    X = np.where(valid[None], X, 0.0)
    log(f"  融合输入 {X.shape}（F1 2ch + F2 6ch）")

    net, hist = train_net(X, pseudo.astype(np.float32), valid, a.epochs, a.batch,
                          a.lr, a.patch, a.seed, a.device, log)
    prob = infer(net, X, valid, a.device, log=log)
    mask = (prob > a.thr) & valid

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
                    pred_px=int((m & sel).sum()))

    def ov(a_, b_):
        u = int((a_ | b_).sum())
        return float((a_ & b_).sum() / u) if u else 0.0

    out = dict(site=site, endpoints=ep_info, pseudo_px=int(pseudo.sum()),
               ae_mse=ae_mse, n_epochs=len(hist),
               raw_model=sc(mask, valid), d10_model=sc(mask, valid & win),
               raw_pseudo=sc(pseudo, valid), d10_pseudo=sc(pseudo, valid & win),
               raw_b1_tight=sc((f1[0] > 0.5) & valid, valid),
               model_vs_pseudo=ov(mask, pseudo),
               prob_auc=float(roc_auc_score(
                   label[valid].astype(int),
                   prob[valid])) if label[valid].any() else float("nan"))

    log(f"  纯分支1（dist≤3px，即伪标签）：IoU={out['raw_pseudo']['iou']:.4f} "
        f"F1={out['raw_pseudo']['f1']:.4f} P={out['raw_pseudo']['precision']:.4f} "
        f"R={out['raw_pseudo']['recall']:.4f}")
    log(f"  纯分支1（W>0.5，dist<1.5px）：IoU={out['raw_b1_tight']['iou']:.4f} "
        f"F1={out['raw_b1_tight']['f1']:.4f} "
        f"P={out['raw_b1_tight']['precision']:.4f} R={out['raw_b1_tight']['recall']:.4f}")
    d = out["raw_model"]
    log(f"  模型掩膜（prob>{a.thr:g}）：IoU={d['iou']:.4f} F1={d['f1']:.4f} "
        f"P={d['precision']:.4f} R={d['recall']:.4f} "
        f"(TP {d['tp']:,}/FP {d['fp']:,}/FN {d['fn']:,})")
    log(f"  模型 vs 伪标签 IoU={out['model_vs_pseudo']:.4f}（越接近 1=越彻底复刻分支 1）；"
        f"prob 对 SHP 的 AUC={out['prob_auc']:.3f}")

    os.makedirs(FIG, exist_ok=True)
    np.savez_compressed(os.path.join(RES, f"fusion_fields_{site}.npz"),
                        f1=f1.astype(np.float32), f2=f2.astype(np.float32),
                        prob=prob.astype(np.float32), mask=mask.astype(np.uint8),
                        pseudo=pseudo.astype(np.uint8))
    import torch
    torch.save(net.state_dict(), os.path.join(RES, f"fusion_net_{site}.pt"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(2, 2, figsize=(11.6, 10.4), dpi=140)
    ax[0, 0].imshow(np.where(valid, f1[0], np.nan), cmap="viridis")
    ax[0, 0].set_title("F1 ch0 路径距离场", fontsize=10)
    ax[0, 1].imshow(np.where(valid, prob, np.nan), cmap="magma")
    ax[0, 1].contour(pseudo, levels=[0.5], colors="#66ff66", linewidths=0.9)
    ax[0, 1].set_title("模型概率（绿=伪标签）", fontsize=10)
    ax[1, 0].imshow(np.where(valid, mask.astype(float), np.nan), cmap="gray")
    ax[1, 0].contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.2)
    ax[1, 0].set_title(f"模型掩膜（IoU={d['iou']:.4f}）", fontsize=10)
    ax[1, 1].imshow(np.where(valid, hs, np.nan), cmap="gray")
    ax[1, 1].contour(label, levels=[0.5], colors="#00e5ff", linewidths=1.2)
    ax[1, 1].set_title("Hillshade + SHP 真值", fontsize=10)
    for a_ in ax.flat:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{site} · 双分支融合（8ch → CNN）+ 自监督伪标签", fontsize=11)
    fig.tight_layout()
    fp = os.path.join(FIG, f"fusion_{site}.png")
    fig.savefig(fp); plt.close(fig)

    out.update(seconds=round(time.time() - t0, 1), figure=os.path.relpath(fp, ROOT),
               hist=[round(h, 6) for h in hist])
    return out


def main():
    p = argparse.ArgumentParser(description="双分支特征融合 + 自监督训练（伪标签=F1 路径膨胀 3px）")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--ep", type=str, default=None,
                   help='手动端点 "r,c;r,c"（像素行,列），缺省则按 CSSC P{pct} 自动取')
    p.add_argument("--ep-shp", action="store_true",
                   help="用 SHP 骨架最远两点作端点（半监督：仅坐标，不读面要素）")
    p.add_argument("--pct", type=float, default=95.0)
    p.add_argument("--r", type=int, default=5, help="CSSC 膨胀半径 px")
    p.add_argument("--gate", type=float, default=3.0, help="F1 距离场衰减带宽 px")
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    p.add_argument("--ae-epochs", type=int, default=20)
    p.add_argument("--ae-patches", type=int, default=1500)
    p.add_argument("--retrain-ae", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "fusion_train.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True); lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"双分支融合训练 fusion_train · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"F1=2ch(路径距离场, openness) | F2=6ch(AE 主成分4, Hillshade, CSSC)")
    log(f"伪标签=F1 路径膨胀 3px；BCE(pos_weight)；loc {a.patch}px batch {a.batch} "
        f"lr {a.lr} epochs≤{a.epochs} device={a.device}")
    log("SHP 全程不参与训练，只在最后评分")
    if a.ep_shp:
        log("注意：本轮 --ep-shp 把 SHP 骨架最远两点用作**坐标输入**（不读面要素），"
            "属半监督设定，结果应作为给定起止点的参考上界")
    src = "shp_ep" if a.ep_shp else ("cli_ep" if a.ep else "cssc_auto")

    out = []
    for s in a.sites:
        r = run_site(s, a, log)
        if r:
            out.append(r)

    log("\n" + "=" * 78)
    log(f"{'站点':>8s} {'方案':>8s} {'IoU':>7s} {'F1':>7s} {'P':>7s} {'R':>7s} "
        f"{'vs伪标签':>9s} {'prob-AUC':>9s}")
    log("-" * 70)
    for r in out:
        for k, nm in (("raw_pseudo", "伪标签"), ("raw_model", "模型")):
            d = r[k]
            log(f"{r['site']:>8s} {nm:>8s} {d['iou']:>7.4f} {d['f1']:>7.4f} "
                f"{d['precision']:>7.4f} {d['recall']:>7.4f} "
                f"{r['model_vs_pseudo']:>9.4f} {r['prob_auc']:>9.3f}")

    jp = os.path.join(RES, f"fusion_train_{src}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="双分支融合 + 自监督（伪标签=F1 路径膨胀3px）",
                       ep_source=src,
                       gate=a.gate, dir_lambda=a.dir_lambda, turn_lambda=a.turn_lambda,
                       epochs=a.epochs, patch=a.patch, batch=a.batch, lr=a.lr,
                       thr=a.thr, device=a.device, sites=out), f,
                  ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
