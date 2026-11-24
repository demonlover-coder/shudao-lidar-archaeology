#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pointcloud_hyper_filter.py —— 不依赖 SHP 的高阶纯点云物理滤波

仅用局部多维度物理特征从原始 LAS 点云中提取古栈道石板点，不使用任何矢量先验。

流程
----
1. 粗筛（回波法）：keep = (rn == 1) | (rn == nr)，剔除全部中间回波。
2. 对粗筛点做体素抽稀（默认 0.15 m）后建 KD-Tree，半径 r = 1.5 m 取邻域。
3. 邻域 PCA（lambda1 > lambda2 > lambda3；T = lambda1 特征向量，N = lambda3 特征向量）：
   A 平整度   Planarity  = (lambda2 - lambda3) / lambda1      > thr_plan
   B 各向异性 Anisotropy = (lambda1 - lambda3) / lambda1      > thr_aniso
   C 横断面不对称：过当前点、沿横向轴左右各取 c_half 范围，|ΔZ| > thr_dz
   D 局部洼地 TPI：Z - Z_mean(邻域) < thr_tpi
   得分：A&B → +1，C → +1，D → +1；Score ∈ {0,1,2,3}
4. Score >= score_min 判为路面点；导出 LAS。
5. 以 SHP 真值统计召回率 / 精确率 / 背景误留率，并出图。

两个实现口径说明（与原始需求文字的差异，均为可复现的量化选择）
------------------------------------------------------------
* C 的横向轴默认取 `cross`：u = normalize(ez × T)，即与 T 垂直的水平方向（横断面左右）。
  原因：PCA 拟合出的 N 是局部平面法向，邻域内所有点的 q·N ≈ 0，按 N 的正负分裂只会在
  平面内噪声上随机二分，ΔZ 恒为 0，特征退化。若确要按 N 分裂，用 `--c-axis normal`。
* 邻域密度很高（两站原始点距约 3~5 cm，r = 1.5 m 球内含数千点），因此先在 0.15 m 体素上
  打分（球内约 300 点，K = 512 不截断，半径仍严格为 1.5 m），再把 Score 按最近体素点继承
  回全分辨率点，保证导出点云仍是全密度。

输入
----
`data_real/{site}_xyz_local.npy`  (N,3) float32 局部坐标
`data_real/{site}_lasefeat.npy`   (N,3) float32 [return_number, number_of_returns, intensity]
`data_real/{site}_label.npy`      (N,)  bool    SHP 面内真值

用法
----
python pointcloud_hyper_filter.py --sites 杨家坝 西坝村
python pointcloud_hyper_filter.py --sites 杨家坝 --voxel 0.15 --c-axis cross
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get("SHUDAO_DATA_ROOT", os.path.abspath(os.path.join(HERE, "..", "data")))   # 数据根目录
DATA = os.path.join(PROJ, "data_real")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")

SITES = ("杨家坝", "西坝村")


# ---------------------------------------------------------------- 数据加载
def load_rule1(site: str):
    """读 npz 缓存，按规则 1 粗筛。返回 xyz / intensity / rn / nr / label。"""
    p_xyz = os.path.join(DATA, f"{site}_xyz_local.npy")
    p_ft = os.path.join(DATA, f"{site}_lasefeat.npy")
    p_lb = os.path.join(DATA, f"{site}_label.npy")
    for p in (p_xyz, p_ft, p_lb):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    n = np.load(p_lb, mmap_mode="r").shape[0]
    ft = np.load(p_ft, mmap_mode="r")
    rn = ft[:, 0]
    nr = ft[:, 1]

    keep = np.zeros(n, dtype=bool)
    step = 5_000_000
    for s in range(0, n, step):
        e = min(s + step, n)
        rn_c = np.rint(rn[s:e]).astype(np.int16)
        nr_c = np.rint(nr[s:e]).astype(np.int16)
        keep[s:e] = (rn_c == 1) | (rn_c == nr_c)

    idx = np.flatnonzero(keep)
    xyz_src = np.load(p_xyz, mmap_mode="r")
    out = np.empty((idx.size, 3), np.float32)
    inten = np.empty(idx.size, np.int16)
    rr = np.empty(idx.size, np.uint8)
    nn = np.empty(idx.size, np.uint8)
    for s in range(0, idx.size, step):
        e = min(s + step, idx.size)
        sub = idx[s:e]
        out[s:e] = xyz_src[sub]
        inten[s:e] = np.rint(np.asarray(ft[sub, 2])).astype(np.int16)
        rr[s:e] = np.rint(np.asarray(ft[sub, 0])).astype(np.uint8)
        nn[s:e] = np.rint(np.asarray(ft[sub, 1])).astype(np.uint8)

    label = np.load(p_lb, mmap_mode="r")[idx]
    return out, inten, rr, nn, np.asarray(label, dtype=bool), n


# ---------------------------------------------------------------- 体素抽稀
def voxel_representative_index(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """2D（XY）体素抽稀，每格取 z 最低点（近似地面）。voxel<=0 时返回全量下标。

    地形面是 (x, y) 的函数，因此体素只在水平面内划分；按 3D 体素划分会把同一 XY 位置
    不同高程的点全部保留，抽稀失效。取 z 最低点等价于在格内做 DTM 式代表点选择。
    """
    if voxel is None or voxel <= 0:
        return np.arange(xyz.shape[0], dtype=np.int64)
    mn = xyz[:, :2].min(axis=0).astype(np.float64)
    k = np.floor((xyz[:, :2].astype(np.float64) - mn) / voxel).astype(np.int64)
    dims = k.max(axis=0) + 1
    key = k[:, 0] * dims[1] + k[:, 1]
    order = np.lexsort((xyz[:, 2], key))          # 主键 = 格号，次键 = z 升序
    ks = key[order]
    first = np.empty(order.size, dtype=bool)
    first[0] = True
    np.not_equal(ks[1:], ks[:-1], out=first[1:])
    return order[first]


# ---------------------------------------------------------------- 局部特征
def local_features(tree: cKDTree, pts: np.ndarray, *, r: float, kmax: int,
                   batch: int, c_axis: str, c_half: float, min_pts: int = 10):
    """逐点计算 Planarity / Anisotropy / ΔZ / TPI。

    返回 (plan, aniso, dz, tpi)，无效点为 nan。
    """
    n = pts.shape[0]
    plan = np.full(n, np.nan, np.float32)
    aniso = np.full(n, np.nan, np.float32)
    dz = np.full(n, np.nan, np.float32)
    tpi = np.full(n, np.nan, np.float32)
    pts64 = pts.astype(np.float64)

    for st in range(0, n, batch):
        ed = min(st + batch, n)
        pq = pts64[st:ed]
        dist, ind = tree.query(pq, k=kmax, distance_upper_bound=r, workers=-1)
        if kmax == 1:
            dist = dist[:, None]
            ind = ind[:, None]

        ok = np.isfinite(dist)
        ip = np.where(ok, ind, 0)
        q = pts64[ip] - pq[:, None, :]                 # 邻域相对当前点的偏移
        cnt = ok.sum(axis=1)
        good = cnt >= min_pts
        if not np.any(good):
            continue

        okm = ok & good[:, None]
        qm = np.where(okm[..., None], q, 0.0)
        cntf = np.maximum(cnt, 1).astype(np.float64)

        # ---- PCA
        cov = np.einsum("bki,bkj->bij", qm, qm, optimize=True) / cntf[:, None, None]
        w, v = np.linalg.eigh(cov)                    # 升序：w[:,0] < w[:,1] < w[:,2]
        l3_, l2_, l1_ = w[:, 0], w[:, 1], w[:, 2]
        T = v[:, :, 2]
        N = v[:, :, 0]

        # ---- A 平整度 / B 各向异性
        denom = np.where(l1_ > 1e-12, l1_, np.nan)
        plan_b = (l2_ - l3_) / denom
        aniso_b = (l1_ - l3_) / denom

        # ---- 横向轴
        if c_axis == "normal":
            axis = N
            uok = np.ones(len(axis), bool)
        else:
            u = np.stack([-T[:, 1], T[:, 0], np.zeros(len(T))], axis=1)
            norm = np.linalg.norm(u, axis=1)
            uok = norm > 0.05                        # T 近竖直时横向轴无定义
            axis = u / np.maximum(norm, 1e-12)[:, None]

        # ---- C 横断面不对称
        proj = np.einsum("bki,bi->bk", qm, axis)
        band = okm & (np.abs(proj) <= c_half)
        gl = band & (proj > 0)
        gr = band & (proj <= 0)
        nl = gl.sum(axis=1)
        nr = gr.sum(axis=1)
        zq = qm[:, :, 2]
        ml = np.where(nl >= 3, np.where(gl, zq, 0.0).sum(axis=1) / np.maximum(nl, 1), np.nan)
        mr = np.where(nr >= 3, np.where(gr, zq, 0.0).sum(axis=1) / np.maximum(nr, 1), np.nan)
        dz_b = ml - mr

        # ---- D 局部洼地 TPI（当前点 z 偏移为 0）
        tpi_b = -(np.where(okm, zq, 0.0).sum(axis=1) / cntf)

        bad = (~good) | (~uok)
        plan_b = np.where(bad, np.nan, plan_b)
        aniso_b = np.where(bad, np.nan, aniso_b)
        dz_b = np.where(bad, np.nan, dz_b)
        tpi_b = np.where(bad, np.nan, tpi_b)

        plan[st:ed] = plan_b.astype(np.float32)
        aniso[st:ed] = aniso_b.astype(np.float32)
        dz[st:ed] = dz_b.astype(np.float32)
        tpi[st:ed] = tpi_b.astype(np.float32)

    return plan, aniso, dz, tpi


def score_points(plan, aniso, dz, tpi, *, thr_plan, thr_aniso, thr_dz, thr_tpi):
    """A&B → +1，C → +1，D → +1；无效点 -1。"""
    s = np.zeros(plan.shape[0], dtype=np.int8)
    s1 = (plan > thr_plan) & (aniso > thr_aniso)
    s2 = np.abs(dz) > thr_dz
    s3 = tpi < thr_tpi
    s[s1] += 1
    s[s2] += 1
    s[s3] += 1
    s[~np.isfinite(plan) | ~np.isfinite(dz) | ~np.isfinite(tpi)] = -1
    return s


# ---------------------------------------------------------------- 指标
def evaluate(score: np.ndarray, label: np.ndarray, score_min: int) -> dict:
    keep = score >= score_min
    n = int(label.size)
    n_road = int(label.sum())
    n_bg = n - n_road
    k_road = int((keep & label).sum())
    k_bg = int((keep & ~label).sum())
    nk = int(keep.sum())
    recall = k_road / max(n_road, 1)
    precision = k_road / max(nk, 1)
    keep_ratio = nk / max(n, 1)
    bg_keep = k_bg / max(n_bg, 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-12)
    return {
        "score_min": int(score_min),
        "n": n,
        "n_road": n_road,
        "n_background": n_bg,
        "n_kept": nk,
        "n_kept_road": k_road,
        "recall_shp": round(recall, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
        "keep_ratio": round(keep_ratio, 6),
        "background_keep_ratio": round(bg_keep, 6),
        "enrichment": round(recall / max(keep_ratio, 1e-12), 4),
    }


def score_histogram(score: np.ndarray, label: np.ndarray) -> list:
    rows = []
    for s in range(-1, 4):
        m = score == s
        n = int(m.sum())
        road = int((m & label).sum())
        rows.append({
            "score": int(s),
            "n": n,
            "n_road": road,
            "road_share": round(road / n, 6) if n else None,
        })
    return rows


def sweep_thresholds(plan, aniso, dz, tpi, label, *, thr_plan, thr_aniso,
                     dz_grid, tpi_grid):
    """在 (thr_dz, thr_tpi) 网格上扫描，报告最佳 F1 与对应召回。"""
    grid = []
    for tdz in dz_grid:
        for ttp in tpi_grid:
            s = score_points(plan, aniso, dz, tpi, thr_plan=thr_plan,
                             thr_aniso=thr_aniso, thr_dz=tdz, thr_tpi=ttp)
            best = None
            for sm in (2, 3):
                m = evaluate(s, label, sm)
                if best is None or m["f1"] > best["f1"]:
                    best = m
            best["thr_dz"] = round(float(tdz), 4)
            best["thr_tpi"] = round(float(ttp), 4)
            grid.append(best)
    grid.sort(key=lambda d: -d["f1"])
    return grid


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """秩统计 AUC（并列值取平均秩）。y 为 bool 正类。"""
    from scipy.stats import rankdata

    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(x, method="average")
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def feature_auc(plan, aniso, dz, tpi, label) -> dict:
    return {
        "planarity": round(_auc(plan, label), 4),
        "anisotropy": round(_auc(aniso, label), 4),
        "abs_dz": round(_auc(np.abs(dz), label), 4),
        "dz": round(_auc(dz, label), 4),
        "tpi": round(_auc(tpi, label), 4),
        "neg_tpi": round(_auc(-tpi, label), 4),
        "n_valid": int(np.isfinite(plan).sum()),
        "n_road_valid": int((np.isfinite(plan) & label).sum()),
    }


def swing_share(plan, aniso, dz, tpi, label, thr_plan, thr_aniso, thr_dz, thr_tpi) -> list:
    """四个判据在 SHP 内 / 背景上的触发比例（用于判断方向是否反了）。"""
    rows = []
    for name, m in (("A 平整度", plan > thr_plan),
                    ("B 各向异性", aniso > thr_aniso),
                    ("C 不对称", np.abs(dz) > thr_dz),
                    ("D 洼地", tpi < thr_tpi)):
        rows.append({
            "rule": name,
            "share_shp": round(float(m[label].mean()), 4),
            "share_background": round(float(m[~label].mean()), 4),
        })
    return rows


def sweep_broad(plan, aniso, dz, tpi, label, *,
                plan_grid, aniso_grid, dz_grid, tpi_grid) -> list:
    """扫描四判据阈值与方向，给出该特征族（保持 A&B + C + D 三段式）的最好成绩。

    方向翻转用乘 -1 实现：`plan_dir=lt` 等价于"平整度越低越像路面"。
    """
    grid = []
    dza = np.abs(dz)
    bad = ~np.isfinite(plan) | ~np.isfinite(dz) | ~np.isfinite(tpi)
    for p_sign in (1, -1):
        pa = p_sign * plan
        for p_thr in plan_grid:
            ab = pa > p_sign * p_thr
            for a_sign in (1, -1):
                aa = a_sign * aniso
                for a_thr in aniso_grid:
                    ab2 = ab & (aa > a_sign * a_thr)
                    for tdz in dz_grid:
                        c = dza > tdz
                        for ttp in tpi_grid:
                            d = tpi < ttp
                            s = (ab2.astype(np.int8) + c.astype(np.int8)
                                 + d.astype(np.int8))
                            s[bad] = -1
                            for sm in (1, 2, 3):
                                m = evaluate(s, label, sm)
                                m["thr_plan"] = round(float(p_thr), 3)
                                m["plan_dir"] = "gt" if p_sign > 0 else "lt"
                                m["thr_aniso"] = round(float(a_thr), 3)
                                m["aniso_dir"] = "gt" if a_sign > 0 else "lt"
                                m["thr_dz"] = round(float(tdz), 3)
                                m["thr_tpi"] = round(float(ttp), 3)
                                grid.append(m)
    grid.sort(key=lambda d: -d["f1"])
    return grid


# ---------------------------------------------------------------- 出图
def make_figure(site: str, xyz: np.ndarray, score: np.ndarray, label: np.ndarray,
                metrics: dict, score_min: int, plot_n: int, out_png: str):
    rng = np.random.default_rng(0)
    n = xyz.shape[0]
    sub = rng.choice(n, size=min(plot_n, n), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    # 面板 1：空间分布，按 Score 上色
    ax = axes[0]
    xs, ys = xyz[sub, 0], xyz[sub, 1]
    sc = score[sub]
    order = np.argsort(sc)                      # 高分后画，避免被覆盖
    colors = {-1: "#cccccc", 0: "#d9d9d9", 1: "#9ecae1", 2: "#fd8d3c", 3: "#a50f15"}
    for s in (-1, 0, 1, 2, 3):
        m = order[sc[order] == s]
        if m.size == 0:
            continue
        ax.scatter(xs[m], ys[m], s=0.05, c=colors[s], linewidths=0,
                   label=f"Score={s} ({m.size})", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{site}：保留点按 Score 上色（抽样 {sub.size:,} 点）")
    ax.legend(markerscale=8, fontsize=8, loc="best", framealpha=0.9)

    # 面板 2：Score 分布 —— SHP 内 vs 背景
    ax = axes[1]
    rows = score_histogram(score, label)
    s_vals = [r["score"] for r in rows if r["score"] >= 0]
    road_share = [r["road_share"] or 0.0 for r in rows if r["score"] >= 0]
    counts = [r["n_road"] for r in rows if r["score"] >= 0]
    ax.bar([str(v) for v in s_vals], road_share, color="#3182bd", alpha=0.85)
    ax.axhline(metrics["n_road"] / metrics["n"], color="k", ls="--", lw=1,
               label=f"基础率 {metrics['n_road'] / metrics['n']:.3%}")
    ax.set_ylim(0, max(0.35, max(road_share) * 1.15))
    ax.set_xlabel("Score")
    ax.set_ylabel("SHP 面内点占该 Score 的比例")
    ax.set_title("各 Score 档的蜀道纯度（柱顶标注 SHP 点数）")
    for i, (v, c) in enumerate(zip(road_share, counts)):
        ax.text(i, v + 0.005, f"{c:,}", ha="center", fontsize=7)
    ax.legend(fontsize=8)

    # 面板 3：阈值 → 召回 / 精确 / 保留率
    ax = axes[2]
    sms = [0, 1, 2, 3]
    rec, pre, kr = [], [], []
    for sm in sms:
        m = evaluate(score, label, sm)
        rec.append(m["recall_shp"] * 100)
        pre.append(m["precision"] * 100)
        kr.append(m["keep_ratio"] * 100)
    ax.plot(sms, rec, "o-", color="#a50f15", label="召回率（SHP 内捞回）")
    ax.plot(sms, pre, "s-", color="#3182bd", label="精确率")
    ax.plot(sms, kr, "^-", color="#969696", label="全场保留率")
    ax.axvline(score_min, color="k", ls=":", lw=1)
    ax.text(score_min + 0.05, max(rec) * 0.9, f"score_min={score_min}", fontsize=8)
    ax.set_xticks(sms)
    ax.set_xlabel("Score ≥")
    ax.set_ylabel("百分比 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("阈值-指标曲线")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{site} · 纯点云物理滤波（Score≥{score_min}）："
                 f"召回 {metrics['recall_shp']:.2%} / 精确 {metrics['precision']:.2%} / "
                 f"全场保留 {metrics['keep_ratio']:.2%}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    o = os.path.abspath(out_png)
    os.makedirs(os.path.dirname(o), exist_ok=True)
    fig.savefig(o, dpi=220)
    plt.close(fig)
    return o


# ---------------------------------------------------------------- 导出 LAS
def write_las(path: str, xyz: np.ndarray, intensity: np.ndarray,
              rn: np.ndarray, nr: np.ndarray, score: np.ndarray, keep: np.ndarray):
    import laspy

    x, y, z = xyz[keep, 0], xyz[keep, 1], xyz[keep, 2]
    hdr = laspy.LasHeader(point_format=0, version="1.2")
    hdr.offsets = [float(x.min()), float(y.min()), float(z.min())]
    hdr.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(hdr)
    las.x = x.astype(np.float64)
    las.y = y.astype(np.float64)
    las.z = z.astype(np.float64)
    las.intensity = np.clip(intensity[keep], 0, 65535).astype(np.uint16)
    las.return_number = np.clip(rn[keep], 1, 7).astype(np.uint8)
    las.number_of_returns = np.clip(nr[keep], 1, 7).astype(np.uint8)
    las.classification = np.ones(int(keep.sum()), np.uint8)
    las.user_data = np.clip(score[keep], 0, 255).astype(np.uint8)
    o = os.path.abspath(path)
    os.makedirs(os.path.dirname(o), exist_ok=True)
    las.write(o)
    return o


# ---------------------------------------------------------------- 主流程
def run_site(site: str, args) -> dict:
    t0 = time.time()
    xyz, inten, rn, nr, label, n_all = load_rule1(site)
    n_rule1 = xyz.shape[0]
    print(f"[{site}] 原始 {n_all:,} → 规则 1 保留 {n_rule1:,} "
          f"({n_rule1 / n_all:.2%})，其中 SHP 内 {int(label.sum()):,}", flush=True)

    vox_idx = voxel_representative_index(xyz, args.voxel)
    pts = np.ascontiguousarray(xyz[vox_idx])
    print(f"[{site}] 体素 {args.voxel} m 抽稀 → {pts.shape[0]:,} 个打分点", flush=True)

    tree = cKDTree(pts)
    plan, aniso, dz, tpi = local_features(
        tree, pts, r=args.r, kmax=args.kmax, batch=args.batch,
        c_axis=args.c_axis, c_half=args.c_half, min_pts=args.min_pts)

    score_v = score_points(plan, aniso, dz, tpi, thr_plan=args.thr_plan,
                           thr_aniso=args.thr_aniso, thr_dz=args.thr_dz,
                           thr_tpi=args.thr_tpi)
    valid_vox = score_v >= 0
    print(f"[{site}] 特征有效体素 {int(valid_vox.sum()):,} / {pts.shape[0]:,}", flush=True)

    # 把体素 Score 继承回全分辨率
    sub_tree = cKDTree(pts[valid_vox])
    score_vs = score_v[valid_vox]
    score = np.full(n_rule1, -1, np.int8)
    step = 2_000_000
    for s in range(0, n_rule1, step):
        e = min(s + step, n_rule1)
        _, nn = sub_tree.query(xyz[s:e].astype(np.float64), k=1, workers=-1)
        score[s:e] = score_vs[nn]
    print(f"[{site}] Score 继承完成，用时 {time.time() - t0:.1f} s", flush=True)

    m = evaluate(score, label, args.score_min)
    m["site"] = site
    m["n_input"] = int(n_all)
    m["n_rule1"] = int(n_rule1)
    m["voxel_points"] = int(pts.shape[0])
    m["voxel_points_valid"] = int(valid_vox.sum())
    m["rule1_ratio"] = round(n_rule1 / n_all, 6)

    hist = score_histogram(score, label)

    # 诊断：单特征 AUC、四判据触发比例、宽网格扫描（均在有效体素点上）
    lab_v = label[vox_idx][valid_vox]
    pv, av, dv, tv = plan[valid_vox], aniso[valid_vox], dz[valid_vox], tpi[valid_vox]
    aucs = feature_auc(pv, av, dv, tv, lab_v)
    shares = swing_share(pv, av, dv, tv, lab_v, args.thr_plan, args.thr_aniso,
                         args.thr_dz, args.thr_tpi)

    # 阈值扫描（在体素点上做，快）
    sweep = sweep_thresholds(pv, av, dv, tv, lab_v,
                             thr_plan=args.thr_plan, thr_aniso=args.thr_aniso,
                             dz_grid=args.dz_grid, tpi_grid=args.tpi_grid)
    broad = sweep_broad(pv, av, dv, tv, lab_v,
                        plan_grid=args.plan_grid, aniso_grid=args.aniso_grid,
                        dz_grid=args.dz_grid, tpi_grid=args.tpi_grid)

    # 出图
    fig_path = make_figure(site, xyz, score, label, m, args.score_min,
                           args.plot_points,
                           os.path.join(FIG, f"pointcloud_hyper_filter_{site}.png"))

    # 导出 LAS
    las_path = None
    if not args.no_las:
        keep = score >= args.score_min
        las_path = write_las(os.path.join(DATA, f"{site}_hyper_filter.las"),
                             xyz, inten, rn, nr, score, keep)

    out = {
        "site": site,
        "params": {
            "rule1": "(rn == 1) | (rn == nr)",
            "r_m": args.r, "kmax": args.kmax, "voxel_m": args.voxel,
            "c_axis": args.c_axis, "c_half_m": args.c_half,
            "thr_planarity": args.thr_plan, "thr_anisotropy": args.thr_aniso,
            "thr_dz_m": args.thr_dz, "thr_tpi_m": args.thr_tpi,
            "score_min": args.score_min,
        },
        "metrics": m,
        "score_histogram": hist,
        "feature_auc": aucs,
        "rule_trigger_share": shares,
        "sweep_top10": sweep[:10],
        "sweep_broad_top10": broad[:10],
        "runtime_s": round(time.time() - t0, 1),
        "fig": fig_path,
        "las": las_path,
    }
    print(f"[{site}] 召回 {m['recall_shp']:.2%} / 精确 {m['precision']:.2%} / "
          f"F1 {m['f1']:.4f} / 全场保留 {m['keep_ratio']:.2%} / "
          f"背景误留 {m['background_keep_ratio']:.2%} / 富集比 {m['enrichment']:.2f}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="不依赖 SHP 的高阶纯点云物理滤波")
    ap.add_argument("--sites", nargs="+", default=list(SITES), choices=list(SITES))
    ap.add_argument("--r", type=float, default=1.5, help="邻域半径 (m)")
    ap.add_argument("--kmax", type=int, default=512, help="邻域点数上限")
    ap.add_argument("--batch", type=int, default=8192, help="批处理查询点数")
    ap.add_argument("--voxel", type=float, default=0.15, help="打分体素边长 (m)，0 = 不抽稀")
    ap.add_argument("--min-pts", type=int, default=10, help="PCA 最少邻域点数")
    ap.add_argument("--c-axis", choices=["cross", "normal"], default="cross")
    ap.add_argument("--c-half", type=float, default=1.0, help="横断面左右取样范围 (m)")
    ap.add_argument("--thr-plan", type=float, default=0.5)
    ap.add_argument("--thr-aniso", type=float, default=0.6)
    ap.add_argument("--thr-dz", type=float, default=0.2)
    ap.add_argument("--thr-tpi", type=float, default=-0.2)
    ap.add_argument("--score-min", type=int, default=2)
    ap.add_argument("--dz-grid", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--tpi-grid", nargs="+", type=float, default=[-0.1, -0.2, -0.3, -0.5])
    ap.add_argument("--plan-grid", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--aniso-grid", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    ap.add_argument("--plot-points", type=int, default=400_000)
    ap.add_argument("--no-las", action="store_true")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    out_path = os.path.join(RES, "pointcloud_hyper_filter.json")
    prev = []
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                old = json.load(f)
            prev = [r for r in (old if isinstance(old, list) else [old])
                    if r.get("site") not in args.sites]
        except Exception:
            prev = []

    results = list(prev)
    for site in args.sites:
        results.append(run_site(site, args))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
