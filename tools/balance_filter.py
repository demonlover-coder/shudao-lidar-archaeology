# -*- coding: utf-8 -*-
"""召回-精度平衡消融（balance_filter.py）。

目标：把召回率抬到 0.75 以上，同时精度不低于 0.20。三组策略对照纯 2D P≥0.5 基线。

策略
----
A **核心 + 高度软切**：`P≥0.5` 全保留；`0.1≤P<0.5` 的缓冲带只保留 `AGL≤--agl-a`（默认 2 m）。
B **宽管 + 贴地**：起止点跑分支 1 图论路径 → 3D 中线，保留 `D3d≤--tube`（默认 7.5 m）
   且 `AGL≤--agl-b`（默认 1.5 m）的点。
C **膨胀并集**：`P≥--thr-lo`（默认 0.1）全保留；再加上 `P≥0.5` 区域向外膨胀
   `--dil-px`（默认 10 px）内的首/末回波点。
D **KMeans 自适应缓冲带**（本版新增，C 的自适应替换）：`P≥0.5` 核心全保留；
   `0.1≤P<0.5` 的缓冲带取其 AGL 跑一维 `KMeans(n_clusters=2)`，只保留中心较低的
   一簇（地面），剔除中心较高的一簇（树冠），分界 = 两中心中点。

`--sweep` 追加参数扫描（A 的 AGL 档位、B 的管半径×贴地阈值、C 的膨胀半径×低阈值），
用来判断“目标 R≥0.75 & P≥0.20”是否存在可达点。

口径
----
* 概率图：`results/model_guided_prob_{site}.npz`（fusion_train CNN 全图推理），双线性取值。
* AGL：点高程 − 1 m DTM 双线性内插；域外点置 1e3（等价于永不通过高度约束）。
* 首/末回波：`(rn==1)|(rn==nr)`。
* C 的膨胀是栅格操作，点在膨胀区内用最近像元判定（与 A/B 的双线性口径略不同，符合各自定义）。
* SHP 真值 `data_real/{site}_label.npy` 只在最后算 P/R/F1。

用法
----
    python tools/balance_filter.py
    python tools/balance_filter.py --sweep
    python tools/balance_filter.py --sites 杨家坝 --agl-a 3 --tube 10 --agl-b 3
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from fusion_train import build_f1, load_terrain, shp_endpoints             # noqa: E402
from model_guided_3d_filter import bilinear                               # noqa: E402

# 模块名以数字开头，不能直接 import 语句
_t3 = importlib.import_module("3d_tube_filter")
metrics = _t3.metrics
path_polyline_3d = _t3.path_polyline_3d
tube_keep = _t3.tube_keep


def metrics_iou(keep, label):
    """在 3d_tube_filter.metrics 的基础上补一个点空间 IoU。"""
    m = metrics(keep, label)
    m["iou"] = float(m["tp"] / max(m["tp"] + m["fp"] + m["fn"], 1))
    return m

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")
SITES = ("杨家坝", "西坝村")
CHUNK = 4_000_000

# C 的扫描组合 (低阈值, 膨胀半径 px)
C_GRID = ((0.10, 10), (0.10, 20), (0.05, 10), (0.05, 20))

# 附加：2D 概率阈值前沿扫描（同一份逐点概率，零额外读取成本）
SCAN_THR = (0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.07, 0.05, 0.02)


# ---------------------------------------------------------------- 逐点特征
def load_pointwise(site, a, valid, dem, meta, log):
    """一次分块遍历算出 pv / AGL / 首末回波 / C 各变体的栅格命中，供所有策略复用。"""
    z = np.load(os.path.join(RES, f"model_guided_prob_{site}.npz"))
    prob, vmask = z["prob"].astype(np.float64), z["valid"] == 1
    H, W = prob.shape
    x0, y1, gsd = float(meta["x0"]), float(meta["y1"]), float(meta["gsd_m"])
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    ft = np.load(os.path.join(REAL, f"{site}_lasefeat.npy"), mmap_mode="r")
    n = xyz.shape[0]

    vf = vmask.astype(np.float64)
    p_base = (prob >= a.thr) & vmask
    # C 的栅格：低阈值区 + (P≥thr 膨胀区)
    variants = list(C_GRID) if a.sweep else [(a.thr_lo, a.dil_px)]
    r_lo = {lo: ((prob >= lo) & vmask) for lo, _ in variants}
    r_dil = {d: (binary_dilation(p_base, disk(d)) & vmask) for _, d in variants}

    pv = np.zeros(n, dtype=np.float32)
    agl = np.full(n, 1e3, dtype=np.float32)
    echo = np.zeros(n, dtype=bool)
    hit_lo = {k: np.zeros(n, dtype=bool) for k in variants}
    hit_dil = {k: np.zeros(n, dtype=bool) for k in variants}

    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        p = np.asarray(xyz[s:e], dtype=np.float64)
        f = np.asarray(ft[s:e])
        p += np.asarray(o, dtype=np.float64)[None, :]
        c = (p[:, 0] - x0) / gsd - 0.5
        r = (y1 - p[:, 1]) / gsd - 0.5
        ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
        ii = np.flatnonzero(ins)
        if ii.size:
            v = bilinear(prob, r[ii], c[ii])
            vb = bilinear(vf, r[ii], c[ii])
            inside = vb > 0.5
            pv[s + ii] = np.where(inside, v, 0.0)
            agl[s + ii] = np.where(inside, p[ii, 2] - bilinear(dem, r[ii], c[ii]), 1e3)
            rn_, c_ = np.rint(r[ii]).astype(np.int32), np.rint(c[ii]).astype(np.int32)
            for k in variants:
                hit_lo[k][s + ii] = r_lo[k[0]][rn_, c_]
                hit_dil[k][s + ii] = r_dil[k[1]][rn_, c_]
        rn, nr = np.rint(f[:, 0]).astype(np.int32), np.rint(f[:, 1]).astype(np.int32)
        echo[s:e] = (rn == 1) | (rn == nr)
        if (s // CHUNK) % 5 == 0:
            log(f"    逐点特征 {e / 1e6:.0f}M / {n / 1e6:.1f}M（{time.time() - t0:.0f}s）")

    label = np.asarray(np.load(os.path.join(REAL, f"{site}_label.npy"),
                               mmap_mode="r"), dtype=bool)
    log(f"  逐点特征完成（{time.time() - t0:.0f}s）：P 分位 P50/P90/P99 = "
        + " / ".join(f"{v:.3f}" for v in np.percentile(pv, [50, 90, 99]))
        + f"；AGL 有效点 {int((agl < 1e3).sum()) / n * 100:.1f}%；"
          f"首末回波 {echo.mean() * 100:.1f}%")
    return dict(pv=pv, agl=agl, echo=echo, label=label, n=n,
                hit_lo=hit_lo, hit_dil=hit_dil)


# ---------------------------------------------------------------- KMeans
def kmeans_agl_split(agl_band, seed, max_fit, log, tag=""):
    """对缓冲带点的 AGL 跑一维 KMeans(2)，返回地面/树冠分界与统计。

    一维情形下“保留中心较低的簇”严格等价于 `AGL ≤ 两中心中点`，因此先用拟合得到的
    中心算出分界，再对全量缓冲带做阈值判断，避免对几百万点做 predict。
    点数超过 `max_fit` 时只在子样本上拟合（一维 + 固定种子，中心几乎不变）。
    """
    from sklearn.cluster import KMeans

    m = int(agl_band.size)
    rng = np.random.default_rng(seed)
    fit = agl_band if m <= max_fit else agl_band[rng.choice(m, size=max_fit, replace=False)]
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(
        fit.reshape(-1, 1).astype(np.float64))
    c = np.sort(km.cluster_centers_.ravel())
    bound = float((c[0] + c[1]) / 2.0)
    lo = agl_band <= bound
    info = dict(bound_m=round(bound, 3),
                center_low_m=round(float(c[0]), 3),
                center_high_m=round(float(c[1]), 3),
                n_band=m, n_fit=int(fit.size),
                n_ground=int(lo.sum()), n_canopy=int((~lo).sum()),
                med_low_m=round(float(np.median(agl_band[lo])), 3) if lo.any() else None,
                med_high_m=round(float(np.median(agl_band[~lo])), 3) if (~lo).any() else None)
    log(f"  [{tag}] KMeans(2) on AGL：{m:,} 点（拟合样本 {fit.size:,}）→ "
        f"地面簇中心 {c[0]:.2f} m（{info['n_ground']:,} 点），"
        f"树冠簇中心 {c[1]:.2f} m（{info['n_canopy']:,} 点），"
        f"**自动分界 = {bound:.2f} m**")
    return info


def kmeans_keep(band, agl, a, log, tag):
    """对 `band` 内点的 AGL 跑 KMeans(2) 并只保留低簇。返回 (keep_band, info)。"""
    m = int(band.sum())
    if m < 10:                                     # 点太少，KMeans 无意义，整段保留
        log(f"  [{tag}] 仅 {m} 点，跳过 KMeans，整段保留")
        return band.copy(), dict(bound_m=None, n_band=m, n_ground=m, n_canopy=0,
                                 center_low_m=None, center_high_m=None,
                                 med_low_m=None, med_high_m=None, n_fit=0)
    info = kmeans_agl_split(agl[band], a.kmeans_seed, a.kmeans_fit, log, tag)
    return band & (agl <= info["bound_m"]), info


# ---------------------------------------------------------------- 微地形
def build_micro_from_points(site, pt, dem, meta, a, log):
    """D+ 地面簇 → AGL 最低 pct% → 0.5m 体素抽稀 → TIN → 微地形 DTM。

    返回 (micro, info)。`micro` 与原栅格同形状，支撑区外等于原值。
    """
    from micro_dtm_agl import build_micro_dtm, voxel_min_index

    x0, y1, gsd = float(meta["x0"]), float(meta["y1"]), float(meta["gsd_m"])
    pv, agl = pt["pv"], pt["agl"]
    base = pv >= a.thr
    c_main = (pt["hit_lo"][(a.thr_lo, a.dil_px)]
              | (pt["echo"] & pt["hit_dil"][(a.thr_lo, a.dil_px)]))
    addon = c_main & ~base
    info_k = kmeans_agl_split(agl[addon], a.kmeans_seed, a.kmeans_fit, log, "微地形分簇")
    ground = addon & (agl <= info_k["bound_m"])
    if a.micro_pct >= 100.0:
        low = ground
    else:
        cut = float(np.percentile(agl[ground], a.micro_pct))
        low = ground & (agl <= cut)
        log(f"  微地形取点：地面簇 {int(ground.sum()):,} → AGL≤P{a.micro_pct:g}"
            f"({cut:.3f} m) 后 {int(low.sum()):,} 点")

    idx = np.flatnonzero(low)
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    p = np.asarray(xyz[idx], dtype=np.float64) + np.asarray(o, dtype=np.float64)[None, :]
    gx, gy, gz = p[:, 0], p[:, 1], p[:, 2]
    del p

    k = voxel_min_index(gx, gy, gz, a.micro_cell)
    gx, gy, gz = gx[k], gy[k], gz[k]
    log(f"  微地形体素抽稀（{a.micro_cell:g} m，每格取最低点）→ {gx.size:,} 点入 TIN")

    micro, support, stats = build_micro_dtm(gx, gy, gz, dem, x0, y1, gsd,
                                            dilate_px=a.micro_dilate,
                                            smooth=a.micro_smooth, log=log)
    info = dict(kmeans=info_k, n_low=int(idx.size), n_tin=int(gx.size), **stats)
    return micro, info


def refresh_agl(pt, site, micro, meta, log, tag="AGL_new"):
    """用给定 DTM 栅格重算全部点 AGL（分块，不一次性展开 X/Y/Z）。"""
    x0, y1, gsd = float(meta["x0"]), float(meta["y1"]), float(meta["gsd_m"])
    H, W = micro.shape
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    n = xyz.shape[0]
    mg = np.where(np.isfinite(micro), micro, 0.0)
    agl = np.full(n, 1e3, dtype=np.float32)
    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        p = np.asarray(xyz[s:e], dtype=np.float64)
        p += np.asarray(o, dtype=np.float64)[None, :]
        c = (p[:, 0] - x0) / gsd - 0.5
        r = (y1 - p[:, 1]) / gsd - 0.5
        ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
        ii = np.flatnonzero(ins)
        if ii.size:
            agl[s + ii] = p[ii, 2] - bilinear(mg, r[ii], c[ii])
    old = pt["agl"]
    m = old < 1e3
    log(f"  {tag}（{time.time() - t0:.0f}s）：全域中位 "
        f"{np.median(old[m]):.3f} → {np.median(agl[m]):.3f} m；"
        f"变化>0.5m 的点 {int((np.abs(agl[m] - old[m]) > 0.5).sum()):,}"
        f"（{(np.abs(agl[m] - old[m]) > 0.5).mean() * 100:.1f}%）")
    pt["agl0"] = old
    pt["agl"] = agl
    return agl


# ---------------------------------------------------------------- 策略
def make_rows(pt, keeps, a, tubes, log):
    pv, agl, echo, label = pt["pv"], pt["agl"], pt["echo"], pt["label"]
    base = pv >= a.thr
    rows = [dict(name=f"基线 纯2D P≥{a.thr:g}", grp="base", **metrics_iou(base, label))]
    is_sweep = a.sweep

    # ---- A：核心 + 高度软切
    agl_a = [a.agl_a, 1.0, 3.0, 5.0] if is_sweep else [a.agl_a]
    for th in sorted(set(agl_a)):
        keep = base | ((pv >= a.thr_lo) & (pv < a.thr) & (agl <= th))
        tag = "" if abs(th - a.agl_a) < 1e-9 else " 扫描"
        rows.append(dict(name=f"A 核心+缓冲带 AGL≤{th:g}m{tag}", grp="A",
                         **metrics_iou(keep, label)))

    # ---- B：宽管 + 贴地
    b_grid = [(t, a.agl_b) for t in tubes]
    if is_sweep:
        b_grid = [(t, th) for t in tubes for th in (a.agl_b, 3.0)]
    for (t, th) in b_grid:
        keep = keeps[t] & (agl <= th)
        tag = "" if (abs(t - a.tube) < 1e-9 and abs(th - a.agl_b) < 1e-9) else " 扫描"
        rows.append(dict(name=f"B 宽管≤{t:g}m + AGL≤{th:g}m{tag}", grp="B",
                         **metrics_iou(keep, label)))

    # ---- C：膨胀并集
    c_grid = list(C_GRID) if is_sweep else [(a.thr_lo, a.dil_px)]
    c_keep = {}
    for (lo, d) in c_grid:
        keep = pt["hit_lo"][(lo, d)] | (echo & pt["hit_dil"][(lo, d)])
        c_keep[(lo, d)] = keep
        tag = "" if (abs(lo - a.thr_lo) < 1e-9 and d == a.dil_px) else " 扫描"
        rows.append(dict(name=f"C P≥{lo:g} ∪ 膨胀{d}px内首末回波{tag}", grp="C",
                         **metrics_iou(keep, label)))

    # ---- D：缓冲带 0.1≤P<0.5 上跑 KMeans(2)，只留低 AGL 簇（用户字面定义）
    band = (pv >= a.thr_lo) & (pv < a.thr)

    # ---- E：核心 + 缓冲带按 AGL_new 硬切（本次修正的主方案）
    e_grid = [a.agl_e, 0.5, 1.5, 2.0] if is_sweep else [a.agl_e]
    for th in sorted(set(e_grid)):
        keep = base | (band & (agl <= th))
        tag = "" if abs(th - a.agl_e) < 1e-9 else " 扫描"
        rows.append(dict(name=f"E 核心+缓冲带 AGL_new≤{th:g}m{tag}", grp="E",
                         **metrics_iou(keep, label)))

    kb, info_d = kmeans_keep(band, agl, a, log, "D 缓冲带 0.1≤P<0.5")
    keep_d = base | kb
    bd = info_d["bound_m"]
    rows.append(dict(name=f"D 缓冲带KMeans" + (f"(AGL≤{bd:.2f}m)" if bd else "(跳过分界)"),
                     grp="D", kmeans=info_d, **metrics_iou(keep_d, label)))

    # ---- D+：把 KMeans 作用到 C 的全部增量区（C−核心），这才是"基于 C 的自适应"
    #    C 的召回来主要靠膨胀区内的首末回波，0.1≤P<0.5 的窄带只占增量的一小部分。
    c_main = c_keep[(a.thr_lo, a.dil_px)]
    addon = c_main & ~base
    kb2, info_d2 = kmeans_keep(addon, agl, a, log, "D+ C增量区")
    keep_d2 = base | kb2
    bd2 = info_d2["bound_m"]
    rows.append(dict(name=f"D+ C增量区KMeans" + (f"(AGL≤{bd2:.2f}m)" if bd2 else ""),
                     grp="D", kmeans=info_d2, **metrics_iou(keep_d2, label)))
    log(f"  D/D+ 的缓冲带规模对比：0.1≤P<0.5 = {int(band.sum()):,} 点；"
        f"C−核心 = {int(addon.sum()):,} 点")

    # ---- 附加：2D 阈值前沿（回答“目标是否可达”）
    if a.scan:
        for t in SCAN_THR:
            rows.append(dict(name=f"前沿 纯2D P≥{t:g}", grp="S",
                             **metrics_iou(pv >= t, label)))
    return rows


def plot(site, rows, a, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图上只画 基线 + A/B/C/D/E 的主配置（扫描行不进图）
    main = [r for r in rows if r["grp"] in ("base", "A", "B", "C", "D", "E")
            and "扫描" not in r["name"]]
    x = np.arange(len(main))
    w = 0.27
    fig, ax = plt.subplots(figsize=(14.6, 5.6), dpi=170)
    b1 = ax.bar(x - w, [r["precision"] for r in main], w, color="#4c8bf5", label="Precision")
    b2 = ax.bar(x, [r["recall"] for r in main], w, color="#e8590c", label="Recall")
    b3 = ax.bar(x + w, [r["f1"] for r in main], w, color="#2f9e44", label="F1")
    for bars, key in ((b1, "precision"), (b2, "recall"), (b3, "f1")):
        for b, r in zip(bars, main):
            ax.text(b.get_x() + b.get_width() / 2, r[key] + 0.012, f"{r[key]:.2f}",
                    ha="center", fontsize=8.5)
    ax.axhline(0.75, ls="--", lw=1.2, c="#e8590c", alpha=0.8)
    ax.text(-1.12, 0.762, "召回目标 0.75", color="#e8590c", fontsize=9.5,
            ha="left", va="bottom")
    ax.axhline(0.20, ls="--", lw=1.2, c="#4c8bf5", alpha=0.8)
    ax.text(-1.12, 0.212, "精度下限 0.20", color="#4c8bf5", fontsize=9.5,
            ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"].replace(" ", "\n", 1) for r in main], fontsize=9.5)
    for xi, r in zip(x, main):                 # 达标组打星标
        if r["recall"] >= 0.75 and r["precision"] >= 0.20:
            ax.text(xi, 1.015, "★ 达标", ha="center", fontsize=10.5,
                    fontweight="bold", color="#2f9e44")
    ax.set_ylim(0, 1.07); ax.set_ylabel("分数")
    ax.set_xlim(-1.18, len(main) - 0.45)      # 左侧留白给参考线标注
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    # 图例放到坐标区下方，避免与右上角的达标星标冲突
    ax.legend(fontsize=10, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False)
    ax.set_title(f"{site} · 召回-精度平衡消融", fontsize=12)
    fig.tight_layout()
    fp = os.path.join(FIG, f"balance_filter_{site}.png")
    fig.savefig(fp); plt.close(fig)
    return fp


def main():
    p = argparse.ArgumentParser(description="召回-精度平衡消融")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--base", choices=("mean", "min"), default="min",
                   help="用哪个栅格当基准地形算 AGL：min=像元最低高程（默认），mean=像元平均高程")
    p.add_argument("--thr", type=float, default=0.5, help="核心概率阈值")
    p.add_argument("--thr-lo", type=float, default=0.10, help="缓冲带/并集的低阈值")
    p.add_argument("--agl-a", type=float, default=2.0, help="策略 A 缓冲带的高度上限 m")
    p.add_argument("--agl-e", type=float, default=1.0, help="策略 E 缓冲带的高度上限 m")
    p.add_argument("--no-micro", action="store_true",
                   help="关闭微地形重建（直接用 dem_min 当 DTM 算 AGL）")
    p.add_argument("--micro-pct", type=float, default=5.0,
                   help="微地形只用地面簇内 AGL 最低的 pct%% 建 TIN")
    p.add_argument("--micro-cell", type=float, default=0.5, help="微地形体素抽稀格宽 m")
    p.add_argument("--micro-dilate", type=int, default=5, help="微地形支撑区膨胀像元")
    p.add_argument("--micro-smooth", type=int, default=0, help="微地形中值滤波窗口")
    p.add_argument("--tube", type=float, default=7.5, help="策略 B 的 3D 管半径 m")
    p.add_argument("--tube-sweep", nargs="*", type=float, default=[10.0])
    p.add_argument("--agl-b", type=float, default=1.5, help="策略 B 的贴地上限 m")
    p.add_argument("--dil-px", type=int, default=10, help="策略 C 的膨胀半径（像元）")
    p.add_argument("--kmeans-seed", type=int, default=0, help="策略 D 的 KMeans 随机种子")
    p.add_argument("--kmeans-fit", type=int, default=2_000_000,
                   help="策略 D 拟合 KMeans 的最大样本数（超出则随机子采样）")
    p.add_argument("--sweep", action="store_true", help="追加参数扫描")
    p.add_argument("--scan", action="store_true",
                   help="追加 2D 概率阈值前沿扫描（判断目标是否可达）")
    p.add_argument("--ep", type=str, default=None)
    p.add_argument("--gate", type=float, default=3.0)
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "balance_filter.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True); lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"召回-精度平衡消融 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"目标：R≥0.75 且 P≥0.20；基线 P≥{a.thr:g}；"
        f"sweep={'on' if a.sweep else 'off'}，scan={'on' if a.scan else 'off'}")
    log("SHP 只在最后评分")

    out = []
    for s in a.sites:
        t0 = time.time()
        log(f"\n[{s}]")
        dem, _, valid = load_terrain(s)
        if a.base == "min":                   # 实验 B：改用像元最低高程当基准地形
            from fix_tpi31_auc import fill_nodata
            dem = fill_nodata(np.load(os.path.join(DATA, f"{s}_1m.npz"))["dem_min"]
                              .astype(np.float64), valid)
            log(f"  基准地形 = dem_min（像元最低高程）")
        meta = json.load(open(os.path.join(DATA, f"meta_{s}.json"), encoding="utf-8"))

        if a.ep:
            ep = [np.array([float(v) for v in q.split(",")]) for q in a.ep.split(";")]
        else:
            ep, _ = shp_endpoints(s, valid, log=log)
        r1 = build_f1(s, valid, ep, gate=a.gate, dir_lambda=a.dir_lambda,
                      turn_lambda=a.turn_lambda, turn_min=a.turn_min, log=log)
        if r1 is None:
            continue
        line, linfo = path_polyline_3d(s, r1["path"], dem, meta, log=log)
        tubes = sorted({float(a.tube), *[float(t) for t in a.tube_sweep]})
        keeps, _ = tube_keep(s, line, tubes, log=log)

        pt = load_pointwise(s, a, valid, dem, meta, log)

        # ---- 微地形：地面簇低 AGL 5% → TIN → AGL_new（本次修正的核心）
        micro_info = None
        if not a.no_micro:
            micro, micro_info = build_micro_from_points(s, pt, dem, meta, a, log)
            refresh_agl(pt, s, micro, meta, log, tag="AGL_new")
            np.save(os.path.join(RES, f"micro_dtm_{s}_bf.npy"),
                    micro.astype(np.float32))
        rows = make_rows(pt, keeps, a, tubes, log)

        d_rows = [r for r in rows if r["grp"] == "D"]
        for r in d_rows:
            k = r["kmeans"]
            b = k["bound_m"]
            log(f"  KMeans 自动地面/树冠分界 · {r['name']}："
                + (f"{b:.2f} m（地面簇 {k['n_ground']:,} 点 / 树冠簇 "
                   f"{k['n_canopy']:,} 点）" if b is not None else "不可用"))

        for r in rows:
            met = r["recall"] >= 0.75 and r["precision"] >= 0.20
            log(f"  {r['name']:<34s} 保留 {r['keep']:>10,d}（{r['keep_frac'] * 100:6.2f}%）"
                f"  P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}"
                f" IoU={r['iou']:.4f}" + ("  ★达标" if met else ""))
        hit = [r for r in rows if r["recall"] >= 0.75 and r["precision"] >= 0.20]
        bestf1 = max(rows, key=lambda r: r["f1"])
        log(f"  达标方案：{len(hit)} 个（{', '.join(r['name'] for r in hit) if hit else '无'}）")
        log(f"  最佳 F1：{bestf1['name']}（P={bestf1['precision']:.3f} "
            f"R={bestf1['recall']:.3f} F1={bestf1['f1']:.3f}）")
        feas = [r for r in rows if r["precision"] >= 0.20]
        if feas:
            br = max(feas, key=lambda r: r["recall"])
            log(f"  P≥0.20 约束下的召回上限：{br['recall']:.4f}"
                f"（{br['name']}，P={br['precision']:.3f} F1={br['f1']:.3f}）")

        fp = plot(s, rows, a, log)
        log(f"  保存 {os.path.relpath(fp, ROOT)}")
        out.append(dict(site=s, line=linfo, rows=rows,
                        kmeans={r["name"]: r["kmeans"] for r in d_rows},
                        micro=micro_info,
                        figure=os.path.relpath(fp, ROOT),
                        seconds=round(time.time() - t0, 1)))

    log("\n" + "=" * 78)
    log("KMeans 自动地面/树冠分界：")
    for o_ in out:
        for nm, k in o_["kmeans"].items():
            if k["bound_m"] is None:
                log(f"  {o_['site']} · {nm}：不可用（点太少）")
                continue
            log(f"  {o_['site']} · {nm}：**{k['bound_m']:.2f} m**"
                f"（簇中心 {k['center_low_m']:.2f} / {k['center_high_m']:.2f} m；"
                f"中位 {k['med_low_m']} / {k['med_high_m']} m；"
                f"缓冲带 {k['n_band']:,} 点，留下 {k['n_ground']:,}）")
    log("-" * 92)
    log(f"{'站点':>8s} {'策略':>36s} {'保留点':>12s} {'保留率':>8s} {'P':>7s} "
        f"{'R':>7s} {'F1':>7s} {'IoU':>7s}")
    log("-" * 100)
    for o_ in out:
        for r in o_["rows"]:
            log(f"{o_['site']:>8s} {r['name']:>36s} {r['keep']:>12,d} "
                f"{r['keep_frac'] * 100:>7.2f}% {r['precision']:>7.4f} "
                f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['iou']:>7.4f}")

    jp = os.path.join(RES, f"balance_filter_{a.base}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="A 核心+高度软切 / B 宽管+贴地 / C 膨胀并集，对照纯2D P≥0.5",
                       target=dict(recall=0.75, precision=0.20),
                       params=dict(thr=a.thr, thr_lo=a.thr_lo, agl_a=a.agl_a,
                                   tube=a.tube, agl_b=a.agl_b, dil_px=a.dil_px),
                       sweep=a.sweep, sites=out), f, ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
