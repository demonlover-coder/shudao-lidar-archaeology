# -*- coding: utf-8 -*-
"""把已验证的 CSSC / DRR 因子铺成**逐像元代价图**（1 m 网格），供 Dijkstra 使用。

为什么必须预计算
----------------
DRR/CSSC 是**剖面级**特征：每个像元都要沿走廊法向取 32 个采样、沿切向再取 32 个，
共 64 次双线性采样。若在寻路里边走边算，每条边都要重做一遍、且无法向量化。
所以先在 1 m 网格上一次性铺成 2D 场，寻路只做查表。

方向场
------
法向直接取 `path_extraction.direction_field()`——**与寻路自身用的方向先验完全同源**
（h1_phase 复数域平滑 size=5，`n_row=−cosθ, n_col=+sinθ`），避免"验证时一套方向、
惩罚时另一套方向"。

口径（与 `path_cs_factors.py` 的 B 口径逐项一致）
-------------------------------------------------
- 剖面 ±15.5 m，1 m 步长（1 m 净对照上 DRR AUC 0.670/0.646、CSSC 0.721/0.889）
- DRR = std(去趋势法向) / std(去趋势切向)
- CSSC = 横断面二阶导（先 2 m 平滑）显著符号交替**次数**，thr=0.03 1/m

产出
----
`results/cs_fields_{site}.npz`：`cssc` / `drr`（float32，valid 外 NaN）
`results/cs_fields.json`：全像元 AUC（骨架 vs 45~55 m 带）、与 openness 的秩相关、
                         逐像元分布分位

用法
----
    python tools/build_cs_fields.py
    python tools/build_cs_fields.py --site 西坝村
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, uniform_filter1d
from scipy.stats import spearmanr
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from fix_tpi31_auc import fill_nodata                       # noqa: E402
from path_extraction import direction_field, smooth_phase_complex  # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

HALF = 15.5
STEP = 1.0
SMOOTH_M = 2.0
CURV_THR = 0.03
BLOCK = 96
OFF_LO, OFF_HI = 45.0, 55.0


def detrend_rows(A):
    """按行去线性趋势（A 每行是一条剖面）。"""
    n = A.shape[1]
    t = np.arange(n, dtype=np.float64)
    t = t - t.mean()
    t = t / np.linalg.norm(t)
    return A - A.mean(axis=1, keepdims=True) - np.outer(A @ t, t)


def alternation_count(A, thr):
    """各行的"显著符号交替次数"（跳过不显著段，只数相邻非零符号的翻转）。"""
    s = np.where(np.abs(A) > thr, np.sign(A), 0.0)
    cur = np.zeros(A.shape[0], dtype=np.float64)
    cnt = np.zeros(A.shape[0], dtype=np.int64)
    for j in range(s.shape[1]):
        v = s[:, j]
        m = v != 0.0
        cnt += (m & (cur != 0.0) & (v != cur)).astype(np.int64)
        cur = np.where(m, v, cur)
    return cnt


def cs_maps(site, block=BLOCK):
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    valid = d["valid"] == 1
    dem = fill_nodata(d["dem_mean"].astype(np.float64), valid)
    n_row, n_col, _, thr_conf = direction_field(site)
    t_row, t_col = -n_col, n_row            # 切向 = 法向转 90°（罗盘右手）

    H, W = dem.shape
    drr = np.full((H, W), np.nan, dtype=np.float32)
    cssc = np.full((H, W), np.nan, dtype=np.float32)
    lags = np.arange(-HALF, HALF + STEP * 0.5, STEP)
    k_smooth = max(1, int(round(SMOOTH_M / STEP)))

    for r0 in range(0, H, block):
        r1 = min(r0 + block, H)
        rr0, cc0 = np.nonzero(valid[r0:r1])
        if rr0.size == 0:
            continue
        rr = rr0 + r0
        cc = cc0
        zn = np.empty((rr.size, lags.size), dtype=np.float64)
        zt = np.empty((rr.size, lags.size), dtype=np.float64)
        inside = np.ones(rr.size, dtype=bool)
        for k, s in enumerate(lags):
            for d1, d2, out in ((n_row[rr, cc], n_col[rr, cc], zn),
                                (t_row[rr, cc], t_col[rr, cc], zt)):
                yr = rr + s * d1
                xr = cc + s * d2
                inside &= (yr >= 0) & (yr <= H - 1) & (xr >= 0) & (xr <= W - 1)
                out[:, k] = map_coordinates(dem, [yr, xr], order=1, mode="nearest")
        if not inside.any():
            continue

        zn, zt = zn[inside], zt[inside]
        rn = detrend_rows(zn)
        rt = detrend_rows(zt)
        sn = rn.std(axis=1, ddof=1)
        st = rt.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dv = np.where(st > 1e-6, sn / st, np.nan)

        zs = uniform_filter1d(zn, k_smooth, axis=1, mode="nearest") if k_smooth > 1 else zn
        d2 = np.gradient(np.gradient(zs, STEP, axis=1), STEP, axis=1)
        cv = alternation_count(d2, CURV_THR).astype(np.float32)

        drr[rr[inside], cc[inside]] = dv.astype(np.float32)
        cssc[rr[inside], cc[inside]] = cv

    # valid 内极值（极少）用中位填补，避免在 _rank01 里因 NaN 变成"免费通道"
    for arr in (drr, cssc):
        m = valid & ~np.isfinite(arr)
        if m.any():
            arr[m] = np.nanmedian(arr[valid])
        arr[~valid] = np.nan
    return drr, cssc, valid, thr_conf


def separability(site, drr, cssc, valid):
    """全像元口径：骨架像元（正）vs 45~55 m 带像元（负）。"""
    mask = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    sk = skeletonize(mask)
    dist = distance_transform_edt(~mask)
    neg = (dist >= OFF_LO) & (dist <= OFF_HI) & valid
    out = {}
    for name, arr in (("drr", drr), ("cssc", cssc)):
        p = arr[sk & valid]
        q = arr[neg]
        p = p[np.isfinite(p)]
        q = q[np.isfinite(q)]
        y = np.r_[np.ones(len(p)), np.zeros(len(q))]
        x = np.r_[p, q]
        from sklearn.metrics import roc_auc_score
        out[name] = {"cor_mean": float(p.mean()), "cor_median": float(np.median(p)),
                     "bg_mean": float(q.mean()), "bg_median": float(np.median(q)),
                     "auc": float(roc_auc_score(y, x)),
                     "n_cor": int(len(p)), "n_bg": int(len(q))}
    return out


def corr_with_openness(site, drr, cssc, valid, n=60_000):
    """与 openness 的秩相关（>0.7 → 边际增益有限，必须在汇报里标注）。"""
    meta = json.load(open(os.path.join(DATA, f"feature_matrix_{site}.meta.json"),
                          encoding="utf-8"))
    k = meta["factors"].index("openness")
    X = np.load(os.path.join(DATA, f"feature_matrix_{site}.npy"))[:, k]
    g = np.full(valid.size, np.nan)
    g[np.flatnonzero(valid.ravel())] = X
    g = g.reshape(valid.shape)
    idx = np.flatnonzero(valid.ravel())
    rng = np.random.default_rng(0)
    if idx.size > n:
        idx = rng.permutation(idx)[:n]
    o = g.ravel()[idx]
    out = {}
    for name, arr in (("drr", drr), ("cssc", cssc)):
        a = arr.ravel()[idx]
        m = np.isfinite(o) & np.isfinite(a)
        out[name] = {"n": int(m.sum()),
                     "rho_spearman": float(spearmanr(o[m], a[m]).statistic)}
    return out


def render(sites, maps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(len(sites), 2, figsize=(10.4, 4.9 * len(sites)), squeeze=False)
    for i, site in enumerate(sites):
        mask = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
        for j, (name, cmap, label) in enumerate(
                (("drr", "viridis", "DRR 法向/切向粗糙度比"),
                 ("cssc", "magma", "CSSC 横断面符号交替次数"))):
            ax = axes[i, j]
            a = maps[site][name]
            v = a[np.isfinite(a)]
            lo, hi = np.percentile(v, 2), np.percentile(v, 98)
            ax.imshow(np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1), cmap=cmap)
            ax.contour(mask, levels=[0.5], colors="#00e5ff", linewidths=1.4)
            ax.set_title(f"{site} · {label}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("CSSC / DRR 逐像元代价场（1 m 网格，法向来自 h1_phase）；蓝线为真值走廊", fontsize=11)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"写出 {os.path.relpath(out, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="all", choices=["杨家坝", "西坝村", "all"])
    a = ap.parse_args()
    sites = ["杨家坝", "西坝村"] if a.site == "all" else [a.site]

    rep, maps = {}, {}
    for s in sites[:]:
        drr, cssc, valid, thr_conf = cs_maps(s)
        p = os.path.join(RES, f"cs_fields_{s}.npz")
        np.savez_compressed(p, drr=drr, cssc=cssc, valid=valid.astype(np.uint8),
                            h1_gate_deg=float(thr_conf))
        maps[s] = {"drr": drr, "cssc": cssc}
        r = {"file": os.path.basename(p), "shape": list(drr.shape),
             "h1_gate_deg": float(thr_conf),
             "separability": separability(s, drr, cssc, valid),
             "corr_openness": corr_with_openness(s, drr, cssc, valid),
             "quantile": {}}
        for name, arr in (("drr", drr), ("cssc", cssc)):
            v = arr[valid]
            v = v[np.isfinite(v)]
            r["quantile"][name] = {f"p{q}": float(np.percentile(v, q))
                                   for q in (1, 10, 25, 50, 75, 90, 99)}
        rep[s] = r
        print(f"[{s}] 写出 {os.path.relpath(p, ROOT)}")
        for name in ("drr", "cssc"):
            st = r["separability"][name]
            print(f"   {name}: 骨架均值 {st['cor_mean']:.3f} / 背景 {st['bg_mean']:.3f} → "
                  f"AUC {st['auc']:.4f}；与 openness 秩相关 "
                  f"{r['corr_openness'][name]['rho_spearman']:+.3f}")

    json.dump(rep, open(os.path.join(RES, "cs_fields.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"写出 {os.path.relpath(os.path.join(RES, 'cs_fields.json'), ROOT)}")
    render(sites, maps, os.path.join(FIG, "cs_fields.png"))


if __name__ == "__main__":
    main()
