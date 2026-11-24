# -*- coding: utf-8 -*-
"""DRR / CSSC / SPI 三因子：真值走廊 vs 背景（偏离 50 m）。

用户口径
--------
- 走廊 100 个真实点位；背景 100 个点位（到走廊距离 45~55 m，均值≈50 m）；
- 每点用 h1_phase 取 31 m（±15.5 m）法向 / 切向剖面；
- DRR = 法向 std / 切向 std（**去线性趋势后**才是"粗糙度"，原始 std 比一并输出）；
- CSSC = 横断面二阶导数的**显著符号交替次数**得分（S 形 = 1 次交替）；
- SPI = 沿切向 0.1 m 高程序列 ACF 在滞后 0.3~5 m 内的峰；
- 输出两组均值 / 箱线图 / AUC。

必须先看的陷阱（本脚本强制处理）
--------------------------------
自适应 0.1 m DEM 只有走廊核心是 0.1 m；偏离 50 m 处按 `p2_adaptive_gsd_pixel_*`
的 `gsd` 规则是 1.0 m（少量 4.0 m）。三个因子全部**分辨率敏感**：
  - DRR：0.1 m 切向剖面的高频方差天然更大 → 走廊 DRR 系统性偏低；
  - CSSC：二阶导数 ∝ 1/step²，0.1 m 步长的噪声被放大 100 倍；
  - SPI：1 m / 4 m 采样在 ACF 上会造出 1 m / 4 m 伪周期峰。
所以每个站点跑三套口径，**只有 B 是干净对照**：
  A   自适应 0.1 m，0.1 m 步长（用户指定，含分辨率混淆）
  A1  同一 DEM，按 10 抽 1 降到 1 m 步长（诊断：分离是否来自 0.1 m 细节）
  B   均匀 1 m（`dem_mean`，两组同分辨率，唯一干净对照）
判据：A 上分离而 A1/B 上不分离，或背景 `spi_peak_lag` 中位 ≈ 4.0 m
     → 结论是分辨率伪影，不是地形因子。

方向约定（**不要照抄 path_cross_section.py 的 offset_path**）
-------------------------------------------------------------
罗盘方位 θ（0=北，90=东）对应 UTM 切向 `(sinθ, cosθ)`；左法向是 `(−cosθ, +sinθ)`。
`path_cross_section.py` 里用的 `(cos(θ+90°), sin(θ+90°)) = (−sinθ, cosθ)` 与切向
点积为 cos2θ，**只在 θ=45° 时才是法向**（正北走向时会沿路径取样），本脚本不沿用。
`h1_phase` 是**轴**（φ 与 φ+180° 等价），正负号对 std/ACF 无影响。

用法
----
    python tools/path_cs_factors.py
    python tools/path_cs_factors.py --site 西坝村 --n 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates, uniform_filter1d
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score

Image.MAX_IMAGE_PIXELS = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from path_chaikin_viz import ordered_paths          # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

HALF = 15.5            # 剖面半长 → 总长 31 m
STEP_AD = 0.1          # 自适应 DEM 采样步长
DECIM = 10             # A1：10 抽 1 → 1 m
STEP_1M = 1.0
OFF_LO, OFF_HI = 45.0, 55.0
SMOOTH_M = 2.0         # CSSC 平滑尺度（≈ 台阶/坡肩宽度）
CURV_THR = 0.03        # 显著曲率阈值 (1/m)，约对应曲率半径 < 33 m
LAG_MIN, LAG_MAX = 0.3, 5.0
SEED = 0


# ------------------------------------------------------------------ 数据 ----
def load_adaptive(site):
    """自适应 0.1 m 画布（NaN 为 nodata，非 -9999）。"""
    p = os.path.join(RES, f"{site}_adaptive_0p1m.tif")
    with Image.open(p) as im:
        tags = dict(im.tag_v2)
        gsd = float(tags[33550][0])
        x0 = float(tags[33922][3])
        y1 = float(tags[33922][4])
        arr = np.asarray(im, dtype=np.float32)
    return arr, x0, y1, gsd


def load_1m(site):
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
    dem = np.asarray(d["dem_mean"], dtype=np.float32)
    dem = np.where(np.asarray(d["valid"]) == 1, dem, np.nan).astype(np.float32)
    return dem, float(meta["x0"]), float(meta["y1"]), 1.0, np.asarray(d["valid"]) == 1


# -------------------------------------------------------------- 方向/剖面 ----
def compass(x, y):
    """切线罗盘方位（0=北，90=东）。UTM +x=东、+y=北。"""
    return np.degrees(np.arctan2(np.gradient(x), np.gradient(y))) % 360.0


def sample_theta(field_deg, r, c):
    """角度场按 sin/cos 双线性插值后再 atan2，避免 0/360 断裂（支持标量输入）。"""
    rad = np.radians(field_deg)
    rr = np.atleast_1d(np.asarray(r, dtype=np.float64))
    cc = np.atleast_1d(np.asarray(c, dtype=np.float64))
    s = map_coordinates(np.sin(rad), [rr, cc], order=1, mode="nearest")
    co = map_coordinates(np.cos(rad), [rr, cc], order=1, mode="nearest")
    out = np.degrees(np.arctan2(s, co)) % 360.0
    return out if np.ndim(r) else float(out[0])


def profile(dem, x, y, theta_deg, half, step, x0, y1, gsd, axis="n"):
    """沿罗盘方位 theta_deg 的法向(n) / 切向(t) 取剖面，返回 z（NaN=越界或 nodata）。"""
    s = np.arange(-half, half + step * 0.5, step)
    t = np.radians(theta_deg)
    if axis == "n":                       # 左法向 (−cosθ, +sinθ)
        dx, dy = -np.cos(t), np.sin(t)
    else:                                 # 切向 (+sinθ, +cosθ)
        dx, dy = np.sin(t), np.cos(t)
    xs = x + s * dx
    ys = y + s * dy
    r = (y1 - ys) / gsd
    c = (xs - x0) / gsd
    H, W = dem.shape
    ok = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
    z = np.full(s.shape, np.nan, dtype=np.float64)
    if ok.any():
        z[ok] = map_coordinates(dem, [r[ok], c[ok]], order=1, mode="nearest")
    return z


# ------------------------------------------------------------------ 因子 ----
def detrend(z):
    n = len(z)
    t = np.arange(n, dtype=np.float64)
    return z - np.polyval(np.polyfit(t, z, 1), t)


def f_drr(zn, zt, detrended=True):
    if detrended:
        rn, rt = detrend(zn).std(ddof=1), detrend(zt).std(ddof=1)
    else:
        rn, rt = zn.std(ddof=1), zt.std(ddof=1)
    return float(rn / rt) if np.isfinite(rt) and rt > 1e-6 else np.nan


def f_cssc(zn, step, smooth_m=SMOOTH_M, thr=CURV_THR):
    """横断面二阶导数的显著符号交替**次数** n（每 31 m 剖面）。

    实测方向：走廊 n 更大（坡折更多：路肩 + 路面 + 边坡），背景是一条光滑坡面。
    因此 n 本身即得分（越高越像人工开凿断面）；`cssc_s=1/(1+|n−1|)` 是
    "S 形=1 次交替"的旧口径，实测方向相反，仅作诊断保留。
    """
    k = max(1, int(round(smooth_m / step)))
    zs = uniform_filter1d(zn, k, mode="nearest") if k > 1 else zn
    d2 = np.gradient(np.gradient(zs, step), step)
    sig = np.abs(d2) > thr
    seq = []
    for g, okk in zip(np.sign(d2), sig):
        if not okk:
            continue
        if not seq or g != seq[-1]:
            seq.append(g)
    n = max(0, len(seq) - 1)
    return float(1.0 / (1.0 + abs(n - 1))), int(n)


def f_spi(zt, step, lag_min=LAG_MIN, lag_max=LAG_MAX):
    """切向剖面去趋势后的 ACF：滞后 [lag_min, lag_max] 内最大局部峰。"""
    r = detrend(zt)
    n = len(r)
    if n < 6:
        return np.nan, np.nan, False
    f = np.fft.rfft(r, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if not np.isfinite(acf[0]) or acf[0] <= 0:
        return np.nan, np.nan, False
    acf = acf / acf[0]
    i0 = max(1, int(np.ceil(lag_min / step)))
    i1 = min(n - 2, int(np.floor(lag_max / step)))
    if i1 <= i0:
        return np.nan, np.nan, False
    seg = acf[i0:i1 + 1]
    j = int(np.argmax(seg))
    is_local = bool(0 < j < len(seg) - 1 and seg[j] > seg[j - 1] and seg[j] > seg[j + 1])
    return float(seg[j]), float((i0 + j) * step), is_local


def factors(zn, zt, step):
    cssc_s, n = f_cssc(zn, step)
    spi, lag, loc = f_spi(zt, step)
    return {"drr": f_drr(zn, zt), "drr_raw": f_drr(zn, zt, False),
            "cssc": float(n), "cssc_s": cssc_s,
            "spi": spi, "spi_lag": lag, "spi_local": loc}


# ------------------------------------------------------------------ 取点 ----
def corridor_points(mask, x0, y1, gsd, n):
    """骨架链按弧长比例分配 → 每条链等距取点，返回 (N,3) [x, y, 切线方位]。"""
    sk = skeletonize(mask)
    chains = [c for c in ordered_paths(sk) if len(c) >= 4]
    if not chains:
        return np.zeros((0, 3))
    info = []
    for ch in chains:
        x = x0 + ch[:, 1] * gsd
        y = y1 - ch[:, 0] * gsd
        info.append((x, y, float(np.hypot(np.diff(x), np.diff(y)).sum())))
    tot = sum(v[2] for v in info) or 1.0
    out = []
    for x, y, L in info:
        k = int(round(n * L / tot))
        if k < 2:
            continue
        d = np.hypot(np.diff(x), np.diff(y))
        cum = np.concatenate([[0.0], np.cumsum(d)])
        arc = np.linspace(0.0, cum[-1], k)
        xr = np.interp(arc, cum, x)
        yr = np.interp(arc, cum, y)
        out.append(np.c_[xr, yr, compass(xr, yr)])
    if not out:
        return np.zeros((0, 3))
    P = np.vstack(out)
    if len(P) > n:
        P = P[np.linspace(0, len(P) - 1, n).round().astype(int)]
    return P


def skeleton_ktree(mask, x0, y1, gsd):
    sk = skeletonize(mask)
    r, c = np.nonzero(sk)
    x = x0 + c * gsd
    y = y1 - r * gsd
    return cKDTree(np.c_[x, y]), x, y, compass(x, y)


def background_points(mask, valid, h1, h1deg, thr, x0, y1, gsd, tree, xs, ys, ths,
                      dem_ad, x0a, y1a, gsa, dem_1, x01, y11, gs1, rng, n, cap=6000):
    """在 dist∈[45,55] m 的候选里拒绝采样，要求两套 DEM 的 31 m 剖面全 finite。"""
    dist = distance_transform_edt(~mask) * gsd
    cand = (dist >= OFF_LO) & (dist <= OFF_HI) & valid
    rr, cc = np.nonzero(cand)
    if len(rr) == 0:
        return np.zeros((0, 3)), np.zeros((0,))
    order = rng.permutation(len(rr))[:cap]
    kept = []
    for k in order:
        yv = y1 - rr[k] * gsd
        xv = x0 + cc[k] * gsd
        _, j = tree.query([xv, yv])
        th_road = float(ths[j])
        th = th_road
        if h1deg[int(rr[k]), int(cc[k])] >= thr:
            th = float(sample_theta(h1, rr[k], cc[k]))
        zn = profile(dem_ad, xv, yv, th, HALF, STEP_AD, x0a, y1a, gsa, "n")
        zt = profile(dem_ad, xv, yv, th, HALF, STEP_AD, x0a, y1a, gsa, "t")
        if not (np.isfinite(zn).all() and np.isfinite(zt).all()):
            continue
        zn1 = profile(dem_1, xv, yv, th, HALF, STEP_1M, x01, y11, gs1, "n")
        zt1 = profile(dem_1, xv, yv, th, HALF, STEP_1M, x01, y11, gs1, "t")
        if not (np.isfinite(zn1).all() and np.isfinite(zt1).all()):
            continue
        kept.append((xv, yv, th, float(dist[int(rr[k]), int(cc[k])])))
        if len(kept) >= n:
            break
    if not kept:
        return np.zeros((0, 3)), np.zeros((0,))
    K = np.asarray(kept)
    return K[:, :3], K[:, 3]


# ------------------------------------------------------------------ 统计 ----
def stats(pos, neg):
    p = np.asarray(pos, dtype=np.float64)
    q = np.asarray(neg, dtype=np.float64)
    p = p[np.isfinite(p)]
    q = q[np.isfinite(q)]
    if len(p) < 3 or len(q) < 3:
        return None
    a = float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q]))
    return {"cor_mean": float(p.mean()), "cor_median": float(np.median(p)),
            "cor_std": float(p.std(ddof=1)),
            "cor_q1": float(np.percentile(p, 25)), "cor_q3": float(np.percentile(p, 75)),
            "bg_mean": float(q.mean()), "bg_median": float(np.median(q)),
            "bg_std": float(q.std(ddof=1)),
            "bg_q1": float(np.percentile(q, 25)), "bg_q3": float(np.percentile(q, 75)),
            "auc": a, "auc_dist": float(max(a, 1.0 - a)),
            "n_cor": int(len(p)), "n_bg": int(len(q))}


FACTOR_KEYS = ("drr", "drr_raw", "cssc", "cssc_s", "spi")


def run_site(site, n_pts):
    rng = np.random.default_rng(SEED)
    dem_ad, x0a, y1a, gsa = load_adaptive(site)
    dem_1, x01, y11, gs1, valid = load_1m(site)
    mask = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    h = np.load(os.path.join(RES, f"horizon_factors_{site}.npz"))
    h1 = h["h1_phase"].astype(np.float64)
    h1deg = h["h1_deg"].astype(np.float64)
    thr = float(np.percentile(h1deg[valid], 60.0))
    gmap = np.load(os.path.join(RES, f"p2_adaptive_gsd_pixel_{site}.npz"))["gsd"]

    # ---- 走廊点
    C = corridor_points(mask, x01, y11, gs1, n_pts)
    print(f"\n=== {site} === 骨架取点 {len(C)}，h1_deg 门控阈值 {thr:.2f}")

    # ---- 背景点
    tree, xs, ys, ths = skeleton_ktree(mask, x01, y11, gs1)
    B, bdist = background_points(mask, valid, h1, h1deg, thr, x01, y11, gs1,
                                 tree, xs, ys, ths, dem_ad, x0a, y1a, gsa,
                                 dem_1, x01, y11, gs1, rng, n_pts)
    print(f"背景点 {len(B)}（实际偏移 中位 {np.median(bdist):.1f} m，"
          f"范围 {bdist.min():.1f}~{bdist.max():.1f}）" if len(B) else "背景点 0")

    # ---- 逐点测度
    rec = {"A": [], "A1": [], "B": []}
    meta = {"cor": [], "bg": []}
    for tag, P, w in (("cor", C, 1.0), ("bg", B, -1.0)):
        for x, y, th in P:
            r1 = (y11 - y) / gs1
            c1 = (x - x01) / gs1
            zn = profile(dem_ad, x, y, th, HALF, STEP_AD, x0a, y1a, gsa, "n")
            zt = profile(dem_ad, x, y, th, HALF, STEP_AD, x0a, y1a, gsa, "t")
            zn1 = profile(dem_1, x, y, th, HALF, STEP_1M, x01, y11, gs1, "n")
            zt1 = profile(dem_1, x, y, th, HALF, STEP_1M, x01, y11, gs1, "t")
            if not (np.isfinite(zn).all() and np.isfinite(zt).all()
                    and np.isfinite(zn1).all() and np.isfinite(zt1).all()):
                continue
            rec["A"].append((tag, factors(zn, zt, STEP_AD)))
            rec["A1"].append((tag, factors(zn[::DECIM], zt[::DECIM], STEP_1M)))
            rec["B"].append((tag, factors(zn1, zt1, STEP_1M)))
            meta[tag].append({"x": float(x), "y": float(y), "theta": float(th),
                              "gsd": float(gmap[int(round(r1)), int(round(c1))])
                              if 0 <= r1 < gmap.shape[0] and 0 <= c1 < gmap.shape[1]
                              else np.nan,
                              "h1_deg": float(h1deg[int(round(r1)), int(round(c1))])
                              if 0 <= r1 < gmap.shape[0] and 0 <= c1 < gmap.shape[1]
                              else np.nan})

    rep = {"site": site, "n_cor": len(meta["cor"]), "n_bg": len(meta["bg"]),
           "h1_gate_deg": thr,
           "gsd_cor": _gsd_stat(meta["cor"]), "gsd_bg": _gsd_stat(meta["bg"]),
           "offset_m": {"median": float(np.median(bdist)) if len(bdist) else None,
                        "min": float(bdist.min()) if len(bdist) else None,
                        "max": float(bdist.max()) if len(bdist) else None},
           "variants": {}}

    out = {"report": rep, "raw": rec}
    for vname, step, dem_desc in (("A", STEP_AD, "自适应 0.1 m，0.1 m 步长"),
                                  ("A1", STEP_1M, "自适应 DEM，10 抽 1 → 1 m 步长"),
                                  ("B", STEP_1M, "均匀 1 m（dem_mean，干净对照）")):
        rows = rec[vname]
        vrep = {"dem": dem_desc, "step_m": step, "factors": {},
                "spi_peak_lag": {}}
        for key in FACTOR_KEYS:
            p = [r[key] for t, r in rows if t == "cor" and np.isfinite(r[key])]
            q = [r[key] for t, r in rows if t == "bg" and np.isfinite(r[key])]
            st = stats(p, q)
            if st:
                vrep["factors"][key] = st
        for key in ("spi_lag",):
            p = [r[key] for t, r in rows if t == "cor" and np.isfinite(r[key])]
            q = [r[key] for t, r in rows if t == "bg" and np.isfinite(r[key])]
            loc_c = np.asarray([bool(r["spi_local"]) for t, r in rows if t == "cor"])
            loc_b = np.asarray([bool(r["spi_local"]) for t, r in rows if t == "bg"])
            vrep["spi_peak_lag"] = {
                "cor_median": float(np.median(p)) if p else None,
                "bg_median": float(np.median(q)) if q else None,
                "bg_frac_near_4m": float(np.mean(np.abs(np.asarray(q) - 4.0) < 0.3))
                if q else None,
                "bg_frac_near_1m": float(np.mean(np.abs(np.asarray(q) - 1.0) < 0.15))
                if q else None,
                # 真实的"内部局部峰"占比；若≈0，说明 argmax 卡在滞后下限，
                # SPI 退化为"最短滞后处的 ACF 衰减率"，不是周期峰
                "cor_local_frac": float(loc_c.mean()) if loc_c.size else None,
                "bg_local_frac": float(loc_b.mean()) if loc_b.size else None}
        rep["variants"][vname] = vrep
    return out


def _gsd_stat(rows):
    g = np.asarray([r["gsd"] for r in rows], dtype=np.float64)
    g = g[np.isfinite(g)]
    if not len(g):
        return {}
    u, cnt = np.unique(np.round(g, 2), return_counts=True)
    return {"levels": {str(a): int(b) for a, b in zip(u, cnt)},
            "median": float(np.median(g))}


# ------------------------------------------------------------------ 输出 ----
def print_table(all_rep):
    print("\n=== 主因子 AUC（0.5=随机；auc_dist=|AUC−0.5| 判定有无信号）===")
    print(f"{'站点':<7}{'口径':<6}{'因子':<9}{'走廊均值':>10}{'背景均值':>10}"
          f"{'AUC':>8}{'auc_dist':>10}")
    for site, rep in all_rep.items():
        for v in ("A", "A1", "B"):
            vrep = rep["variants"].get(v)
            if not vrep:
                continue
            for key in ("drr", "cssc", "spi"):
                st = vrep["factors"].get(key)
                if not st:
                    continue
                print(f"{site:<7}{v:<6}{key:<9}{st['cor_mean']:>10.3f}"
                      f"{st['bg_mean']:>10.3f}{st['auc']:>8.3f}{st['auc_dist']:>10.3f}")
    print("\n=== 诊断因子（不参与主判定）===")
    print(f"{'站点':<7}{'口径':<6}{'因子':<9}{'走廊均值':>10}{'背景均值':>10}{'AUC':>8}")
    for site, rep in all_rep.items():
        for v in ("A", "A1", "B"):
            vrep = rep["variants"].get(v)
            if not vrep:
                continue
            for key in ("drr_raw", "cssc_s"):
                st = vrep["factors"].get(key)
                if st:
                    print(f"{site:<7}{v:<6}{key:<9}{st['cor_mean']:>10.3f}"
                          f"{st['bg_mean']:>10.3f}{st['auc']:>8.3f}")
    print("\n=== SPI 峰位 / 真实局部峰占比（local≈0 即 SPI 退化）===")
    for site, rep in all_rep.items():
        for v in ("A", "B"):
            d = rep["variants"].get(v, {}).get("spi_peak_lag") or {}
            if d.get("bg_median") is None:
                continue
            print(f"{site} {v}: 峰位 走廊 {d['cor_median']:.2f} / 背景 {d['bg_median']:.2f} m；"
                  f"内部局部峰占比 走廊 {d['cor_local_frac']:.1%} / "
                  f"背景 {d['bg_local_frac']:.1%}；背景≈4m {d['bg_frac_near_4m']:.1%}")
    print("\n=== gsd 组成（1 m 网格逐像元 gsd 图）===")
    for site, rep in all_rep.items():
        print(f"{site} 走廊 {rep['gsd_cor']}  背景 {rep['gsd_bg']}")


def render(all_raw, all_rep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    keys = (("drr", "DRR 法向/切向粗糙度比（去趋势）"),
            ("cssc", "CSSC 横断面显著符号交替次数"),
            ("spi", "SPI 切向 ACF 周期峰"))
    rows = [(s, v, "自适应 0.1 m" if v == "A" else "均匀 1 m 对照")
            for s in all_rep for v in ("A", "B")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.2, 3.05 * len(rows)),
                             squeeze=False)
    rng = np.random.default_rng(1)
    for i, (site, v, desc) in enumerate(rows):
        for j, (key, title) in enumerate(keys):
            ax = axes[i, j]
            groups, colors = [], []
            for tag, col in (("cor", "#d95f02"), ("bg", "#1b9e77")):
                vals = [r[key] for t, r in all_raw[site]["raw"][v]
                        if t == tag and np.isfinite(r[key])]
                groups.append(np.asarray(vals) if vals else np.array([np.nan]))
                colors.append(col)
            bp = ax.boxplot(groups, patch_artist=True, widths=0.52,
                            medianprops=dict(color="#252525", lw=1.4),
                            flierprops=dict(marker=".", ms=2.5, alpha=0.35))
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.55)
            for k, (g, col) in enumerate(zip(groups, colors)):
                g = g[np.isfinite(g)]
                if g.size:
                    ax.scatter(k + 1 + rng.uniform(-0.16, 0.16, g.size), g,
                               s=6, color=col, alpha=0.45, lw=0, zorder=3)
            ax.set_xticks([1, 2], ["走廊", "背景"], fontsize=8)
            st = all_rep[site]["variants"][v]["factors"].get(key, {})
            auc = st.get("auc")
            sub = (f"AUC={auc:.3f}（|AUC−.5|={st.get('auc_dist', float('nan')):.3f}）"
                   if auc is not None else "AUC=NA")
            ax.set_title(f"{site} · {title}\n{desc}　{sub}", fontsize=9)
            ax.grid(alpha=0.22, axis="y")
    fig.suptitle("走廊 vs 背景（50 m 偏移）：DRR / CSSC / SPI 在自适应 0.1 m 与均匀 1 m 上的对照",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"写出 {os.path.relpath(out, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="all", choices=["杨家坝", "西坝村", "all"])
    ap.add_argument("--n", type=int, default=100, help="每组点位数（默认 100）")
    a = ap.parse_args()
    sites = ["杨家坝", "西坝村"] if a.site == "all" else [a.site]

    all_raw, all_rep = {}, {}
    for s in sites:
        o = run_site(s, a.n)
        all_raw[s] = o
        all_rep[s] = o["report"]

    p = os.path.join(RES, "path_cs_factors.json")
    json.dump({"n_target": a.n, "half_len_m": HALF, "smooth_m": SMOOTH_M,
               "curv_thr": CURV_THR, "lag_range_m": [LAG_MIN, LAG_MAX],
               "offset_range_m": [OFF_LO, OFF_HI],
               "note": ("A=自适应0.1m/0.1m步长（含分辨率混淆）；A1=同DEM 1m步长（诊断）；"
                        "B=均匀1m（干净对照）。判定：A分离而A1/B不分离 → 分辨率伪影"),
               "sites": all_rep},
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"写出 {os.path.relpath(p, ROOT)}")

    print_table(all_rep)
    render(all_raw, all_rep, os.path.join(FIG, "path_cs_factors.png"))


if __name__ == "__main__":
    main()
