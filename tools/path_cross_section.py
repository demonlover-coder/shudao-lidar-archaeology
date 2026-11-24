# -*- coding: utf-8 -*-
"""路径横断面三大特征：CAI / CSASC / LAAS。

定义（按用户规格实现，含两处必要的口径修正，见下）
--------------------------------------------------
沿路径按 `sample_dist=5 m` 等距重采样，取每点切线 heading、法向；
向左右各延伸 `half_width_m=10 m` 双线性取高程，得 Z_L / Z_R：

    D      = Z_L − Z_R                       横断面高差
    CAI    = mean(|D|)                       窗口内高差幅值（越大越不对称）
    CSASC  = max(P(D>0), P(D<0))             窗口内高差符号一致性（1=永远同侧高）
    LAAS   = mean(|sin(heading − h1_phase)|) 窗口内切线与走廊法向的对齐度（1=完全垂直）

口径修正 1（LAAS 方向）
    实测 h1_phase 是**走廊法向**（与 EDT 法向中位差 8.8°），所以"对齐"意味着
    切线与 h1_phase **垂直**，|cos| ≈ 0 才是对的，不是 1。故 LAAS 取 |sin|。
    同时输出 `laas_cos`（用户原式 mean|cos|），便于对照：它的高值代表**错**。

口径修正 2（CAI 与坡度混淆）
    ±10 m 处的 D 主要由**局部横坡**决定：35% 坡上 D≈7 m，与有无古道无关。
    故额外输出 `cai_resid = mean(|D − D_expected|)`，其中
    D_expected = 2·half_width·(∇z·n̂) 由中心线处的 DEM 梯度给出，
    把"自然横坡"扣掉，剩下的才是道路开凿形成的不对称。

负样本：沿法向平移 ±50/±100/±200 m 的平行线（同地形、不同选线）。

输出
----
results/path_cross_section.json + figures/path_cross_section.png

用法
----
    python tools/path_cross_section.py
    python tools/path_cross_section.py --site 杨家坝 --half-width 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from path_chaikin_viz import ordered_paths          # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
GSD = 1.0
SAMPLE = 5.0
HALF_W = 10.0
WINDOW = 100.0
OFFSETS = (-200.0, -100.0, -50.0, 50.0, 100.0, 200.0)
BAND = (0.9, 1.0, 1.1)      # 在 ±(9,10,11) m 三点取均值，抗单像元噪声


# ---------------------------------------------------------------- 采样 ----
def utm_to_rc(x, y, x0, y1, gsd):
    """UTM → (row, col)。y 轴向南翻转。"""
    return (y1 - np.asarray(y)) / gsd, (np.asarray(x) - x0) / gsd


def rc_to_utm(r, c, x0, y1, gsd):
    return x0 + np.asarray(c) * gsd, y1 - np.asarray(r) * gsd


def resample_utm(x, y, step):
    """按弧长等距重采样（UTM 米），返回 (xs, ys, arc)。"""
    d = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    n = int(np.floor(cum[-1] / step)) + 1
    arc = np.arange(n) * step
    return np.interp(arc, cum, x), np.interp(arc, cum, y), arc


def heading_compass(x, y):
    """切线的罗盘方位角（0=北，90=东）。UTM 的 +x=东、+y=北。"""
    dx, dy = np.gradient(x), np.gradient(y)
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def sample_field(field, r, c, order=1):
    return map_coordinates(field, [r, c], order=order, mode="nearest")


def sample_h1_phase(h1, r, c):
    """h1_phase 是角度，必须按 sin/cos 双线性插值后再取 atan2（避免 0/360 断裂）。"""
    rad = np.radians(h1)
    s = map_coordinates(np.sin(rad), [r, c], order=1, mode="nearest")
    co = map_coordinates(np.cos(rad), [r, c], order=1, mode="nearest")
    return np.degrees(np.arctan2(s, co)) % 360.0


def cross_section(dem, x0, y1, gsd, xs, ys, heading, half_width):
    """左右各取 half_width（同一距离上取 3 点均值），返回 Z_L, Z_R, D, D_exp。

    方向约定（修正版）：罗盘 θ 的切向是 `(sinθ, cosθ)`，左法向是 `(−cosθ, +sinθ)`、
    右法向是 `(+cosθ, −sinθ)`。原实现用 `(cos(θ+90°), sin(θ+90°)) = (−sinθ, cosθ)`，
    与切向点积为 cos2θ —— **只在 θ=45° 时才是法向**，正北走向会沿路径取样。
    """
    rad = np.radians(heading)
    nlx, nly = -np.cos(rad), np.sin(rad)      # 左手侧法向
    nrx, nry = np.cos(rad), -np.sin(rad)      # 右手侧法向
    zl, zr = [], []
    for f in BAND:
        d = half_width * f
        zl.append(sample_field(dem, *utm_to_rc(xs + d * nlx, ys + d * nly,
                                               x0, y1, gsd)))
        zr.append(sample_field(dem, *utm_to_rc(xs + d * nrx, ys + d * nry,
                                               x0, y1, gsd)))
    Z_L = np.nanmean(np.stack(zl), axis=0)
    Z_R = np.nanmean(np.stack(zr), axis=0)

    # 中心线处的梯度（用 ±half_width 内的差分估计），投影到法向 → 自然横坡预期高差
    z_c = sample_field(dem, *utm_to_rc(xs, ys, x0, y1, gsd))
    g = half_width * 0.5
    zx = (sample_field(dem, *utm_to_rc(xs + g, ys, x0, y1, gsd))
          - sample_field(dem, *utm_to_rc(xs - g, ys, x0, y1, gsd))) / (2 * g)
    zy = (sample_field(dem, *utm_to_rc(xs, ys + g, x0, y1, gsd))
          - sample_field(dem, *utm_to_rc(xs, ys - g, x0, y1, gsd))) / (2 * g)
    # 沿左法向的坡度；对称横坡下 D_exp ≈ 2·hw·(∇z·n̂_L)
    d_exp = 2.0 * half_width * (zx * nlx + zy * nly)
    return Z_L, Z_R, Z_L - Z_R, d_exp, z_c


# ---------------------------------------------------------------- 指标 ----
def window_metrics(D, d_exp, heading, phase, step, win):
    """滑窗（步进 = win/2 与 step 的较大者），返回各窗口指标数组。"""
    k = int(round(win / step))
    if k < 3 or len(D) < k:
        return None
    st = max(1, k // 2)
    out = {key: [] for key in ("cai", "cai_resid", "csasc", "laas", "laas_cos")}
    for i in range(0, len(D) - k + 1, st):
        w = D[i:i + k]
        we = d_exp[i:i + k]
        if not np.all(np.isfinite(w)):
            continue
        out["cai"].append(float(np.mean(np.abs(w))))
        out["cai_resid"].append(float(np.mean(np.abs(w - we))))
        sg = np.sign(w)
        sg[sg == 0] = 1.0
        pos = float((sg > 0).mean())
        out["csasc"].append(max(pos, 1.0 - pos))
        dh = np.radians((heading[i:i + k] - phase[i:i + k]) % 360.0)
        out["laas"].append(float(np.mean(np.abs(np.sin(dh)))))
        out["laas_cos"].append(float(np.mean(np.abs(np.cos(dh)))))
    if not out["cai"]:
        return None
    return {k2: np.asarray(v) for k2, v in out.items()}


def extract_cs_features(path_coords, dem, gsd=GSD, sample_dist=SAMPLE,
                        half_width_m=HALF_W, window_m=WINDOW,
                        x0=None, y1=None, h1_phase=None):
    """对一条 UTM 路径计算三大指标。

    path_coords: (N,2) UTM (x, y)
    dem: 2D 高程数组；x0/y1: 栅格左上角 UTM 与上边界 y
    h1_phase: 与 dem 同形的角度数组（走廊法向，度）
    """
    p = np.asarray(path_coords, dtype=np.float64)
    xs, ys, arc = resample_utm(p[:, 0], p[:, 1], sample_dist)
    hd = heading_compass(xs, ys)
    r, c = utm_to_rc(xs, ys, x0, y1, gsd)
    ph = sample_h1_phase(h1_phase, r, c)
    Z_L, Z_R, D, D_exp, _ = cross_section(dem, x0, y1, gsd, xs, ys, hd,
                                          half_width_m)
    m = window_metrics(D, D_exp, hd, ph, sample_dist, window_m)
    return {"arc": arc.tolist(), "D": D.tolist(),
            "heading": hd.tolist(), "phase": ph.tolist(),
            "Z_L": Z_L.tolist(), "Z_R": Z_R.tolist(),
            "metrics": ({k: v.tolist() for k, v in m.items()} if m else None),
            "summary": ({k: float(np.mean(v)) for k, v in m.items()}
                        if m else None)}


# ---------------------------------------------------------------- 主流程 ----
def load_site(site):
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"),
                          encoding="utf-8"))
    h = np.load(os.path.join(RES, f"horizon_factors_{site}.npz"))
    return (np.asarray(d["dem_mean"], dtype=np.float64),
            np.asarray(d["valid"]) == 1,
            np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool),
            h["h1_phase"].astype(np.float64),
            float(meta["x0"]), float(meta["y1"]))


def offset_path(xs, ys, dist):
    """沿**左法向** `(−cosθ, +sinθ)` 平移一条 UTM 折线（dist>0 = 左手侧）。

    修正前用 `(cos(θ+90°), sin(θ+90°))`，与切向点积为 cos2θ，非真正交。
    """
    hd = heading_compass(xs, ys)
    rad = np.radians(hd)
    return np.c_[xs - dist * np.cos(rad), ys + dist * np.sin(rad)], hd


def run_site(site, args):
    dem, valid, mask, h1, x0, y1 = load_site(site)
    demo = dem.copy()
    demo[~valid] = np.nan
    chains = sorted(ordered_paths(skeletonize(mask)), key=lambda q: -len(q))[:2]
    print(f"\n=== {site} === 走廊段 {len(chains)}，"
          f"网格 {dem.shape}，h1_phase 有效 {np.isfinite(h1).mean():.1%}")

    out = {"site": site, "segments": []}
    for li, chain in enumerate(chains):
        x, y = rc_to_utm(chain[:, 0], chain[:, 1], x0, y1, GSD)
        x, y, _ = resample_utm(x, y, SAMPLE)      # 5 m 抽稀后再算切线

        road = extract_cs_features(np.c_[x, y], demo, GSD, SAMPLE,
                                   args.half_width, WINDOW, x0, y1, h1)
        if road["summary"] is None:
            continue
        seg = {"line": li, "len_m": float(len(x) * SAMPLE), "road": road,
               "neg": {}}
        print(f" 段{li} 长约 {seg['len_m']:.0f} m，窗口 {len(road['metrics']['cai'])}")
        print("   古道          " + "  ".join(
            f"{k}={road['summary'][k]:.3f}" for k in
            ("cai", "cai_resid", "csasc", "laas", "laas_cos")))

        for off in OFFSETS:
            q, _ = offset_path(x, y, off)
            n = extract_cs_features(q, demo, GSD, SAMPLE, args.half_width,
                                    WINDOW, x0, y1, h1)
            if n["summary"] is None:
                continue
            seg["neg"][f"{off:+.0f}"] = n
            print(f"   偏移 {off:+6.0f} m  " + "  ".join(
                f"{k}={n['summary'][k]:.3f}" for k in
                ("cai", "cai_resid", "csasc", "laas", "laas_cos")))

        # 判别力：古道窗口 vs 全部偏移窗口
        seg["auc"] = {}
        if seg["neg"]:
            for key in ("cai", "cai_resid", "csasc", "laas", "laas_cos"):
                p = np.asarray(road["metrics"][key])
                q = np.concatenate([np.asarray(v["metrics"][key])
                                    for v in seg["neg"].values()])
                if len(p) > 2 and len(q) > 2:
                    a = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))],
                                      np.r_[p, q])
                    seg["auc"][key] = {"auc": float(a),
                                       "auc_dist": float(max(a, 1 - a)),
                                       "road": float(np.mean(p)),
                                       "neg": float(np.mean(q))}
        out["segments"].append(seg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="all")
    ap.add_argument("--half-width", type=float, default=HALF_W)
    a = ap.parse_args()
    sites = ["杨家坝", "西坝村"] if a.site == "all" else [a.site]

    res = {"sites": {}}
    for s in sites:
        res["sites"][s] = run_site(s, a)

    p = os.path.join(RES, "path_cross_section.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n写出 {os.path.relpath(p, ROOT)}")

    print("\n=== 判别力汇总（AUC，0.5=随机；auc_dist=偏离随机的程度）===")
    print(f"{'站点':<8}{'指标':<12}{'古道均值':>10}{'偏移均值':>10}{'AUC':>8}{'|AUC-.5|':>10}")
    for s in sites:
        for seg in res["sites"][s]["segments"]:
            for k, v in seg.get("auc", {}).items():
                print(f"{s:<8}{k:<12}{v['road']:>10.3f}{v['neg']:>10.3f}"
                      f"{v['auc']:>8.3f}{v['auc_dist'] - 0.5:>10.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    keys = (("cai", "CAI  |D| 均值"), ("csasc", "CSASC  符号一致性"),
            ("laas", "LAAS  |sin(Δ)| 对齐度"))
    fig, axes = plt.subplots(3, len(sites), figsize=(5.6 * len(sites), 9),
                             squeeze=False)
    for j, s in enumerate(sites):
        for i, (k, title) in enumerate(keys):
            ax = axes[i, j]
            for seg in res["sites"][s]["segments"]:
                m = seg["road"]["metrics"][k]
                ax.plot(np.arange(len(m)) * (WINDOW / 2), m, "-", lw=1.6,
                        label=f"古道·段{seg['line']}")
                for off, v in seg["neg"].items():
                    if off.startswith("-50"):
                        ax.plot(np.arange(len(v["metrics"][k])) * (WINDOW / 2),
                                v["metrics"][k], "--", lw=1.0, alpha=0.8,
                                label="偏移 −50 m")
                    if off.startswith("+200"):
                        ax.plot(np.arange(len(v["metrics"][k])) * (WINDOW / 2),
                                v["metrics"][k], ":", lw=1.0, alpha=0.8,
                                label="偏移 +200 m")
            ax.set_title(f"{s} · {title}", fontsize=10)
            ax.set_xlabel("沿路径距离 (m)")
            ax.grid(alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=7.5)
    fig.tight_layout()
    fp = os.path.join(FIG, "path_cross_section.png")
    fig.savefig(fp, dpi=150)
    print(f"写出 {os.path.relpath(fp, ROOT)}")


if __name__ == "__main__":
    main()
