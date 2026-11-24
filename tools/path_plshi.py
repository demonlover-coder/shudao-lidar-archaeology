# -*- coding: utf-8 -*-
"""PLSHI（路径纵坡匀质度指数）可行性校验。

目的不是"实现它"，而是先回答三个决定生死的量化问题：
  Q1 平缓路段的悖论：`CV = std(s)/mean(|s|)` 在 s→0 时发散，平地段会被判为"最不均质"。
  Q2 信噪比：1m DEM 的噪声 σ 在短基线上的坡度噪声 σ√2/L，会不会盖过真实纵坡（3%~8%）？
  Q3 判别力：真值走廊 vs 平行偏移线（±50/100/200m）的 PLSHI 是否可分？

方法
----
走廊中心线 = `results/mask_1m_{site}.npy` 骨架的测地直径有序点列（复用
path_chaikin_viz.ordered_paths）。沿弧长等距重采样（0.5 m），双线性取 dem_mean。
纵坡 s_i = (z_{i+k} − z_i) / L（k = L/step），L 为坡度基线，扫描 {2,5,10,20,40} m。
滑窗 W=100 m 计算：
    CV  = std(s) / (mean|s| + eps)
    CV* = std(s) / (mean|s| + s0)      # s0=0.03（3% 参考坡度），修正平缓悖论
    PLSHI = 1 − CV/(CV+1)  →  仅为 CV 的单调变换，对 AUC 无影响（诚实声明）
    RW  = 窗口内坡度符号不变的平均连续长度（m）
负样本 = 沿法向平移 ±50/±100/±200 m 的平行线（同地形、不同选线）。

输出
----
results/path_plshi.json + figures/path_plshi.png

用法
----
    python tools/path_plshi.py                 # 两站
    python tools/path_plshi.py --site 杨家坝
    python tools/path_plshi.py --synthetic     # 只跑合成数值实验（Q1）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter
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
STEP = 0.5            # 弧长采样间隔 (m)
BASELINES = (2.0, 5.0, 10.0, 20.0, 40.0)
WINDOW = 100.0        # PLSHI 滑窗长度 (m)
S0 = 0.03             # 修正版 CV* 的参考坡度下限
OFFSETS = (50.0, 100.0, 200.0)


def load_site(site):
    z = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    dem = np.asarray(z["dem_mean"], dtype=np.float64)
    valid = np.asarray(z["valid"]) == 1
    mask = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    return dem, valid, mask


def dem_noise_sigma(dem, valid):
    """DEM 噪声估计：5×5 去均值残差的稳健标准差（含真实微地形，故为上界）。"""
    d = np.where(valid, dem, np.nan)
    m = np.isfinite(d)
    f = np.where(m, d, np.nanmean(d[m]))
    res = d - uniform_filter(f, size=5)
    r = res[m]
    r = r[np.isfinite(r)]
    mad = np.median(np.abs(r - np.median(r)))
    # var(x - 邻域均值) = σ²(1 - 1/25)（白噪声假设）
    return float(1.4826 * mad / np.sqrt(1.0 - 1.0 / 25.0))


def smooth_chain(p, win_px):
    """对骨架点列做滑动平均（win_px 单位=像素=米），消除 1px 骨架的横向抖动。

    骨架抖动 × 横坡 = 伪造的纵坡震荡，会让 PLSHI 测到"骨架噪声"而不是"道路工程"。
    该对照组用于把这一混淆分离出去。
    """
    if win_px <= 1 or len(p) < win_px:
        return p
    k = np.ones(int(win_px)) / float(win_px)
    pad = int(win_px) // 2
    out = p.copy()
    for c in range(2):
        v = np.pad(p[:, c], pad, mode="edge")
        sm = np.convolve(v, k, mode="same")
        out[:, c] = sm[pad:pad + len(p)]
    return out


def resample_line(dem, valid, pts_rc):
    """pts_rc: (N,2) 行/列（浮点）。等距重采样 + 双线性取值（域外 NaN）。"""
    p = np.asarray(pts_rc, dtype=np.float64)
    if len(p) < 2:
        return None, None
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < WINDOW:
        return None, None
    n = int(np.floor(total / STEP)) + 1
    arc = np.arange(n) * STEP
    out = np.empty((n, 2))
    for k in range(2):
        out[:, k] = np.interp(arc, cum, p[:, k])
    z = map_coordinates(dem, [out[:, 0], out[:, 1]], order=1, mode="nearest")
    ok = map_coordinates(valid.astype(np.float64), [out[:, 0], out[:, 1]],
                         order=0, mode="constant", cval=0.0) > 0.5
    z = np.where(ok, z, np.nan)
    # 弧长坐标（米），用于滑窗对齐
    return z, out


def slopes(z, L):
    """纵坡序列 s_i（无量纲），步长 STEP；基线 L 米 → 跨 k 个采样点。"""
    k = int(round(L / STEP))
    if k < 1 or len(z) <= k:
        return None
    s = (z[k:] - z[:-k]) / L
    ok = np.isfinite(s)
    return s if ok.mean() > 0.5 else None


def window_stats(s, L, win=WINDOW):
    """沿序列滑窗（步进 10 m）计算 CV / CV* / PLSHI / RW。"""
    k = int(round(win / STEP)) - int(round(L / STEP))
    if k < 8 or len(s) < k:
        return None
    step = max(1, int(round(10.0 / STEP)))
    cvs, cvs2, rws, means = [], [], [], []
    for i in range(0, len(s) - k + 1, step):
        w = s[i:i + k]
        if not np.all(np.isfinite(w)):
            continue
        mu = float(np.mean(np.abs(w)))
        sd = float(np.std(w))
        cvs.append(sd / (mu + 1e-6))
        cvs2.append(sd / (mu + S0))
        means.append(float(np.mean(w)))
        sg = np.sign(w)
        ch = np.flatnonzero(np.diff(sg) != 0)
        if len(ch):
            runs = np.diff(np.concatenate(([0], ch + 1, [len(w)])))
            rws.append(float(np.mean(runs)) * STEP)
        else:
            rws.append(float(len(w)) * STEP)
    if not cvs:
        return None
    cvs = np.array(cvs)
    return dict(cv=cvs, cv_s0=np.array(cvs2), rw=np.array(rws),
                mean_s=np.array(means),
                plshi=1.0 - cvs / (cvs + 1.0),
                plshi_s0=1.0 - np.array(cvs2) / (np.array(cvs2) + 1.0))


def offset_lines(pts_rc, dist_m):
    """沿各点法向平移 dist_m 像素（1px=1m），返回平移后的点列。"""
    p = np.asarray(pts_rc, dtype=np.float64)
    d = np.gradient(p, axis=0)
    nrm = np.hypot(d[:, 0], d[:, 1])
    nrm[nrm < 1e-9] = 1.0
    t = d / nrm[:, None]
    # (row, col) 平面内旋转 90°：(dr, dc) → (-dc, dr)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return p + dist_m * n


def analyze_site(site, log=True, smooth_px=0, tag="raw"):
    dem, valid, mask = load_site(site)
    sigma = dem_noise_sigma(dem, valid)
    skel = skeletonize(mask)
    lines = ordered_paths(skel)
    lines = sorted(lines, key=lambda q: -len(q))[:2]        # 取最长两段
    if smooth_px > 1:
        lines = [smooth_chain(q, smooth_px) for q in lines]
    if log:
        print(f"\n=== {site} [{tag}] ===  骨架段 {len(lines)}，"
              f"DEM 噪声估计 σ≈{sigma:.3f} m")
    out = {"site": site, "sigma_noise": sigma, "n_lines": len(lines),
           "tag": tag, "smooth_px": smooth_px, "baselines": {}}

    for L in BASELINES:
        noise_s = sigma * np.sqrt(2.0) / L
        rec = {"noise_slope": float(noise_s), "pos": {}, "neg": {}}
        pos_w, neg_w = [], []
        for li, line in enumerate(lines):
            z, rc = resample_line(dem, valid, line)
            if z is None:
                continue
            s = slopes(z, L)
            if s is None:
                continue
            st = window_stats(s, L)
            if st is None:
                continue
            rec["pos"][f"line{li}"] = dict(
                n_win=len(st["cv"]), cv=float(np.median(st["cv"])),
                cv_s0=float(np.median(st["cv_s0"])),
                plshi=float(np.median(st["plshi"])),
                rw=float(np.median(st["rw"])),
                mean_s=float(np.median(st["mean_s"])))
            pos_w.append(st["cv"])
            for off in OFFSETS:
                for sgn in (1, -1):
                    z2, rc2 = resample_line(dem, valid, offset_lines(rc, sgn * off))
                    if z2 is None:
                        continue
                    s2 = slopes(z2, L)
                    if s2 is None:
                        continue
                    st2 = window_stats(s2, L)
                    if st2 is None:
                        continue
                    neg_w.append(st2["cv"])
        if pos_w and neg_w:
            p = np.concatenate(pos_w)
            q = np.concatenate(neg_w)
            y = np.r_[np.ones(len(p)), np.zeros(len(q))]
            sc = np.r_[p, q]
            rec["auc_cv"] = float(roc_auc_score(y, -sc))   # CV 越低越像古道
            rec["n_pos"] = int(len(p))
            rec["n_neg"] = int(len(q))
            rec["cv_pos_med"] = float(np.median(p))
            rec["cv_neg_med"] = float(np.median(q))
            if log:
                print(f"  L={L:>4.1f}m  坡度噪声≈{noise_s:.3f}"
                      f"  CV 古道 {np.median(p):.3f} vs 偏移 {np.median(q):.3f}"
                      f"  AUC(CV低=古道) {rec['auc_cv']:.3f}"
                      f"  [n+={len(p)} n-={len(q)}]")
        out["baselines"][f"{L:g}"] = rec
    return out


def synthetic():
    """Q1 平缓悖论：不同 (纵坡, 噪声) 组合下的 CV 与 PLSHI。"""
    rng = np.random.default_rng(0)
    n = 400
    print("\n=== 合成数值实验（1 m 采样，σ 为高程噪声）===")
    print(f"{'场景':<26}{'CV':>8}{'CV*':>8}{'PLSHI':>8}{'PLSHI*':>8}")
    cases = [
        ("纯平 + 无噪声 (s=0)", 0.0, 0.0),
        ("纯平 + 2cm 噪声", 0.0, 0.02),
        ("纯平 + 10cm 噪声", 0.0, 0.10),
        ("均匀 5% 坡 + 无噪声", 0.05, 0.0),
        ("均匀 5% 坡 + 10cm 噪声", 0.05, 0.10),
        ("自然坡 ±5% 随机震荡", -1.0, 0.0),
        ("自然坡 ±5% + 10cm 噪声", -1.0, 0.10),
    ]
    rows = []
    for name, grade, sig in cases:
        if grade == -1.0:
            s = rng.normal(0.0, 0.05, n)
        elif grade == 0.0:
            s = np.zeros(n)
        else:
            s = np.full(n, grade)
        if sig > 0:
            z = np.cumsum(s) * 1.0 + rng.normal(0, sig, n)
            s = np.diff(z)
        mu, sd = np.mean(np.abs(s)), np.std(s)
        cv = sd / (mu + 1e-6)
        cv2 = sd / (mu + S0)
        rows.append(dict(case=name, cv=float(cv), cv_s0=float(cv2),
                         plshi=float(1 - cv / (cv + 1)),
                         plshi_s0=float(1 - cv2 / (cv2 + 1))))
        print(f"{name:<24}{cv:>9.3f}{cv2:>9.3f}"
              f"{rows[-1]['plshi']:>9.3f}{rows[-1]['plshi_s0']:>9.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="all")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    res = {"synthetic": synthetic(), "sites": {}}
    sites = ["杨家坝", "西坝村"] if a.site == "all" else [a.site]
    for s in sites:
        res["sites"][s] = {"raw": analyze_site(s, smooth_px=0, tag="raw"),
                           "smooth15": analyze_site(s, smooth_px=15,
                                                    tag="中心线平滑15m")}

    p = os.path.join(RES, "path_plshi.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n写出 {os.path.relpath(p, ROOT)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    ok = [s for s in sites]
    fig, axes = plt.subplots(1, len(ok), figsize=(6 * len(ok), 4.2), squeeze=False)
    for j, s in enumerate(ok):
        ax = axes[0, j]
        for tag, style in (("raw", "o-"), ("smooth15", "s-")):
            b = res["sites"][s][tag]["baselines"]
            Ls = [float(k) for k in b if "auc_cv" in b[k]]
            if Ls:
                ax.plot(Ls, [b[f"{L:g}"]["auc_cv"] for L in Ls], style,
                        label=f"AUC·{tag}")
        ax.axhline(0.5, color="k", ls=":", lw=1, label="随机 0.5")
        ax2 = ax.twinx()
        b = res["sites"][s]["raw"]["baselines"]
        Ls = [float(k) for k in b if "noise_slope" in b[k]]
        ax2.plot(Ls, [b[f"{L:g}"]["noise_slope"] for L in Ls], "^--",
                 color="#377eb8", label="坡度噪声 σ√2/L")
        ax2.axhline(0.05, color="#377eb8", ls=":", lw=1)
        ax2.text(Ls[0], 0.052, "真实纵坡量级 5%", fontsize=7.5, color="#377eb8")
        ax2.set_ylabel("坡度噪声（无量纲）", color="#377eb8")
        ax.set_xscale("log"); ax.set_xlabel("坡度计算基线 L (m)")
        ax.set_ylabel("AUC（>0.5 = 古道纵坡更匀质）"); ax.set_ylim(0.2, 0.9)
        ax.set_title(f"{s} · PLSHI 判别力 vs 坡度基线")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fp = os.path.join(FIG, "path_plshi.png")
    fig.savefig(fp, dpi=160)
    print(f"写出 {os.path.relpath(fp, ROOT)}")


if __name__ == "__main__":
    main()
