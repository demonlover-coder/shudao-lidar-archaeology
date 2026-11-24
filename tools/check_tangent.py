# -*- coding: utf-8 -*-
"""蜀道中心线切线的可靠性核查。

为什么需要这个
--------------
1. 真值 SHP 是**面要素**，没有原始线几何 → 切线只能由面推导，不存在绝对真值。
2. 方向先验（h1 相位边权）依赖走廊法向，中心线偏移指标依赖沿法向度量，
   两者都以"切线准确"为前提。
3. 1 px 骨架的方向被量化到 45° 的整数倍，原始 `np.gradient` 的逐步转角很大。

做法
----
对同一段走廊，用三种**独立**方式推切线，比较彼此一致性：

  A. 骨架链梯度（原始）              —— 1px 量化，最抖
  B. 骨架链梯度 + 15 m 滑动平均       —— 实际用于偏移线/PLSHI 的方式
  C. EDT 梯度（距离变换的法向取直）   —— 用于 h1 相位验证的方式
  D. 结构张量（模糊掩膜的梯度）       —— 第三种独立估计

指标：两两中位夹角、p90、≤15° 与 ≤30° 的占比（夹角按 mod 180 计，因为切线无向）。
另测原始骨架的逐步转角抖动，以及平滑后相邻 10 m 的转角变化（衡量曲率可信度）。

输出
----
results/tangent_accuracy.json + figures/tangent_accuracy.png

用法
----
    python tools/check_tangent.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from path_chaikin_viz import ordered_paths          # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
GSD = 1.0
SMOOTH = (5.0, 15.0, 30.0)


def heading(dr, dc):
    """(row, col) 增量 → 罗盘方位角（0=北，90=东），无向（mod 180）。"""
    return np.degrees(np.arctan2(dc, -dr)) % 180.0


def field_heading(field, pts):
    """标量场（此处为 EDT）梯度 → 法向 → 切线（旋转 90°），返回切线方位。"""
    gy = map_coordinates(field, [pts[:, 0] + 1.0, pts[:, 1]], order=1,
                         mode="nearest") - map_coordinates(
        field, [pts[:, 0] - 1.0, pts[:, 1]], order=1, mode="nearest")
    gx = map_coordinates(field, [pts[:, 0], pts[:, 1] + 1.0], order=1,
                         mode="nearest") - map_coordinates(
        field, [pts[:, 0], pts[:, 1] - 1.0], order=1, mode="nearest")
    # 法向 (gy, gx) → 切线 (gx, -gy)
    return heading(gx, -gy)


def ang_diff(a, b):
    """两个无向方向的最小夹角（0~90°）。"""
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def smooth_chain(p, win_px):
    if win_px <= 1 or len(p) < win_px:
        return p
    k = np.ones(int(win_px)) / float(win_px)
    pad = int(win_px) // 2
    out = p.copy()
    for c in range(2):
        v = np.pad(p[:, c], pad, mode="edge")
        out[:, c] = np.convolve(v, k, mode="same")[pad:pad + len(p)]
    return out


def chain_heading(p, half=1):
    """点列切线：中心差分。half=1 → 原始（最抖）。"""
    d = p[half * 2:] - p[:-half * 2] if half > 0 else np.gradient(p, axis=0)
    return heading(d[:, 0], d[:, 1])


def pca_tangent(mask, pts, radius=10.0):
    """第三独立估计：局部掩膜**边界**点的 PCA 主轴（与骨架/EDT 无关）。

    注：不要用 `gaussian_filter(mask)` 的梯度作第三估计——6~8 m 宽带经 σ=3 模糊后
    带心是一个平台，梯度≈0，测到的是噪声而非法向。
    """
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    b = mask & ~binary_erosion(mask, iterations=1)
    by, bx = np.nonzero(b)
    tree = cKDTree(np.c_[by, bx])
    out = np.full(len(pts), np.nan)
    for i, (r, c) in enumerate(pts):
        nb = tree.query_ball_point((r, c), radius)
        if len(nb) < 8:
            continue
        q = np.c_[by[nb], bx[nb]].astype(np.float64)
        q -= q.mean(0)
        w, v = np.linalg.eigh(q.T @ q)
        d = v[:, -1]                        # 主轴 (row, col)
        out[i] = heading(d[0], d[1])
    return out


def run_site(site):
    mask = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    skel = skeletonize(mask)
    lines = sorted(ordered_paths(skel), key=lambda q: -len(q))[:2]

    edt = distance_transform_edt(mask).astype(np.float64)
    blur = gaussian_filter(mask.astype(np.float64), sigma=3.0)

    out = {"site": site, "segments": []}
    for li, chain in enumerate(lines):
        seg = {"n_px": int(len(chain))}

        # 逐步转角抖动：原始 1px 链
        raw_hd = chain_heading(chain, half=1)
        seg["raw_step_turn_deg"] = {
            "median": float(np.median(np.abs(ang_diff(raw_hd[1:], raw_hd[:-1])))),
            "p90": float(np.percentile(np.abs(ang_diff(raw_hd[1:], raw_hd[:-1])), 90)),
        }

        res = {}
        for s in SMOOTH:
            c = smooth_chain(chain, s)
            hd = chain_heading(c, half=1)
            # 每 10 m 的航向变化（曲率是否可信）
            k = int(round(10.0 / GSD))
            if len(hd) > k:
                turn = np.abs(ang_diff(hd[k:], hd[:-k]))
                res[f"s{s:g}"] = {"heading": hd, "turn10_med": float(np.median(turn)),
                                  "turn10_p90": float(np.percentile(turn, 90))}
            else:
                res[f"s{s:g}"] = {"heading": hd, "turn10_med": np.nan,
                                  "turn10_p90": np.nan}

        pts = res["s15"]["heading"].shape[0]
        idx = np.arange(pts) + int(round(15.0 / GSD)) // 2   # 对齐到平滑后的中心
        idx = np.clip(idx, 0, len(chain) - 1)
        p_mid = chain[idx].astype(np.float64)

        # 参考估计：局部边界 PCA 主轴（对带宽中心的中点处仍然良定义）
        hd_ref = pca_tangent(mask, p_mid)
        good = np.isfinite(hd_ref)

        seg["vs_ref"] = {}
        for key in res:
            h = res[key]["heading"]
            if len(h) < 20:
                continue
            n = min(len(h), len(hd_ref))
            h, hr = h[:n], hd_ref[:n]
            m = good[:n]
            if m.sum() < 10:
                continue
            d1 = ang_diff(h[m], hr[m])
            seg["vs_ref"][key] = {
                "median": float(np.median(d1)),
                "p90": float(np.percentile(d1, 90)),
                "le15": float((d1 <= 15).mean()),
                "le30": float((d1 <= 30).mean()),
            }
        seg["n_eval"] = int(good.sum())
        for key in res:
            seg[key] = {"turn10_med": res[key]["turn10_med"],
                        "turn10_p90": res[key]["turn10_p90"]}
        out["segments"].append(seg)

    print(f"\n=== {site} ===")
    for li, seg in enumerate(out["segments"]):
        print(f" 段{li} 长 {seg['n_px']} px  有效比对点 {seg['n_eval']}")
        print(f"   原始 1px 链逐步转角: 中位 {seg['raw_step_turn_deg']['median']:.1f}°"
              f"  p90 {seg['raw_step_turn_deg']['p90']:.1f}°")
        for key, name in (("s5", "平滑 5m"), ("s15", "平滑 15m"), ("s30", "平滑 30m")):
            a = seg["vs_ref"].get(key)
            if not a:
                continue
            print(f"   {name:<9} vs 边界PCA 中位 {a['median']:5.1f}°"
                  f" (≤15° {a['le15']:.0%}, ≤30° {a['le30']:.0%})"
                  f"  p90 {a['p90']:5.1f}°"
                  f" | 每10m转角 {seg[key]['turn10_med']:.1f}°")
    return out


def main():
    sites = ["杨家坝", "西坝村"]
    res = {s: run_site(s) for s in sites}
    p = os.path.join(RES, "tangent_accuracy.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n写出 {os.path.relpath(p, ROOT)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 1, figsize=(6.4, 4.2), squeeze=False)
    keys, names = ["s5", "s15", "s30"], ["5m", "15m", "30m"]
    w = 0.35
    for j, (metric, title) in enumerate((("vs_ref", "vs 局部边界 PCA"),)):
        ax = axes[0, 0]
        for i, s in enumerate(sites):
            med = [np.mean([seg[metric][k]["median"] for seg in res[s]["segments"]
                            if k in seg[metric]]) for k in keys]
            p90 = [np.mean([seg[metric][k]["p90"] for seg in res[s]["segments"]
                            if k in seg[metric]]) for k in keys]
            x = np.arange(len(keys)) + (i - 0.5) * w
            ax.bar(x, med, w * 0.9, label=f"{s} 中位", alpha=0.85)
            ax.errorbar(x, med, yerr=[[0] * 3, np.array(p90) - np.array(med)],
                        fmt="none", ecolor="k", capsize=3, lw=1)
        ax.axhline(45, color="r", ls="--", lw=1.2)
        ax.text(0, 46, "45° = 方向先验保序临界", fontsize=8, color="r")
        ax.axhline(15, color="g", ls=":", lw=1)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels([f"中心线平滑 {n}" for n in names])
        ax.set_ylabel("切线夹角 (度)")
        ax.set_title(f"骨架切线 {title}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(60, ax.get_ylim()[1]))
    fig.tight_layout()
    fp = os.path.join(FIG, "tangent_accuracy.png")
    fig.savefig(fp, dpi=160)
    print(f"写出 {os.path.relpath(fp, ROOT)}")


if __name__ == "__main__":
    main()
