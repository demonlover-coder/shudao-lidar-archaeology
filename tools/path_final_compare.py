# -*- coding: utf-8 -*-
"""最终对比图（全域口径）：原始 Dijkstra  vs  状态增广（连续性惩罚）。

左：tpi31 底图 + 原始 Dijkstra 路径（红色虚线）+ 真值（白色实线）
右：同一底图 + 加入转角连续性惩罚的路径（黄色虚线）+ 真值（白色实线）

杨家坝另有"拽回区"放大图：自动定位基线路径偏离真值最严重的一段，
把同一地理窗口在两种解下并排放大，作为"物理先验起作用"的图证。

口径
----
* 全域（`--roi-radius 0`），与 D12 #36 锁定的终值同口径：
  杨家坝 0.2180 → 0.2791；西坝村 0.2359 → 0.2231。
* 输入的 `path_pred_*.npy` 是**膨胀 3 px 后**的掩膜；画线前先 `skeletonize`
  还原 1 px 中心线，再做 2 次 Chaikin 平滑（移动量 ≤0.47 px，纯视觉切角）。
  指标不在此处计算，全部引用 JSON，避免"图上量出来"的口径漂移。

用法
----
    python tools/path_final_compare.py              # 两站，杨带放大
    python tools/path_final_compare.py --site 杨家坝
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, uniform_filter
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from path_chaikin_viz import chaikin, ordered_paths      # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
BASE_TAG = "open_dir1.5"                 # 原始 Dijkstra（无转角惩罚）
TURN_TAG = "open_dir1.5_turn0.2m1"       # 状态增广 + 连续性惩罚

C_GT = "white"
C_BASE = "#e41a1c"       # 红
C_TURN = "#ffd400"       # 黄
GSD_M = 1.0
BAR_M = 100.0
# 显示窗口余量（px）：杨家坝走廊几乎铺满测区，小余量即可；西坝村走廊只占中段窄带
VIEW_MARGIN = {"杨家坝": 120, "西坝村": 185}


def _pred(tag, site):
    p = os.path.join(RES, f"path_pred_{tag}_{site}.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p).astype(bool)


def _poly(mask):
    """膨胀掩膜 → 1 px 中心线 → 最长有序折线 → Chaikin 平滑（2 次）。"""
    if mask.sum() == 0:
        return np.zeros((0, 2))
    lines = ordered_paths(skeletonize(mask))
    if not lines:
        return np.zeros((0, 2))
    return chaikin(max(lines, key=len), iterations=2)


def _dist_profile(poly, edt):
    """折线各点到真值骨架的距离（亚像元线性插值）。"""
    if len(poly) == 0:
        return np.zeros(0)
    return map_coordinates(edt, [poly[:, 0], poly[:, 1]], order=1)


def find_pullback(site, base_poly, turn_poly, edt):
    """定位"基线偏离最严重、且被优化解明确拉回"的中段窗口。

    只在基线折线的**中段**（去掉首尾各 15%）里找，避免把端点效应当成果；
    窗口 = 以平滑后偏差曲线的峰值为中心、长度 90 个点（≈90 px）的区间。
    """
    d = _dist_profile(base_poly, edt)
    n = len(d)
    if n < 40:
        return None
    ds = uniform_filter(d, size=15)
    lo, hi = int(n * 0.15), int(n * 0.85)
    k = int(np.argmax(ds[lo:hi])) + lo
    h = 45
    a, b = max(0, k - h), min(n, k + h)
    seg = base_poly[a:b]
    y0, x0 = seg.min(axis=0)
    y1, x1 = seg.max(axis=0)
    m = 55.0                                     # 外扩，让读者看到"为什么会被拽"
    box = (int(max(0, y0 - m)), int(max(0, x0 - m)),
           int(min(edt.shape[0], y1 + m)), int(min(edt.shape[1], x1 + m)))
    inside = ((turn_poly[:, 0] >= box[0]) & (turn_poly[:, 0] < box[2])
              & (turn_poly[:, 1] >= box[1]) & (turn_poly[:, 1] < box[3]))
    dt = _dist_profile(turn_poly[inside], edt) if inside.any() else np.zeros(0)
    db = d[a:b]
    return dict(box=box, peak_pt=[float(base_poly[k, 0]), float(base_poly[k, 1])],
                base_mean=float(db.mean()), base_max=float(db.max()),
                turn_mean=float(dt.mean()) if dt.size else float("nan"),
                turn_max=float(dt.max()) if dt.size else float("nan"),
                turn_pts=int(inside.sum()))


def _draw(ax, bg, label, base_poly, turn_poly, box=None, title=""):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    ax.imshow(bg, cmap="terrain", interpolation="nearest")
    for poly, c, ls, lw in ((base_poly, C_BASE, "--", 1.7),
                            (turn_poly, C_TURN, "--", 1.9)):
        if len(poly) < 2:
            continue
        ax.plot(poly[:, 1], poly[:, 0], lw=lw + 2.0, ls=ls, color="black",
                alpha=0.55, solid_capstyle="round", zorder=3)
        ax.plot(poly[:, 1], poly[:, 0], lw=lw, ls=ls, color=c,
                solid_capstyle="round", zorder=4)
    ys, xs = np.nonzero(label)
    if ys.size:
        ax.contour(label, levels=[0.5], colors="black", linewidths=2.9, zorder=5)
        ax.contour(label, levels=[0.5], colors=C_GT, linewidths=1.7, zorder=6)
    if box is not None:
        y0, x0, y1, x1 = box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="black", lw=3.6, ls="-", zorder=7, alpha=0.7))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="#00e5ff", lw=1.9, ls="-", zorder=8))
        ax.annotate("跑偏—拽回区\n放大见下排", xy=(x1, y0), xytext=(8, -12),
                    textcoords="offset points", color="black", fontsize=8.4,
                    ha="left", va="top", zorder=9,
                    bbox=dict(fc="#00e5ff", ec="black", alpha=0.85, pad=2.0))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10.5)
    h = [Line2D([], [], color=C_GT, lw=1.9, label="真值走廊（SHP）"),
         Line2D([], [], color=C_BASE, lw=1.8, ls="--", label="原始 Dijkstra"),
         Line2D([], [], color=C_TURN, lw=1.9, ls="--", label="状态增广 + 连续性惩罚")]
    ax.legend(handles=h, loc="lower right", fontsize=7.6, framealpha=0.72)


def scale_bar(ax, m=BAR_M):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px = m / GSD_M
    X = x0 + (x1 - x0) * 0.06
    Y = y1 - (y1 - y0) * 0.06
    ax.plot([X, X + px], [Y, Y], color="black", lw=4.2, solid_capstyle="butt", zorder=9)
    ax.plot([X, X + px], [Y, Y], color="white", lw=2.2, solid_capstyle="butt", zorder=10)
    ax.text(X + px / 2, Y + (y1 - y0) * 0.028, f"{m:.0f} m", color="white",
            ha="center", va="top", fontsize=8.5,
            bbox=dict(fc="black", ec="none", alpha=0.5, pad=1.2), zorder=10)


def run_site(site, metric, plt):
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    v = tpi31[valid]
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    bg = np.where(valid, np.clip((tpi31 - lo) / max(hi - lo, 1e-9), 0, 1), np.nan)

    ys, xs = np.nonzero(valid)
    pad = 25
    r0, r1 = max(0, ys.min() - pad), min(valid.shape[0], ys.max() + pad)
    c0, c1 = max(0, xs.min() - pad), min(valid.shape[1], xs.max() + pad)
    # 显示窗口取"真值走廊外包框 + 余量"：西坝村走廊只占测区中段一条窄带，
    # 按整个 valid 出图会让 6 px 级偏移完全看不见（求解本身仍是全域）。
    if label.any():
        ly, lx = np.nonzero(label)
        m = VIEW_MARGIN.get(site, 120)
        r0, r1 = max(r0, ly.min() - m), min(r1, ly.max() + m)
        c0, c1 = max(c0, lx.min() - m), min(c1, lx.max() + m)
    sl = (slice(r0, r1), slice(c0, c1))
    bg, lab = bg[sl], label[sl]

    b_poly = _poly(_pred(BASE_TAG, site)) - np.array([r0, c0])
    t_poly = _poly(_pred(TURN_TAG, site)) - np.array([r0, c0])

    edt = distance_transform_edt(~skeletonize(label))
    edt_c = edt[sl]
    info = find_pullback(site, b_poly, t_poly, edt_c)

    m0 = next(s for s in metric["base"]["sites"] if s["site"] == site)
    m1 = next(s for s in metric["turn"]["sites"] if s["site"] == site)
    note = (f"IoU {m0['iou']:.4f} → {m1['iou']:.4f}"
            f"    中心线 {m0['centerline_px']:.2f} → {m1['centerline_px']:.2f} px"
            f"    F1 {m0['centerline_f1']:.4f} → {m1['centerline_f1']:.4f}")

    if not info or site != "杨家坝":
        if site != "杨家坝":                      # 西坝村：只出全域二联图
            fig, axes = plt.subplots(1, 2, figsize=(13.4, 8.2))
            _draw(axes[0], bg, lab, b_poly, np.zeros((0, 2)),
                  title=f"(a) {site} · 原始 Dijkstra：全域 IoU {m0['iou']:.4f}")
            _draw(axes[1], bg, lab, np.zeros((0, 2)), t_poly,
                  title=f"(b) {site} · 状态增广：全域 IoU {m1['iou']:.4f}")
            scale_bar(axes[0]); scale_bar(axes[1])
            fig.suptitle(f"{site} · 全域口径最终对比（红=原始 Dijkstra，"
                         f"黄=状态增广 turn_min=1, λ_turn=0.2）\n{note}", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.935))
            fp = os.path.join(FIG, f"path_final_compare_{site}.png")
            fig.savefig(fp, dpi=170)
            plt.close(fig)
            return fp, None, note
        return None, info, note

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 12.6))
    _draw(axes[0, 0], bg, lab, b_poly, np.zeros((0, 2)), info["box"],
          f"(a) {site} · 原始 Dijkstra：全域 IoU {m0['iou']:.4f}")
    _draw(axes[0, 1], bg, lab, np.zeros((0, 2)), t_poly, info["box"],
          f"(b) {site} · 状态增广：全域 IoU {m1['iou']:.4f}")

    y0, x0, y1, x1 = info["box"]
    zl = (slice(y0, y1), slice(x0, x1))
    for ax, poly, ttl in ((axes[1, 0], b_poly, "(c) 拽回区放大 · 原始"),
                          (axes[1, 1], t_poly, "(d) 拽回区放大 · 状态增广")):
        ax.imshow(bg[zl], cmap="terrain", interpolation="nearest")
        if len(poly) > 1:
            ax.plot(poly[:, 1] - x0, poly[:, 0] - y0, lw=3.6, ls="--", color="black",
                    alpha=0.55, zorder=3)
            ax.plot(poly[:, 1] - x0, poly[:, 0] - y0, lw=1.9, ls="--",
                    color=C_BASE if "原始" in ttl else C_TURN, zorder=4)
        ax.contour(lab[zl], levels=[0.5], colors="black", linewidths=2.9, zorder=5)
        ax.contour(lab[zl], levels=[0.5], colors=C_GT, linewidths=1.8, zorder=6)
        ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
        ax.set_xticks([]); ax.set_yticks([])
        d_mean = info["base_mean"] if "原始" in ttl else info["turn_mean"]
        d_max = info["base_max"] if "原始" in ttl else info["turn_max"]
        ax.set_title(f"{ttl}：区内偏移 均值 {d_mean:.2f} px / 最大 {d_max:.2f} px",
                     fontsize=10)
    scale_bar(axes[0, 0]); scale_bar(axes[0, 1])

    fig.suptitle(
        f"{site} · 全域口径最终对比（红=原始 Dijkstra，黄=状态增广 turn_min=1, λ_turn=0.2）\n"
        f"{note}；蓝框 = 自动定位的\"中段跑偏—被拽回\"区", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fp = os.path.join(FIG, f"path_final_compare_{site}.png")
    fig.savefig(fp, dpi=170)
    plt.close(fig)
    return fp, info, note


def main():
    ap = argparse.ArgumentParser(description="全域口径最终对比图")
    ap.add_argument("--site", default="all", choices=["杨家坝", "西坝村", "all"])
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    metric = dict(
        base=json.load(open(os.path.join(RES, f"path_extraction_{BASE_TAG}_d3.json"),
                            encoding="utf-8")),
        turn=json.load(open(os.path.join(RES, f"path_extraction_{TURN_TAG}_d3.json"),
                            encoding="utf-8")))

    sites = ["杨家坝", "西坝村"] if a.site == "all" else [a.site]
    summary = {}
    for site in sites:
        fp, info, note = run_site(site, metric, plt)
        if fp:
            print(f"[{site}] 写出 {os.path.relpath(fp, ROOT)}")
        else:
            print(f"[{site}] 跳过（仅杨家坝出放大图）")
        print(f"  {note}")
        if info:
            print(f"  拽回区 box={info['box']} 峰点=({info['peak_pt'][0]:.0f},"
                  f"{info['peak_pt'][1]:.0f})  区内偏移 均值 "
                  f"{info['base_mean']:.2f}→{info['turn_mean']:.2f} px、最大 "
                  f"{info['base_max']:.2f}→{info['turn_max']:.2f} px"
                  f"（优化解在框内 {info['turn_pts']} 个采样点）")
            summary[site] = info

    if summary:
        p = os.path.join(RES, "path_final_compare.json")
        json.dump(dict(base_tag=BASE_TAG, turn_tag=TURN_TAG,
                       note="全域口径（roi-radius 0）；偏移=到真值骨架的距离，亚像元插值",
                       pullback=summary), open(p, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"写出 {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
