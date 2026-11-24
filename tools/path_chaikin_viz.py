# -*- coding: utf-8 -*-
"""路径可视化：把 Dijkstra 的 1 px 原始路径做 2 次 Chaikin 平滑后再膨胀上色。

**纯绘图，不参与评估**——科学指标锁定在 `results/path_extraction_open_dir1.5_d3.json`
（西 IoU 0.2359 / 中心线 6.36 px，杨 0.2180 / 4.98 px），本脚本不改任何判定。

流程：
  1. 取 `results/path_raw_open_dir1.5_{site}.npy`（1 px 路径，可能含多段）；
  2. 每段按 8 邻域图求**测地直径**（两次 Dijkstra）得到有序点列，剔除短枝；
  3. 2 次 Chaikin 平滑（端点保持不动，避免路径变短丢失端点匹配）；
  4. 平滑折线栅格化 → 圆盘膨胀 3 px → 叠加在 tpi31 地形图上。

用法：
    python tools/path_chaikin_viz.py
    python tools/path_chaikin_viz.py --site 西坝村 --iter 3 --radius 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import disk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
RAW_TAG = "open_dir1.5"
S8 = np.ones((3, 3), dtype=np.int32)


def ordered_paths(mask):
    """1 px 路径 → 每段的有序点列 [(N,2) 行/列]。

    用**测地直径**取最长通路：两次 Dijkstra（第一次从任意点取最远点 A，第二次从 A
    取最远点 B，回溯 B→A）。这样即使多段 OR 在一起留下短枝，也只会留下主干。
    """
    lab, n = cc_label(mask, structure=S8)
    out = []
    for cid in range(1, n + 1):
        ys, xs = np.nonzero(lab == cid)
        if ys.size < 4:
            continue
        key = {(y, x): k for k, (y, x) in enumerate(zip(ys.tolist(), xs.tolist()))}
        rows, cols, w = [], [], []
        for (y, x), k in key.items():
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    j = key.get((y + dy, x + dx))
                    if j is not None:
                        rows.append(k); cols.append(j)
                        w.append(np.hypot(dy, dx))
        G = csr_matrix((w, (rows, cols)), shape=(len(key), len(key)))

        _, pred = dijkstra(G, indices=0, directed=False, return_predecessors=True)
        d = np.asarray(pred)
        a = 0                                  # 第一趟只需能到达即可，A 由第二趟定
        dist, pred = dijkstra(G, indices=a, directed=False, return_predecessors=True)
        dist = np.where(np.isfinite(dist), dist, -1.0)
        A = int(np.argmax(dist))
        dist, pred = dijkstra(G, indices=A, directed=False, return_predecessors=True)
        dist = np.where(np.isfinite(dist), dist, -1.0)
        B = int(np.argmax(dist))
        chain, cur = [], B
        while cur != -9999 and cur >= 0:
            chain.append(cur)
            if cur == A:
                break
            cur = int(pred[cur])
        chain.reverse()
        if len(chain) >= 4:
            out.append(np.stack([ys[chain], xs[chain]], axis=1).astype(np.float64))
    return out


def chaikin(points, iterations=2):
    """Chaikin 切角：每段取 1/4、3/4 点；**首尾各复制一次**以保持端点不动。"""
    p = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        if len(p) < 3:
            break
        a, b = p[:-1], p[1:]
        q = 0.75 * a + 0.25 * b
        r = 0.25 * a + 0.75 * b
        mid = np.empty((q.shape[0] * 2, 2), dtype=np.float64)
        mid[0::2], mid[1::2] = q, r
        p = np.vstack([p[0], mid, p[-1]])
    return p


def rasterize(polys, shape):
    """折线 → bool 掩膜（按像素四舍五入取整，段内线性插值加密）。"""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for p in polys:
        for (y0, x0), (y1, x1) in zip(p[:-1], p[1:]):
            n = int(max(abs(y1 - y0), abs(x1 - x0)) * 2) + 1
            ys = np.round(np.linspace(y0, y1, n)).astype(int).clip(0, H - 1)
            xs = np.round(np.linspace(x0, x1, n)).astype(int).clip(0, W - 1)
            m[ys, xs] = True
    return m


def metrics(raw_mask, polys_smooth, gsd=1.0):
    """平滑前后长度与偏移（仅作可视化参考，非评估指标）。"""
    from scipy.ndimage import distance_transform_edt, map_coordinates
    d = distance_transform_edt(~raw_mask)       # 到原始 1px 路径的距离（px）
    L = lambda p: float(np.hypot(*np.diff(p, axis=0).T).sum()) * gsd
    dev = np.concatenate([map_coordinates(d, [p[:, 0], p[:, 1]], order=1,
                                          mode="nearest") for p in polys_smooth])
    return dict(n_poly=len(polys_smooth),
                len_smooth_m=float(sum(L(p) for p in polys_smooth)),
                dev_max_px=float(dev.max()), dev_median_px=float(np.median(dev)))


def render(sites, iters, radius, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    try:
        ev = json.load(open(os.path.join(RES, "path_extraction_open_dir1.5_d3.json"),
                            encoding="utf-8"))
        ev = {r["site"]: r for r in ev["sites"]}
    except FileNotFoundError:
        ev = {}

    fig, axes = plt.subplots(1, len(sites), figsize=(7.2 * len(sites), 7.4))
    axes = np.atleast_1d(axes)
    rep = {}
    for ax, site in zip(axes, sites):
        raw = np.load(os.path.join(RES, f"path_raw_{RAW_TAG}_{site}.npy"))
        label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
        tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
        valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
        v = tpi31[valid]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        bg = np.where(valid, np.clip((tpi31 - lo) / max(hi - lo, 1e-9), 0, 1), np.nan)

        raw_polys = ordered_paths(raw)
        sm_polys = [chaikin(p, iters) for p in raw_polys]
        sm_mask = rasterize(sm_polys, raw.shape)
        band = binary_dilation(sm_mask, structure=disk(radius))

        ax.imshow(bg, cmap="terrain")
        band_fill = np.where(band, 1.0, np.nan)
        ax.imshow(band_fill, cmap="Reds", alpha=0.55, vmin=0, vmax=1.6)
        for p in sm_polys:
            ax.plot(p[:, 1], p[:, 0], "-", color="#d7191c", lw=1.7,
                    solid_capstyle="round")
        ax.contour(label, levels=[0.5], colors="white", linewidths=1.8)

        e = ev.get(site, {})
        sub = (f"\nIoU={e.get('iou', float('nan')):.4f}　"
               f"中心线偏移={e.get('centerline_px', float('nan')):.2f}px　"
               f"中心线F1={e.get('centerline_f1', float('nan')):.4f}") if e else ""
        ax.set_title(f"{site}：Chaikin×{iters} 平滑 + 膨胀 {radius}px（示意）{sub}",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(handles=[Line2D([], [], color="white", lw=2, label="真值走廊(SHP)"),
                           Line2D([], [], color="#d7191c", lw=2, label="预测路径(平滑后)")],
                  loc="upper right", fontsize=8, framealpha=0.85)

        rep[site] = metrics(raw, sm_polys)
        rep[site]["n_raw_poly"] = len(raw_polys)
        rep[site]["raw_px"] = int(raw.sum())
        print(f"[{site}] 段数 {len(raw_polys)} 原始 {int(raw.sum())} px → "
              f"平滑后 {rep[site]['len_smooth_m']:.1f} m；"
              f"相对原始路径偏移 中位 {rep[site]['dev_median_px']:.2f} px / "
              f"最大 {rep[site]['dev_max_px']:.2f} px")

    fig.suptitle("蜀道走廊路径提取（cost=open + h1 相位方向先验 λ=1.5）："
                 "1px 原始路径经 2×Chaikin 平滑后渲染", fontsize=11)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"写出 {os.path.relpath(out, ROOT)}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="all", choices=["西坝村", "杨家坝", "all"])
    ap.add_argument("--iter", type=int, default=2, help="Chaikin 迭代次数")
    ap.add_argument("--radius", type=int, default=3, help="平滑路径膨胀半径（px）")
    a = ap.parse_args()
    sites = ["西坝村", "杨家坝"] if a.site == "all" else [a.site]
    out = os.path.join(FIG, "path_chaikin.png" if a.site == "all"
                       else f"path_chaikin_{a.site}.png")
    rep = render(sites, a.iter, a.radius, out)
    p = os.path.join(RES, "path_chaikin_viz.json")
    json.dump(dict(raw_tag=RAW_TAG, chaikin_iter=a.iter, dilate_radius=a.radius,
                   note="仅渲染用；科学指标以 path_extraction_open_dir1.5_d3.json 为准",
                   sites=rep), open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"写出 {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
