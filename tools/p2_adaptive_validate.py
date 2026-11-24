# -*- coding: utf-8 -*-
"""自适应 GSD 的梯度保留率验证（选定 50 m × 50 m 走廊窗口）。

输入（都读已有产物，不重建分配图）：
    results/p2_adaptive_gsd_pixel_{site}.npz   逐像元 GSD 分配（1 m 网格）
    源 DEM tif                                  杨家坝 0.05 m / 西坝村 0.1 m

三张图对照（同一窗口，统一重采样到 0.1 m 网格）：
    (a) 源 DEM（0.1 m 原生）
    (b) 传统均匀 1 m（10×10 面积平均后常值块）
    (c) 自适应策略（走廊 0.1 m / 背景 4 m，中间档 0.5、1.0 m）

指标：Sobel 梯度幅值保留率 = mean(|∇x|) / mean(|∇a|)，分三区：
    全窗 / 走廊核心（dist ≤ 1 m）/ 背景（4 m 档）

窗口选择：在 4 m 对齐的 13×13 格（52 m）中取走廊像元最多者，
再取内嵌 500×500 px（整 50 m）作为度量窗口。合成时多留 1 m 边，
保证 4 m 块平均不被裁剪边界污染。

输出：
    figures/adaptive_validation.png
    results/adaptive_validation.json

用法：
    python tools/p2_adaptive_validate.py --site 西坝村
    python tools/p2_adaptive_validate.py --site 杨家坝
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_gsd_render import SRC, M, block_mean, compose, uniform_1m  # noqa: E402
from audit_data import BASE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

SUB4 = 13                     # 52 m = 13 个 4 m 块（合成区，保证块边界对齐）
CELLS = SUB4 * 4              # 52 m 在 1 m 网格上的格数
WIN = 50                      # 度量窗口边长（m）
OFF = (CELLS - WIN) // 2      # 52 m → 50 m 的内嵌偏移（1 m）
CODE = {0.1: 1, 0.5: 2, 1.0: 3, 4.0: 4}


def load_alloc(site: str) -> dict:
    """读已有 GSD 分配 npz（1 m 网格）。label 用 dist==0 还原（EDT 恒等）。"""
    p = os.path.join(RES, f"p2_adaptive_gsd_pixel_{site}.npz")
    d = np.load(p)
    return {"gsd": d["gsd"], "dist": d["dist"], "covered": d["covered"],
            "x0": float(d["x0"]), "y1": float(d["y1"]), "label": d["dist"] == 0}


def pick_window(label: np.ndarray) -> tuple[int, int, int]:
    """在 4 m 对齐的 13×13 格中找走廊像元最多的一块，返回 (n_px, r0, c0)。"""
    H, W = label.shape
    best_n, best = -1, (0, 0)
    for r in range(0, H - CELLS + 1, 4):
        for c in range(0, W - CELLS + 1, 4):
            n = int(label[r:r + CELLS, c:c + CELLS].sum())
            if n > best_n:
                best_n, best = n, (r, c)
    return best_n, best[0], best[1]


def read_crop(site: str, y: int, x: int, h: int, w: int):
    """按源像素坐标裁源 DEM（省内存，不读整幅 1 亿像元）。"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    p = os.path.join(BASE, SRC[site])
    with Image.open(p) as im:
        src_gsd = float(im.tag_v2[33550][0])
        x0, y1 = float(im.tag_v2[33922][3]), float(im.tag_v2[33922][4])
        a = np.array(im.crop((x, y, x + w, y + h)), dtype=np.float32)
    a[a <= -9998.0] = np.nan
    return a, src_gsd, x0, y1


def codes_from(g1m: np.ndarray) -> np.ndarray:
    """1 m GSD 图 → 0.1 m 码图（1/2/3/4 = 0.1/0.5/1.0/4.0 m）。"""
    c = np.zeros(g1m.shape, dtype=np.uint8)
    for g, v in CODE.items():
        c[np.isclose(g1m, g)] = v
    return np.repeat(np.repeat(c, M, 0), M, 1)


def gmag(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值（掩膜外置 0，避免 NaN 传播）。"""
    from scipy.ndimage import sobel
    z = np.where(mask, a, 0.0)
    return np.hypot(sobel(z, axis=1), sobel(z, axis=0))


def lp1m(x: np.ndarray) -> np.ndarray:
    """1 m 低通（10×10 块平均后常值块还原到 0.1 m 网格）。"""
    P = block_mean(x, M)
    up = np.repeat(np.repeat(P, M, 0), M, 1)
    return up[:x.shape[0], :x.shape[1]]


def fine_mag(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """1 m 尺度以下的高频梯度幅值：|∇x| − |∇LP1m(x)|（非负）。"""
    return np.clip(gmag(x, mask) - gmag(lp1m(x), mask), 0.0, None)


def retention(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    ga, gb = gmag(a, mask), gmag(b, mask)
    s = float(ga[mask].mean())
    return float(gb[mask].mean() / s) if s > 0 else float("nan")


def ret_fine(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    ea = float((fine_mag(a, mask) ** 2)[mask].mean())
    eb = float((fine_mag(b, mask) ** 2)[mask].mean())
    return float(eb / ea) if ea > 0 else float("nan")


def corr_grad(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    ga = gmag(a, mask)[mask]
    gb = gmag(b, mask)[mask]
    if ga.std() == 0 or gb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ga, gb)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    e = (a - b)[mask]
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def run(site: str) -> dict:
    from scipy.ndimage import binary_erosion

    m = load_alloc(site)
    n_lab, r0, c0 = pick_window(m["label"])
    print(f"[{site}] 选中窗口 1m 格 r={r0} c={c0}（4 m 对齐），"
          f"52 m 内走廊 {n_lab} px")

    R0, C0, N = r0 * M, c0 * M, CELLS * M              # 0.1 m 合成区：520×520

    # 先读 1 个源像元拿 gsd（再按 k 扩大裁剪范围）
    _, src_gsd, x0, y1 = read_crop(site, 0, 0, 1, 1)
    k_src = int(round(0.1 / src_gsd))
    src, _, _, _ = read_crop(site, R0 * k_src, C0 * k_src, N * k_src, N * k_src)
    native = block_mean(src, k_src)                    # 0.1 m 原生
    del src

    codes01 = codes_from(m["gsd"][r0:r0 + CELLS, c0:c0 + CELLS])
    adaptive = compose(native, codes01)
    uni1 = uniform_1m(native)

    # 内嵌整 50 m 窗口
    o = OFF * M
    sl = (slice(o, o + WIN * M), slice(o, o + WIN * M))
    a, b, c = native[sl], uni1[sl], adaptive[sl]
    codes = codes01[sl]
    dist = np.repeat(np.repeat(m["dist"][r0:r0 + CELLS, c0:c0 + CELLS], M, 0), M, 1)[sl]

    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    mask = binary_erosion(mask, iterations=2)
    mk_core = mask & (dist <= 1.0)
    mk_bg = mask & (codes == 4)

    out = {
        "site": site,
        "window": {
            "size_m": WIN,
            "row_1m": int(r0), "col_1m": int(c0),
            "x_min": round(x0 + (C0 + OFF * M) * 0.1, 2),
            "y_max": round(y1 - (R0 + OFF * M) * 0.1, 2),
        },
        "src_gsd": src_gsd,
        "corridor_px_in_window": int((dist <= 1.0).sum()),
        "corridor_frac": round(float((dist <= 1.0).mean()), 4),
        "mask_px": int(mask.sum()),
        "core_px": int(mk_core.sum()),
        "background_px": int(mk_bg.sum()),
        "mean_grad_source": round(float(gmag(a, mask)[mask].mean()), 5),
        "grad_retention": {},
        "grad_retention_fine": {},
        "grad_corr": {},
        "rmse": {},
    }
    labels = {"uniform_1m": b, "adaptive": c}
    for name, arr in labels.items():
        out["grad_retention"][name] = {
            "all": round(retention(a, arr, mask), 4),
            "core": round(retention(a, arr, mk_core), 4),
            "background": round(retention(a, arr, mk_bg), 4),
        }
        out["grad_retention_fine"][name] = {
            "all": round(ret_fine(a, arr, mask), 4),
            "core": round(ret_fine(a, arr, mk_core), 4),
            "background": round(ret_fine(a, arr, mk_bg), 4),
        }
        out["grad_corr"][name] = round(corr_grad(a, arr, mask), 4)
        out["rmse"][name] = round(rmse(a, arr, mask), 4)

    print(f"  走廊占窗口 {out['corridor_frac'] * 100:.1f}%"
          f"（{out['corridor_px_in_window']} px @ 0.1 m）")
    for name in ("uniform_1m", "adaptive"):
        g = out["grad_retention"][name]
        f = out["grad_retention_fine"][name]
        print(f"  {name:<11} 朴素幅值比 全窗 {g['all']:.3f} / 核心 {g['core']:.3f}"
              f" / 背景 {g['background']:.3f}")
        print(f"  {'':<11} 高频能量比 全窗 {f['all']:.3f} / 核心 {f['core']:.3f}"
              f" / 背景 {f['background']:.3f}；梯度相关 {out['grad_corr'][name]:.3f}"
              f"；RMSE {out['rmse'][name]:.3f} m")

    figure(site, out, a, b, c, codes, dist)
    return out


def figure(site, out, a, b, c, codes, dist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(18, 11.5))
    gs = GridSpec(2, 6, height_ratios=[1.55, 1.0], hspace=0.30, wspace=0.35)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6])]

    vmin, vmax = np.nanmin(a), np.nanmax(a)
    for ax, arr, t in ((axes[0], a, "(a) 源 DEM（原生 0.1 m）"),
                       (axes[1], b, "(b) 传统 1 m 降采样"),
                       (axes[2], c, "(c) 自适应（走廊 0.1 m / 背景 4 m）")):
        im = ax.imshow(np.ma.masked_invalid(arr), cmap="terrain", vmin=vmin, vmax=vmax)
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="高程 (m)")

    xs = np.arange(WIN * M)
    ys = np.arange(WIN * M)
    for arr, col, ls in ((codes == 1, "#d7191c", "-"),
                         (codes == 4, "#2c7bb6", "--")):
        if arr.any() and (~arr).any():
            axes[2].contour(xs, ys, arr.astype(float), levels=[0.5], colors=col,
                            linewidths=1.1, linestyles=ls)
    axes[2].legend(handles=[Line2D([], [], color="#d7191c", lw=1.2, label="0.1 m 区边界"),
                            Line2D([], [], color="#2c7bb6", lw=1.2, ls="--",
                                   label="4 m 区边界")],
                   loc="lower right", fontsize=9, frameon=False)

    groups = ["全窗", "走廊核心 (dist≤1 m)", "背景 (4 m 档)"]
    keys = ["all", "core", "background"]
    panels = [
        (fig.add_subplot(gs[1, 0:3]), "grad_retention",
         "Sobel 梯度幅值保留率（朴素口径）",
         "mean|G| / mean|G源|"),
        (fig.add_subplot(gs[1, 3:6]), "grad_retention_fine",
         "1 m 以下高频梯度能量保留率（判别口径）",
         "高频梯度能量比"),
    ]
    x = np.arange(len(groups))
    w = 0.34
    for ax, key, title, ylab in panels:
        vals_b = [out[key]["uniform_1m"][k] for k in keys]
        vals_c = [out[key]["adaptive"][k] for k in keys]
        bs_b = ax.bar(x - w / 2, vals_b, w, label="(b) 传统 1 m", color="#2c7bb6")
        bs_c = ax.bar(x + w / 2, vals_c, w, label="(c) 自适应", color="#d7191c")
        for bs in (bs_b, bs_c):
            for r in bs:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                        f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=9.5)
        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax.set_xticks(x, groups)
        ax.set_ylim(0, 1.30)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper right")
        ax.grid(axis="y", color="0.85", lw=0.6)
        ax.set_axisbelow(True)

    win = out["window"]
    fig.suptitle(f"{site}　50 m × 50 m 走廊窗口　"
                 f"({win['x_min']:.1f}, {win['y_max']:.1f}) UTM　"
                 f"走廊占比 {out['corridor_frac'] * 100:.1f}%")
    out_png = os.path.join(FIG, "adaptive_validation.png" if site == "西坝村"
                           else f"adaptive_validation_{site}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  写出 {os.path.relpath(out_png, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="西坝村", choices=["西坝村", "杨家坝", "all"])
    args = ap.parse_args()

    sites = ["西坝村", "杨家坝"] if args.site == "all" else [args.site]
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    out = [run(s) for s in sites]
    p = os.path.join(RES, "adaptive_validation.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"写出 {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
