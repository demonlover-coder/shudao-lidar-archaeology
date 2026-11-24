# -*- coding: utf-8 -*-
"""按逐像元 GSD 分配图，从**源 DEM**（方案 B）生成 0.1 m 画布。

不查 P2。渲染规则：
    0.1 m 档 → 源 DEM 原生分辨率（西 0.1 m 原生；杨 0.05 m 2×2 平均到 0.1 m）
    0.5/1.0/4.0 档 → 对应 k×k 面积平均后，以 k×k **常值块**贴回 0.1 m 画布

即"细档保原生、粗档丢高频"，不做任何插值上采样，保证误差来源唯一可解释
（误差 = 块平均造成的低频化，不含重采样振铃）。

输出：
    figures/adaptive_render_{site}.png      120 m 窗口三联图：
                                            源 0.1 m / 自适应 / 均匀 1 m
    results/adaptive_render_{site}.json     数据量与 RMSE 对照
    results/{site}_adaptive_0p1m.tif        可选（--save-tif），带地理标签

用法：
    python tools/adaptive_gsd_render.py --site 西坝村
    python tools/adaptive_gsd_render.py --site all --save-tif
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_adaptive_gsd import build_map  # noqa: E402
from audit_data import BASE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

SRC = {"杨家坝": r"杨家坝\杨家河坝碥道-dem.tif",
       "西坝村": r"西坝村\西坝村栈道-dem.tif"}

M = 10                       # 0.1 m 画布相对 1 m 网格的倍率
K = {0.5: 5, 1.0: 10, 4.0: 40}


def read_source(site: str) -> tuple[np.ndarray, float, float, float]:
    """读源 DEM → (float32 数组, 源 GSD, x0, y1)。nodata 转 NaN。"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    p = os.path.join(BASE, SRC[site])
    with Image.open(p) as im:
        tags = im.tag_v2
        src_gsd = float(tags[33550][0])
        x0, y1 = float(tags[33922][3]), float(tags[33922][4])
        arr = np.array(im, dtype=np.float32)      # 必须 copy：PIL 缓冲只读
    arr[arr <= -9998.0] = np.nan
    print(f"  源 DEM {arr.shape[1]}×{arr.shape[0]} @ {src_gsd} m")
    return arr, src_gsd, x0, y1


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """k×k 面积平均（NaN 忽略），输出 (H//k, W//k)。"""
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    v = a[:h, :w].reshape(h // k, k, w // k, k)
    cnt = np.isfinite(v).sum(axis=(1, 3))
    s = np.nansum(v, axis=(1, 3))
    out = np.full((h // k, w // k), np.nan, dtype=np.float32)
    np.divide(s, cnt, out=out, where=cnt > 0)
    return out


def to_native(a: np.ndarray, src_gsd: float, shape01: tuple[int, int]) -> np.ndarray:
    """源 DEM → 0.1 m 原生画布（不足处补 NaN）。"""
    k = int(round(0.1 / src_gsd))
    nat = a if k == 1 else block_mean(a, k)
    H01, W01 = shape01
    out = np.full(shape01, np.nan, dtype=np.float32)
    h, w = min(nat.shape[0], H01), min(nat.shape[1], W01)
    out[:h, :w] = nat[:h, :w]
    return out


def expand_codes(gsd_map: np.ndarray, shape01: tuple[int, int]) -> np.ndarray:
    """1 m GSD 图 → 0.1 m 码图（uint8：1=0.1, 2=0.5, 3=1.0, 4=4.0）。"""
    code = np.zeros(gsd_map.shape, dtype=np.uint8)
    code[np.isclose(gsd_map, 0.1)] = 1
    code[np.isclose(gsd_map, 0.5)] = 2
    code[np.isclose(gsd_map, 1.0)] = 3
    code[np.isclose(gsd_map, 4.0)] = 4
    up = np.repeat(np.repeat(code, M, axis=0), M, axis=1)
    H01, W01 = shape01
    out = np.full(shape01, 4, dtype=np.uint8)      # 画布外按 4 m
    h, w = min(up.shape[0], H01), min(up.shape[1], W01)
    out[:h, :w] = up[:h, :w]
    return out


CODE = {0.5: 2, 1.0: 3, 4.0: 4}     # 与 expand_codes 一致（1 = 0.1 m 原生，无需贴）


def compose(native: np.ndarray, codes01: np.ndarray) -> np.ndarray:
    """按码图把各档块贴进 0.1 m 画布。

    逐档做 k×k 面积平均（`pyr`），再以 k×k **常值块** 复制回 0.1 m 网格；
    条带裁剪保证任意画布尺寸下行的相位对齐（`r0 - p0*k` 补偿）。
    """
    H01, W01 = native.shape
    pyr = {g: block_mean(native, k) for g, k in K.items()}
    out = np.empty((H01, W01), dtype=np.float32)
    step = 40 * 8                                   # 32 m 条带，对齐 40 px
    for r0 in range(0, H01, step):
        r1 = min(r0 + step, H01)
        strip = native[r0:r1].copy()
        cm = codes01[r0:r1]
        for g, k in K.items():
            P = pyr[g]
            p0 = r0 // k
            p1 = min(-(-r1 // k), P.shape[0])
            if p0 >= p1:
                continue
            up = np.repeat(np.repeat(P[p0:p1], k, 0), k, 1)
            off = r0 - p0 * k
            up = up[off:off + (r1 - r0)]
            if up.shape[0] < (r1 - r0):
                up = np.vstack([up, np.full((r1 - r0 - up.shape[0], up.shape[1]),
                                            np.nan, dtype=np.float32)])
            if up.shape[1] < W01:                   # k=40 时 W 可能不整除
                up = np.hstack([up, np.full((up.shape[0], W01 - up.shape[1]),
                                            np.nan, dtype=np.float32)])
            mask = cm == CODE[g]
            strip[mask] = up[mask]
        out[r0:r1] = strip
    return out


def uniform_1m(native: np.ndarray) -> np.ndarray:
    P = block_mean(native, 10)
    up = np.repeat(np.repeat(P, 10, 0), 10, 1)
    out = np.full(native.shape, np.nan, dtype=np.float32)
    h, w = min(up.shape[0], out.shape[0]), min(up.shape[1], out.shape[1])
    out[:h, :w] = up[:h, :w]
    return out


def metrics(native, adaptive, uni1, gsd_map, core: np.ndarray, codes01: np.ndarray) -> dict:
    def rmse(a, b, m):
        e = (a - b)[m]
        e = e[np.isfinite(e)]
        return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")

    fin = np.isfinite(native)
    res = {
        "canvas_px": int(native.size),
        "valid_px": int(fin.sum()),
        "samples_adaptive": float(sum(
            (gsd_map == g).sum() / (g * g) for g in (0.1, 0.5, 1.0, 4.0))),
        "samples_uniform_0p1m": int(native.size),
        "samples_uniform_1m": int(gsd_map.size),
        "rmse_adaptive_vs_native": rmse(adaptive, native, fin),
        "rmse_uniform1m_vs_native": rmse(uni1, native, fin),
        "rmse_adaptive_core": rmse(adaptive, native, fin & core),
        "rmse_uniform1m_core": rmse(uni1, native, fin & core),
    }
    for g, c in CODE.items():
        res[f"area_lv{g}_m2"] = float((gsd_map == g).sum())
        res[f"rmse_adaptive_lv{g}"] = rmse(adaptive, native, fin & (codes01 == c))
    res["bytes_adaptive_f32"] = int(res["samples_adaptive"] * 4)
    res["bytes_uniform_0p1m_f32"] = int(res["samples_uniform_0p1m"] * 4)
    res["bytes_uniform_1m_f32"] = int(res["samples_uniform_1m"] * 4)
    return res


def figure(site, native, adaptive, uni1, codes01, out_png, size_m: float = 30.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 窗口中心取核心带（0.1 m）像元；成对取中位索引，保证落在核心带上
    ys, xs = np.where(codes01 == 1)
    if ys.size == 0:
        ys, xs = np.array([native.shape[0] // 2]), np.array([native.shape[1] // 2])
    k = ys.size // 2
    cy, cx = int(ys[k]), int(xs[k])
    size = int(size_m * M)
    size = min(size, min(native.shape))
    y0 = min(max(cy - size // 2, 0), native.shape[0] - size)
    x0 = min(max(cx - size // 2, 0), native.shape[1] - size)
    sl = (slice(y0, y0 + size), slice(x0, x0 + size))

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, arr, t in ((axes[0], native[sl], "源 DEM（原生 0.1 m）"),
                       (axes[1], adaptive[sl], "自适应贴图（块平均）"),
                       (axes[2], uni1[sl], "均匀 1 m 对照")):
        im = ax.imshow(np.ma.masked_invalid(arr), cmap="terrain")
        ax.set_title(t)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="高程 (m)")

    cmap = ListedColormap(["#d7191c", "#fdae61", "#ffffbf", "#2c7bb6"])
    norm = BoundaryNorm([0, 1.5, 2.5, 3.5, 4.5], cmap.N)
    im = axes[3].imshow(codes01[sl], cmap=cmap, norm=norm, interpolation="nearest")
    axes[3].set_title("GSD 档位分配")
    axes[3].axis("off")
    cb = fig.colorbar(im, ax=axes[3], fraction=0.046, ticks=[1, 2, 3, 4])
    cb.ax.set_yticklabels(["0.1 m", "0.5 m", "1.0 m", "4.0 m"])

    fig.suptitle(f"{site}：{size_m:.0f} m 窗口 @ 0.1 m 画布（按档块平均，无插值）")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_tif(path, arr, x0, y1):
    import tifffile
    extratags = [
        (33550, "d", 3, (0.1, 0.1, 0.0), False),                 # ModelPixelScale
        (33922, "d", 6, (0.0, 0.0, 0.0, x0, y1, 0.0), False),    # ModelTiepoint
        (42113, "s", len("nan") + 1, "nan", False),              # GDAL_NODATA
    ]
    tifffile.imwrite(path, arr.astype(np.float32), compression="zlib",
                     extratags=extratags)


def run(site: str, save: bool) -> dict:
    m = build_map(site)
    H, W = m["H"], m["W"]
    shape01 = (H * M, W * M)
    print(f"[{site}] 0.1 m 画布 {shape01[0]}×{shape01[1]} = {np.prod(shape01)/1e6:.1f}M 像元")

    src, src_gsd, sx0, sy1 = read_source(site)
    native = to_native(src, src_gsd, shape01)
    del src

    codes01 = expand_codes(m["gsd"], shape01)
    adaptive = compose(native, codes01)
    uni1 = uniform_1m(native)

    core = np.repeat(np.repeat(m["dist"] <= 1.0, M, 0), M, 1)
    core01 = np.zeros(shape01, dtype=bool)
    h, w = min(core.shape[0], shape01[0]), min(core.shape[1], shape01[1])
    core01[:h, :w] = core[:h, :w]

    s = metrics(native, adaptive, uni1, m["gsd"], core01, codes01)
    s.update({"site": site, "canvas": list(shape01), "src_gsd": src_gsd})
    print(f"  样本数：自适应 {s['samples_adaptive']:.0f} / 均匀 0.1 m "
          f"{s['samples_uniform_0p1m']:,} → 压缩 {s['samples_uniform_0p1m']/max(s['samples_adaptive'],1):.1f}×")
    print(f"  float32 体积：{s['bytes_adaptive_f32']/1e6:.1f} MB vs "
          f"{s['bytes_uniform_0p1m_f32']/1e6:.1f} MB")
    print(f"  RMSE vs 源：自适应 {s['rmse_adaptive_vs_native']:.3f} m，"
          f"均匀 1 m {s['rmse_uniform1m_vs_native']:.3f} m")
    print(f"  核心带 RMSE：自适应 {s['rmse_adaptive_core']:.4f} m，"
          f"均匀 1 m {s['rmse_uniform1m_core']:.4f} m")
    for g in (0.5, 1.0, 4.0):
        print(f"    {g:>4.1f} m 档：面积 {s[f'area_lv{g}_m2']:>9.0f} m²  "
              f"RMSE {s[f'rmse_adaptive_lv{g}']:.3f} m")

    out_png = os.path.join(FIG, f"adaptive_render_{site}.png")
    figure(site, native, adaptive, uni1, codes01, out_png)
    print(f"  写出 {os.path.relpath(out_png, ROOT)}")

    if save:
        p = os.path.join(RES, f"{site}_adaptive_0p1m.tif")
        save_tif(p, adaptive, sx0, sy1)
        print(f"  写出 {os.path.relpath(p, ROOT)}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="西坝村", choices=["西坝村", "杨家坝", "all"])
    ap.add_argument("--save-tif", action="store_true")
    args = ap.parse_args()

    sites = ["西坝村", "杨家坝"] if args.site == "all" else [args.site]
    os.makedirs(FIG, exist_ok=True)
    out = [run(s, args.save_tif) for s in sites]
    p = os.path.join(RES, "adaptive_render.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"写出 {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
