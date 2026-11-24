# -*- coding: utf-8 -*-
"""P2 下游导出：GeoTIFF（L0~L3 + 地形因子 + IDW 混合）+ 可视化。

只读 P2 权重与 1m 网格，产物全部是新文件；不改任何现有文件。
GeoTIFF 用 PIL 写（无 GDAL/rasterio 依赖）。

用法：
    python tools/p2_export_all.py --site 西坝村
    python tools/p2_export_all.py --site 杨家坝 --no-fuse
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P                                   # noqa: E402
from p2_terrain import CHANNELS, terrain_stack       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")

# GeoKeyDirectory：投影坐标系 EPSG:32648（UTM 48N），PixelIsArea
GEOKEY = (1, 1, 0, 3,
          1024, 0, 1, 1,       # GTModelType = projected
          1025, 0, 1, 1,       # GTRasterType = PixelIsArea
          3072, 0, 1, 32648)   # ProjectedCSType = EPSG:32648


def _use_cjk_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def write_tif(path, arr, gsd, x0, y1, nodata=-9999.0):
    """用 PIL 写单波段 float32 GeoTIFF（PixelIsArea，tiepoint 为左上角）。"""
    info = {
        33550: (gsd, gsd, 0.0),                     # ModelPixelScale
        33922: (0.0, 0.0, 0.0, x0, y1, 0.0),        # ModelTiepoint（左上角）
        34735: GEOKEY,                              # GeoKeyDirectory
        42113: str(nodata),                         # GDAL_NODATA
    }
    Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F").save(
        path, tiffinfo=info, compression="tiff_deflate")
    print(f"  写出 {os.path.relpath(path, ROOT)}  shape={tuple(np.shape(arr))}")


def project_covered(site, x0, y1, gsd, H, W, dilate=1):
    """**地面点** (x,y) → H×W 覆盖掩码（3×3 膨胀）。

    P2 训练只用**地面点**（否则 (x,y)→z 不单值），所以"P2 可信域"= **有地面点覆盖**，
    不能用含树冠的全点云（树冠处 P2 的 z 是冠层高度混叠）。
    这里读 `p2_{site}_points_ground.npz`；`xyz` 为按 bbox 归一化的 [0,1]³，
    `bbox`(2,3) 给出 UTM 的 [lo, hi]（含 z），据此反算像元行列。
    """
    d = np.load(os.path.join(DATA, f"p2_{site}_points_ground.npz"))
    xyz, bb = d["xyz"], d["bbox"]
    bx0, by0 = float(bb[0, 0]), float(bb[0, 1])
    bx1, by1 = float(bb[1, 0]), float(bb[1, 1])
    cov = np.zeros((H, W), dtype=bool)
    n = int(xyz.shape[0])
    step = 5_000_000
    for s in range(0, n, step):
        p = xyz[s:s + step]
        xu = bx0 + p[:, 0].astype(np.float64) * (bx1 - bx0)
        yu = by0 + p[:, 1].astype(np.float64) * (by1 - by0)
        col = np.floor((xu - x0) / gsd).astype(np.int64)
        row = np.floor((y1 - yu) / gsd).astype(np.int64)
        ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        cov[row[ok], col[ok]] = True
    print(f"  地面点 {n:,} → 覆盖像元 {int(cov.sum()):,}（{cov.sum() / cov.size:.2%}）")
    if dilate:
        cov = binary_dilation(cov, iterations=dilate)
        print(f"  3×3 膨胀后覆盖像元 {int(cov.sum()):,}（{cov.sum() / cov.size:.2%}）")
    return cov


def idw_fill(dem, obs, radius=8, iters=2):
    """obs==0 处用最近邻 + 局部均值（uniform_filter）填充，返回同形数组。"""
    filled = np.asarray(dem, dtype=np.float32).copy()
    miss = ~obs
    if not miss.any():
        return filled
    idx = distance_transform_edt(miss, return_distances=False, return_indices=True)
    nearest = np.asarray(dem, dtype=np.float32)[idx[0], idx[1]]
    filled[miss] = nearest[miss]
    for _ in range(iters):
        sm = uniform_filter(filled, size=2 * radius + 1)
        filled[miss] = sm[miss]
    return filled


def plot_multires(levels, gsd, out, site):
    plt = _use_cjk_font()
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 6))
    for l, ax in enumerate(axes):
        im = ax.imshow(levels[l], cmap="terrain")
        ax.set_title(f"L{l}  1/{2 ** l}  {levels[l].shape}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{site} P2 多分辨率 DEM（gsd={gsd} m）")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  写出 {os.path.relpath(out, ROOT)}")


def plot_terrain(stack, out, site):
    plt = _use_cjk_font()
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(5 * len(CHANNELS), 6))
    for ax, c, arr in zip(axes, CHANNELS, stack):
        lo, hi = np.percentile(arr, (1, 99))
        im = ax.imshow(arr, cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(f"{c}\n[{lo:.3f}, {hi:.3f}]")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{site} 地形因子（L0，{len(CHANNELS)} 通道）")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  写出 {os.path.relpath(out, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["杨家坝", "西坝村"])
    ap.add_argument("--gsd", type=float, default=None, help="缺省从 meta 读")
    ap.add_argument("--n_levels", type=int, default=4)
    ap.add_argument("--no-terrain", dest="terrain", action="store_false",
                    help="跳过地形因子")
    ap.add_argument("--no-fuse", dest="fuse", action="store_false",
                    help="跳过 IDW 混合")
    a = ap.parse_args()

    os.makedirs(FIG, exist_ok=True)
    x0, y1, wm, hm, gsd, d = P.dem_window(a.site)
    if a.gsd is not None:
        gsd = a.gsd
    H, W = d["dem_mean"].shape
    print(f"[{a.site}] 窗口 x0={x0:.1f} y1={y1:.1f} {W}×{H} @ gsd={gsd} m，设备 {P.DEV}")

    model = P.load_model(a.site)
    packed = model.pack_dem(x0, y1, W * gsd, H * gsd, gsd,
                            n_levels=a.n_levels, device=P.DEV)
    print(f"  pack_dem → {tuple(packed.shape)}（L0~L{a.n_levels - 1}）")
    levels_m = [P.to_utm(a.site, packed[l]).cpu().numpy().astype(np.float64)
                for l in range(a.n_levels)]

    # 1) 每层 GeoTIFF
    for l in range(a.n_levels):
        write_tif(os.path.join(DATA, f"p2_{a.site}_L{l}.tif"), levels_m[l], gsd, x0, y1)

    # 2) 地形因子 GeoTIFF
    stack = None
    if a.terrain:
        stack = terrain_stack(levels_m[0], gsd)
        for c, arr in zip(CHANNELS, stack):
            write_tif(os.path.join(DATA, f"p2_{a.site}_terrain_{c}.tif"), arr, gsd, x0, y1)

    # 3) P2 + IDW 混合
    if a.fuse:
        cov = project_covered(a.site, x0, y1, gsd, H, W)
        final = idw_fill(levels_m[0], cov)
        write_tif(os.path.join(DATA, f"p2_{a.site}_final.tif"), final, gsd, x0, y1)
        print(f"  混合：覆盖区={int(cov.sum()):,} px（取 P2），"
              f"其余 {int((~cov).sum()):,} px 用 IDW 填")

    # 4) 可视化
    plot_multires(levels_m, gsd,
                  os.path.join(FIG, f"p2_export_{a.site}_multires.png"), a.site)
    if a.terrain:
        plot_terrain(stack, os.path.join(FIG, f"p2_export_{a.site}_terrain.png"), a.site)
    print(f"[{a.site}] 完成")


if __name__ == "__main__":
    main()
