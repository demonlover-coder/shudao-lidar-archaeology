# -*- coding: utf-8 -*-
"""图 2 研究区位置图：两处测区的山体阴影与真实相对位置。

输入为点云派生的 1 m 分析栅格（data/p2_*_final.tif，EPSG:32648）。
左两幅为测区 hillshade + 比例尺 + 指北针；右幅为两测区在同一投影下的
真实相对位置（只给几何关系，不叠加未经核实的地形底图）。
"""
import os
import numpy as np
import rasterio
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SHUDAO_DATA_ROOT", ROOT / "data"))
OUT = ROOT / "figures" / "fig2_study_area.png"

SITES = [
    dict(name="杨家坝", tif=ROOT / "data" / "p2_杨家坝_final.tif",
         orig=DATA_ROOT / "survey-shp" / "杨家坝" / "杨家河坝碥道-dem.tif",
         mask=ROOT / "results" / "mask_1m_杨家坝.npy", label="(a)"),
    dict(name="西坝村", tif=ROOT / "data" / "p2_西坝村_final.tif",
         orig=DATA_ROOT / "survey-shp" / "西坝村" / "西坝村栈道-dem.tif",
         mask=ROOT / "results" / "mask_1m_西坝村.npy", label="(b)"),
]


def load(path, shape=None):
    with rasterio.open(path) as ds:
        if shape is not None:
            dem = ds.read(1, out_shape=shape, resampling=rasterio.enums.Resampling.average).astype("float32")
        else:
            dem = ds.read(1).astype("float32")
        nodata = ds.nodata
        b = ds.bounds
        res = (b.right - b.left) / dem.shape[1]
    if nodata is not None:
        dem[dem == nodata] = np.nan
    return dem, b, res


def crop_valid(dem, pad=6):
    m = np.isfinite(dem)
    rows = np.where(m.any(1))[0]
    cols = np.where(m.any(0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, dem.shape[0] - 1)
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, dem.shape[1] - 1)
    return dem[r0:r1 + 1, c0:c1 + 1], r0, c0


def crop_to_corridor(dem, mask_path, pad_m=140):
    """以走廊外接矩形加缓冲裁剪，避开栅格边缘的插值条纹。"""
    m = np.load(mask_path) > 0
    if not m.any():
        return dem
    rows = np.where(m.any(1))[0]
    cols = np.where(m.any(0))[0]
    r0 = max(rows[0] - pad_m, 0)
    r1 = min(rows[-1] + pad_m, dem.shape[0] - 1)
    c0 = max(cols[0] - pad_m, 0)
    c1 = min(cols[-1] + pad_m, dem.shape[1] - 1)
    return dem[r0:r1 + 1, c0:c1 + 1]


def add_scalebar(ax, length_m, y_frac=0.07, x_frac=0.07, lw=2.6, fs=8):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * x_frac
    y = y0 + (y1 - y0) * y_frac
    ax.plot([x, x + length_m], [y, y], color="k", lw=lw, solid_capstyle="butt", zorder=6)
    ax.text(x + length_m / 2, y + (y1 - y0) * 0.02, f"{length_m:g} m",
            ha="center", va="bottom", fontsize=fs, zorder=6)


def add_north(ax, x_frac=0.90, y_frac=0.88, size=0.09):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * x_frac
    y = y0 + (y1 - y0) * y_frac
    dy = (y1 - y0) * size
    ax.annotate("", xy=(x, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4), zorder=6)
    ax.text(x, y + dy * 1.12, "N", ha="center", va="bottom", fontsize=9, zorder=6)


fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.3, 0.58], wspace=0.46,
                      left=0.075, right=0.985, top=0.76, bottom=0.15)

ls = LightSource(azdeg=315, altdeg=45)
axes = []
for i, s in enumerate(SITES):
    ax = fig.add_subplot(gs[0, i])
    dem, b, res = load(s["tif"])
    if s.get("orig") and Path(s["orig"]).exists():
        dem_o, _, _ = load(s["orig"], shape=dem.shape)
        # 原始 DEM 与 1 m 分析网格共用同一范围，直接重采样即可对齐
        dem = dem_o
    dem_c = crop_to_corridor(dem, s["mask"])
    mask = ~np.isfinite(dem_c)
    filled = dem_c.copy()
    if mask.any():
        idx = ndimage.distance_transform_edt(mask, return_distances=False,
                                             return_indices=True)
        filled = dem_c[tuple(idx)]
    lo, hi = np.nanpercentile(filled, [1, 99])
    hs = ls.hillshade(filled, vert_exag=1.5, dx=res, dy=res, fraction=1.0)
    hs = (hs - 0.15) / 0.85          # 轻微拉伸，避免平地过暗
    hs = np.clip(hs, 0, 1)
    hs[mask] = np.nan
    ext = [0, dem_c.shape[1] * res, 0, dem_c.shape[0] * res]
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext, origin="upper",
              interpolation="bilinear")
    ax.add_patch(Rectangle((0, 0), ext[1], ext[3], fill=False, ec="k", lw=0.8))
    ax.set_xlim(0, ext[1]); ax.set_ylim(0, ext[3])
    ax.set_aspect("equal")
    ax.set_xlabel("东向 (m)", fontsize=8, labelpad=2)
    ax.set_ylabel("北向 (m)", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7, length=2.5)
    zlo, zhi = np.nanmin(dem), np.nanmax(dem)
    ax.set_title(f'{s["label"]} {s["name"]}（海拔 {zlo:.0f}–{zhi:.0f} m）\n'
                 f'图示范围 {ext[1]:.0f} × {ext[3]:.0f} m', fontsize=8.4, pad=6)
    add_scalebar(ax, 200 if ext[1] < 700 else 400)
    add_north(ax, x_frac=0.86 if ext[1] < 700 else 0.90, y_frac=0.84)
    axes.append(ax)

# 右幅：真实相对位置
axc = fig.add_subplot(gs[0, 2])
info = []
for s in SITES:
    with rasterio.open(s["tif"]) as ds:
        b = ds.bounds
    info.append(dict(name=s["name"], w=b.right - b.left, h=b.top - b.bottom,
                     cx=(b.left + b.right) / 2, cy=(b.bottom + b.top) / 2))

x0 = min(d["cx"] - d["w"] / 2 for d in info) - 1500
x1 = max(d["cx"] + d["w"] / 2 for d in info) + 5200
y0 = min(d["cy"] - d["h"] / 2 for d in info) - 2200
y1 = max(d["cy"] + d["h"] / 2 for d in info) + 2200

for d in info:
    axc.add_patch(Rectangle((d["cx"] - d["w"] / 2, d["cy"] - d["h"] / 2), d["w"], d["h"],
                            fc="#b03a2e", ec="k", lw=0.6, zorder=3))
    if d["name"] == "杨家坝":
        axc.annotate(d["name"], xy=(d["cx"], d["cy"] + d["h"] / 2),
                     xytext=(d["cx"], d["cy"] + d["h"] / 2 + 900),
                     fontsize=7.8, va="bottom", ha="center",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="k"), zorder=4)
    else:
        axc.annotate(d["name"], xy=(d["cx"] - d["w"] / 2, d["cy"]),
                     xytext=(d["cx"] - d["w"] / 2 - 700, d["cy"] + 300),
                     fontsize=7.8, va="center", ha="right",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="k"), zorder=4)

dist = np.hypot(info[0]["cx"] - info[1]["cx"], info[0]["cy"] - info[1]["cy"])
axc.plot([info[0]["cx"], info[1]["cx"]], [info[0]["cy"], info[1]["cy"]],
         ls=(0, (4, 3)), color="#4a6fa5", lw=0.9, zorder=2)
axc.text((info[0]["cx"] + info[1]["cx"]) / 2 - 700,
         (info[0]["cy"] + info[1]["cy"]) / 2,
         f"{dist / 1000:.1f} km", fontsize=7.6, color="#4a6fa5",
         rotation=72, ha="right", va="center")

axc.set_xlim(x0, x1); axc.set_ylim(y0, y1)
axc.set_aspect("equal")
axc.set_xticks([684000, 690000, 696000])
axc.set_xticklabels(["684", "690", "696"])
axc.set_yticks([3712000, 3724000, 3736000, 3748000])
axc.set_yticklabels(["3712", "3724", "3736", "3748"])
axc.set_xlabel("UTM 东向 (km)", fontsize=7.6, labelpad=2)
axc.set_ylabel("UTM 北向 (km)", fontsize=7.6, labelpad=2)
axc.tick_params(labelsize=7, length=2.5)
axc.set_title("(c) 两测区相对位置\n(EPSG:32648)", fontsize=8.4, pad=6)
add_scalebar(axc, 10000, x_frac=0.62, y_frac=0.03, fs=7.4)
add_north(axc, x_frac=0.86, y_frac=0.95)

fig.text(0.5, 0.012,
         "测区中心：杨家坝 106.9864°E / 33.5351°N；西坝村 107.1006°E / 33.8381°N（WGS84）。"
         "底图为点云派生的 1 m 最小高程栅格，(c) 不叠加地形底图。",
         ha="center", fontsize=7.0, color="#333")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, bbox_inches="tight")
print("saved", OUT)
