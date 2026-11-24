# -*- coding: utf-8 -*-
"""微地形 DTM 重建 + AGL 重算（micro_dtm_agl.py）。

流程
----
1. 取地面簇点：一维 KMeans 标签为 0 的点（簇中心较低者）。
   · 标签来源一：`--labels xxx.npy`，与点云同序的 0/1 数组；
   · 标签来源二（缺省）：按 D+ 口径现场算——取 `C−核心` 增量带，对该带点的 AGL 跑
     一维 `KMeans(2)`，低簇即地面簇（复用 balance_filter 的实现，口径与上一轮完全一致）。
2. **局部 TIN 插值**：只对这些地面簇点建三角网，插到 1 m DTM 网格上。
   插值前先做 0.5 m 体素抽稀（每格取 z 最低点），既降噪又把点数从百万级压到万级。
   用 `scipy.interpolate.griddata(method="linear")` 得到 TIN 线性插值面。
3. **微地形替换**：仅在“地面点支撑区”内用新面替换原 DTM，支撑区 = 含地面点的
   栅格再向外膨胀 `--dilate` 像元。
4. **AGL 重算**：全部点 `AGL = z - 微地形DTM`（双线性取值）。
5. 输出：微地形 npy（有 rasterio 时另存 GeoTIFF）+ 全部点的 AGL 数组 + 日志统计。

NaN 处理（关键，按顺序）
----------------------
(a) `griddata(method="linear")` 在凸包外和三角网空洞处返回 NaN → 用 `method="nearest"`
    对这些像元做最近邻外推（最近邻永不为 NaN，只要有 ≥1 个地面点）。
(b) 但最近邻外推会把地形“平推”到很远的无数据区，所以仍用支撑区掩膜约束替换范围：
    支撑区外保留原 DTM，不参与替换。
(c) 若全站一个地面点都没有（TIN 无法构建）→ 整幅保留原 DTM 并告警。
(d) 原 DTM 本身的 nodata（NaN）→ 用内插面填补；若内插面也没值，则保持 NaN。

用法
----
    # 项目内数据（npz 网格 + npy 点云），标签现场算
    python tools/micro_dtm_agl.py --site 杨家坝

    # 真实 LAS + GeoTIFF DTM + 外部标签
    python tools/micro_dtm_agl.py --las data/xxx.las --dtm data/dtm.tif \
        --labels results/kmeans_label.npy --cell 0.5 --dilate 5

    # 指定 KMeans 分界（跳过拟合）或关闭抽稀
    python tools/micro_dtm_agl.py --site 杨家坝 --bound 5.67 --cell 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, median_filter
from skimage.morphology import disk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from fusion_train import load_terrain                                     # noqa: E402
from model_guided_3d_filter import bilinear                               # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")


# ---------------------------------------------------------------- 输入
def read_las_points(path, chunk=4_000_000):
    """读 LAS/LAZ 的全部 XYZ（已应用 scale/offset，单位米）。分块以控内存。"""
    import laspy

    las = laspy.read(path) if chunk is None else laspy.open(path)
    X, Y, Z = [], [], []
    if hasattr(las, "chunk_iterator"):
        for pts in las.chunk_iterator(chunk):
            X.append(np.asarray(pts.x, dtype=np.float64))
            Y.append(np.asarray(pts.y, dtype=np.float64))
            Z.append(np.asarray(pts.z, dtype=np.float64))
    else:
        X.append(np.asarray(las.x, dtype=np.float64))
        Y.append(np.asarray(las.y, dtype=np.float64))
        Z.append(np.asarray(las.z, dtype=np.float64))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Z)


def read_dtm_tif(path):
    """读 GeoTIFF 为 (dem[H,W], x0, y1, gsd)。要求北向上、像素为正方形。"""
    import rasterio

    with rasterio.open(path) as ds:
        dem = ds.read(1).astype(np.float64)
        t = ds.transform
        if abs(t.a) != abs(t.e):
            raise ValueError(f"像元非正方形：{t.a} vs {t.e}")
        if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
            raise ValueError("不支持带旋转的仿射变换")
        dem[dem == ds.nodata] = np.nan if ds.nodata is not None else dem[dem == ds.nodata]
        x0, y1, gsd = float(t.c), float(t.f), float(abs(t.a))
    return dem, x0, y1, gsd


def write_dtm_tif(path, dem, x0, y1, gsd, crs=None):
    """把微地形写成 GeoTIFF（无 rasterio 时跳过）。"""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        print("  未安装 rasterio，跳过 GeoTIFF 输出（npy 已保存）")
        return None
    prof = dict(driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1,
                dtype="float32", transform=from_origin(x0, y1, gsd, gsd),
                nodata=np.nan, compress="deflate")
    if crs is not None:
        prof["crs"] = crs
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(np.where(np.isfinite(dem), dem, np.nan).astype(np.float32), 1)
    return path


# ---------------------------------------------------------------- 抽稀
def voxel_min_index(X, Y, Z, cell):
    """XY 体素抽稀，每格取 z 最低点（DTM 的标准做法：抑制植被残留）。

    地形面是 (x,y) 的函数，因此只在水平面内划分体素；按 3D 体素划分会把同一
    XY 位置的高层点全部留下，抽稀失效。cell<=0 时返回全部下标。
    """
    if cell is None or cell <= 0:
        return np.arange(X.size)
    k = np.floor((np.stack([X, Y], axis=1) - np.array([X.min(), Y.min()])) / cell)
    k = k.astype(np.int64)
    key = k[:, 0] * (k[:, 1].max() + 1) + k[:, 1]
    order = np.lexsort((Z, key))                     # 主键=格号，次键=z 升序
    ks = key[order]
    first = np.empty(order.size, dtype=bool)
    first[0] = True
    np.not_equal(ks[1:], ks[:-1], out=first[1:])
    return order[first]


# ---------------------------------------------------------------- 核心：微地形
def build_micro_dtm(gx, gy, gz, dem, x0, y1, gsd,
                    dilate_px=5, smooth=0, log=print):
    """地面点 → TIN 线性插值 → 与原地形融合。返回 (micro, support, stats)。

    `micro` 与原 `dem` 同形状，未替换区等于原值；`support` 标记替换区。
    """
    H, W = dem.shape
    Xc = x0 + (np.arange(W) + 0.5) * gsd             # 像元中心坐标
    Yc = y1 - (np.arange(H) + 0.5) * gsd
    GX, GY = np.meshgrid(Xc, Yc)

    # ---- 1) 支撑区：先标出“含地面点”的像元，再向外膨胀
    c = np.rint((gx - x0) / gsd - 0.5).astype(np.int64)
    r = np.rint((y1 - gy) / gsd - 0.5).astype(np.int64)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    occ = np.zeros((H, W), dtype=bool)
    occ[r[ok], c[ok]] = True
    support = binary_dilation(occ, disk(dilate_px)) if dilate_px > 0 else occ.copy()
    n_occ = int(occ.sum())
    log(f"  支撑区：含地面点像元 {n_occ:,}（{n_occ / (H * W) * 100:.2f}%）；"
        f"膨胀 {dilate_px}px 后 {int(support.sum()):,}")

    # ---- 2) TIN 线性插值
    t0 = time.time()
    tin = griddata(np.stack([gx, gy], axis=1), gz, (GX, GY), method="linear")
    n_nan = int(np.isnan(tin).sum())
    lin_ok = np.isfinite(tin)                        # 真实三角网覆盖（外推前）
    log(f"  griddata linear：{gx.size:,} 点 → 网格 {H}×{W}（{time.time() - t0:.1f}s），"
        f"NaN {n_nan:,}（{n_nan / tin.size * 100:.1f}%），"
        f"真实三角网覆盖 {int(lin_ok.sum()):,} px")

    # ---- 3) NaN 处理 (a)：最近邻外推（只对 NaN 像元算，省时间）
    if n_nan:
        nn = griddata(np.stack([gx, gy], axis=1), gz,
                      (GX[np.isnan(tin)], GY[np.isnan(tin)]), method="nearest")
        tin = tin.copy()
        tin[np.isnan(tin)] = nn
    log(f"  NaN 填补后剩余 {int(np.isnan(tin).sum()):,}")

    # ---- 4) 可选平滑（抑制单个错分点造成的高频尖刺）
    if smooth and smooth > 1:
        sm = median_filter(np.where(np.isfinite(tin), tin, dem), size=int(smooth),
                           mode="nearest")
        tin = np.where(np.isfinite(tin), sm, tin)
        log(f"  中值滤波 {int(smooth)}×{int(smooth)}")

    # ---- 5) NaN 处理 (b)(c)：支撑区外的 NaN 与“无地面点”情形一律退回原 DTM
    micro = dem.copy()
    use = support & np.isfinite(tin)
    micro[use] = tin[use]
    if not use.any():
        log("  [!] 支撑区内无有效插值，整幅保留原 DTM")

    # ---- 6) 统计（分开报“真实三角网内”与“外推带”，判断残差来源）
    d = micro - dem
    dm = d[use]
    dm_lin = d[use & lin_ok]
    dm_ext = d[use & ~lin_ok]
    q = (lambda v, p: float(np.percentile(np.abs(v), p)) if v.size else None)
    stats = dict(n_ground=int(gx.size), n_occ_px=n_occ,
                 support_px=int(support.sum()), replaced_px=int(use.sum()),
                 nan_before_fill=int(n_nan), tin_px=int(lin_ok.sum()),
                 ext_px=int((use & ~lin_ok).sum()),
                 diff_med=float(np.median(dm)) if dm.size else None,
                 diff_p05=float(np.percentile(dm, 5)) if dm.size else None,
                 diff_p95=float(np.percentile(dm, 95)) if dm.size else None,
                 diff_abs_p95=q(dm, 95),
                 diff_abs_p50=q(dm, 50),
                 diff_abs_p90=q(dm, 90),
                 diff_abs_p95_lin=q(dm_lin, 95),
                 diff_abs_p50_lin=q(dm_lin, 50),
                 diff_abs_p95_ext=q(dm_ext, 95),
                 diff_med_lin=float(np.median(dm_lin)) if dm_lin.size else None)
    if dm.size:
        fmt = lambda v: "—" if v is None else f"{v:.3f}"
        med_lin = "—" if stats["diff_med_lin"] is None else f"{stats['diff_med_lin']:+.3f}"
        log(f"  微地形−原DTM：中位 {stats['diff_med']:+.3f} m，"
            f"P5/P95 {stats['diff_p05']:+.3f}/{stats['diff_p95']:+.3f} m，"
            f"|Δ| P50/P90/P95 {fmt(stats['diff_abs_p50'])}/"
            f"{fmt(stats['diff_abs_p90'])}/{fmt(stats['diff_abs_p95'])} m")
        log(f"    ├ 真实三角网内（{dm_lin.size:,} px）：|Δ| P95 "
            f"{fmt(stats['diff_abs_p95_lin'])} m，中位 {med_lin} m")
        log(f"    └ 最近邻外推带（{dm_ext.size:,} px）：|Δ| P95 "
            f"{fmt(stats['diff_abs_p95_ext'])} m")
    return micro, use, stats


# ---------------------------------------------------------------- 标签
def ground_label_from_band(site, valid, dem, meta, a, bound, log):
    """按 D+ 口径现场算地面簇：C−核心增量带 → AGL 一维 KMeans → 低簇。"""
    from balance_filter import kmeans_agl_split, load_pointwise

    pt = load_pointwise(site, a, valid, dem, meta, log)
    band = (pt["pv"] >= a.thr_lo) & (pt["pv"] < a.thr)
    c_keep = pt["hit_lo"][(a.thr_lo, 10.0)] | (pt["echo"] & pt["hit_dil"][(a.thr_lo, 10.0)])
    addon = c_keep & ~(pt["pv"] >= a.thr)
    if bound is None:
        info = kmeans_agl_split(pt["agl"][addon], a.kmeans_seed, a.kmeans_fit,
                                log, "微地形地面簇")
        bound = info["bound_m"]
        log(f"  KMeans 自动分界 = {bound:.2f} m")
    ground = addon & (pt["agl"] <= bound)
    log(f"  地面簇点 {int(ground.sum()):,}（增量带 {int(addon.sum()):,}，"
        f"缓冲带 0.1≤P<0.5 仅 {int(band.sum()):,}）")
    return ground, bound, pt["agl"]


# ---------------------------------------------------------------- 主流程
def main():
    p = argparse.ArgumentParser(description="微地形 DTM 重建与 AGL 重算")
    p.add_argument("--site", default="杨家坝", help="项目内站点（用 npy 点云 + npz 网格）")
    p.add_argument("--base", choices=("mean", "min"), default="mean",
                   help="项目 npz 里用哪个栅格当“原始 DTM”：mean=像元平均高程（旧口径），"
                        "min=像元最低高程（更接近真实地面）")
    p.add_argument("--las", default=None, help="改用外部 LAS/LAZ（需同时给 --dtm）")
    p.add_argument("--dtm", default=None, help="GeoTIFF DTM（配合 --las）")
    p.add_argument("--labels", default=None,
                   help="与点云同序的 0/1 地面标签 npy；缺省则按 D+ 口径现场算")
    p.add_argument("--bound", type=float, default=None,
                   help="直接指定地面/树冠分界（m），跳过 KMeans 拟合")
    p.add_argument("--cell", type=float, default=0.5,
                   help="TIN 前的体素抽稀格宽 m（0=不抽稀）")
    p.add_argument("--dilate", type=int, default=5, help="支撑区膨胀像元数")
    p.add_argument("--smooth", type=int, default=0, help="中值滤波窗口（0/1=关闭）")
    p.add_argument("--pct", type=float, default=100.0,
                   help="只用地簇内 AGL 最低的 pct%% 建 TIN（默认 100=整簇）")
    p.add_argument("--kmeans-seed", type=int, default=0)
    p.add_argument("--kmeans-fit", type=int, default=2_000_000)
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "micro_dtm_agl.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True); lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"微地形 DTM 重建 · {time.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  数据源={'LAS ' + os.path.basename(a.las) if a.las else '项目 ' + a.site}")

    # ---- 数据源一：项目内 npy 点云 + npz 网格
    if not a.las:
        site = a.site
        dem, _, valid = load_terrain(site)
        if a.base == "min":                       # 换用像元最低高程当原始 DTM
            from fix_tpi31_auc import fill_nodata
            dem = fill_nodata(np.load(os.path.join(DATA, f"{site}_1m.npz"))["dem_min"]
                              .astype(np.float64), valid)
        meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
        x0, y1, gsd = float(meta["x0"]), float(meta["y1"]), float(meta["gsd_m"])
        o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
        xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
        X = np.asarray(xyz[:, 0], dtype=np.float64) + float(o[0])
        Y = np.asarray(xyz[:, 1], dtype=np.float64) + float(o[1])
        Z = np.asarray(xyz[:, 2], dtype=np.float64) + float(o[2])
        n = X.size
        log(f"  点云 {n:,}；网格 {dem.shape[1]}×{dem.shape[0]} @ {gsd:g} m")

        if a.labels:
            gl = np.asarray(np.load(a.labels, mmap_mode="r")) == 0
            bound = a.bound
            log(f"  标签来自 {os.path.basename(a.labels)}：地面 {int(gl.sum()):,}")
        else:
            ns = SimpleNamespace(thr=0.5, thr_lo=0.1, sweep=False, dil_px=10,
                                 kmeans_seed=a.kmeans_seed, kmeans_fit=a.kmeans_fit)
            gl, bound, _ = ground_label_from_band(site, valid, dem, meta, ns,
                                                  a.bound, log)
        tag = f"{site}_{a.base}"
    else:
        X, Y, Z = read_las_points(a.las)
        n = X.size
        if not a.dtm:
            raise SystemExit("使用 --las 时必须同时给 --dtm（GeoTIFF）")
        dem_raw, x0, y1, gsd = read_dtm_tif(a.dtm)
        dem = np.where(np.isfinite(dem_raw), dem_raw, np.nan)
        log(f"  点云 {n:,}；DTM {dem.shape[1]}×{dem.shape[0]} @ {gsd:g} m")
        if not a.labels:
            raise SystemExit("外部 LAS 必须提供 --labels（0=地面簇）")
        gl = np.asarray(np.load(a.labels, mmap_mode="r")) == 0
        tag = os.path.splitext(os.path.basename(a.las))[0]
        valid = np.isfinite(dem)

    # ---- 原始 AGL（只依赖原 DTM，先算出来，供分位数筛选使用）
    r = (y1 - Y) / gsd - 0.5
    c = (X - x0) / gsd - 0.5
    agl_old = Z - bilinear(np.where(np.isfinite(dem), dem, 0.0), r, c)
    in_dtm = (r >= -0.5) & (r <= dem.shape[0] - 0.5) & (c >= -0.5) & (c <= dem.shape[1] - 0.5)

    # ---- 实验 A：只取地面簇内 AGL 最低的 pct% 建 TIN（抑制低层植被混入）
    if a.pct < 100.0:
        n_gl0 = int(gl.sum())
        cut = float(np.percentile(agl_old[gl], a.pct))
        gl = gl & (agl_old <= cut)
        log(f"  分位筛选：地面簇内 AGL 的 P{a.pct:g} = {cut:.3f} m → "
            f"{n_gl0:,} → {int(gl.sum()):,} 点（{gl.sum() / max(n_gl0, 1) * 100:.1f}%）")

    # ---- 地面点 → 抽稀 → TIN
    gx, gy, gz = X[gl], Y[gl], Z[gl]
    log(f"  地面簇候选 {gx.size:,} 点")
    if gx.size == 0:
        raise SystemExit("没有地面簇点，无法建 TIN")

    keep = voxel_min_index(gx, gy, gz, a.cell)
    gx, gy, gz = gx[keep], gy[keep], gz[keep]
    log(f"  体素抽稀（{a.cell:g} m，每格取最低点）→ {gx.size:,} 点用于 TIN")

    micro, support, stats = build_micro_dtm(gx, gy, gz, dem, x0, y1, gsd,
                                            dilate_px=a.dilate, smooth=a.smooth, log=log)

    # ---- AGL 重算（双线性）
    agl_new = Z - bilinear(np.where(np.isfinite(micro), micro, 0.0), r, c)

    ch = np.abs(agl_new - agl_old) > 0.5
    log(f"  AGL：全域中位 {np.median(agl_old):.3f} → {np.median(agl_new):.3f} m；"
        f"地面簇中位 {np.median(agl_old[gl]):.3f} → {np.median(agl_new[gl]):.3f} m")
    log(f"  AGL 变化 >0.5 m 的点 {int(ch.sum()):,}（{ch.mean() * 100:.1f}%），"
        f"其中域内 {int((ch & in_dtm).sum()):,}")

    # ---- 输出（带分位后缀，避免覆盖整簇版本）
    sfx = f"p{a.pct:g}" if a.pct < 100.0 else "full"
    pn = os.path.join(RES, f"micro_dtm_{tag}_{sfx}.npy")
    np.save(pn, micro.astype(np.float32))
    ps = os.path.join(RES, f"micro_dtm_support_{tag}_{sfx}.npy")
    np.save(ps, support)
    pa = os.path.join(RES, f"agl_micro_{tag}_{sfx}.npy")
    np.save(pa, agl_new.astype(np.float32))
    log(f"  保存 {os.path.relpath(pn, ROOT)} / {os.path.basename(ps)} / "
        f"{os.path.basename(pa)}")
    ptif = write_dtm_tif(os.path.join(RES, f"micro_dtm_{tag}_{sfx}.tif"), micro,
                         x0, y1, gsd)
    if ptif:
        log(f"  保存 {os.path.relpath(ptif, ROOT)}")

    pj = os.path.join(RES, f"micro_dtm_agl_{tag}_{sfx}.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(dict(site=tag, params=dict(cell=a.cell, dilate=a.dilate,
                                             smooth=a.smooth, bound=bound,
                                             pct=a.pct),
                        n_points=int(n), base=a.base, **stats,
                       agl=dict(old_med=float(np.median(agl_old)),
                                new_med=float(np.median(agl_new)),
                                old_med_ground=float(np.median(agl_old[gl])),
                                new_med_ground=float(np.median(agl_new[gl])),
                                n_changed_gt05=int(ch.sum()))),
                  f, ensure_ascii=False, indent=2)
    log(f"  保存 {os.path.relpath(pj, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
