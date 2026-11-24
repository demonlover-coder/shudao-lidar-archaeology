# -*- coding: utf-8 -*-
"""真实 CSF 地面滤波对照（尽力而为），失败则退回"渐进 TIN 代理"。

指标与 `tools/corridor_filter.py` **完全同口径**（逐点、同一批 LAS 点）：
    road_point       真值掩膜内且 AGL≤0.5 m 的点
    veg_point        AGL≥1.0 m 的点（全场）
    road_preservation 保留的蜀道路面点 / 全部蜀道路面点
    veg_removal      1 − 保留植被点 / 全部植被点（全场）
    keep_ratio       保留点数 / 窗口内总点数
    road_fp          保留但不在真值掩膜内的点数

AGL 基准面 = 1 m `dem_min`（与 corridor_filter 一致）。SHP 只用于评分。

真值：杨家坝 = results/mask_1m_杨家坝.npy；西坝村 = 新 SHP 主板（最大连通域）。

两种方式
--------
1. `method="csf"`（优先）
   - 依赖 `pip install cloth-simulation-filter`（PyPI 的 `CSF` 是同名金融数据包，
     **不是**地面滤波；真实 CSF 的发行名为 cloth-simulation-filter 1.1.7）
   - 参数：cloth_resolution=1.0 m、class_threshold=0.3 m、rigidness=2、
     iterations=500、bSloopSmooth=False（除前两项外均为库默认）
   - CSF 需要一次性载入点云；全量 91M 点约 2.2 GB（Python + C++ 各一份），
     本机空闲内存不足 → 按步长抽稀（JSON 里注明 `full_scan` 与 `stride`）
2. `method="progressive_tin_proxy"`（CSF 不可用/超时时的退路）
   - Axelsson 渐进加密 TIN 的简化版：
       ① 10 m 块最低点 → Delaunay TIN；
       ② 粗 TIN 残差 ≤3 m 的 1 m 像元视为候选地面；
       ③ 只用候选像元的 4 m 块最低点重建 TIN；
       ④ 点高程 − TIN ≤0.3 m 判为地面。
   - **不是 CSF**，文件名/JSON 明确标注；全量流式扫描。

输出：`results/csf_filter.json`（CSF）或 `results/tin_proxy_filter.json`（代理）。

用法：
    python tools/csf_filter.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import label as cc_lab

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import p2_dem as P  # noqa: E402
from eval_new_shp import LAS, RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402
from eval_freeaxis import fill_nodata  # noqa: E402

CH = 3_000_000
SS = 4
SITES = ("杨家坝", "西坝村")
CFG = {
    "杨家坝": {"truth": "old", "d3": 3.0},
    "西坝村": {"truth": "new_main", "d3": 4.0},
}
THR = 0.3          # 地面高差阈值（CSF class_threshold / TIN 残差阈值）
CLOTH_RES = 1.0
TIN_COARSE = 10    # 粗 TIN 块边长（m）
TIN_FINE = 4       # 加密 TIN 块边长（m）
TIN_COARSE_THR = 3.0   # 粗 TIN 候选地面阈值（m）
CSF_STRIDE = {"杨家坝": 1, "西坝村": 2}   # 分站抽稀步长（内存约束；见 JSON full_scan）
CSF_STRIDE_FALLBACK = 4                   # 内存不足时的降级步长


def truth_mask(site, valid, x0, y1, gsd, H, W):
    if CFG[site]["truth"] == "old":
        return np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
    _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
    hr = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
    gt = (hr.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5) & valid
    lb, n = cc_lab(gt)
    if n > 1:
        sz = np.bincount(lb.ravel())
        sz[0] = 0
        gt = lb == sz.argmax()
    return gt


# ------------------------------------------------------------------ TIN 代理
def block_min(dem, mask, bs):
    """按 bs×bs 块取最低高程，返回**最低像元的真实坐标** (row, col) 与高程。

    低位值必须挂回它自己的像元位置（而不是块中心），否则在 30°+ 陡坡上
    块中心与最低角的高差可达数米，TIN 会整体压低、把地面点全判成非地面。
    """
    H, W = dem.shape
    Hb = int(np.ceil(H / bs))
    Wb = int(np.ceil(W / bs))
    a = np.full((Hb * bs, Wb * bs), np.inf, np.float64)
    a[:H, :W] = np.where(mask, dem, np.inf)
    b = a.reshape(Hb, bs, Wb, bs)
    bm = b.min(axis=(1, 3))
    off = np.argmin(b.transpose(0, 2, 1, 3).reshape(Hb, Wb, bs * bs), axis=2)
    rr, cc = np.nonzero(np.isfinite(bm))
    rows = rr * bs + off[rr, cc] // bs
    cols = cc * bs + off[rr, cc] % bs
    return np.column_stack([rows, cols]), bm[rr, cc]


def tin_surface(pts, vals, H, W):
    """块最低点 → 线性 TIN → 1 m 全面幅（凸包外按最近邻填）。"""
    if pts.shape[0] < 4:
        raise RuntimeError("TIN 控制点不足")
    gy, gx = np.mgrid[0:H, 0:W]
    z = griddata(pts, vals, (gy, gx), method="linear")
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = griddata(pts, vals, (gy[bad], gx[bad]), method="nearest")
    return z


def build_tin_proxy(dem, valid):
    """渐进 TIN：10 m 块最低点 → 候选地面 → 4 m 块最低点重建。"""
    t0 = time.time()
    p1, v1 = block_min(dem, valid, TIN_COARSE)
    tin1 = tin_surface(p1, v1, *dem.shape)
    print(f"    粗 TIN：{p1.shape[0]:,} 控制点（{TIN_COARSE} m 块，{time.time()-t0:.1f}s）",
          flush=True)
    cand = valid & ((dem - tin1) <= TIN_COARSE_THR)
    print(f"    候选地面像元 {int(cand.sum()):,} / valid {int(valid.sum()):,}", flush=True)
    p2, v2 = block_min(dem, cand, TIN_FINE)
    tin2 = tin_surface(p2, v2, *dem.shape)
    print(f"    加密 TIN：{p2.shape[0]:,} 控制点（{TIN_FINE} m 块，"
          f"总 {time.time()-t0:.1f}s）", flush=True)
    return tin2, dict(n_coarse=int(p1.shape[0]), n_fine=int(p2.shape[0]),
                      cand_px=int(cand.sum()))


# ------------------------------------------------------------------ 指标
def stream_metrics(site, tin, gt, x0, y1, gsd, dem, valid, stride=1):
    """流式读 LAS，逐点判定地面（Z − tin ≤ 0.3）并累计指标。

    stride > 1 时只取每 stride 个点（CSF 内存受限时的抽稀，JSON 会注明）。
    """
    H, W = dem.shape
    keep_n = road_n = veg_n = road_fp = 0
    road_tot = veg_tot = n_tot = 0
    import laspy
    t0 = time.time()
    with laspy.open(LAS[site]) as f:
        for pts in f.chunk_iterator(CH):
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            Z = np.asarray(pts.z, np.float64)
            if stride > 1:
                X, Y, Z = X[::stride], Y[::stride], Z[::stride]
            c = np.floor((X - x0) / gsd).astype(np.int64)
            r = np.floor((y1 - Y) / gsd).astype(np.int64)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            if not ok.any():
                continue
            X, Y, Z, c, r = X[ok], Y[ok], Z[ok], c[ok], r[ok]
            agl = Z - dem[r, c]
            tr = gt[r, c]
            keep = (Z - tin[r, c]) <= THR
            road = tr & (agl <= 0.5)
            veg = agl >= 1.0
            n_tot += int(Z.size)
            road_tot += int(road.sum())
            veg_tot += int(veg.sum())
            keep_n += int(keep.sum())
            road_n += int((keep & road).sum())
            veg_n += int((keep & veg).sum())
            road_fp += int((keep & ~tr).sum())
    road_p = road_n / max(road_tot, 1)
    veg_r = 1.0 - veg_n / max(veg_tot, 1)
    keep_r = keep_n / max(n_tot, 1)
    return {
        "n_points_used": n_tot,
        "n_road_points": road_tot,
        "n_veg_points": veg_tot,
        "n_keep": keep_n,
        "keep_ratio": round(keep_r, 4),
        "road_preservation": round(road_p, 4),
        "veg_removal": round(veg_r, 4),
        "road_fp": road_fp,
        "road_fp_rate": round(road_fp / max(keep_n, 1), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }


# ------------------------------------------------------------------ CSF 分支
def read_xyz_stride(site, stride):
    """按全局步长读取 LAS 坐标（预分配，避免 list+concat 的 2× 峰值内存）。"""
    import laspy
    with laspy.open(LAS[site]) as f:
        n_full = int(f.header.point_count)
        m = (n_full + stride - 1) // stride
        out = np.empty((m, 3), np.float64)
        k = off = 0
        for pts in f.chunk_iterator(CH):
            L = len(pts)
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            Z = np.asarray(pts.z, np.float64)
            j = (stride - off % stride) % stride
            sl = slice(j, L, stride)
            cnt = len(range(j, L, stride))
            out[k:k + cnt, 0] = X[sl]
            out[k:k + cnt, 1] = Y[sl]
            out[k:k + cnt, 2] = Z[sl]
            k += cnt
            off += L
    return out[:k], n_full


def try_csf(site, x0, y1, gsd, H, W, dem, stride):
    """运行 CSF；返回逐点结果 dict，或 None（库不可用）。

    CSF 需要整块点云，故按步长抽稀后一次性读入；内存不足时自动降级。
    """
    try:
        import CSF
    except Exception as e:  # noqa: BLE001
        print(f"    CSF 不可用：{e}")
        return None
    for sd in (stride, CSF_STRIDE_FALLBACK):
        try:
            return _run_csf(CSF, site, x0, y1, gsd, H, W, dem, sd)
        except MemoryError:
            import gc
            gc.collect()
            print(f"    内存不足（stride={sd}），降级到 stride={CSF_STRIDE_FALLBACK}",
                  flush=True)
    return None


def _run_csf(CSF, site, x0, y1, gsd, H, W, dem, stride):
    xyz, n_full = read_xyz_stride(site, stride)
    print(f"    CSF 输入 {xyz.shape[0]:,} 点（全量 {n_full:,}，步长 {stride}）",
          flush=True)
    t0 = time.time()
    csf = CSF.CSF()
    csf.params.cloth_resolution = CLOTH_RES
    csf.params.class_threshold = THR
    csf.params.rigidness = 2
    if hasattr(csf.params, "iterations"):
        csf.params.iterations = 500
    elif hasattr(csf.params, "interations"):
        csf.params.interations = 500
    csf.params.bSloopSmooth = False
    csf.setPointCloud(xyz)
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    if hasattr(csf, "do_filtering"):          # cloth-simulation-filter 1.1.x
        csf.do_filtering(ground, non_ground, False)
        gi = np.asarray(list(ground), dtype=np.int64)
    else:                                      # 旧版 CSF 绑定
        csf.do()
        csf.getClassification(ground, non_ground)
        gi = np.asarray(list(ground), dtype=np.int64)
    print(f"    CSF 完成：地面 {gi.size:,} / {xyz.shape[0]:,}（{time.time()-t0:.0f}s）",
          flush=True)
    c = np.floor((xyz[:, 0] - x0) / gsd).astype(np.int64)
    r = np.floor((y1 - xyz[:, 1]) / gsd).astype(np.int64)
    ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    is_ground = np.zeros(xyz.shape[0], bool)
    is_ground[gi] = True
    agl = np.full(xyz.shape[0], np.nan)
    agl[ok] = xyz[ok, 2] - dem[r[ok], c[ok]]
    return dict(r=r, c=c, ok=ok, ground=is_ground, agl=agl,
                n_used=int(xyz.shape[0]), n_full=int(n_full), stride=int(stride))


def csf_metrics(site, res, gt):
    ok = res["ok"]
    r, c = res["r"][ok], res["c"][ok]
    ground = res["ground"][ok]
    agl = res["agl"][ok]
    tr = gt[r, c]
    road = tr & (agl <= 0.5)
    veg = agl >= 1.0
    keep_n = int(ground.sum())
    road_n = int((ground & road).sum())
    veg_n = int((ground & veg).sum())
    return {
        "n_points_used": int(ok.sum()),
        "n_road_points": int(road.sum()),
        "n_veg_points": int(veg.sum()),
        "n_keep": keep_n,
        "keep_ratio": round(keep_n / max(int(ok.sum()), 1), 4),
        "road_preservation": round(road_n / max(int(road.sum()), 1), 4),
        "veg_removal": round(1.0 - veg_n / max(int(veg.sum()), 1), 4),
        "road_fp": int((ground & ~tr).sum()),
        "road_fp_rate": round(int((ground & ~tr).sum()) / max(keep_n, 1), 4),
    }


# ------------------------------------------------------------------ 主流程
def main():
    os.makedirs(RES, exist_ok=True)
    use_csf = False
    try:
        import CSF  # noqa: F401
        use_csf = True
    except Exception:  # noqa: BLE001
        use_csf = False
    method = "csf" if use_csf else "progressive_tin_proxy"
    print(f"方法：{method}" + ("" if use_csf else "（CSF 不可用，退回简化渐进 TIN 代理）"))

    results = {}
    csf_status = ("CSF 1.1.7 可用（cloth-simulation-filter；PyPI 的 `CSF` 是同名金融包，"
                  "不是地面滤波）" if use_csf
                  else "CSF 不可用，改用 progressive_tin_proxy 代理")
    for site in SITES:
        print(f"\n[{site}] 读取网格/真值 ...", flush=True)
        x0, y1, wm, hm, gsd, d = P.dem_window(site)
        H, W = d["dem_mean"].shape
        valid = d["valid"] == 1
        dem = fill_nodata(d["dem_min"], valid)
        gt = truth_mask(site, valid, x0, y1, gsd, H, W)
        print(f"  网格 {H}×{W}，真值 {int(gt.sum()):,} px")

        if use_csf:
            try:
                res = try_csf(site, x0, y1, gsd, H, W, dem, CSF_STRIDE[site])
            except Exception as e:  # noqa: BLE001
                print(f"    CSF 运行失败：{e} → 本站改用渐进 TIN 代理", flush=True)
                res = None
            if res is not None:
                m = csf_metrics(site, res, gt)
                results[site] = dict(
                    site=site, method="csf",
                    full_scan=bool(res["stride"] == 1), stride=int(res["stride"]),
                    n_full=int(res["n_full"]),
                    params=dict(cloth_resolution=CLOTH_RES, class_threshold=THR,
                                rigidness=2, iterations=500, bSloopSmooth=False),
                    truth_source=("results/mask_1m_%s.npy" % site
                                  if CFG[site]["truth"] == "old" else
                                  "SHP %s → poly_mask_hr → 1 m 最大连通域"
                                  % SHP_NEW[site]["poly"]),
                    note=("CSF 布料模拟滤波（cloth-simulation-filter 1.1.7）；"
                          "因内存限制按步长 %d 抽稀，指标基于抽稀点。"
                          % res["stride"]),
                    metrics=m)
                print(f"  指标 {m}")
                continue
            use_csf = False
            method = "progressive_tin_proxy"

        print("  构建渐进 TIN ...", flush=True)
        tin, tinfo = build_tin_proxy(dem, valid)
        m = stream_metrics(site, tin, gt, x0, y1, gsd, dem, valid)
        results[site] = dict(
            site=site, method="progressive_tin_proxy", full_scan=True,
            params=dict(grid_block_coarse_m=TIN_COARSE, grid_block_fine_m=TIN_FINE,
                        coarse_candidate_thr_m=TIN_COARSE_THR,
                        ground_residual_thr_m=THR),
            tin=tinfo,
            truth_source=("results/mask_1m_%s.npy" % site
                          if CFG[site]["truth"] == "old" else
                          "SHP %s → poly_mask_hr → 1 m 最大连通域"
                          % SHP_NEW[site]["poly"]),
            note=("Axelsson 渐进 TIN 的简化代理：10 m 块最低点 TIN → 残差≤3 m 的候选像元 "
                  "→ 4 m 块最低点重建 TIN → 点残差≤0.3 m 判地面。"
                  "**不是 CSF**；全量流式扫描。"),
            metrics=m)
        print(f"  指标 {m}")

    suffix = ("csf_filter" if all(results[s]["method"] == "csf" for s in SITES)
              else "tin_proxy_filter")
    for s in SITES:
        results[s]["csf_status"] = csf_status
    out = os.path.join(RES, f"{suffix}.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n写出 {out}")

    print("\n### 地面滤波对照（road_preservation / veg_removal / keep_ratio）\n")
    print("| 站点 | 方法 | 蜀道保留率 | 植被去除率(全场) | 总保留率 | road_fp |")
    print("|---|---|---|---|---|---|")
    for site in SITES:
        m = results[site]["metrics"]
        print(f"| {site} | {results[site]['method']} | {m['road_preservation']:.4f} | "
              f"{m['veg_removal']:.4f} | {m['keep_ratio']:.4f} | {m['road_fp']:,} |")


if __name__ == "__main__":
    main()
