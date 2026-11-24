# -*- coding: utf-8 -*-
"""诊断：指标长期偏低是否由 SHP 真值本身引起。

四组检查（全部只读）：
  A. 几何自洽   —— 中轴线 vs 本体范围：线是否落在面内、面宽是否与线一致
  B. 数据覆盖   —— 真值有多少落在 DEM 无效区 / 点云空洞（那里概率恒为 0，召回必被锁死）
  C. 真值漂移   —— 旧面 vs 新面：IoU、最优整体平移量（整数格扫描）
  D. 物理证据   —— 走廊内 vs 外围的点云离地高度、点数密度是否有可分性
  E. 高程一致   —— PolyLineZ 的 Z 与 DEM 是否吻合（SHP 是否画在别的底图上）

用法：
    python tools/diag_shp_truth.py --site 杨家坝
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import (LINE_R, RES, ROOT, SHP_BASE, SHP_NEW, bilinear,  # noqa: E402
                          line_dist_map, parts_of, poly_mask, read_shp_any)

REAL = os.path.join(
    os.environ.get("SHUDAO_DATA_ROOT", os.path.join(ROOT, "data")), "data_real"
)
LOG = os.path.join(RES, "diag_shp_truth.log")
LINE_OVERRIDE = {"西坝村": r"西坝村\XBC_zhanda_line.shp"}


def line_file_of(site):
    return SHP_NEW.get(site, {}).get("line") or LINE_OVERRIDE.get(site)


def dist_of(site, shapes, x0, y1, gsd, H, W):
    p = os.path.join(RES, f"dist_centerline_{site}.npy")
    if os.path.exists(p):
        d = np.load(p)
        if d.shape == (H, W):
            return d
    d = line_dist_map(shapes, x0, y1, gsd, H, W)
    np.save(p, d)
    return d


def log(s):
    print(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def read_line_z(path):
    """读 PolyLineZ 的 Z 数组（与 XY 顶点一一对应）。"""
    raw = open(path, "rb").read()
    shapes, pos = [], 100
    while pos + 8 <= len(raw):
        _n, clen = struct.unpack_from(">ii", raw, pos)
        pos += 8
        end = pos + clen * 2
        st = struct.unpack_from("<i", raw, pos)[0]
        if st in (13, 15):
            nparts, npoints = struct.unpack_from("<ii", raw, pos + 36)
            xy = np.frombuffer(raw, dtype="<f8", count=npoints * 2,
                               offset=pos + 44 + 4 * nparts).reshape(-1, 2)
            zoff = pos + 44 + 4 * nparts + 16 * npoints + 16
            z = np.frombuffer(raw, dtype="<f8", count=npoints, offset=zoff)
            shapes.append({"parts": struct.unpack_from("<%di" % nparts, raw, pos + 44),
                           "pts": xy, "z": z})
        pos = end
    return shapes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="杨家坝")
    a = ap.parse_args()
    site = a.site
    log(f"\n{'='*74}\n[diag_shp_truth] {site}")

    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem_min = d["dem_min"].astype(np.float64)
    old = (d["label"] == 1) if "label" in d.files else np.zeros((H, W), bool)
    cc = np.arange(W)[None, :]
    rr = np.arange(H)[:, None]
    xs = x0 + (cc + 0.5) * gsd
    ys = y1 - (rr + 0.5) * gsd

    # ---------------- A. 几何自洽
    lf = line_file_of(site)
    ftype, line_shapes = read_shp_any(os.path.join(SHP_BASE, lf))
    dist = dist_of(site, line_shapes, x0, y1, gsd, H, W)
    ftype2, poly_shapes = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
    new_poly = poly_mask(poly_shapes, x0, y1, gsd, H, W)
    seglen = [float(np.sum(np.hypot(*np.diff(xy, axis=0).T))) for sh in line_shapes for xy in parts_of(sh)]
    log("\nA. 几何自洽")
    log(f"  中轴线 {len(line_shapes)} 段，各段长度 {[round(v,1) for v in seglen]} m，合计 {sum(seglen):.1f} m")
    log(f"  本体范围 {len(poly_shapes)} 面，栅格 {int(new_poly.sum()):,} px；"
        f"等效平均宽度 {sum(seglen) and new_poly.sum()*gsd*gsd/sum(seglen):.2f} m")
    # 线是否落在面内：按 0.5 m 采样
    samples = []
    for sh in line_shapes:
        for xy in parts_of(sh):
            for k in range(len(xy) - 1):
                p0, p1 = xy[k], xy[k + 1]
                n = max(int(np.hypot(*(p1 - p0)) / 0.5), 1)
                t = np.linspace(0, 1, n, endpoint=False)[:, None]
                samples.append(p0 + t * (p1 - p0))
    smp = np.vstack(samples)
    sc = np.clip(((smp[:, 0] - x0) / gsd).astype(int), 0, W - 1)
    sr = np.clip(((y1 - smp[:, 1]) / gsd).astype(int), 0, H - 1)
    in_poly = float(new_poly[sr, sc].mean())
    in_valid = float(valid[sr, sc].mean())
    log(f"  中轴线采样 {smp.shape[0]:,} 点：落在本体范围内 {in_poly:.1%}；落在 DEM 有效区 {in_valid:.1%}")
    # 面到线的距离分布
    pd = dist[new_poly]
    log(f"  本体范围像元到中轴线距离：中位 {np.median(pd):.2f} m，P90 {np.percentile(pd,90):.2f} m，"
        f"最大 {pd.max():.2f} m")

    # ---------------- B. 数据覆盖（决定性）
    log("\nB. 数据覆盖（真值落在无效区 = 召回天花板）")
    for tag, mask in (("旧面", old), ("新面", new_poly)):
        log(f"  {tag}：{int(mask.sum()):,} px，其中无效区 {int((mask & ~valid).sum()):,} px "
            f"（{(mask & ~valid).sum()/max(mask.sum(),1):.1%}）")
    for r in (1.0, 2.5, 5.0):
        m = dist <= r
        log(f"  中轴线 ≤{r:g} m：{int(m.sum()):,} px，其中无效区 {int((m & ~valid).sum()):,} px "
            f"（{(m & ~valid).sum()/max(m.sum(),1):.1%}）")

    # ---------------- C. 新旧真值一致性 / 漂移
    log("\nC. 新旧真值一致性")
    inter = int((old & new_poly).sum())
    log(f"  旧面 {int(old.sum()):,} px vs 新面 {int(new_poly.sum()):,} px："
        f"交集 {inter:,}，IoU {inter/max(int(old.sum()+new_poly.sum())-inter,1):.4f}")
    log(f"  新面中落在旧面内 {inter/max(int(new_poly.sum()),1):.1%}；旧面中落在新面内 "
        f"{inter/max(int(old.sum()),1):.1%}")
    best = (-1, 0, 0)
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            sh = np.zeros_like(new_poly)
            ys0, ys1 = max(0, dy), H + min(0, dy)
            xs0, xs1 = max(0, dx), W + min(0, dx)
            sh[ys0:ys1, xs0:xs1] = new_poly[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
            u = int((sh | old).sum())
            if u:
                iou = int((sh & old).sum()) / u
                if iou > best[0]:
                    best = (iou, dx, dy)
    log(f"  新面整体平移最优：dx={best[1]} m, dy={best[2]} m → IoU {best[0]:.4f}"
        f"（零平移 {inter/max(int(old.sum()+new_poly.sum())-inter,1):.4f}）")

    # ---------------- E. 高程一致（SHP Z vs DEM）
    log("\nE. 高程一致性（PolyLineZ 的 Z vs DEM）")
    try:
        zs = read_line_z(os.path.join(SHP_BASE, lf))
        dz = []
        for sh in zs:
            c = ((sh["pts"][:, 0] - x0) / gsd).astype(int)
            r = ((y1 - sh["pts"][:, 1]) / gsd).astype(int)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H) & valid[np.clip(r, 0, H-1), np.clip(c, 0, W-1)]
            if ok.sum():
                dz.append(sh["z"][ok] - dem_min[r[ok], c[ok]])
        dz = np.concatenate(dz)
        log(f"  与 dem_min 的差：中位 {np.median(dz):+.2f} m，P10/P90 "
            f"{np.percentile(dz,10):+.2f} / {np.percentile(dz,90):+.2f} m，"
            f"|Δ|≤1 m 占 {(np.abs(dz)<=1).mean():.1%}")
    except Exception as e:  # noqa: BLE001
        log(f"  （Z 解析失败：{e}）")

    # ---------------- D. 物理证据（点云）
    log("\nD. 物理证据（点云离地高度 / 地面回波密度）")
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    n = xyz.shape[0]
    lamps = {"corridor2.5": [], "ring5_15": [], "outside15": []}
    cnt = np.zeros(4, dtype=np.int64)          # 走廊/环/外/无效
    lowcnt = np.zeros(H * W, dtype=np.int64)   # 每格 AGL≤0.5 m 的点数
    allcnt = np.zeros(H * W, dtype=np.int64)
    for s in range(0, n, 2_000_000):
        e = min(s + 2_000_000, n)
        p = np.asarray(xyz[s:e], dtype=np.float64) + o[None, :]
        c = np.clip(((p[:, 0] - x0) / gsd).astype(int), 0, W - 1)
        r = np.clip(((y1 - p[:, 1]) / gsd).astype(int), 0, H - 1)
        ok = valid[r, c]
        dd = dist[r, c]
        agl = p[:, 2] - dem_min[r, c]
        m1 = ok & (dd <= 2.5)
        m2 = ok & (dd > 5) & (dd <= 15)
        m3 = ok & (dd > 15)
        cnt += [int(m1.sum()), int(m2.sum()), int(m3.sum()), int((~ok).sum())]
        for k, m in (("corridor2.5", m1), ("ring5_15", m2), ("outside15", m3)):
            lamps[k].append(agl[m][::7])          # 1/7 抽样，控制内存
        idx = r * W + c
        lowcnt += np.bincount(idx[ok & (agl <= 0.5)], minlength=H * W)
        allcnt += np.bincount(idx[ok], minlength=H * W)
    out = {}
    for k, v in lamps.items():
        v = np.concatenate(v)
        q = np.percentile(v, [25, 50, 75, 90])
        out[k] = {"n_sample": int(v.size), "p25": round(float(q[0]), 2),
                  "p50": round(float(q[1]), 2), "p75": round(float(q[2]), 2),
                  "p90": round(float(q[3]), 2),
                  "frac_le0p3": round(float((v <= 0.3).mean()), 4),
                  "frac_le0p5": round(float((v <= 0.5).mean()), 4),
                  "frac_le1m": round(float((v <= 1.0).mean()), 4)}
        log(f"  {k:<12} n={v.size:>9,}  AGL 中位 {q[1]:5.2f} m  "
            f"≤0.3 m {out[k]['frac_le0p3']:.1%}  ≤0.5 m {out[k]['frac_le0p5']:.1%}  "
            f"≤1 m {out[k]['frac_le1m']:.1%}")
    log(f"  点数分布：走廊 {cnt[0]:,}（{cnt[0]/n:.1%}） / 5–15 m 环 {cnt[1]:,}（{cnt[1]/n:.1%}）"
        f" / >15 m {cnt[2]:,}（{cnt[2]/n:.1%}） / 无效区 {cnt[3]:,}（{cnt[3]/n:.1%}）")

    # 密度对比（每像元点数）
    npx_c = int((dist <= 2.5).sum())
    npx_v = int(valid.sum())
    log(f"  密度：走廊 {cnt[0]/max(npx_c,1):.1f} 点/格（{npx_c:,} 格）；全有效区 "
        f"{int((cnt[0]+cnt[1]+cnt[2]))/max(npx_v,1):.1f} 点/格")

    # 地面回波密度：真值是否对准"路面"
    from scipy.ndimage import gaussian_filter
    low = lowcnt.reshape(H, W).astype(np.float32)
    dens = gaussian_filter(low, 1.0)
    ring = (dist > 5) & (dist <= 15) & valid
    ref = float(dens[ring].mean())
    log(f"\n  地面回波（AGL≤0.5 m）密度：")
    for tag, mk in (("中轴≤2.5 m", (dist <= 2.5) & valid),
                    ("本体范围", new_poly & valid), ("旧面", old & valid),
                    ("5–15 m 环", ring)):
        log(f"    {tag:<10} {dens[mk].mean():7.1f} 点/格  相对环带 {dens[mk].mean()/max(ref,1e-6):5.2f}×")
    # 平移扫描：用**局地对比**（核心带 vs 自身 5–15 m 环）判断真值是否整体偏移。
    # 用全局均值会被"谷底地面回波更多"带偏，局地对比可消除这种梯度。
    def shifted(a, dx, dy):
        o = np.full_like(a, np.nan)
        ys0, ys1 = max(0, dy), H + min(0, dy)
        xs0, xs1 = max(0, dx), W + min(0, dx)
        o[ys0:ys1, xs0:xs1] = a[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        return o

    c00 = float(dens[(dist <= 2.5) & valid].mean() - dens[ring].mean())
    bestS = (-9e9, 0, 0)
    for dy in range(-15, 16):
        for dx in range(-15, 16):
            ds = shifted(dist, dx, dy)
            co = (ds <= 2.5) & valid
            rg = (ds > 5) & (ds <= 15) & valid
            if co.sum() < 500 or rg.sum() < 2000:
                continue
            v = float(dens[co].mean() - dens[rg].mean())
            if v > bestS[0]:
                bestS = (v, dx, dy)
    log(f"  局地对比（核心带 − 自身环带，地面回波密度）：零平移 {c00:+.1f} 点/格；"
        f"最优 dx={bestS[1]} m, dy={bestS[2]} m → {bestS[0]:+.1f} 点/格")

    # 地形平坦度（碥道应更平）
    gy, gx = np.gradient(np.where(valid, d["dem_mean"].astype(np.float64), np.nan))
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    std = d["dem_std"].astype(np.float64)
    log("\n  地形平坦度（湾道/碥道 应比周边更平）")
    for tag, mk in (("中轴≤2.5 m", (dist <= 2.5) & valid),
                    ("5–15 m 环", ring), (">15 m", (dist > 15) & valid)):
        sl = np.nanmedian(slope[mk])
        st = np.median(std[mk])
        log(f"    {tag:<10} 坡度中位 {sl:5.1f}°  1 m 内高程 std 中位 {st:5.2f} m")

    # ---------------- F. 纵向剖面测试（判断线路是否整体偏移）
    # 真实道路沿等高线走：纵向坡率应平缓；若整体横移，线会横切等高线，
    # 纵向坡率陡增。该指标不受"谷底地面回波更多"这类全局梯度干扰。
    def profile(dx, dy, dem):
        xx = smp[:, 0] + dx
        yy = smp[:, 1] + dy
        c = ((xx - x0) / gsd - 0.5)
        r = ((y1 - yy) / gsd - 0.5)
        ok = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
        zz = bilinear(dem, r[ok], c[ok])
        g = np.abs(np.diff(zz)) / 0.5                      # 采样步长 0.5 m
        return float(np.median(g)), float(np.percentile(g, 90)), float(zz[-1] - zz[0])

    log("\nF. 纵向剖面（道路应沿等高线，坡率平缓）")
    for tag, dem in (("dem_min", dem_min), ("dem_mean", d["dem_mean"].astype(np.float64))):
        m0, p0, dh0 = profile(0, 0, dem)
        bestF = (9e9, 0, 0, 0)
        for dy in range(-12, 13):
            for dx in range(-12, 13):
                m, p, dh = profile(dx, dy, dem)
                if m < bestF[0]:
                    bestF = (m, dx, dy, p)
        log(f"  {tag:<9} 零平移：中位纵向坡率 {m0*100:5.1f}%（P90 {p0*100:5.1f}%），"
            f"高差 {dh0:+6.1f} m")
        log(f"  {'':<9} 最优平移 dx={bestF[1]:+d} m, dy={bestF[2]:+d} m → "
            f"中位 {bestF[0]*100:.1f}%（P90 {bestF[3]*100:.1f}%）")

    json.dump({"site": site, "agl": out, "counts": cnt.tolist(),
               "line_len_m": round(sum(seglen), 1),
               "line_in_valid": round(in_valid, 4), "line_in_poly": round(in_poly, 4),
               "old_px": int(old.sum()), "new_px": int(new_poly.sum()),
               "shift_old_vs_new": {"dx": best[1], "dy": best[2], "iou": round(best[0], 4)},
               "shift_max_ground": {"dx": bestS[1], "dy": bestS[2],
                                     "gain": round(float(bestS[0]), 2)}},
              open(os.path.join(RES, f"diag_shp_truth_{site}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
