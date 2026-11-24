# -*- coding: utf-8 -*-
"""逐像元距离场自适应分辨率（GSD）分配图。

规则（逐像元，按优先级覆盖；档位 = LEVELS × `scale`，`--half` 即 scale=0.5）：

    P >= p_hi (0.5)             → 0.1·s m   极高置信度，核心石板路面
    p_lo <= P < p_hi (0.1~0.5)  → 0.5·s m   过渡带（矮植被 / 路边缘）
    dist <= 5.0·s                → 1.0·s m   紧贴带
    否则                         → 4.0·s m   盲区 / 背景

    `--half` 下即 0.05 / 0.25 / 0.5 / 2.0 m。
    没有概率图时退化为纯距离分段（0.1·s / 0.5·s / 1.0·s）。
    dist = 到走廊掩膜的最近欧氏距离（1 m 网格，distance_transform_edt）。

    走廊掩膜来源：缺省 = SHP 标注（`_1m.npz` 的 label）；
    给 `--cloud` 时 = **交付点云的 1 m 占用格**（即模型自己找出来的走廊，
    新站点无 SHP 时用这条路径）。

物理门控（无地面点覆盖 = p2_{site}_points_ground.npz 投影 + 3×3 膨胀）：
    核心带内（dist <= 1.0·s） → 降级到 1.0·s m 做平滑插值，并写入 interp 标记
    核心带外                  → 4.0·s m（背景盲区不值细采样）
    interp=1 表示该像元的 GSD 决策**没有地面回波支撑**，下游可据此降权。

输出：
    figures/p2_adaptive_gsd_pixel_{site}.png   左=GSD 分配，右=距离场
    results/p2_adaptive_gsd_pixel_{site}.npz   gsd / dist / covered / interp（1 m 网格）

存储含义（重要）：
    这是**采样规划图**，不是一张变分辨率 DEM。逐像元变分辨率无法用单个
    1 m 栅格承载（0.1 m 档在 1 m 栅格里没有位置），真正落地要按档输出分块
    GeoTIFF，或按此图从源 DEM 分块重采样。
    S 用标注真值 → 不能当作"自动定位蜀道"的证据。

用法：
    python tools/p2_adaptive_gsd.py --site all
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from p2_export_all import project_covered  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

# 基准档位（scale=1）；scale≠1 时逐档乘以 scale（--half → 0.05/0.25/0.5/2.0）
LEVELS = [0.1, 0.5, 1.0, 4.0]
LEVEL_COLORS = ["#d7191c", "#fdae61", "#ffffbf", "#2c7bb6"]


def local_std(dem: np.ndarray, valid: np.ndarray, size: int = 3) -> np.ndarray:
    """3×3 局部高程标准差，**只在 valid 上统计**（无效区记 0）。

    直接把含 -9999 的 dem_mean 送进 uniform_filter 会让方差被 1e8 量级污染，
    这里先用掩膜把无效像元置 0，再按有效像元数归一化。
    """
    from scipy.ndimage import uniform_filter
    v = valid.astype(np.float64)
    z = np.where(valid, dem, 0.0)
    n = uniform_filter(v, size, mode="constant", cval=0.0)
    m1 = uniform_filter(z, size, mode="constant", cval=0.0)
    m2 = uniform_filter(z ** 2, size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = m2 / n - (m1 / n) ** 2
    var = np.clip(var, 0.0, None)
    return np.where(n > 0, np.sqrt(var), 0.0)


def read_cloud_xy(path: str, chunk: int = 4_000_000):
    """读 LAS/LAZ 的 X/Y（已应用 scale/offset，UTM 米）。分块控内存。"""
    import laspy
    X, Y = [], []
    with laspy.open(path) as f:
        for pts in f.chunk_iterator(chunk):
            X.append(np.asarray(pts.x, dtype=np.float64))
            Y.append(np.asarray(pts.y, dtype=np.float64))
    return np.concatenate(X), np.concatenate(Y)


def cloud_layers(path, x0, y1, gsd, H, W, dilate=1):
    """交付点云 → (走廊掩膜, 覆盖掩膜, 有效点数)。

    走廊 = 有点的 1 m 像元 + 3×3 闭运算（补单格空洞，避免碎点把核心带切碎）；
    覆盖 = 再向外膨胀 `dilate` 像元（判断"此处有无实际支撑数据"）。
    """
    X, Y = read_cloud_xy(path)
    col = np.floor((X - x0) / gsd).astype(np.int64)
    row = np.floor((y1 - Y) / gsd).astype(np.int64)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    occ = np.zeros((H, W), dtype=bool)
    occ[row[ok], col[ok]] = True
    corridor = binary_closing(occ, np.ones((3, 3), dtype=bool))
    covered = binary_dilation(occ, iterations=dilate) if dilate else occ.copy()
    print(f"  交付点云 {X.size:,} 点（域内 {int(ok.sum()):,}）→ 走廊 "
          f"{int(corridor.sum()):,}px（{corridor.sum() / corridor.size:.2%}），"
          f"覆盖 {int(covered.sum()):,}px")
    return corridor, covered, int(ok.sum())


def load_prob(path: str, H: int, W: int):
    """读模型概率图 → (prob[H,W] 域外置 0, valid 掩膜)。"""
    z = np.load(path)
    p = z["prob"].astype(np.float64)
    if p.shape != (H, W):
        raise ValueError(f"概率图尺寸 {p.shape} 与 DEM 网格 {(H, W)} 不一致：{path}")
    v = (z["valid"] == 1) if "valid" in z.files else (p > 0)
    print(f"  概率图 {os.path.basename(path)}：P≥0.5 {int((p[v] >= 0.5).sum()):,}px，"
          f"0.1≤P<0.5 {int(((p[v] >= 0.1) & (p[v] < 0.5)).sum()):,}px")
    return np.where(v, p, 0.0), v


def build_map(site: str, cloud: str | None = None, scale: float = 1.0,
              cover_from_cloud: bool = False, prob_path: str | None = None,
              p_hi: float = 0.5, p_lo: float = 0.1,
              dist_scale: float = 1.0) -> dict:
    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape

    prob = None
    if prob_path:
        prob, _ = load_prob(prob_path, H, W)

    # 走廊来源只允许"交付点云"或"模型概率图"，**禁止回退到 SHP 标注**
    # （mask_1m / npz.label 均由 SHP 栅格化，用作流程输入即构成评测泄漏）。
    if cloud:
        label, cov_cloud, npt = cloud_layers(cloud, x0, y1, gsd, H, W, dilate=1)
        covered = cov_cloud if cover_from_cloud else project_covered(
            site, x0, y1, gsd, H, W, dilate=1)
    elif prob is not None:
        label = (prob >= p_hi)
        covered = project_covered(site, x0, y1, gsd, H, W, dilate=1)
        npt = None
    else:
        raise ValueError(
            f"[{site}] 必须提供 --cloud 或 --prob：禁止默认使用 SHP 标注走廊")
    dist = distance_transform_edt(~label)

    dem = d["dem_mean"].astype(np.float64)
    valid = d["valid"] == 1
    R = local_std(dem, valid)          # 3×3 局部高程标准差

    s = float(scale)
    d_core, d_near = 1.0 * dist_scale, 5.0 * dist_scale
    core = (prob >= p_hi) & valid if prob is not None else (dist <= d_core)
    band = (prob >= p_lo) & ~core & valid if prob is not None else None

    # 分配优先级（与设计口径一致）：P≥0.5 → 核心档；0.1≤P<0.5 → 缓冲档；
    # 距离≤5 m → 紧贴档；其余 → 盲区/背景档。
    # 档位 = LEVELS × scale（--half 时即 0.05 / 0.25 / 0.5 / 2.0）。
    gsd_map = np.full((H, W), LEVELS[3] * s, dtype=np.float32)   # 盲区/背景
    gsd_map[dist <= d_near] = LEVELS[2] * s                       # 紧贴带
    if prob is not None:
        gsd_map[band] = LEVELS[1] * s                             # 缓冲带 0.1≤P<0.5
        gsd_map[core] = LEVELS[0] * s                             # 核心 P≥0.5
    else:
        gsd_map[(dist > d_core) & (dist <= d_near)] = LEVELS[1] * s
        gsd_map[dist <= d_core] = LEVELS[0] * s
    # R 门控只在背景区生效：dist>5·s 且 R>0.4·s → 强制 ≥ 背景档。
    # 背景基档已经是 LEVELS[2]，max(LEVELS[2], LEVELS[2]) 为恒等操作
    # （保留代码以表达策略，不产生实际效果）。
    bg = (dist > d_near) & (R > 0.4 * s)
    gsd_map[bg] = np.maximum(gsd_map[bg], LEVELS[2] * s)

    # 物理门控：无地面点。核心带降到紧贴档（平滑插值）并标记，其他直接盲区档。
    blind = ~covered
    gsd_map[blind] = LEVELS[3] * s
    gsd_map[blind & core] = LEVELS[2] * s

    return {"site": site, "label": label, "dist": dist, "covered": covered,
            "interp": blind, "gsd": gsd_map, "H": H, "W": W, "x0": x0, "y1": y1,
            "gsd0": gsd, "scale": s, "levels": [v * s for v in LEVELS],
            "cloud": cloud, "cloud_px": npt,
            "prob": prob, "core": core, "band": band,
            "p_hi": p_hi, "p_lo": p_lo,
            "d_core": d_core, "d_near": d_near}


def stats(m: dict) -> dict:
    gsd_map, dist, covered = m["gsd"], m["dist"], m["covered"]
    H, W = gsd_map.shape
    levels = m["levels"]
    out = {"site": m["site"], "grid": [H, W], "scale": m["scale"],
           "levels_m": [round(v, 4) for v in levels], "levels": {}}
    total = 0.0
    for gv in levels:
        mask = np.isclose(gsd_map, gv, rtol=0, atol=max(gv, 1e-6) * 1e-3)
        area = float(mask.sum()) * m["gsd0"] ** 2       # m²
        samples = area / (gv * gv) if area else 0.0
        total += samples
        out["levels"][f"{gv:.4g}"] = {"px": int(mask.sum()),
                                     "area_m2": round(area, 1),
                                     "samples": round(samples, 1)}
    out["samples_total"] = round(total, 1)
    out["samples_uniform_1m"] = round(H * W * m["gsd0"] ** 2 / 1.0 ** 2, 1)
    out["samples_uniform_finest"] = round(
        H * W * m["gsd0"] ** 2 / levels[0] ** 2, 1)
    out["samples_uniform_0p1m"] = out["samples_uniform_finest"]   # 旧键名兼容
    out["core_band_px"] = int((dist <= m["d_core"]).sum())
    out["core_band_interp_px"] = int(((dist <= m["d_core"]) & ~covered).sum())
    out["interp_px"] = int((~covered).sum())
    out["corridor_px"] = int((dist <= 0).sum())
    out["p_core_px"] = int(m["core"].sum()) if m["prob"] is not None else None
    out["p_band_px"] = int(m["band"].sum()) if m["band"] is not None else None
    out["alloc"] = ("P≥%.2f→核心 / %.2f≤P<%.2f→缓冲 / dist≤%g→紧贴 / 其余→背景"
                    % (m["p_hi"], m["p_lo"], m["p_hi"], m["d_near"])
                    if m["prob"] is not None else "按距离分段")
    return out


def figure(m: dict, s: dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from skimage.measure import find_contours

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    levels = m["levels"]
    sc = m["scale"]
    cmap = ListedColormap(LEVEL_COLORS)
    norm = BoundaryNorm([0, 0.3 * sc, 0.75 * sc, 2.5 * sc, 8 * sc], cmap.N)
    im0 = axes[0].imshow(m["gsd"], cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title(f"{m['site']}：逐像元 GSD 分配（阈值 ×{sc:g}）")
    axes[0].axis("off")
    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046, ticks=levels)
    cb.ax.set_yticklabels([f"{g:g} m" for g in levels])

    if m["prob"] is not None:
        im1 = axes[1].imshow(np.where(m["covered"], m["prob"], np.nan),
                             cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("走廊概率 P（灰=无地面点）")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, label="P")
    else:
        im1 = axes[1].imshow(np.where(m["covered"], m["dist"], np.nan),
                             cmap="magma", vmax=60)
        axes[1].set_title("到蜀道距离（灰=无地面点）")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, label="距离 (m)")
    axes[1].axis("off")

    # 白色轮廓：给了概率图就画 P≥0.5 核心，否则画走廊掩膜
    contour = m["core"] if m["prob"] is not None else m["label"]
    for c in find_contours(contour.astype(float), 0.5):
        for ax in axes:
            ax.plot(c[:, 1], c[:, 0], "-", color="w", lw=0.7, alpha=0.65)

    ys, xs = np.where(m["interp"] & (m["dist"] <= m["d_core"]))
    if ys.size:
        axes[0].plot(xs, ys, ".", ms=1.0, color="k")
        axes[0].text(0.01, 0.01,
                     f"黑点 = 核心带无地面点（降 {1.0 * m['scale']:g} m 插值）",
                     transform=axes[0].transAxes, fontsize=9, color="k")

    lv = []
    for g in m["levels"][:3]:
        v = s["levels"][f"{g:.4g}"]
        lv.append(f"{g:g} m {v['area_m2']:.0f} m²")
    fig.suptitle("   ".join(lv), y=0.02, fontsize=11, color="0.35")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="西坝村", choices=["西坝村", "杨家坝", "all"])
    ap.add_argument("--cloud", default=None,
                    help="用交付点云（LAS/LAZ）的 1 m 占用格定义走廊，替代 SHP 标注")
    ap.add_argument("--half", action="store_true",
                    help="所有档位阈值统一除以 2：0.1→0.05、0.5→0.25、1→0.5、4→2；"
                         "距离带另由 --dist-scale 控制")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="显式指定阈值倍数（--half 等价 --scale 0.5）")
    ap.add_argument("--cover-from-cloud", action="store_true",
                    help="用交付点云自身作为覆盖依据（缺省仍用全量地面点投影）")
    ap.add_argument("--prob", default=None,
                   help="模型概率图 npz（键 prob）；缺省自动找 cold_start_{site}_ep.npz"
                        " → model_guided_prob_{site}.npz，找不到就退回按距离分段")
    ap.add_argument("--p-hi", type=float, default=0.5, help="核心档概率阈值")
    ap.add_argument("--p-lo", type=float, default=0.1, help="缓冲档概率阈值")
    ap.add_argument("--dist-scale", type=float, default=1.0,
                   help="距离带倍数（默认 1，即核心 1 m / 紧贴 5 m）")
    ap.add_argument("--tag", default="", help="输出文件名后缀，避免覆盖旧结果")
    args = ap.parse_args()

    scale = 0.5 if args.half else float(args.scale)
    sites = ["西坝村", "杨家坝"] if args.site == "all" else [args.site]
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    all_stats = []

    for site in sites:
        prob_path = args.prob
        if prob_path is None:
            for cand in (f"cold_start_{site}_ep.npz", f"cold_start_{site}.npz",
                         f"model_guided_prob_{site}.npz"):
                p = os.path.join(RES, cand)
                if os.path.exists(p):
                    prob_path = p
                    break
        tag = args.tag or ((f"_s{scale:g}" if scale != 1.0 else "")
                          + ("_cloud" if args.cloud else "")
                          + ("_prob" if prob_path else ""))
        m = build_map(site, cloud=args.cloud, scale=scale,
                      cover_from_cloud=args.cover_from_cloud,
                      prob_path=prob_path, p_hi=args.p_hi, p_lo=args.p_lo,
                      dist_scale=args.dist_scale)
        s = stats(m)
        s["tag"] = tag
        s["cloud"] = os.path.basename(args.cloud) if args.cloud else None
        s["prob_file"] = os.path.basename(prob_path) if prob_path else None
        all_stats.append(s)

        print(f"[{site}] 网格 {s['grid'][0]}×{s['grid'][1]}，阈值倍数 {scale:g}"
              + (f"，走廊来源 {os.path.basename(args.cloud)}" if args.cloud else "，走廊来源 SHP")
              + (f"，概率图 {os.path.basename(prob_path)}" if prob_path else "，按距离分段"))
        print(f"  分配：{s['alloc']}"
              + (f"；P≥{args.p_hi:g} 核心 {s['p_core_px']:,}px，"
                 f"{args.p_lo:g}≤P<{args.p_hi:g} 缓冲 {s['p_band_px']:,}px"
                 if prob_path else ""))
        for gv in m["levels"]:
            v = s["levels"][f"{gv:.4g}"]
            print(f"  {gv:>6.3f} m: {v['px']:>8d} 像元  {v['area_m2']:>10.0f} m²  "
                  f"需 {v['samples']:>12.0f} 个查询样本")
        print(f"  总样本 {s['samples_total']:.0f}；均匀 1 m = {s['samples_uniform_1m']:.0f}"
              f"（{100 * s['samples_total'] / max(s['samples_uniform_1m'], 1):.1f}%）；"
              f"均匀 {m['levels'][0]:g} m = {s['samples_uniform_finest']:.0f}")
        print(f"  核心带 {s['core_band_px']} 像元，其中无地面点降级 {s['core_band_interp_px']}"
              f"（{100 * s['core_band_interp_px'] / max(s['core_band_px'], 1):.2f}%）；"
              f"全图 interp 标记 {s['interp_px']} 像元")

        np.savez_compressed(
            os.path.join(RES, f"p2_adaptive_gsd_pixel_{site}{tag}.npz"),
            gsd=m["gsd"].astype(np.float32),
            dist=m["dist"].astype(np.float32),
            covered=m["covered"].astype(bool),
            interp=m["interp"].astype(bool),
            x0=m["x0"], y1=m["y1"], gsd0=m["gsd0"], scale=scale,
            levels=np.asarray(m["levels"]))
        png = os.path.join(FIG, f"p2_adaptive_gsd_pixel_{site}{tag}.png")
        figure(m, s, png)
        print(f"  写出 {os.path.relpath(png, ROOT)}")

    p = os.path.join(RES, f"p2_adaptive_gsd_pixel{tag}.json")
    json.dump(all_stats, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"写出 {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
