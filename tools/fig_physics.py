# -*- coding: utf-8 -*-
"""物理真实性图：高程剖面（λc=0 vs λc=6 vs 真值）+ 3D 地形叠加图。

    figures/physics_profile.png       两站剖面：沿程距离-绝对高程
    figures/graphical_abstract.png    3D 地形 + 红线(λc=0 抄近道) / 青线(λc=6) / 绿线(真值)

路径缓存：results/path_lc0_{site}.npz、results/pathreal_{site}.npz
（两者用同一对端点，差异只在 λc）

用法：
    python tools/fig_physics.py
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_closing
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import cold_start_transfer as C                                       # noqa: E402
import path_extraction as P                                           # noqa: E402
import p2_dem as PD                                                   # noqa: E402
from eval_freeaxis import fill_nodata                                 # noqa: E402
from fig_recognition import load_truth                                # noqa: E402
from fig_pred_vs_truth import order_path                              # noqa: E402
from p2_adaptive_gsd import FIG                                       # noqa: E402

RES = os.path.join(ROOT, "results")
SITES = ("杨家坝", "西坝村")


def get_paths(site, dem, valid, log=print):
    """返回 λc=0 / λc=6 两条路径（同一对端点），带缓存。"""
    f6 = os.path.join(RES, f"pathreal_{site}.npz")
    f0 = os.path.join(RES, f"path_lc0_{site}.npz")
    z = np.load(os.path.join(RES, f"cold_start_{site}_lc6.npz"))
    ep = P.merge_and_pick_pair(np.argwhere(P.find_endpoints(z["path"].astype(bool))))
    op, nr, nc, conf, _ = C.macro_fields(dem, valid)
    out = {}
    for tag, lam in (("lc6", 6.0), ("lc0", 0.0)):
        f = f6 if tag == "lc6" else f0
        if os.path.exists(f):
            out[tag] = np.load(f)["path"].astype(bool)
            continue
        a = SimpleNamespace(dir_lambda=0.0, turn_lambda=0.0, turn_min=2,
                            slope_lambda=0.0, z_lambda=0.0 if lam == 0 else 2.0,
                            lam_c=lam, max_grade=0.0, grade_lag=5)
        p, _, _, _ = C.macro_path(op, valid, ep, nr, nc, conf, a, log=log, dem=dem)
        np.savez_compressed(f, path=p.astype(np.uint8))
        out[tag] = p
    return out


def profile(mask, dem):
    pts = order_path(mask)
    if pts is None:
        return None
    pts = np.array(pts)
    elev = dem[pts[:, 0], pts[:, 1]]
    step = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    dist = np.concatenate([[0.0], np.cumsum(step)])
    return dict(d=dist, z=elev, pts=pts)


def climb_stats(pr, win=20.0):
    """最大 win 米窗口内的高差与坡度（%）。"""
    d, z = pr["d"], pr["z"]
    best, at = 0.0, (0.0, 0.0)
    for i in range(len(d)):
        j = np.searchsorted(d, d[i] + win)
        j = min(j, len(d) - 1)
        dz = z[j] - z[i]
        if abs(dz) > abs(best):
            best, at = float(dz), (float(d[i]), float(d[j]))
    return dict(max_dz=round(best, 2), at=[round(v, 1) for v in at],
                ascent=round(float(np.sum(np.maximum(np.diff(z), 0))), 1),
                length=round(float(d[-1]), 1))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    dat = {}
    for site in SITES:
        x0, y1, wm, hm, gsd, d = PD.dem_window(site)
        valid = d["valid"] == 1
        dem = fill_nodata(d["dem_mean"], valid)
        paths = get_paths(site, dem, valid)
        gt = load_truth(site, valid, x0, y1, gsd, dem.shape[0], dem.shape[1])
        gsk = skeletonize(binary_closing(gt, np.ones((3, 3), bool)))
        pr = {k: profile(v, dem) for k, v in paths.items()}
        pr["truth"] = profile(gsk, dem)
        st = {k: climb_stats(v) for k, v in pr.items() if v}
        dat[site] = dict(paths=paths, pr=pr, st=st, dem=dem, valid=valid,
                         x0=x0, y1=y1, gsd=gsd)
        print(f"[{site}] λc0 高差 {st['lc0']['max_dz']} m / 总爬升 {st['lc0']['ascent']} m；"
              f"λc6 {st['lc6']['max_dz']} m / {st['lc6']['ascent']} m；"
              f"真值 {st['truth']['max_dz']} m")

    # ---------------- 剖面图 ----------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for ax, site in zip(axes, SITES):
        R = dat[site]
        for k, col, lab, ls in (("lc0", "#e02424", "λc=0（无物理约束）", "-"),
                                ("lc6", "#00a6b8", "λc=6（本文）", "-"),
                                ("truth", "#0b9d58", "SHP 真值", "--")):
            p = R["pr"][k]
            if p is None:
                continue
            ax.plot(p["d"], p["z"], color=col, lw=2.2 if k != "truth" else 1.8,
                    ls=ls, label=lab)
        st = R["st"]
        s0, s6 = st["lc0"], st["lc6"]
        ax.set_title(f"{site}：沿程高程剖面（20 m 窗口最大高差："
                     f"λc=0 {s0['max_dz']:+.1f} m → λc=6 {s6['max_dz']:+.1f} m）",
                     fontsize=11)
        ax.set_xlabel("沿程距离 (m)", fontsize=9)
        ax.set_ylabel("绝对高程 (m)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.text(0.015, 0.03,
                f"总爬升：λc=0 {s0['ascent']:.0f} m / λc=6 {s6['ascent']:.0f} m / "
                f"真值 {st['truth']['ascent']:.0f} m\n"
                f"长度：{s0['length']:.0f} / {s6['length']:.0f} / "
                f"{st['truth']['length']:.0f} m",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(fc="white", ec="#999999", alpha=0.85, pad=4))
    fig.suptitle("物理约束的剖面证据：λc 阻止了横切陡坡的暴力爬升", fontsize=14)
    out = os.path.join(FIG, "physics_profile.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"写出 {out}")

    # ---------------- 3D 图形摘要 ----------------
    fig = plt.figure(figsize=(15, 6.4))
    for i, site in enumerate(SITES, 1):
        R = dat[site]
        dem, valid = R["dem"], R["valid"]
        ds = max(2, int(round(max(dem.shape) / 160)))
        zz = np.where(valid, dem, np.nan)[::ds, ::ds]
        H, W = zz.shape
        yy, xx = np.mgrid[0:H, 0:W]
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.plot_surface(xx, yy, zz, cmap="gist_earth", rstride=1, cstride=1,
                        linewidth=0, antialiased=True, alpha=0.35,
                        vmin=np.nanmin(zz), vmax=np.nanmax(zz))
        ax.contour(xx, yy, np.nan_to_num(zz, nan=np.nanmin(zz)),
                   levels=np.arange(np.floor(np.nanmin(zz) / 20) * 20,
                                    np.nanmax(zz) + 20, 20),
                   colors="k", linewidths=0.35, alpha=0.45)
        for k, col, lab, lw in (("lc0", "#e02424", "λc=0（抄近道）", 3.0),
                                ("lc6", "#00e5ff", "λc=6（沿等高线）", 3.0),
                                ("truth", "#00c853", "SHP 真值", 2.4)):
            p = R["pr"][k]
            if p is None:
                continue
            pts = p["pts"]
            ax.plot(pts[:, 1] / ds, pts[:, 0] / ds,
                    dem[pts[:, 0], pts[:, 1]] + 8.0,
                    color=col, lw=lw, label=lab, solid_capstyle="round")
        ax.set_title(f"{site}", fontsize=12, pad=0)
        ax.set_zlabel("高程 (m)", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=42, azim=-68)
        ax.set_box_aspect((W, H, 0.45 * max(W, H)))
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("图形摘要：λc=0（红，横切抄近道） vs λc=6（青，沿等高线绕行） vs 真值（绿）",
                 fontsize=13)
    out2 = os.path.join(FIG, "graphical_abstract.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"写出 {out2}")

    json.dump({s: dat[s]["st"] for s in SITES},
              open(os.path.join(RES, "physics_profile.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("写出 results/physics_profile.json")

    # ---------------- 2D 等高线版图形摘要（最直观） ----------------
    from fig_recognition import hillshade_img
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    for ax, site in zip(axes, SITES):
        R = dat[site]
        dem, valid, gsd = R["dem"], R["valid"], R["gsd"]
        x0, y1 = R["x0"], R["y1"]
        H, W = dem.shape
        ext = [x0, x0 + W * gsd, y1 - H * gsd, y1]
        ax.imshow(hillshade_img(dem, valid), cmap="gray", vmin=0.05, vmax=0.85,
                  extent=ext, interpolation="nearest")
        ax.contour(np.linspace(ext[0], ext[1], W), np.linspace(ext[3], ext[2], H),
                   np.where(valid, dem, np.nan), levels=np.arange(
                       np.floor(np.nanmin(dem[valid]) / 10) * 10,
                       np.nanmax(dem[valid]) + 10, 10),
                   colors="#555555", linewidths=0.35, alpha=0.55)
        for k, col, lab, lw in (("lc0", "#e02424", "λc=0（抄近道）", 2.0),
                                ("lc6", "#00c2d1", "λc=6（沿等高线）", 2.0),
                                ("truth", "#0b9d58", "SHP 真值", 1.7)):
            p = R["pr"][k]
            if p is None:
                continue
            pts = p["pts"]
            ax.plot(x0 + (pts[:, 1] + 0.5) * gsd, y1 - (pts[:, 0] + 0.5) * gsd,
                    color=col, lw=lw, label=lab)
        ax.set_aspect("equal")
        ax.set_title(f"{site}（等高距 10 m）", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=7)
    fig.suptitle("图形摘要（2D）：红线横切等高线，青线沿等高线绕行，绿线为真值",
                 fontsize=13)
    out3 = os.path.join(FIG, "graphical_abstract_2d.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"写出 {out3}")


if __name__ == "__main__":
    main()
