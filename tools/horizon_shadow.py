# -*- coding: utf-8 -*-
"""地平线角（horizon angle）因子的判别力与相位物理含义验证。

因子（`p2_terrain.horizon_harmonics`，8 罗盘方位 × 射线 30 m）：
    h1_deg/cos/sin  一阶：两壁**不对称**分量（哪一侧更高）
    h2_deg/cos/sin  二阶：**V 形夹持**分量（对称谷底的主谐波）
    h0_deg          8 向平均仰角

两项验证：
  1) **AUC**（走廊像元 vs 背景），逐因子；
  2) **相位对齐**：把 `hN_phase` 与走廊**法向**比较。法向取自走廊掩膜内部
     EDT 的梯度（∇dist 指向中轴，与走廊切线垂直），比较时取 mod 180°
     （V 谷两侧对称，最大仰角在 ±法向两端）。用背景区作对照。

输出：`results/horizon_shadow.json`、`figures/horizon_shadow_{site}.png`

用法：
    python tools/horizon_shadow.py                 # 两站
    python tools/horizon_shadow.py --site 杨家坝 --radius 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_tpi31_auc import fill_nodata            # noqa: E402
from p2_terrain import horizon_harmonics         # noqa: E402
from shudao_auc import roc_auc                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
FACTORS = ("h0_deg", "h1_deg", "h1_cos", "h1_sin", "h2_deg", "h2_cos", "h2_sin")


def ang_diff_180(a, b):
    """两方位角的最小夹角（mod 180°，度，∈[0,90]）。"""
    d = np.abs((a - b) % 180.0)
    return np.minimum(d, 180.0 - d)


def normal_field(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """走廊掩膜内部 EDT 的梯度方向（罗盘度）+ 距离（px）。"""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    edt = distance_transform_edt(label)
    gy, gx = np.gradient(gaussian_filter(edt, 1.5))
    # 罗盘方位：0=N（-row），顺时针 → atan2(dcol, -drow)
    ang = np.degrees(np.arctan2(gx, -gy)) % 360.0
    return ang, edt


def run(site: str, radius: float) -> dict:
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    v = d["valid"] == 1
    dem = fill_nodata(d["dem_mean"].astype(np.float64), v)

    h = horizon_harmonics(dem, gsd=1.0, max_dist_m=radius)
    ang, edt = normal_field(label)

    core = label & (edt >= 1.5)                       # 避开中轴（∇dist≈0）

    out = {"site": site, "radius_m": radius,
           "corridor_px": int(label.sum()), "core_px": int(core.sum()),
           "auc": {}, "phase": {}}
    for f in FACTORS:
        out["auc"][f] = round(roc_auc(h[f][v], label[v]), 4)

    # 零假设（相位与法向无关）：|Δ| 在 [0,90] 上近似均匀 → 中位 45°、≤30° 占 1/3
    for k in (1, 2):
        ph = h[f"h{k}_phase"]
        rc = ang_diff_180(ph[core], ang[core])
        out["phase"][f"h{k}"] = {
            "median_deg": round(float(np.median(rc)), 2),
            "frac_within_30": round(float((rc <= 30).mean()), 4),
            "null_median_deg": 45.0,
            "null_frac_within_30": round(1.0 / 3.0, 4),
        }
        print(f"[{site}] h{k} 相位 vs 走廊法向：中位 {np.median(rc):5.1f}°"
              f"（≤30° 占 {(rc <= 30).mean() * 100:4.1f}%）；"
              f"零假设应为 45.0° / 33.3%")

    print(f"[{site}] AUC：" + "　".join(f"{f}={out['auc'][f]:.4f}" for f in FACTORS))
    figure(site, h, label, core, ang, out)
    return out


def figure(site, h, label, core, ang, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for ax, key, t in ((axes[0], "h2_deg", "h2_deg（V 形夹持强度，度）"),
                       (axes[1], "h1_deg", "h1_deg（两壁不对称，度）")):
        im = ax.imshow(h[key], cmap="magma")
        ax.contour(label.astype(float), levels=[0.5], colors="#00e5ff", linewidths=0.7)
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    from scipy.ndimage import distance_transform_edt
    edt = distance_transform_edt(label)
    core = label & (edt >= 1.5)
    r1 = ang_diff_180(h["h1_phase"][core], ang[core])
    r2 = ang_diff_180(h["h2_phase"][core], ang[core])
    ax.hist(r2, bins=np.arange(0, 95, 5), density=True, alpha=0.45,
            color="#2c7bb6", label=f"h2 相位（中位 {np.median(r2):.0f}°）")
    ax.hist(r1, bins=np.arange(0, 95, 5), density=True, alpha=0.65,
            color="#d7191c", label=f"h1 相位（中位 {np.median(r1):.0f}°）")
    ax.axhline(1.0 / 90.0, color="0.35", lw=1.4, ls="--",
               label="零假设（相位无关，均匀）")
    ax.set_xlabel("|相位 − 走廊法向| （度，mod 180）")
    ax.set_ylabel("概率密度")
    ax.set_title(f"相位对齐：h1 中位 {np.median(r1):.1f}° / h2 {np.median(r2):.1f}°"
                 f"（零假设 45°）")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color="0.85", lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle(f"{site} 地平线角谐波（8 向 × {out['radius_m']:.0f} m）　"
                 f"AUC h0={out['auc']['h0_deg']:.4f} h1={out['auc']['h1_deg']:.4f} "
                 f"h2={out['auc']['h2_deg']:.4f}　"
                 f"h1_cos={out['auc']['h1_cos']:.4f}")
    fig.tight_layout()
    p = os.path.join(FIG, f"horizon_shadow_{site}.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  写出 {os.path.relpath(p, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="all", choices=["西坝村", "杨家坝", "all"])
    ap.add_argument("--radius", type=float, default=30.0, help="射线半径（米）")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    sites = ["西坝村", "杨家坝"] if args.site == "all" else [args.site]
    out = [run(s, args.radius) for s in sites]
    p = os.path.join(RES, "horizon_shadow.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"写出 {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
