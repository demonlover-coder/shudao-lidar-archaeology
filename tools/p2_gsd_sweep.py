# -*- coding: utf-8 -*-
"""P2 分辨率自适应扫描：固定站点与窗口，遍历 gsd，验证形状按比例变化。

判据：gsd 减半 → H/W 翻倍；5 联图无条带/棋盘伪影。

用法：
    python tools/p2_gsd_sweep.py                 # 默认 西坝村
    python tools/p2_gsd_sweep.py --site 杨家坝
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
GSD_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="西坝村", choices=["西坝村", "杨家坝"])
    args = ap.parse_args()
    SITE = args.site

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    os.makedirs(FIG, exist_ok=True)
    x0, y1, wm, hm, gsd0, _ = P.dem_window(SITE)
    print(f"[{SITE}] 固定窗口 x0={x0:.1f} y1={y1:.1f} {wm:.0f}×{hm:.0f} m，设备 {P.DEV}")

    model = P.load_model(SITE)
    zs, shapes = [], []
    for g in GSD_LIST:
        packed = model.pack_dem(x0, y1, wm, hm, g, n_levels=1, device=P.DEV)
        z = P.to_utm(SITE, packed[0]).cpu().numpy().astype(np.float64)
        zs.append(z)
        shapes.append(z.shape)
        print(f"  gsd={g:<5} → packed {tuple(packed.shape)}  L0 {z.shape}")

    # 形状比例校验（1m 为基准）
    base = shapes[GSD_LIST.index(1.0)]
    for g, sh in zip(GSD_LIST, shapes):
        exp = (int(round(hm / g)), int(round(wm / g)))
        ok = sh == exp
        rel = (sh[0] / base[0], sh[1] / base[1])
        print(f"  校验 gsd={g:<5} 期望 {exp} 实测 {sh} {'OK' if ok else 'NG'} "
              f"相对 1m 倍率 {rel[0]:.2f}×{rel[1]:.2f}（期望 {1 / g:.2f}×）")

    fig, axes = plt.subplots(1, len(GSD_LIST), figsize=(5 * len(GSD_LIST), 6))
    for ax, g, z in zip(axes, GSD_LIST, zs):
        im = ax.imshow(z, cmap="terrain")
        ax.set_title(f"gsd={g} m\n{z.shape}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{SITE} P2 分辨率自适应（同一 UTM 窗口，gsd 遍历）")
    fig.tight_layout()
    # 默认站点沿用原名（保持历史引用有效），其它站点加站点后缀
    name = "p2_gsd_sweep.png" if SITE == "西坝村" else f"p2_gsd_sweep_{SITE}.png"
    out = os.path.join(FIG, name)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"写出 {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
