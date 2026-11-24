# -*- coding: utf-8 -*-
"""生成 v4 输入扩展用的地形因子数据（P2 L0 → 8 通道 → .npy）。

用法：
    python tools/p2_terrain_to_v4.py --site 西坝村
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P                              # noqa: E402
from p2_terrain import CHANNELS, terrain_stack  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["杨家坝", "西坝村"])
    a = ap.parse_args()

    x0, y1, wm, hm, gsd, d = P.dem_window(a.site)
    valid = d["valid"] == 1
    print(f"[{a.site}] 窗口 {wm:.0f}×{hm:.0f} m @ gsd={gsd} m，设备 {P.DEV}")

    model = P.load_model(a.site)
    packed = model.pack_dem(x0, y1, wm, hm, gsd, n_levels=1, device=P.DEV)
    l0 = P.to_utm(a.site, packed[0]).cpu().numpy().astype(np.float64)
    stack = terrain_stack(l0, gsd)
    print(f"  terrain_stack {tuple(stack.shape)}（通道 {CHANNELS}）")

    out = os.path.join(DATA, f"p2_terrain_{a.site}.npy")
    np.save(out, stack.astype(np.float32))
    print(f"  写出 {os.path.relpath(out, ROOT)}  shape={stack.shape} dtype=float32")

    print(f"  {'通道':<12}{'min':>10}{'max':>10}{'mean':>10}{'std':>10}")
    for c, arr in zip(CHANNELS, stack):
        v = arr[valid]
        print(f"  {c:<12}{v.min():>10.4f}{v.max():>10.4f}{v.mean():>10.4f}{v.std():>10.4f}")


if __name__ == "__main__":
    main()
