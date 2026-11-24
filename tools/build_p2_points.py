# -*- coding: utf-8 -*-
"""P2（纯坐标 HashGrid+MLP 地形表征）任务 1：点云准备。

读 LAS → 提取 (x, y, z) → 按**站点自身包围盒**归一化到 [0,1]³ → 保存 npz。

为什么按站点自身 bbox 归一化：P2 的主判据是**同站自验**（杨训杨验 / 西训西验），
坐标只需要在站内自洽；跨站只作泛化对照（两站相距 35 km、高差 300 m，
纯坐标网络无法迁移，见 docs/p2_report.md）。

不做任何滤波：按任务书「提取 (x,y,z)」。分类直方图一并打印，供判断是否需要
只取地面点（--ground-only）。

用法：
    python tools/build_p2_points.py                       # 两站，全点
    python tools/build_p2_points.py --ground-only         # 只留分类==2（地面）
    python tools/build_p2_points.py --sites 杨家坝
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_data import BASE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# 站点 → LAS 文件（点云数为任务书给定的 21M / 70M 那两个文件）
SITE_LAS = {
    "杨家坝": r"杨家坝\yangjiaba_lastreturn.las",
    "西坝村": r"西坝村\西坝村栈道-lidar_回波法.las",
}

# LAS 1.2 Point Format 3 = 34 字节/记录
PF3 = np.dtype([
    ("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("intensity", "<u2"),
    ("flags", "u1"), ("cls", "u1"), ("scan_angle", "i1"), ("user", "u1"),
    ("psid", "<u2"), ("gps", "<f8"), ("red", "<u2"), ("green", "<u2"), ("blue", "<u2"),
])

CLS_NAME = {0: "从未分类", 1: "未分类", 2: "地面", 3: "低植被", 4: "中植被",
            5: "高植被", 6: "建筑", 7: "低点(噪声)", 9: "水体", 10: "铁路",
            11: "路面", 12: "重叠/保留", 13: "护网", 14: "导线", 15: "塔杆",
            16: "连接线", 17: "桥面", 18: "高噪声"}


def las_header(path: str) -> dict:
    with open(path, "rb") as f:
        head = f.read(375)
    if head[:4] != b"LASF":
        raise SystemExit(f"{path}: 不是 LAS 文件")
    import struct
    ver = (head[24], head[25])
    n = struct.unpack_from("<I", head, 107)[0]
    if ver >= (1, 4):
        n14 = struct.unpack_from("<Q", head, 247)[0]
        if n14:
            n = n14
    sx, sy, sz = struct.unpack_from("<3d", head, 131)
    ox, oy, oz = struct.unpack_from("<3d", head, 155)
    maxx, minx, maxy, miny, maxz, minz = struct.unpack_from("<6d", head, 179)
    return {"point_format": head[104], "n_points": int(n),
            "offset": struct.unpack_from("<I", head, 96)[0],
            "scale": (sx, sy, sz), "xyz_offset": (ox, oy, oz),
            "bbox": (minx, miny, minz, maxx, maxy, maxz)}


def read_xyz(path: str, hdr: dict, chunk: int = 2_000_000,
             ground_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """分块读 LAS，返回 (xyz(U,3) float64 的 UTM 坐标, 分类直方图(32,))。"""
    if hdr["point_format"] != 3:
        raise SystemExit(f"只实现了 Point Format 3，实为 {hdr['point_format']}")
    n = hdr["n_points"]
    sx, sy, sz = hdr["scale"]
    ox, oy, oz = hdr["xyz_offset"]
    hist = np.zeros(32, dtype=np.int64)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    zs = np.empty(n, dtype=np.float64)
    keep = np.ones(n, dtype=bool)
    pos, done = hdr["offset"], 0
    with open(path, "rb") as f:
        f.seek(pos)
        while done < n:
            k = min(chunk, n - done)
            raw = f.read(k * 34)
            if len(raw) < k * 34:
                k = len(raw) // 34
                if k == 0:
                    break
            rec = np.frombuffer(raw, dtype=PF3, count=k)
            xs[done:done + k] = rec["X"] * sx + ox
            ys[done:done + k] = rec["Y"] * sy + oy
            zs[done:done + k] = rec["Z"] * sz + oz
            hist += np.bincount(rec["cls"], minlength=32)[:32]
            if ground_only:
                keep[done:done + k] = rec["cls"] == 2
            done += k
            del raw, rec
    if ground_only:
        xs, ys, zs = xs[keep], ys[keep], zs[keep]
    return np.column_stack([xs, ys, zs]), hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=list(SITE_LAS))
    ap.add_argument("--ground-only", action="store_true",
                    help="只保留分类==2（地面）点")
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = "_ground" if a.ground_only else ""

    for site in a.sites:
        t0 = time.perf_counter()
        path = os.path.join(BASE, SITE_LAS[site])
        hdr = las_header(path)
        xyz, hist = read_xyz(path, hdr, ground_only=a.ground_only)
        n = len(xyz)
        lo = xyz.min(axis=0)
        hi = xyz.max(axis=0)
        rng = np.where(hi > lo, hi - lo, 1.0)
        xyz = ((xyz - lo) / rng).astype(np.float32)          # → [0,1]³

        # 判据：坐标范围必须落在 [0,1]
        assert xyz.min() >= 0.0 and xyz.max() <= 1.0, "归一化越界"
        assert np.isfinite(xyz).all(), "出现非有限值"

        name = f"p2_{site}_points{tag}"
        np.savez(os.path.join(a.out, f"{name}.npz"),
                 xyz=xyz, bbox=np.stack([lo, hi]).astype(np.float64))
        meta = {"site": site, "las": SITE_LAS[site],
                "n_points": int(n), "n_raw": int(hdr["n_points"]),
                "ground_only": bool(a.ground_only),
                "utm_min": [round(float(v), 3) for v in lo],
                "utm_max": [round(float(v), 3) for v in hi],
                "extent_m": [round(float(v), 2) for v in (hi - lo)],
                "norm_min": [round(float(v), 6) for v in xyz.min(axis=0)],
                "norm_max": [round(float(v), 6) for v in xyz.max(axis=0)],
                "class_hist": {CLS_NAME.get(i, f"cls{i}"): int(c)
                               for i, c in enumerate(hist) if c},
                "seconds": round(time.perf_counter() - t0, 1)}
        json.dump(meta, open(os.path.join(a.out, f"{name}_meta.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[{site}] 点数 {n:,}（原始 {hdr['n_points']:,}）")
        print(f"    UTM  min={meta['utm_min']}  max={meta['utm_max']}")
        print(f"    范围 {meta['extent_m']} m")
        print(f"    归一 min={meta['norm_min']}  max={meta['norm_max']}")
        print(f"    分类 " + "  ".join(f"{k}:{v:,}" for k, v in meta["class_hist"].items()))
        print(f"    写出 data/{name}.npz  耗时 {meta['seconds']}s")

    print(f"\n写出: {a.out}\\p2_{{site}}_points{tag}.npz + _meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
