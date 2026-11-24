# -*- coding: utf-8 -*-
"""地形因子 robust 标准化：从**训练站**统计，套用到两站（决策 2 落地）。

规则（不可违反）：统计量只从**训练站** `p2_terrain_{train_site}.npy` 的有效像元汇总，
存 `data/terrain_stats_{train_site}.json`；推理（含验证站）读同一份 json。
若逐站各算各的统计量，训练/推理看到不同的归一化 → 跨站泛化会被破坏
（与 D12 #6 绝对坐标的教训同源）。

产物：
    data/terrain_stats_{train_site}.json
    data/p2_terrain_{site}_norm.npy   （两站都写，用同一份 stats）

用法：
    python tools/p2_terrain_norm.py                       # 默认训练站 西坝村
    python tools/p2_terrain_norm.py --train-site 杨家坝
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_terrain import (CHANNELS, compute_robust_stats,  # noqa: E402
                        robust_normalize, stats_path)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SITES = ("杨家坝", "西坝村")


def load_terrain(site):
    return np.load(os.path.join(DATA, f"p2_terrain_{site}.npy")).astype(np.float64)


def valid_mask(site):
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    return d["valid"] == 1


def dump_table(title, stack, mask):
    print(f"  {title}")
    print(f"    {'通道':<12}{'min':>10}{'max':>10}{'mean':>10}{'std':>10}")
    for c, arr in zip(CHANNELS, stack):
        v = arr[mask]
        print(f"    {c:<12}{v.min():>10.4f}{v.max():>10.4f}{v.mean():>10.4f}{v.std():>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-site", default="西坝村", choices=list(SITES))
    a = ap.parse_args()
    tr = a.train_site

    # 1) 只从训练站汇总统计量
    tr_stack = load_terrain(tr)
    tr_mask = valid_mask(tr)
    stats = compute_robust_stats(tr_stack, valid=tr_mask)
    jp = stats_path(tr)
    json.dump(stats, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[robust 标准化] 训练站 = {tr}；统计量写出 {os.path.relpath(jp, ROOT)}")
    dump_table(f"训练站 {tr} 原始量纲（valid 像元）：", tr_stack, tr_mask)
    print(f"    统计量 {{p_lo, p_hi, median, iqr}}：")
    for c in CHANNELS:
        s = stats[c]
        print(f"    {c:<12} p1={s['p_lo']:>9.4f} p99={s['p_hi']:>9.4f} "
              f"median={s['median']:>9.4f} iqr={s['iqr']:>9.4f}")

    # 2) 同一份 stats 套用到两站
    for site in SITES:
        stack = load_terrain(site)
        mask = valid_mask(site)
        norm = robust_normalize(stack, stats)
        out = os.path.join(DATA, f"p2_terrain_{site}_norm.npy")
        np.save(out, norm.astype(np.float32))
        tag = "训练站" if site == tr else "验证站"
        print(f"\n  {tag} {site}：写出 {os.path.relpath(out, ROOT)}  "
              f"shape={norm.shape} dtype=float32")
        dump_table(f"{tag} {site} 标准化后（valid 像元，用训练站 stats）：", norm, mask)

    print("\n  提示：v4/v6 推理时读同一份 "
          f"{os.path.relpath(jp, ROOT)}，不要逐站重算。")


if __name__ == "__main__":
    main()
