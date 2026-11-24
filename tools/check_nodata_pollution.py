# -*- coding: utf-8 -*-
"""量化 data/{site}_1m.npz 地形通道的 nodata 污染是否命中走廊标签（决策 4）。

污染判据：valid 像元中 slope_deg > 89°（实测受污染像元 slope 恒为 90.0°）。
门槛（用户给定）：
    "污染 ∩ 走廊 / 走廊总数" < 5%  → 不动
                          ≥ 10% → 必须重生成 npz + 重训
    5%~10% → 报告，待决策

用法：
    python tools/check_nodata_pollution.py            # 西坝村
    python tools/check_nodata_pollution.py --site 杨家坝
"""
from __future__ import annotations

import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(site: str) -> dict:
    p = os.path.join(ROOT, "data", f"{site}_1m.npz")
    d = np.load(p)
    keys = list(d.keys())
    print(f"== {site}")
    print(f"   文件: {p}")
    print(f"   键: {keys}")
    if "label" not in keys:
        raise SystemExit(f"  缺少 'label'，无法算走廊交集；请确认标签键名")
    if "slope_deg" not in keys or "valid" not in keys:
        raise SystemExit("  缺少 'slope_deg' 或 'valid'")

    slope = np.asarray(d["slope_deg"], dtype=np.float64)
    label = np.asarray(d["label"])
    valid = np.asarray(d["valid"]) == 1

    polluted = (slope > 89.0) & valid
    corridor = label == 1

    n_valid = int(valid.sum())
    n_corr = int(corridor.sum())
    n_pol = int(polluted.sum())
    n_pol_corr = int((polluted & corridor).sum())

    in_corr = n_pol_corr / n_corr if n_corr else float("nan")
    in_pol = n_pol_corr / n_pol if n_pol else float("nan")
    pol_rate = n_pol / n_valid if n_valid else float("nan")

    print(f"   valid 像元 {n_valid}，走廊像元 {n_corr}（占 valid {n_corr / n_valid:.2%}）")
    print(f"   污染像元（slope>89° & valid）：{n_pol}（占 valid {pol_rate:.2%}）")
    print(f"   污染 ∩ 走廊：{n_pol_corr}")
    print(f"   ▶ 走廊中污染比例（污染∩走廊 / 走廊总数）= {in_corr:.2%}")
    print(f"   污染中走廊比例 = {in_pol:.2%}")
    return {"site": site, "n_valid": n_valid, "n_corr": n_corr, "n_pol": n_pol,
            "n_pol_corr": n_pol_corr, "corridor_polluted_frac": in_corr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="西坝村")
    args = ap.parse_args()
    r = check(args.site)
    f = r["corridor_polluted_frac"]
    verdict = ("不动（<5%）" if f < 0.05 else
               "必须重生成 npz + 重训（≥10%）" if f >= 0.10 else
               "介于 5%~10%，待决策")
    print(f"   → 裁决：{verdict}\n")


if __name__ == "__main__":
    main()
