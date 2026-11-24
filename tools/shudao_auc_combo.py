# -*- coding: utf-8 -*-
"""多因子组合 AUC：等权 rank 相加，看是否强于单因子 tpi31。

背景
----
单因子有效 AUC（`shudao_auc.py`，**填充修正口径**，`tools/fix_tpi31_auc.py`）：
tpi31 **0.6691（西）/ 0.7148（杨）**（原 0.7285/0.7711 含 nodata 边界污染虚高约 0.058），
是全工作区唯一"跨站方向一致"的因子。本脚本问：把 tpi31 与其他因子**组合**能否到 0.80+？

⚠ 方向对齐（关键修正）
----
`tpi31` 是**负向**因子（走廊 tpi31 偏低，中位数 −1.74/−1.03 m）。若按原始 rank 直接相加
（`rankdata(tpi31) + rankdata(slope)`），tpi31 与 slope（正向）会**互相抵消** → AUC 掉回 0.5。
故本脚本给出两栏：
  - **naive**：按原始 rank 相加（复现"忘记翻符号"的结果，作对照）；
  - **aligned**：每个因子先按其**该站点**单因子 AUC 的符号翻正，再 rank 归一化等权相加。
（sign 由同站点标签选出，属"轻度用标签"，故同时报 naive 作无偷看的对照。）

因子来源与标签、评估域同 `tools/shudao_auc.py`（npz 原生 + p2_terrain 8 通道；
标签 = results/mask_1m_{site}.npy = npz label；评估域 = valid）。

用法
----
    python tools/shudao_auc_combo.py                 # 两站都跑
    python tools/shudao_auc_combo.py --site 西坝村
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from shudao_auc import collect_factors, roc_auc   # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

# 组合定义（首个 = 单因子基线）；名称需存在于 collect_factors
COMBO_SETS = (
    ("tpi31", ("tpi31",)),
    ("tpi31+slope", ("tpi31", "slope")),
    ("tpi31+slope+roughness", ("tpi31", "slope", "roughness")),
)


def rank01(x: np.ndarray) -> np.ndarray:
    """平均秩 → [0,1]（与样本量归一）。"""
    r = rankdata(np.asarray(x, dtype=np.float64), method="average")
    return r / r.size


def run_site(site: str) -> dict:
    npz = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    assert np.array_equal(label, npz["label"].astype(bool)), f"{site}: mask 与 npz label 不一致"
    valid = npz["valid"] == 1
    assert not (label & ~valid).any(), f"{site}: 存在 valid 之外的走廊像元"

    factors = {name: arr for name, _src, _note, arr in collect_factors(npz)}
    n_val = int(valid.sum())
    n_pos = int((label & valid).sum())
    print(f"\n{'=' * 70}\n[{site}] 有效 {n_val:,}，走廊 {n_pos:,}（基准率 {n_pos / n_val:.4f}）")
    print(f"{'组合':<24s}{'单因子AUC':>10s}{'naive AUC':>11s}{'aligned AUC':>13s}  各因子单因子AUC")

    rows = []
    for set_name, names in COMBO_SETS:
        # 公共掩膜：本组合内所有因子都有穷 + valid
        m = valid.copy()
        for nm in names:
            m &= np.isfinite(factors[nm])
        pos = label & m
        singles = {}
        aligned_parts, naive_parts = [], []
        for nm in names:
            a = factors[nm][m]
            auc_i = roc_auc(a, pos[m])
            singles[nm] = auc_i
            sign = 1.0 if auc_i >= 0.5 else -1.0
            aligned_parts.append(rank01(a * sign))
            naive_parts.append(rank01(a))
        naive = np.mean(naive_parts, axis=0)
        aligned = np.mean(aligned_parts, axis=0)
        auc_naive = roc_auc(naive, pos[m])
        auc_aligned = roc_auc(aligned, pos[m])
        single_first = singles[names[0]]
        detail = "  ".join(f"{nm}={singles[nm]:.4f}" for nm in names)
        print(f"{set_name:<24s}{single_first:>10.4f}{auc_naive:>11.4f}{auc_aligned:>13.4f}  {detail}")
        rows.append(dict(combo=set_name, factors=list(names),
                         single_first=single_first, auc_naive=auc_naive,
                         auc_aligned=auc_aligned, singles=singles, n_used=int(m.sum())))

    base = rows[0]["auc_aligned"]
    best = max(r["auc_aligned"] for r in rows)
    verdict = "多因子有增益（≥0.80）" if best >= 0.80 else \
              ("多因子有增益（>tpi31 单因子）" if best > base + 1e-4 else "多因子无增益（不超过 tpi31 单因子）")
    print(f"  判定：{verdict}（最佳 aligned AUC={best:.4f}，tpi31 单因子={base:.4f}）")
    return dict(site=site, n_valid=n_val, n_corridor=n_pos, combos=rows,
                best_aligned=best, tpi31_single=base, verdict=verdict)


def main():
    ap = argparse.ArgumentParser(description="多因子组合 AUC（等权 rank 相加）")
    ap.add_argument("--site", default=None, choices=["西坝村", "杨家坝"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sites = [a.site] if a.site else ["西坝村", "杨家坝"]
    out = {"sites": [run_site(s) for s in sites]}
    path = a.out or os.path.join(RES, "shudao_auc_combo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
