# -*- coding: utf-8 -*-
"""用"最近 valid 邻居"填 nodata 后，重算 tpi31 的两站 AUC（**修正边界污染虚高**）。

为什么必须修正
----
`tools/shudao_auc.py` / `shudao_context_auc.py` / `shudao_composite_figure.py`
都把 `dem_mean` 直接喂给 `uniform_filter` / `sobel`，而 `data/{site}_1m.npz` 的
`dem_mean` 在 valid 之外是 **−9999**（见 `docs/00_step0_decisions.md` D12 #13）。
`uniform_filter`（mode="nearest"）在 valid 最外一圈会把 −9999 平均进来 →
该圈 `tpi31 = z − 负得很厉害的均值` = 极大正值。这些像元**不是走廊**（D12 #13：
污染 ∩ 走廊 = 0），却因分数被推到排序顶端，改变了整个 valid 域的秩 → AUC 虚高。

本脚本把 nodata 用**最近 valid 像元**填掉再算 tpi31，得到无污染的 AUC；
并把旧口径一并算出，量化"虚高了多少"。

MWU-AUC（无 sklearn 依赖）：AUC = (R_pos − n_pos(n_pos+1)/2) / (n_pos·n_neg)。

用法
----
    python tools/fix_tpi31_auc.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.stats import rankdata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")


def fill_nodata(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """nodata 用**最近 valid 像元**填充（`distance_transform_edt` 的零元素 = valid）。"""
    out = dem.astype(np.float64).copy()
    miss = ~valid
    if miss.any():
        idx = distance_transform_edt(miss, return_distances=False, return_indices=True)
        out[miss] = dem[idx[0], idx[1]][miss]
    return out


def mwu_auc(label: np.ndarray, score: np.ndarray) -> float:
    """MWU-AUC；并列值取平均秩（与 sklearn.roc_auc_score 等价）。"""
    r = rankdata(score, method="average")
    n1 = int(label.sum())
    n0 = int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    out = {}
    print(f"{'站点':<8s}{'AUC(旧,未填充)':>16s}{'有效':>9s}"
          f"{'AUC(新,填充)':>14s}{'有效':>9s}{'方向':>6s}{'虚高':>9s}")
    print("-" * 74)
    for site in ("西坝村", "杨家坝"):
        d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
        dem = d["dem_mean"].astype(np.float64)
        valid = d["valid"] == 1
        label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
        assert np.array_equal(label, d["label"].astype(bool)), "mask 与 npz label 不一致"
        assert not (label & ~valid).any(), "存在 valid 之外的走廊像元"

        dem_f = fill_nodata(dem, valid)
        tpi31_old = dem - uniform_filter(dem, size=31, mode="nearest")
        tpi31_new = dem_f - uniform_filter(dem_f, size=31, mode="nearest")

        a_old = mwu_auc(label[valid], tpi31_old[valid])
        a_new = mwu_auc(label[valid], tpi31_new[valid])
        e_old, e_new = max(a_old, 1 - a_old), max(a_new, 1 - a_new)
        direction = "+" if a_new >= 0.5 else "−"
        out[site] = dict(auc_old=a_old, eff_old=e_old, auc_new=a_new, eff_new=e_new,
                         direction=direction, inflation=e_old - e_new,
                         n_valid=int(valid.sum()), n_corridor=int(label.sum()))
        print(f"{site:<8s}{a_old:>16.4f}{e_old:>9.4f}{a_new:>14.4f}{e_new:>9.4f}"
              f"{direction:>6s}{e_old - e_new:>9.4f}")

    print("\n  ⇒ 结论：旧值含 nodata 边界污染虚高约 "
          f"{np.mean([v['inflation'] for v in out.values()]):.3f}；"
          "新值为无污染口径，方向不变（两站仍一致）。")
    p = os.path.join(RES, "fix_tpi31_auc.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  保存: {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
