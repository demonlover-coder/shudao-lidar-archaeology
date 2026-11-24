# -*- coding: utf-8 -*-
"""因子冗余检查：现有因子的 Spearman 相关矩阵，聚焦 tpi31 是否与他人冗余。

原需求
----
用户想算 `corr(tpi31, cliff_proximity)` 与 `corr(tpi31, contour_conformity)`，
判据：若 > 0.7 则冗余、不必新增该因子。

实测前提（重要）
----
**`cliff_proximity`（崖壁接近度）与 `contour_conformity`（等高线贴合度）在本工作区
并不存在**——无任何实现脚本或数据（grep 仅命中文档里"崖壁/贴合"作形容词）。
故这两个相关性**无法计算**；本脚本改为对**现有全部因子**做 Spearman 冗余检查
（同一目的：确认 tpi31 是否与已有因子冗余），并在输出中显式标注两个缺失因子。
若要真正算崖壁/等高线因子，需先给出它们的定义（距离场？Canny 边缘距离？等值线缓冲？）。

指标
----
Spearman ρ ∈ [−1,1]（秩相关，抗非线性单调关系）；|ρ| ≥ 0.7 视为**冗余**。
评估域 = valid（同 shudao_auc.py），并按逐对有效像元剔除非有限值。

用法
----
    python tools/shudao_corr.py                 # 两站都跑
    python tools/shudao_corr.py --site 西坝村
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from shudao_auc import collect_factors   # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

REDUNDANT = 0.7          # |ρ| ≥ 此值视为冗余
FOCUS = "tpi31"          # 聚焦因子
ABSENT = ("cliff_proximity（崖壁接近度）", "contour_conformity（等高线贴合度）")


def run_site(site: str) -> dict:
    npz = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    valid = npz["valid"] == 1
    pairs = collect_factors(npz)
    names = [p[0] for p in pairs]
    arrs = np.stack([p[3] for p in pairs], axis=0)         # [F, H, W]

    m = valid.copy()
    for i in range(arrs.shape[0]):
        m &= np.isfinite(arrs[i])
    M = arrs[:, m]                                          # [F, N]

    print(f"\n{'=' * 74}\n[{site}] 冗余检查：{len(names)} 个因子，有效像元 {int(m.sum()):,}")
    print(f"  缺失因子（无法计算相关）：{ABSENT[0]}、{ABSENT[1]}")

    rho, _ = spearmanr(M, axis=1)                           # [F, F]
    rho = np.atleast_2d(rho)

    # 矩阵
    print("\n  Spearman ρ 矩阵（行=因子）：")
    print("    " + "".join(f"{n[:8]:>10s}" for n in names))
    for i, n in enumerate(names):
        print(f"    {n[:8]:<8s}" + "".join(f"{rho[i, j]:>10.2f}" for j in range(len(names))))

    # tpi31 聚焦
    fi = names.index(FOCUS)
    order = sorted((j for j in range(len(names)) if j != fi),
                   key=lambda j: abs(rho[fi, j]), reverse=True)
    print(f"\n  {FOCUS} 与其余因子的 |ρ| 排序（≥{REDUNDANT} 标冗余）：")
    focus_rows = []
    for j in order:
        flag = "  ← 冗余" if abs(rho[fi, j]) >= REDUNDANT else ""
        print(f"    {names[j]:<16s} ρ={rho[fi, j]:+.3f}{flag}")
        focus_rows.append(dict(other=names[j], rho=float(rho[fi, j]),
                               abs_rho=float(abs(rho[fi, j])),
                               redundant=bool(abs(rho[fi, j]) >= REDUNDANT)))

    # 全冗余对
    red_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(rho[i, j]) >= REDUNDANT:
                red_pairs.append((names[i], names[j], float(rho[i, j])))
    print(f"\n  全部 |ρ|≥{REDUNDANT} 的因子对（{len(red_pairs)} 对）：")
    for a_, b_, r_ in sorted(red_pairs, key=lambda t: -abs(t[2])):
        print(f"    {a_:<16s} ~ {b_:<16s} ρ={r_:+.3f}")

    concl = (f"{FOCUS} 与现有因子均不冗余（最大 |ρ|={focus_rows[0]['abs_rho']:.3f}）"
             if focus_rows and focus_rows[0]["abs_rho"] < REDUNDANT
             else f"{FOCUS} 与 {focus_rows[0]['other']} 冗余（|ρ|={focus_rows[0]['abs_rho']:.3f}）")
    print(f"\n  结论：{concl}")
    print(f"  注意：因 {ABSENT[0]} / {ABSENT[1]} 不存在，"
          f"\"tpi31 与它们的相关性\"未计算——需先定义这两个因子。")

    return dict(site=site, n_used=int(m.sum()), names=names,
                rho=rho.round(6).tolist(), focus=FOCUS,
                focus_rows=focus_rows, redundant_pairs=red_pairs,
                absent_factors=[ABSENT[0], ABSENT[1]], conclusion=concl)


def main():
    ap = argparse.ArgumentParser(description="现有因子 Spearman 冗余检查（聚焦 tpi31）")
    ap.add_argument("--site", default=None, choices=["西坝村", "杨家坝"])
    a = ap.parse_args()
    sites = [a.site] if a.site else ["西坝村", "杨家坝"]
    out = {"sites": [run_site(s) for s in sites]}
    path = os.path.join(RES, "shudao_corr.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
