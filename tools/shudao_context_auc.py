# -*- coding: utf-8 -*-
"""上下文特征区分走廊：Frangi 线性响应 + tpi31 邻域一致性（**两站对照**）。

动机
----
已有单因子最高有效 AUC ≤ 0.78（tpi31，**填充修正口径** 0.6691 / 0.7148，唯一跨站方向一致；
原 0.7285 / 0.7711 含 nodata 边界污染虚高约 0.058，见 `tools/fix_tpi31_auc.py`）。
本脚本试两个"贴线"上下文特征，看能否**跨站**稳定地区分走廊：

  特征 1  Frangi 线性响应 —— 多尺度 Hessian 特征值构造的**线状结构**增强子
          （skimage.filters.frangi，sigmas=1..4，black_ridges=False → 亮脊）
  特征 2  tpi31 邻域一致性 —— tpi31 在 11px 窗内的**局部方差**（越小越一致）

判据（用户给定，逐特征判定）
----
  两站有效 AUC 都 > 0.7 且方向一致   → ✅ 通过，加进合成图
  单站 > 0.7、另一站 < 0.6          → ❌ 不可迁移，不加（同 slope）
  两站都 < 0.65                     → ❌ 无效

评估口径与 tools/shudao_auc.py 完全一致：标签 = results/mask_1m_{site}.npy
（脚本内断言与 npz label 逐位一致），评估域 = npz valid == 1，
AUC 用 scipy.stats.rankdata 实现的 MWU 形式（环境无 sklearn），
有效 AUC = max(AUC, 1−AUC)，方向可翻符号。

用法
----
    python tools/shudao_context_auc.py                 # 两站对照（默认）
    python tools/shudao_context_auc.py --site 西坝村   # 只跑一站
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import rankdata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from fix_tpi31_auc import fill_nodata            # noqa: E402
from p2_terrain import CHANNELS, terrain_stack   # noqa: E402
from skimage.filters import frangi               # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

# 判据阈值
T_PASS = 0.70      # 两站都需 > 0.70 且方向一致 → 通过
T_WEAK = 0.60      # 单站 > 0.70、另一站 < 0.60 → 不可迁移
T_VOID = 0.65      # 两站都 < 0.65 → 无效


def roc_auc(score: np.ndarray, pos: np.ndarray) -> float:
    """Mann–Whitney U 形式的 ROC-AUC（score 越大越倾向正类）。"""
    score = np.asarray(score, dtype=np.float64)
    pos = np.asarray(pos, dtype=bool)
    n_pos = int(pos.sum())
    n_neg = int(pos.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(score, method="average")
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def context_features(dem: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """返回 [(名称, 备注, 因子)]：Frangi 线性响应 / tpi31 邻域一致性（+ tpi31 参照）。"""
    z = np.asarray(dem, dtype=np.float64)
    z_fill = np.where(np.isfinite(z), z, np.nanmedian(z))

    # 特征 1：Frangi 线性响应（亮脊），sigmas=1..4 px
    fr = frangi(z_fill, sigmas=range(1, 5), black_ridges=False).astype(np.float64)

    # tpi31（与合成图/单因子 AUC 同源，取自 p2_terrain，保证口径一致）
    tpi31 = np.asarray(terrain_stack(z, gsd=1.0)[CHANNELS.index("tpi31")], dtype=np.float64)

    # 特征 2：tpi31 邻域一致性 = 负局部方差（var 窗口 11px，越小越一致）
    m1 = uniform_filter(tpi31, size=11, mode="nearest")
    m2 = uniform_filter(tpi31 ** 2, size=11, mode="nearest")
    var11 = np.maximum(m2 - m1 ** 2, 0.0)          # 数值保护
    consist = -var11                                # 一致性（越一致越大）

    return [
        ("frangi_resp", "Frangi 线性响应(亮脊, sigmas=1..4)", fr),
        ("tpi31_consist", "tpi31 邻域一致性(−var, 11px窗)", consist),
        ("tpi31", "参照：主通道单因子", tpi31),
    ]


def eval_site(site: str) -> dict:
    npz = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    mk_path = os.path.join(RES, f"mask_1m_{site}.npy")
    label = np.load(mk_path).astype(bool)
    assert np.array_equal(label, npz["label"].astype(bool)), f"{mk_path} 与 npz label 不一致"
    valid = npz["valid"] == 1
    assert not (label & ~valid).any(), "存在 valid 之外的走廊像元"

    # ⚠ 先按"最近 valid 邻居"填充 nodata 再算特征：npz 的 `dem_mean` 在 valid 之外是
    #   −9999，直接进 Frangi/31px 滤波会把 valid 最外一圈推成非物理值（D12 #13），
    #   与 `shudao_auc.py` / 合成图**同口径**（见 `tools/fix_tpi31_auc.py`）。
    dem = fill_nodata(npz["dem_mean"].astype(np.float64), valid)
    n_val = int(valid.sum())
    n_pos = int((label & valid).sum())
    print(f"\n[{site}] 有效像元 {n_val:,}，走廊 {n_pos:,}（基准率 {n_pos / n_val:.4f}）")

    header = f"  {'特征':<16s}{'AUC':>8s}{'反序AUC':>9s}{'有效AUC':>9s}{'方向':>7s}{'中位数(走廊)':>14s}{'中位数(背景)':>14s}  备注"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for name, note, arr in context_features(dem):
        m = valid & np.isfinite(arr)
        pos = label & m
        auc = roc_auc(arr[m], pos[m])
        flip = 1.0 - auc
        eff = max(auc, flip)
        direction = "+" if auc >= 0.5 else "−"
        med_p = float(np.median(arr[pos])) if int(pos.sum()) else float("nan")
        med_n = float(np.median(arr[m & ~label])) if int((m & ~label).sum()) else float("nan")
        rows.append(dict(name=name, auc=auc, auc_flip=flip, auc_eff=eff,
                         direction=direction, median_pos=med_p, median_neg=med_n,
                         note=note, n_used=int(m.sum())))
        print(f"  {name:<16s}{auc:>8.4f}{flip:>9.4f}{eff:>9.4f}{direction:>7s}"
              f"{med_p:>14.4f}{med_n:>14.4f}  {note}")

    return dict(site=site, n_valid=n_val, n_corridor=n_pos,
                base_rate=n_pos / n_val, rows=rows)


def classify(a: float, b: float, da: str, db: str) -> str:
    """按用户判据给跨站结论（a/b 为两站有效 AUC，da/db 为方向）。"""
    if a > T_PASS and b > T_PASS and da == db:
        return "PASS（两站 >0.70 且方向一致 → 可加入合成图）"
    if (a > T_PASS and b < T_WEAK) or (b > T_PASS and a < T_WEAK):
        return "FAIL-不可迁移（单站 >0.70、另一站 <0.60，同 slope）"
    if a < T_VOID and b < T_VOID:
        return "FAIL-无效（两站均 <0.65）"
    return "FAIL-边界（未满足通过条件，亦不属明确失效）"


def main():
    ap = argparse.ArgumentParser(description="上下文特征（Frangi / tpi31 邻域一致性）跨站 AUC")
    ap.add_argument("--site", default=None, choices=["西坝村", "杨家坝"],
                    help="只跑单站；默认两站对照")
    a = ap.parse_args()

    sites = [a.site] if a.site else ["西坝村", "杨家坝"]
    res = [eval_site(s) for s in sites]
    by_site = {r["site"]: {x["name"]: x for x in r["rows"]} for r in res}

    print("\n" + "=" * 88)
    print("跨站对照与判据")
    print("=" * 88)
    verdicts = {}
    if len(res) == 2:
        _npz_xb = np.load(os.path.join(DATA, "西坝村_1m.npz"))
        _dem_xb = fill_nodata(_npz_xb["dem_mean"].astype(np.float64), _npz_xb["valid"] == 1)
        names = [n for n, _, _ in context_features(_dem_xb)]
        for n in names:
            xa = by_site["西坝村"][n]
            xb = by_site["杨家坝"][n]
            v = classify(xa["auc_eff"], xb["auc_eff"], xa["direction"], xb["direction"])
            verdicts[n] = v
            print(f"  {n:<16s} 西坝村 有效AUC={xa['auc_eff']:.4f}({xa['direction']})"
                  f"  杨家坝 有效AUC={xb['auc_eff']:.4f}({xb['direction']})")
            print(f"                   → {v}")
    else:
        print(f"  仅跑单站 {sites[0]}，判据需两站对照，此处不给结论。")

    out = os.path.join(RES, "shudao_context_auc.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(thresholds=dict(pass_=T_PASS, weak=T_WEAK, void=T_VOID),
                       sites=res, verdicts=verdicts), f, ensure_ascii=False, indent=2)
    print(f"\n保存: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
