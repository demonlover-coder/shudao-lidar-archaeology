# -*- coding: utf-8 -*-
"""用**已有因子**区分走廊：逐因子 AUC（ROC-AUC / Mann–Whitney U）。

动机
----
走廊标签占比仅约 2%（西坝村 5,859 / 286,184 有效像元 ≈ 2.05%），
逐像元分类被背景淹没 → Accuracy / 单 F1 无意义（D10 已禁）。
**AUC** 衡量"某因子的排序在多大程度上把走廊排到非走廊之前"：
    AUC = 0.5 → 与随机无异；越接近 1（或 0）越能区分；方向可翻符号。
本脚本只做**单因子** AUC，回答"哪个现有因子真的带走廊信号"。

因子来源（"已有因子"，不新造）
----
  A. npz 原生通道：dem_mean / dem_std / slope_deg / aspect_deg / curvature / dom_lum
  B. tools/p2_terrain.terrain_stack(dem_mean, gsd) 的 8 通道：
     slope / sin_aspect / cos_aspect / curvature / roughness / profile_curv / tpi15 / tpi31

标签与评估域
----
  标签 = results/mask_1m_{site}.npy（实测与 npz 的 label 逐位一致，脚本内断言校验）
  评估域 = npz valid == 1（走廊像元全落在 valid 内，实测 label ∩ ¬valid = 0）

实现说明
----
环境无 sklearn，故用 scipy.stats.rankdata 实现 MWU-AUC（与 roc_auc_score 等价）：
    AUC = (R_pos − n_pos(n_pos+1)/2) / (n_pos · n_neg)，并列取平均秩。
环形量（aspect_deg 原始角度）AUC 无意义，仅作对照并标注。

用法
----
    python tools/shudao_auc.py                  # 默认 西坝村
    python tools/shudao_auc.py --site 杨家坝
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

from fix_tpi31_auc import fill_nodata                # noqa: E402
from p2_terrain import CHANNELS, terrain_stack, hillshade_factors, HS_CHANNELS  # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")


def roc_auc(score: np.ndarray, pos: np.ndarray) -> float:
    """Mann–Whitney U 形式的 ROC-AUC（score 越大越倾向正类）。

    pos: bool 掩膜。并列值取平均秩 → 与 sklearn.roc_auc_score 等价。
    """
    score = np.asarray(score, dtype=np.float64)
    pos = np.asarray(pos, dtype=bool)
    n_pos = int(pos.sum())
    n_neg = int(pos.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(score, method="average")
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def collect_factors(npz) -> list[tuple[str, str, str, np.ndarray]]:
    """返回 [(名称, 来源, 备注, [H,W] 因子)]。

    ⚠ `dem` 先按"最近 valid 邻居"填充 nodata 再求导/滤波：npz 的 `dem_mean` 在
      valid 之外是 −9999，直接进 3×3 / 31×31 核会把 valid 最外一圈的
      slope/curvature/tpi31 推到非物理值（D12 #13），使 AUC 虚高约 0.058
      （见 `tools/fix_tpi31_auc.py`：tpi31 0.7285/0.7711 → 0.6691/0.7148）。
      npz **原生通道**（dem_std/slope_deg/...）本身就在边界带带污染，无法在此修复，
      仍按原样参与对照（备注已标注）。
    """
    gsd = 1.0
    valid = npz["valid"] == 1
    dem = fill_nodata(npz["dem_mean"].astype(np.float64), valid)
    fac: list[tuple[str, str, str, np.ndarray]] = [
        ("dem_mean", "npz", "高程均值", npz["dem_mean"].astype(np.float64)),
        ("dem_std", "npz", "窗口高程标准差", npz["dem_std"].astype(np.float64)),
        ("slope_deg", "npz", "坡度(度)；边界带含 nodata 污染(D12 #13)", npz["slope_deg"].astype(np.float64)),
        ("aspect_deg", "npz", "环形量，AUC 无意义(仅对照)", npz["aspect_deg"].astype(np.float64)),
        ("curvature_npz", "npz", "中心−8邻域均值(m)；边界带含污染", npz["curvature"].astype(np.float64)),
        ("dom_lum", "npz", "主导亮度", npz["dom_lum"].astype(np.float64)),
    ]
    stack = terrain_stack(dem, gsd=gsd)          # [8, H, W]，CHANNELS 顺序
    for name, arr in zip(CHANNELS, stack):
        note = ""
        if name == "curvature":
            note = "−∇²z(1/m)，各向同性"
        elif name == "profile_curv":
            note = "沿坡向二阶方向导(1/m)，各向异性"
        elif name.startswith("tpi"):
            note = f"TPI=z−{name[3:]}px邻域均值(m)"
        elif name in ("sin_aspect", "cos_aspect"):
            note = "坡向 sin/cos（环形量展开）"
        fac.append((name, "p2_terrain", note, np.asarray(arr, dtype=np.float64)))

    hs = hillshade_factors(dem, gsd=gsd)          # 8 向 Hillshade 导出的两个因子
    hs_note = {"hs_max": "8向Hillshade最大值(最强受光面)",
               "hs_var": "8向Hillshade方差；≈½cos²45°·sin²(slope)，坡度重参数化"}
    for name in HS_CHANNELS:
        fac.append((name, "p2_terrain_hs", hs_note[name],
                    np.asarray(hs[name], dtype=np.float64)))
    return fac


def main():
    ap = argparse.ArgumentParser(description="已有因子的走廊区分 AUC")
    ap.add_argument("--site", default="西坝村", choices=["西坝村", "杨家坝"])
    a = ap.parse_args()
    site = a.site

    npz = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    mk_path = os.path.join(RES, f"mask_1m_{site}.npy")
    label = np.load(mk_path).astype(bool)
    assert np.array_equal(label, npz["label"].astype(bool)), \
        f"{mk_path} 与 npz label 不一致"
    valid = npz["valid"] == 1
    assert not (label & ~valid).any(), "存在 valid 之外的走廊像元"

    n_val = int(valid.sum())
    n_pos = int((label & valid).sum())
    base = n_pos / n_val
    print(f"[{site}] 有效像元 {n_val:,}，走廊 {n_pos:,}（基准率 {base:.4f} ≈ 1:{1 / base:.0f}）")
    print("评估域 = valid（走廊全在 valid 内）；AUC 越大越能区分，方向可翻符号\n")

    header = f"  {'因子':<16s}{'来源':<12s}{'AUC':>8s}{'反序AUC':>9s}{'有效AUC':>9s}{'方向':>7s}{'中位数(走廊)':>13s}{'中位数(背景)':>13s}  备注"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for name, src, note, arr in collect_factors(npz):
        m = valid & np.isfinite(arr)
        pos = label & m
        auc = roc_auc(arr[m], pos[m])
        flip = 1.0 - auc
        eff = max(auc, flip)
        direction = "+" if auc >= 0.5 else "−"
        med_p = float(np.median(arr[pos])) if int(pos.sum()) else float("nan")
        med_n = float(np.median(arr[m & ~label])) if int((m & ~label).sum()) else float("nan")
        rows.append(dict(name=name, source=src, auc=auc, auc_flip=flip, auc_eff=eff,
                         direction=direction, median_pos=med_p, median_neg=med_n,
                         note=note, n_valid_used=int(m.sum())))
        print(f"  {name:<16s}{src:<12s}{auc:>8.4f}{flip:>9.4f}{eff:>9.4f}"
              f"{direction:>7s}{med_p:>13.4f}{med_n:>13.4f}  {note}")

    rows.sort(key=lambda r: r["auc_eff"], reverse=True)
    print("\n  按『有效AUC』排序（最能区分走廊的因子在前）：")
    for i, r in enumerate(rows, 1):
        tag = "（环形量，无意义）" if r["name"] == "aspect_deg" else ""
        print(f"    {i:>2d}. {r['name']:<16s} AUC={r['auc']:.4f}  有效={r['auc_eff']:.4f}  {r['direction']} {tag}")

    out = os.path.join(RES, f"shudao_auc_{site}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(site=site, n_valid=n_val, n_corridor=n_pos, base_rate=base,
                       factors=rows), f, ensure_ascii=False, indent=2)
    print(f"\n保存: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
