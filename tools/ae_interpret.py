# -*- coding: utf-8 -*-
"""AE 瓶颈维的可解释性分析：命名 + 走廊 AUC + 跨站载荷匹配。

流程
----
    data/ae_latent_{site}_L{latent}.npy   [N_valid, latent]（train_ae.py 产物）
        ↓ 与标准化后的 20 因子算相关 → 载荷矩阵 C[latent, 20]
        ↓ 每维取 |载荷| top-3 命名；对走廊标签算有效 AUC（MWU 形式）
        ↓ 两站按 |cos| 贪心配对，报跨站是否稳定（cos>0.8 视为同物理含义）

⚠ 口径修正（相对用户原始草稿）
----
    走廊标签是**整幅栅格** [H, W]，而 Z 只在 **valid 像元** 上（[N_valid, latent]）。
    直接 `mwu_auc(label, Z[:, i])` 会因长度不符报错。本脚本先按
    `np.flatnonzero(valid.ravel())`（与 build_feature_matrix 行序一致）把标签
    压到 valid 行，再做 AUC。

判据（用户给定）
----
    重建 R² > 0.85                                     → 通过
    存在一维 跨站 cos>0.8 且 两站 AUC 均 > tpi31(0.67/0.71) → AE 有增益，替换主通道
    存在一维 跨站 cos>0.8 但 AUC ≤ tpi31               → 保留 AE 作辅助，主通道仍 tpi31
    无维 跨站 cos>0.8                                  → 同稀疏 PCA 结论（无跨站稳定特征）

用法
----
    python tools/ae_interpret.py --latent 8
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

from build_feature_matrix import FACTORS              # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

# tpi31 修正后基准（tools/fix_tpi31_auc.py）
TPI31_REF = {"西坝村": 0.6691, "杨家坝": 0.7148}


def mwu_auc(label: np.ndarray, score: np.ndarray) -> float:
    """Mann–Whitney U 形式的有效 AUC = max(AUC, 1−AUC)。"""
    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    r = rankdata(score, method="average")
    n1 = int(label.sum())
    n0 = int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    a = (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(a, 1 - a))


def valid_label(site: str) -> np.ndarray:
    """把整幅走廊 mask 压到 valid 行（行序 = np.flatnonzero(valid.ravel())）。"""
    npz = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    valid = npz["valid"] == 1
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    assert not (label & ~valid).any(), f"{site}: 存在 valid 之外的走廊像元"
    return label.ravel()[np.flatnonzero(valid.ravel())]


def load(site: str, latent: int = 8):
    X = np.load(os.path.join(DATA, f"feature_matrix_{site}.npy"))
    Z = np.load(os.path.join(DATA, f"ae_latent_{site}_L{latent}.npy"))
    return X, Z


def analyze(site: str, latent: int = 8) -> dict:
    X, Z = load(site, latent)
    assert Z.shape[0] == X.shape[0], f"{site}: Z 与 X 样本数不一致 {Z.shape} vs {X.shape}"
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    C = np.corrcoef(Z.T, Xs.T)[:latent, latent:]      # [latent, 20]
    lab = valid_label(site)

    out = {"site": site, "latent": latent, "n_valid": int(X.shape[0]),
           "n_corridor": int(lab.sum()), "dims": []}
    print(f"\n[{site}] latent = {latent}  valid={X.shape[0]:,}  走廊={int(lab.sum()):,}")
    for i in range(latent):
        row = C[i]
        top3 = np.argsort(-np.abs(row))[:3]
        name_parts = [f"{row[j]:+.2f}·{FACTORS[j]}" for j in top3]
        auc = mwu_auc(lab, Z[:, i])
        print(f"  dim{i}  AUC={auc:.4f}  {' + '.join(name_parts)}")
        out["dims"].append({
            "dim": i, "auc": float(auc),
            "top3": [{"factor": FACTORS[j], "load": float(row[j])} for j in top3],
            "load_vec": row.tolist(),
        })
    return out


def cross_site(west_json: dict, yang_json: dict):
    """两站对应维的载荷余弦相似度（西站基准，贪心一一配对）。"""
    Cw = np.array([d["load_vec"] for d in west_json["dims"]])
    Cy = np.array([d["load_vec"] for d in yang_json["dims"]])
    used: set[int] = set()
    pairs = []
    for i in range(Cw.shape[0]):
        best, best_j = -1.0, -1
        for j in range(Cy.shape[0]):
            if j in used:
                continue
            cs = float(Cw[i] @ Cy[j] /
                       (np.linalg.norm(Cw[i]) * np.linalg.norm(Cy[j]) + 1e-9))
            if cs > best:
                best, best_j = cs, j
        used.add(best_j)
        pairs.append((i, best_j, best))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="AE 瓶颈维命名 / AUC / 跨站匹配")
    ap.add_argument("--latent", type=int, default=8)
    a = ap.parse_args()

    outs = {}
    for s in ("西坝村", "杨家坝"):
        outs[s] = analyze(s, a.latent)

    print("\n跨站载荷匹配（cos>0.8 视为同物理含义；西站基准贪心配对）:")
    pairs = cross_site(outs["西坝村"], outs["杨家坝"])
    for i, j, cs in pairs:
        aw = outs["西坝村"]["dims"][i]["auc"]
        ay = outs["杨家坝"]["dims"][j]["auc"]
        flag = "✅" if cs > 0.8 else "  "
        print(f"  {flag} 西dim{i} = 杨dim{j}  cos={cs:+.3f}  AUC={aw:.3f}/{ay:.3f}")

    outs["cross_site_pairs"] = [
        {"west": int(i), "yang": int(j), "cos": float(cs)} for i, j, cs in pairs]
    outs["tpi31_ref"] = TPI31_REF
    with open(os.path.join(RES, f"ae_interpret_L{a.latent}.json"), "w",
              encoding="utf-8") as f:
        json.dump(outs, f, ensure_ascii=False, indent=2)
    print(f"\n写出 results/ae_interpret_L{a.latent}.json")


if __name__ == "__main__":
    main()
