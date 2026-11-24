# -*- coding: utf-8 -*-
"""稀疏 PCA：20 因子 → 5 个"超级特征"（每个站**独立**跑一次）。

流程
----
    data/feature_matrix_{site}.npy  [N, 20] 原始量纲
        ↓ robust 标准化（P1/P99 截断 + (x−median)/IQR，逐列，本站自算）
    Xs  [N, 20]
        ↓ sklearn.decomposition.SparsePCA(n_components=5, alpha=0.1, random_state=42)
    W   [5, 20]  载荷矩阵（每一行 = 一个超级特征的加权公式）
    Z   [N, 5]   每个像元的 5 个超级特征取值

为什么用**稀疏** PCA
----
普通 PCA 的载荷是稠密的，20 个因子每个都掺一点，无法命名也无法解释；
SparsePCA 加 L1 惩罚（`alpha`）把次要载荷压到 0，每行只剩少数几个因子 →
"PC0 = 0.82·tpi31 + 0.71·tpi15 + 0.53·relief_31" 这种可命名的公式。
且无需训练、秒级出结果，适合"两站各跑一次看载荷是否一致"。

标准化口径（实测决定，两种都存）
----
    `winsor_z`（**默认**）：P1/P99 截断 + 单位方差 (x−mean)/std
    `robust_iqr`（对照）  ：P1/P99 截断 + (x−median)/IQR（项目原约定）

为什么默认不是 IQR：实测两站 IQR 缩放下各列标准差差 **3.9×**
（`curvature/profile_curv/plan_curv/tpi7` 的 std ≈ 1.9~2.2，
而 `slope/roughness/relief_*/openness/aspect_std` 只有 0.55~0.78）——
长尾列被"标准化不足"，PCA 于是被曲率族垄断（PC0 = 0.49·curvature − 0.48·profile_curv，
方差贡献 0.39/0.51），这不是地形本身的结构而是缩放选择的产物。
PCA/SPCA 以方差为度量，前提就是各列可比 ⇒ **先截断去极值、再单位方差**。
截断在前，故不触发项目"禁止 z-score"的原由（那条针对未截断的原始长尾）。
IQR 口径仍会算一遍存 `results/sparse_pca_{site}_robust_iqr.json` 备查。

符号约定
----
SparsePCA 的分量符号是任意的。本脚本强制：**每行绝对值最大的载荷取正号**
（若为负则整行连同对应 Z 列一起翻符号）。即便这样，两站"最大载荷"元素
可能互换（实测 杨家坝 PC0 由 curvature 领衔、西坝村由 profile_curv 领衔），
导致同一物理分量在两站符号相反 —— 下游 `factor_name_auc.py` 会按
符号对齐后再比余弦，不依赖这里的约定。

输出
----
    data/pc_map_{site}.npy         [5, H, W] float32（valid 之外为 0；Z 按 valid 行序还原）
    results/sparse_pca_{site}.json W / 标准化参数 / 方差贡献 / 重构 R²

用法
----
    python tools/sparse_pca.py                        # 两站，winsor_z
    python tools/sparse_pca.py --scale robust_iqr     # IQR 口径对照
    python tools/sparse_pca.py --site 杨家坝
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from build_feature_matrix import DATA, FACTORS, NAME_HINT   # noqa: E402

RES = os.path.join(ROOT, "results")
N_COMPONENTS = 5
ALPHA = 0.1
SEED = 42


def fit_scale(X: np.ndarray, mode: str, q=(1.0, 99.0)) -> dict:
    """逐列 robust 统计量：先 P1/P99 截断，再按 mode 定标。"""
    lo, hi = np.percentile(X, q, axis=0)
    Xc = np.clip(X, lo, hi)
    med = np.median(Xc, axis=0)
    q25, q75 = np.percentile(Xc, (25.0, 75.0), axis=0)
    iqr = np.where(np.abs(q75 - q25) > 1e-12, q75 - q25, 1.0)
    std = np.where(Xc.std(axis=0) > 1e-12, Xc.std(axis=0), 1.0)
    center = med if mode == "robust_iqr" else Xc.mean(axis=0)
    scale = iqr if mode == "robust_iqr" else std
    return dict(p_lo=lo, p_hi=hi, median=med, iqr=iqr, std=std,
                center=center, scale=scale, mode=mode)


def apply_scale(X: np.ndarray, st: dict) -> np.ndarray:
    return (np.clip(X, st["p_lo"], st["p_hi"]) - st["center"]) / st["scale"]


def print_loadings(W: np.ndarray, title: str, topk: int = 3):
    """打印 5×20 载荷矩阵 + 每行 top-`topk` 因子（带名字与物理含义）。"""
    print(f"\n{title}")
    short = [f[:8] for f in FACTORS]
    print("  " + " " * 5 + "".join(f"{s:>9s}" for s in short))
    for i, row in enumerate(W):
        print(f"  PC{i:<2d} " + "".join(f"{v:>+9.3f}" for v in row))
    print("  " + "-" * (5 + 9 * len(FACTORS)))
    for i, row in enumerate(W):
        idx = np.argsort(-np.abs(row))[:topk]
        formula = " + ".join(f"{row[j]:+.2f}·{FACTORS[j]}" for j in idx)
        nss = int((np.abs(row) > 0.05 * np.abs(row).max()).sum())
        hint = "＋".join(NAME_HINT[FACTORS[j]] for j in idx[:2])
        print(f"  PC{i}: {formula}")
        print(f"       → 命名『{hint}』；非零载荷 {nss}/{len(FACTORS)}"
              f"（|w|>5%·max）")


def run_site(site: str, mode: str = "winsor_z") -> dict:
    from sklearn.decomposition import SparsePCA

    X = np.load(os.path.join(DATA, f"feature_matrix_{site}.npy")).astype(np.float64)
    meta = json.load(open(os.path.join(DATA, f"feature_matrix_{site}.meta.json"),
                          encoding="utf-8"))
    assert list(meta["factors"]) == list(FACTORS)
    print(f"\n[{site}] 因子矩阵 {X.shape}（{len(FACTORS)} 因子），标准化口径 = {mode}")

    st = fit_scale(X, mode)
    Xs = apply_scale(X, st)
    spread = Xs.std(axis=0)
    print(f"  标准化后各列 std：min={spread.min():.3f} max={spread.max():.3f}"
          f" 极差比={spread.max() / max(spread.min(), 1e-12):.2f}×"
          f"（越接近 1 越好；IQR 口径实测 3.9×）")

    t0 = time.time()
    pca = SparsePCA(n_components=N_COMPONENTS, alpha=ALPHA, random_state=SEED)
    pca.fit(Xs)
    Z = pca.transform(Xs)
    W = np.asarray(pca.components_, dtype=np.float64)
    print(f"  SparsePCA(n={N_COMPONENTS}, alpha={ALPHA}, seed={SEED})"
          f" 用时 {time.time() - t0:.1f}s")

    # 符号约定：每行 |最大载荷| 取正
    for i in range(W.shape[0]):
        if W[i][np.argmax(np.abs(W[i]))] < 0:
            W[i] *= -1.0
            Z[:, i] *= -1.0

    # 方差贡献与重构 R²
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)      # 单位载荷
    Xhat = Z @ Wn
    ss_res = float(((Xs - Xhat) ** 2).sum())
    ss_tot = float((Xs ** 2).sum())
    var_ratio = Z.var(axis=0) / Xs.var(axis=0).sum()
    recon_r2 = 1.0 - ss_res / ss_tot

    print_loadings(W, f"载荷矩阵 W[{N_COMPONENTS}×{len(FACTORS)}]（{site}）")
    print(f"  方差贡献 var(Z_i)/Σvar(X)："
          + "  ".join(f"PC{i}={v:.4f}" for i, v in enumerate(var_ratio)))
    print(f"  5 分量重构 R² = {recon_r2:.4f}")

    # 还原到栅格
    npz = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    valid = npz["valid"] == 1
    H, Wg = valid.shape
    rows = np.flatnonzero(valid.ravel())
    assert len(rows) == X.shape[0], "行序与 valid 不一致（npz 变了？）"
    pc_map = np.zeros((N_COMPONENTS, H, Wg), dtype=np.float32)
    for i in range(N_COMPONENTS):
        pc_map[i].ravel()[rows] = Z[:, i].astype(np.float32)
    tag = "" if mode == "winsor_z" else f"_{mode}"
    np.save(os.path.join(DATA, f"pc_map_{site}{tag}.npy"), pc_map)
    print(f"  写出 data/pc_map_{site}{tag}.npy {pc_map.shape}"
          f"（valid 外为 0，valid 内为 Z 分值）")

    out = dict(site=site, factors=list(FACTORS), name_hint=NAME_HINT,
               n_components=N_COMPONENTS, alpha=ALPHA, seed=SEED, scale_mode=mode,
               W=W.tolist(), var_ratio=var_ratio.tolist(), recon_r2=recon_r2,
               n_samples=int(X.shape[0]), col_std=spread.tolist(),
               robust={k: (v if isinstance(v, str) else np.asarray(v).tolist())
                       for k, v in st.items()})
    with open(os.path.join(RES, f"sparse_pca_{site}{tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  写出 results/sparse_pca_{site}{tag}.json")
    return out


def main():
    ap = argparse.ArgumentParser(description="20 因子 → 5 个稀疏主成分（超级特征）")
    ap.add_argument("--site", default=None, choices=["西坝村", "杨家坝"],
                    help="默认两站各跑一次")
    ap.add_argument("--scale", default="winsor_z", choices=["winsor_z", "robust_iqr"],
                    help="标准化口径；winsor_z=P1/P99截断+单位方差（默认）")
    a = ap.parse_args()
    for site in ([a.site] if a.site else ["西坝村", "杨家坝"]):
        run_site(site, mode=a.scale)


if __name__ == "__main__":
    main()
