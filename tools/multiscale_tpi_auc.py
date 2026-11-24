# -*- coding: utf-8 -*-
"""手工加权的多尺度 TPI vs tpi31 单尺度的走廊区分 AUC（**两站对照**）。

动机
----
修正 nodata 污染后 tpi31 单尺度有效 AUC ≈ 0.669 / 0.715（见 `fix_tpi31_auc.py`），
是唯一跨站方向一致的因子。多尺度 TPI 在同一物理量上叠 15/31/63 m 三个尺度，
若能**两站都超过单尺度**，就换掉合成图的主通道；否则维持 tpi31。

三个候选（都在填充口径上算，方向按原始 tpi 符号）
----
    A  0.5·tpi15 + 0.3·tpi31 + 0.2·tpi63   近尺度主导
    B  (tpi15 + tpi31 + tpi63) / 3          等权
    C  0.2·tpi15 + 0.6·tpi31 + 0.2·tpi63   中尺度主导

判据（用户给定）：**两站都超过修正后的 tpi31** → 换主通道；否则保持 tpi31。

口径与 `fix_tpi31_auc.py` / `build_feature_matrix.py` 一致：
nodata 用最近 valid 像元填充，评估域 = valid，标签 = results/mask_1m_{site}.npy。
同时给出**带符号 AUC**（合成图 H 通道需要知道"越大越像走廊"还是反向）。

用法
----
    python tools/multiscale_tpi_auc.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from fix_tpi31_auc import DATA, RES, fill_nodata, mwu_auc   # noqa: E402

CANDS = (
    ("tpi31 单尺度", {"tpi15": 0.0, "tpi31": 1.0, "tpi63": 0.0}),
    ("多尺度 5/3/2", {"tpi15": 0.5, "tpi31": 0.3, "tpi63": 0.2}),
    ("多尺度 等权", {"tpi15": 1 / 3, "tpi31": 1 / 3, "tpi63": 1 / 3}),
    ("多尺度 2/6/2", {"tpi15": 0.2, "tpi31": 0.6, "tpi63": 0.2}),
)


def main():
    out = {}
    for site in ("西坝村", "杨家坝"):
        d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
        dem = fill_nodata(d["dem_mean"].astype(np.float64), d["valid"] == 1)
        valid = d["valid"] == 1
        label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
        assert np.array_equal(label, d["label"].astype(bool)), "mask 与 npz label 不一致"

        tpi = {f"tpi{k}": dem - uniform_filter(dem, size=k, mode="nearest")
               for k in (15, 31, 63)}

        print(f"\n[{site}] 有效 {int(valid.sum()):,}，走廊 {int(label.sum()):,}")
        print(f"  {'候选':<16s}{'AUC(带符号)':>13s}{'有效AUC':>10s}{'方向':>6s}")
        res = {}
        for name, w in CANDS:
            score = sum(w[k] * tpi[k] for k in w)[valid]
            a = mwu_auc(label[valid], score)
            res[name] = dict(auc=a, eff=max(a, 1 - a), direction="+" if a >= 0.5 else "−",
                             weights=w)
            print(f"  {name:<16s}{a:>13.4f}{max(a, 1 - a):>10.4f}"
                  f"{('+' if a >= 0.5 else '−'):>6s}")
        out[site] = res

    base = {s: out[s]["tpi31 单尺度"]["eff"] for s in out}
    print("\n" + "=" * 74)
    print("判据：候选的两站有效 AUC 是否都 > 修正后 tpi31 单尺度")
    print("=" * 74)
    verdict = {}
    for name, _ in CANDS[1:]:
        wins = [out[s][name]["eff"] > base[s] for s in out]
        verdict[name] = bool(all(wins))
        detail = "  ".join(f"{s} {out[s][name]['eff']:.4f} vs {base[s]:.4f}"
                           f"({'胜' if w else '负'})" for s, w in zip(out, wins))
        print(f"  {name:<16s}{'✅ 换主通道' if all(wins) else '❌ 不换':<12s}{detail}")

    winner = next((n for n, _ in CANDS[1:] if verdict[n]), None)
    print(f"\n  ⇒ {'主通道改为：' + winner if winner else '维持 tpi31 单尺度主通道'}")

    out["_verdict"] = verdict
    out["_winner"] = winner
    p = os.path.join(RES, "multiscale_tpi_auc.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  写出 {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
