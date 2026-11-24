# -*- coding: utf-8 -*-
"""第 3 步先验：验证 tpi31 在**真值走廊骨架**上的分布，判断路径提取是否可行。

背景
----
走廊是连续线状、只占 ~1% 像元，逐像素二分类在跨站（全正塌缩）与同站（全负塌缩）
两种设定下都失败（见 results/train_same_site_ss_base16.json）。下一步改把走廊当"路径"：
以 tpi31（唯一跨站稳定因子）为代价，用 Dijkstra 求端点间最小代价路径。

但有个前提必须先验：**如果走廊中轴上的 tpi31 不是持续负值，Dijkstra 就会沿着
"同样低洼但不在走廊上"的地方走**，输出一条"看着像路、完全不贴真值"的路径。
本脚本就量这件事：真值骨架上的 tpi31 是否稳定偏负、比背景低多少。

判据（用户给定）
----
    骨架 tpi31 中位 < −0.5 m  且  <0 比例 > 80%  → ✅ 可做 Dijkstra
    骨架 tpi31 中位 ∈ [−0.5, 0)                   → ⚠️ 弱信号，代价图权重需调
    骨架 tpi31 中位 ≈ 0 或 > 0                     → ❌ tpi31 不足，换代价图

另报：连通域数（决定要跑几次 Dijkstra 与是否需局部切块）与骨架端点数
（≈连通域×2 直接配对；远多于此需先合并近邻端点）。

附加：**候选代价通道排名**（迭代 2 依据）
----
若 tpi31 判据落在 ⚠️/❌（极性不纯或信号弱），需换／加代价通道。本脚本顺带把
`feature_matrix_{site}.npy` 的 20 个因子逐个摊回栅格，量"真值骨架 vs 背景"的
MWU-AUC 与极性纯度（骨架中位落在背景中位的哪一侧的比例），按判别力 |AUC−0.5|
排序，直接给出迭代 2 该用哪个通道、以及是"越低越像走廊"还是"越高越像走廊"。

用法：
    python tools/path_feasibility.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import label as cc_label
from scipy.stats import rankdata
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
SITES = ("西坝村", "杨家坝")


def mwu_auc(pos, neg) -> float:
    """Mann-Whitney U 的 AUC：P(pos > neg)。>0.5 表示骨架偏高，<0.5 表示骨架偏低。"""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    u = r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def rank_channels(site, valid, skel, label):
    """把 20 因子摊回栅格，逐个算"骨架 vs 背景"判别力，返回排序后的列表。"""
    npy = os.path.join(DATA, f"feature_matrix_{site}.npy")
    mj = os.path.join(DATA, f"feature_matrix_{site}.meta.json")
    if not (os.path.exists(npy) and os.path.exists(mj)):
        return None
    X = np.load(npy)
    names = json.load(open(mj, encoding="utf-8"))["factors"]
    rows = np.flatnonzero(valid.ravel())
    m_skel = skel & valid
    m_bg = valid & ~label
    out = []
    for k, name in enumerate(names):
        g = np.full(valid.size, np.nan, dtype=np.float64)
        g[rows] = X[:, k]
        g = g.reshape(valid.shape)
        pos, neg = g[m_skel], g[m_bg]
        auc = mwu_auc(pos, neg)
        if not np.isfinite(auc):
            continue
        med_s, med_b = float(np.median(pos)), float(np.median(neg))
        low_better = auc < 0.5                      # 骨架偏低 → 直接当"越低越像走廊"的代价
        purity = float((pos < med_b).mean() if low_better else (pos > med_b).mean())
        out.append(dict(factor=name, auc=auc, power=abs(auc - 0.5),
                        direction="越低越像走廊" if low_better else "越高越像走廊",
                        skel_median=med_s, bg_median=med_b, purity=purity))
    return sorted(out, key=lambda d: -d["power"])


def verdict_of(median_m: float, frac_neg: float) -> str:
    """按用户判据表映射；表三行不穷尽，故补一条"中位强负但极性不纯"的中间档。"""
    if median_m < -0.5 and frac_neg > 0.8:
        return "✅ 可做 Dijkstra"
    if median_m < -0.5:
        return f"⚠️ 中位强负但 <0 占比仅 {frac_neg:.1%} (<80%)，极性不纯，代价图需调"
    if median_m < 0:
        return "⚠️ 弱信号，代价图权重需调"
    return "❌ tpi31 不足以驱动路径提取，要换代价图"


def main():
    out = {}
    for site in SITES:
        label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
        skel = skeletonize(label)
        # scipy.ndimage.label 返回 (labeled_array, num_features)
        cc, n_cc = cc_label(label, structure=np.ones((3, 3), int))

        terrain = np.load(os.path.join(ROOT, "data", f"p2_terrain_{site}.npy"))  # [8,H,W]
        tpi31 = terrain[7]
        valid = np.load(os.path.join(ROOT, "data", f"{site}_1m.npz"))["valid"] == 1

        on_skel = tpi31[skel & valid]
        bg = tpi31[valid & ~label]
        med_skel = float(np.median(on_skel))
        med_bg = float(np.median(bg))
        frac_neg = float((on_skel < 0).mean())

        nb = convolve(skel.astype(int), np.ones((3, 3)), mode="constant")
        endpoints = skel & (nb == 2)          # 骨架点自身 + 恰 1 个邻居
        n_ep = int(endpoints.sum())

        print(f"[{site}] 走廊 {int(label.sum())} px → 骨架 {int(skel.sum())} px "
              f"(比 {skel.sum() / max(label.sum(), 1):.2%})")
        print(f"  连通域数 = {n_cc}")
        print(f"  tpi31 骨架中位 = {med_skel:+.2f} m")
        print(f"  tpi31 背景中位 = {med_bg:+.2f} m   (骨架−背景 = {med_skel - med_bg:+.2f} m)")
        print(f"  骨架 tpi31 < 0 的比例 = {frac_neg:.1%}")
        print(f"  骨架端点 = {n_ep}（每条路径 2 个；连通域×2 = {n_cc * 2}）")
        v = verdict_of(med_skel, frac_neg)
        print(f"  判据：{v}")

        rank = rank_channels(site, valid, skel, label)
        if rank:
            print(f"  候选代价通道排名（骨架 vs 背景，|AUC−0.5| 降序，前 8）：")
            print(f"    {'因子':<14s}{'AUC':>8s}  {'方向':<12s}{'骨架中位':>11s}"
                  f"{'背景中位':>11s}{'纯度':>8s}")
            for d in rank[:8]:
                print(f"    {d['factor']:<14s}{d['auc']:>8.4f}  {d['direction']:<12s}"
                      f"{d['skel_median']:>11.3f}{d['bg_median']:>11.3f}{d['purity']:>8.1%}")

        out[site] = dict(label_px=int(label.sum()), skel_px=int(skel.sum()),
                         skeleton_ratio=float(skel.sum() / max(label.sum(), 1)),
                         n_cc=int(n_cc), n_endpoints=n_ep,
                         tpi31_skel_median=med_skel, tpi31_bg_median=med_bg,
                         tpi31_skel_minus_bg=med_skel - med_bg,
                         frac_neg_on_skel=frac_neg, verdict=v,
                         cost_channel_rank=rank)

    jp = os.path.join(RES, "path_feasibility.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {jp}")


if __name__ == "__main__":
    main()
