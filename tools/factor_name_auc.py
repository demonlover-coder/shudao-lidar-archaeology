# -*- coding: utf-8 -*-
"""给 5 个稀疏主成分命名 + 跨站 AUC + 载荷稳定性 → 挑出"跨站稳定的超级特征"。

判据（用户给定）
----
    AUC 跨站**方向一致** + 载荷余弦相似度 > 0.8  → ✅ 跨站稳定的超级特征，保留
    只有单站成立                              → ❌ 剔除

三个必须讲清的口径
----
1. **符号对齐**：SparsePCA 的分量符号任意（本管线已把"最大载荷取正"，
   但两站最大载荷元素可能互换，实测 PC0 在杨家坝由 `TRI` 领衔、西坝村由 `slope`）。
   故先按 cos 符号把杨家坝的 W/Z 翻正，再比方向与余弦；
   报告同时给 **raw cos**（可能 ≈ −1，代表完全一致）与 **|cos|**（判据用）。
2. **索引对齐 ≠ 物理对齐**：PCA 分量在两站之间会换序（实测"低洼度"是
   西坝村 PC3、杨家坝 PC2）。故除按用户口径的**同序号对照**外，
   另给**最佳匹配表**（每个 PC 在两站间找 |cos| 最大的对手）。
3. **AUC 强度**：方向一致但两站有效 AUC 都不到 0.6 的成分没有判别力，
   本脚本在判据之外单列"强度"一栏（不改用户判据，只提示）。

AUC 口径与 `tools/shudao_auc.py` 完全一致：标签 = results/mask_1m_{site}.npy
（断言与 npz label 逐位一致），评估域 = npz valid == 1，MWU 形式（无 sklearn 依赖），
有效 AUC = max(AUC, 1−AUC)。

输出
----
    results/factor_name_auc.json
    figures/shudao_pc_rgb_{site}.png    前 3 个 PC → RGB 合成图（白线 = SHP 走廊真值）

用法
----
    python tools/factor_name_auc.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from build_feature_matrix import DATA, FACTORS, NAME_HINT   # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
SITES = ["西坝村", "杨家坝"]
COS_MIN = 0.80      # 载荷余弦阈值
EF_WEAK = 0.60      # 有效 AUC 强度提示阈值


def roc_auc(score: np.ndarray, pos: np.ndarray) -> float:
    """Mann–Whitney U 形式 ROC-AUC（score 越大越倾向正类）；并列取平均秩。"""
    score = np.asarray(score, dtype=np.float64)
    pos = np.asarray(pos, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int(pos.size - pos.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(score, method="average")
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def name_of(row: np.ndarray, k: int = 2) -> str:
    """按 |载荷| 前 k 个因子的物理含义拼名字。"""
    idx = np.argsort(-np.abs(row))[:k]
    return "＋".join(NAME_HINT[FACTORS[j]] for j in idx)


def formula_of(row: np.ndarray, k: int = 3) -> str:
    idx = np.argsort(-np.abs(row))[:k]
    return " + ".join(f"{row[j]:+.2f}·{FACTORS[j]}" for j in idx)


def load() -> dict:
    """读两站 sparse_pca json + pc_map + 标签，返回打包 dict。"""
    out = {}
    for s in SITES:
        j = json.load(open(os.path.join(RES, f"sparse_pca_{s}.json"), encoding="utf-8"))
        npz = np.load(os.path.join(DATA, f"{s}_1m.npz"))
        valid = npz["valid"] == 1
        lab = np.load(os.path.join(RES, f"mask_1m_{s}.npy")).astype(bool)
        assert np.array_equal(lab, npz["label"].astype(bool)), "mask 与 npz label 不一致"
        assert not (lab & ~valid).any(), "存在 valid 之外的走廊像元"
        pc = np.load(os.path.join(DATA, f"pc_map_{s}.npy")).astype(np.float64)
        Z = np.stack([pc[i][valid] for i in range(pc.shape[0])], axis=1)
        out[s] = dict(W=np.asarray(j["W"], dtype=np.float64),
                      var_ratio=np.asarray(j["var_ratio"]),
                      scale_mode=j["scale_mode"], Z=Z,
                      y=lab[valid].astype(bool), valid=valid,
                      label_map=lab)
    return out


def main():
    d = load()
    W1, W2 = d[SITES[0]]["W"], d[SITES[1]]["W"]
    n_pc = W1.shape[0]

    # ---------------------------------------------------- 同序号对照 ----
    print("=" * 120)
    print("一、载荷矩阵对照（同序号）与符号对齐")
    print("=" * 120)
    hdr = (f"  {'PC':<5s}{'raw cos':>9s}{'|cos|':>8s}{'AUC(' + SITES[0] + ')':>12s}"
           f"{'AUC(' + SITES[1] + ')':>12s}{'对齐后AUC(杨)':>14s}{'方向一致':>10s}  命名")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    for i in range(n_pc):
        c_raw = cos(W1[i], W2[i])
        W2a = W2[i] * (1.0 if c_raw >= 0 else -1.0)     # 符号对齐到西坝村
        Wm = 0.5 * (W1[i] + W2a)                        # 对齐后均值载荷（命名用）
        a1 = roc_auc(d[SITES[0]]["Z"][:, i], d[SITES[0]]["y"])
        a2raw = roc_auc(d[SITES[1]]["Z"][:, i], d[SITES[1]]["y"])
        a2 = a2raw if c_raw >= 0 else 1.0 - a2raw       # 对齐后
        e1, e2 = max(a1, 1 - a1), max(a2, 1 - a2)
        same = (a1 - 0.5) * (a2 - 0.5) > 0
        rows.append(dict(pc=i, cos_raw=c_raw, cos_abs=abs(c_raw),
                         auc_s1=a1, auc_s2_raw=a2raw, auc_s2_aligned=a2,
                         eff_s1=e1, eff_s2=e2, dir_s1="+" if a1 >= 0.5 else "−",
                         dir_s2="+" if a2 >= 0.5 else "−", dir_same=bool(same),
                         name=name_of(Wm), formula=formula_of(Wm),
                         formula_s1=formula_of(W1[i]), formula_s2=formula_of(W2[i]),
                         W_mean=Wm.tolist(), var_ratio_s1=float(d[SITES[0]]["var_ratio"][i]),
                         var_ratio_s2=float(d[SITES[1]]["var_ratio"][i])))
        print(f"  PC{i:<3d}{c_raw:>+9.3f}{abs(c_raw):>8.3f}{a1:>12.4f}{a2raw:>12.4f}"
              f"{a2:>14.4f}{('是' if same else '否'):>10s}  {rows[-1]['name']}")

    print("\n  同序号载荷（前 3）")
    for r in rows:
        print(f"    PC{r['pc']}: 西 {r['formula_s1']}")
        print(f"          杨 {r['formula_s2']}")

    # ---------------------------------------------------- 最佳匹配 ----
    print("\n" + "=" * 120)
    print("二、最佳匹配（不看序号，在两站之间找 |cos| 最大的对手，贪心一一配对）")
    print("=" * 120)
    C = W1 @ W2.T                                  # 行已单位化 ⇒ 即余弦矩阵
    order = np.argsort(-np.abs(C), axis=None)
    flat = np.unravel_index(order, C.shape)
    left, used, pairs = set(range(n_pc)), set(), []
    for i, j in zip(flat[0], flat[1]):
        i, j = int(i), int(j)
        if i in left and j not in used:
            pairs.append((i, j))
            left.discard(i)
            used.add(j)
        if not left:
            break
    print(f"  {'西坝村':<8s}{'→ 杨家坝':>10s}{'cos(原始)':>12s}{'|cos|':>8s}"
          f"{'AUC西':>9s}{'有效':>8s}{'对齐后AUC杨':>13s}{'有效':>8s}"
          f"{'方向一致':>9s}{'强度':>7s}{'判定':>16s}  内容")
    print("  " + "-" * 132)
    matched = []
    for i, j in pairs:
        c_raw = cos(W1[i], W2[j])
        a1 = roc_auc(d[SITES[0]]["Z"][:, i], d[SITES[0]]["y"])
        a2r = roc_auc(d[SITES[1]]["Z"][:, j], d[SITES[1]]["y"])
        a2 = a2r if c_raw >= 0 else 1.0 - a2r
        e1, e2 = max(a1, 1 - a1), max(a2, 1 - a2)
        same = (a1 - 0.5) * (a2 - 0.5) > 0
        strong = e1 > EF_WEAK and e2 > EF_WEAK
        ok = same and abs(c_raw) > COS_MIN
        nm = name_of(0.5 * (W1[i] + (1 if c_raw >= 0 else -1) * W2[j]))
        matched.append(dict(s1_pc=i, s2_pc=j, cos_raw=c_raw, cos_abs=abs(c_raw),
                            auc_s1=a1, auc_s2_raw=a2r, auc_s2_aligned=a2,
                            eff_s1=e1, eff_s2=e2, dir_same=bool(same),
                            strength_ok=bool(strong), name=nm,
                            verdict="保留(跨站稳定)" if ok else "剔除(跨站不稳)"))
        print(f"  西PC{i:<4d}{'杨PC' + str(j):>10s}{c_raw:>+12.3f}{abs(c_raw):>8.3f}"
              f"{a1:>9.4f}{e1:>8.4f}{a2:>13.4f}{e2:>8.4f}"
              f"{('是' if same else '否'):>9s}{('达标' if strong else '偏弱'):>7s}"
              f"{matched[-1]['verdict']:>16s}  {nm}")
    keep_m = [f"西PC{m['s1_pc']}=杨PC{m['s2_pc']}" for m in matched
              if m["verdict"].startswith("保留")]
    print(f"\n  配对判据同用户口径：方向一致 且 |cos|>{COS_MIN} → 保留。"
          f"保留 = {keep_m if keep_m else '无'}")

    # ---------------------------------------------------- AUC 表 ----
    print("\n" + "=" * 120)
    print("三、跨站 AUC 表（标签 = SHP 走廊；评估域 = valid）")
    print("=" * 120)
    print(f"  {'PC':<5s}{'命名':<22s}{'AUC西':>9s}{'有效':>8s}{'方向':>5s}"
          f"{'AUC杨(原)':>11s}{'对齐后':>9s}{'有效':>8s}{'方向':>5s}{'|cos|':>8s}"
          f"{'方向一致':>9s}{'强度':>7s}{'判定':>14s}")
    print("  " + "-" * 116)
    keep, drop = [], []
    for r in rows:
        strong = r["eff_s1"] > EF_WEAK and r["eff_s2"] > EF_WEAK
        ok = r["dir_same"] and r["cos_abs"] > COS_MIN
        r["strength_ok"] = bool(strong)
        r["verdict"] = ("保留(跨站稳定)" if ok else "剔除(跨站不稳)")
        (keep if ok else drop).append(r["pc"])
        print(f"  PC{r['pc']:<3d}{r['name']:<22s}{r['auc_s1']:>9.4f}{r['eff_s1']:>8.4f}"
              f"{r['dir_s1']:>5s}{r['auc_s2_raw']:>11.4f}{r['auc_s2_aligned']:>9.4f}"
              f"{r['eff_s2']:>8.4f}{r['dir_s2']:>5s}{r['cos_abs']:>8.3f}"
              f"{('是' if r['dir_same'] else '否'):>9s}{('达标' if strong else '偏弱'):>7s}"
              f"{r['verdict']:>14s}")
    print(f"\n  判据：方向一致 且 |cos|>{COS_MIN} → 保留。"
          f"保留 = {keep if keep else '无'}；剔除 = {drop if drop else '无'}")

    # ---------------------------------------------------- RGB 合成图 ----
    make_figures(d)

    out = dict(thresholds=dict(cos_min=COS_MIN, eff_weak=EF_WEAK),
               sites=SITES, rows=rows, matched=matched, cos_matrix=C.tolist(),
               keep=keep, drop=drop, keep_matched=keep_m)
    p = os.path.join(RES, "factor_name_auc.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {os.path.relpath(p, ROOT)}")


def make_figures(d):
    """前 3 个 PC → RGB 合成图（valid 内分位映射）；白线 = SHP 走廊真值。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(FIG, exist_ok=True)
    j = {s: json.load(open(os.path.join(RES, f"sparse_pca_{s}.json"), encoding="utf-8"))
         for s in SITES}

    for s in SITES:
        dd = d[s]
        pc = np.load(os.path.join(DATA, f"pc_map_{s}.npy")).astype(np.float64)
        valid, lab = dd["valid"], dd["label_map"]
        H, W = valid.shape
        rgb = np.zeros((H, W, 3))
        for ch in range(3):
            v = pc[ch]
            r = np.zeros(H * W)
            r[valid.ravel()] = rankdata(v[valid], method="average")
            r[valid.ravel()] = (r[valid.ravel()] - 1.0) / max(valid.sum() - 1.0, 1.0)
            rgb[..., ch] = r.reshape(H, W)
        rgb[~valid] = 1.0
        names = [name_of(np.asarray(j[s]["W"][i])) for i in range(3)]

        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_axes([0.02, 0.03, 0.72, 0.90])
        ax.imshow(rgb)
        if lab.any():
            ax.contour(lab.astype(float), levels=[0.5], colors="white", linewidths=1.0)
        ax.set_title(f"稀疏 PCA 超级特征 RGB 合成图（{s}）\n"
                     f"R={names[0]}　G={names[1]}　B={names[2]}（各 valid 内分位映射）",
                     fontsize=13)
        ax.axis("off")

        ax2 = fig.add_axes([0.76, 0.03, 0.23, 0.90])
        ax2.axis("off")
        txt = [f"[{s}] 前 3 个 PC 载荷（对齐后均值，前 3 项）"]
        for i in range(3):
            txt.append(f"  PC{i}  {names[i]}")
            txt.append(f"       {formula_of(np.asarray(j[s]['W'][i]))}")
            txt.append(f"       方差贡献 {j[s]['var_ratio'][i]:.3f}")
        txt += ["", "判据：方向一致 且 |cos|>0.80 → 保留", "白色轮廓 = SHP 走廊真值",
                "灰白 = valid 之外（nodata）"]
        ax2.text(0, 1.0, "\n".join(txt), fontsize=10, va="top", ha="left",
                 bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.7"))
        ax2.legend(handles=[Line2D([], [], color="white", lw=2, label="SHP 走廊真值")],
                   loc="lower left", fontsize=10, frameon=False)

        p = os.path.join(FIG, f"shudao_pc_rgb_{s}.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  写出 {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
