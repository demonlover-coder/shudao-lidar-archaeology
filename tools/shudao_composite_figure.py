# -*- coding: utf-8 -*-
"""蜀道走廊合成图（**主通道 = tpi31 单通道**，跨站稳定因子）。

设计（经两轮实测确定，见 results/shudao_auc*.json）
----
    色相 H = tpi31（**负向**：走廊 tpi31 偏低 → 越红越像走廊，越蓝越不像）
             H 用 **valid 内分位（秩）映射**铺开，而非 P2–P98 线性——
             score=−tpi31 左尾很长，线性映射会把中位数抬到红端、色相挤在一起
    明度 V = 高程（robust P2–P98 归一化，深=低、亮=高）
    饱和 S = 置信度（P2 **地面点**测点覆盖掩码；无覆盖 → 去饱和=灰，即"半透明/不可信")
    白色轮廓 = SHP 走廊真值（results/mask_1m_{site}.npy）
    细虚线   = 有测点覆盖区边界
    图例     = tpi31 AUC / 组合 SSI AUC（含"杨站退化、不采用"说明）

为何不用 SSI 作主通道（实测）
----
  等权 rank 组合（tpi31+slope+roughness，方向对齐）aligned AUC：
  **西坝村 0.8327（≥0.80 有增益）/ 杨家坝 0.6072（< tpi31 单因子 0.7148，退化）**。
  组合增益只在 slope/roughness 有效的西坝村成立；杨家坝 slope/roughness 近乎随机
  （AUC 0.505/0.514），等权相加稀释 tpi31 → **固定组合不可跨站迁移**。
  故主通道取**跨站方向一致的 tpi31 单因子**。

⚠ tpi31 的 AUC 已修正（`tools/fix_tpi31_auc.py`，results/fix_tpi31_auc.json）
----
  旧口径（未填 nodata，valid 最外一圈含 −9999 污染）0.7285 / 0.7711
  → **填充口径 0.6691（西）/ 0.7148（杨）**，虚高约 0.058。
  本图 H 通道也用填充后的 DEM 计算，**与图例 AUC 同口径**。

为何不换多尺度 TPI（`tools/multiscale_tpi_auc.py`）
----
  5/3/2、等权、2/6/2 三候选：**杨家坝全部超 tpi31（0.743~0.770），西坝村全部略负
  （0.645~0.651）** → 不满足"两站都超过"，**维持 tpi31 单尺度主通道**。

tpi31 与现有因子不冗余（|ρ|<0.35），仅与 tpi15 冗余（ρ=0.74）——两者是同一信号两尺度。

用法
----
    python tools/shudao_composite_figure.py --site 西坝村
    python tools/shudao_composite_figure.py --site 杨家坝
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

import p2_dem as P                                        # noqa: E402
from fix_tpi31_auc import fill_nodata                     # noqa: E402
from p2_export_all import project_covered                 # noqa: E402
from p2_terrain import CHANNELS, terrain_stack            # noqa: E402
from shudao_auc import roc_auc                            # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")


def _robust_norm(x, mask, q=(2.0, 98.0)):
    lo, hi = np.percentile(x[mask], q)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _rank_norm(x, mask):
    """valid 内分位（秩）归一化到 [0,1]：色相均匀铺开，不受长尾挤压。"""
    out = np.full(x.shape, 0.5, dtype=np.float64)
    r = rankdata(x[mask], method="average")
    out[mask] = (r - 1.0) / max(len(r) - 1.0, 1.0)
    return out


def load_auc(site):
    """读 AUC 结果：tpi31 单因子（**填充修正口径**）+ 组合 SSI。

    tpi31 取 `results/fix_tpi31_auc.json` 的 `eff_new`（nodata 用最近 valid 邻居
    填充后再算，无边界污染），与图中 H 通道**同一口径**；
    旧的 `shudao_auc_{site}.json`（未填充）含约 0.058 的虚高，仅作退化兜底。
    """
    try:
        fix = json.load(open(os.path.join(RES, "fix_tpi31_auc.json"), encoding="utf-8"))
        tpi = fix[site]["eff_new"]
    except Exception:
        single = json.load(open(os.path.join(RES, f"shudao_auc_{site}.json"),
                                 encoding="utf-8"))
        tpi = next(r["auc_eff"] for r in single["factors"] if r["name"] == "tpi31")
    combo = json.load(open(os.path.join(RES, "shudao_auc_combo.json"), encoding="utf-8"))
    cs = next(s for s in combo["sites"] if s["site"] == site)
    ssi = next(r for r in cs["combos"] if r["combo"] == "tpi31+slope+roughness")
    return dict(tpi31_auc=tpi, ssi_auc=ssi["auc_aligned"], verdict=cs["verdict"])


def openness_grid(site):
    """openness 取自 feature_matrix_{site}.npy 第 14 列（已按最近 valid 邻居填 nodata），摊回栅格。

    openness 不在 p2_terrain.CHANNELS 里（那是 8 通道地形栈），只存在于 20 因子特征池
    `build_feature_matrix.FACTORS`（index 14），故此处单独取列。
    """
    meta = json.load(open(os.path.join(DATA, f"feature_matrix_{site}.meta.json"),
                          encoding="utf-8"))
    k = meta["factors"].index("openness")
    X = np.load(os.path.join(DATA, f"feature_matrix_{site}.npy"))
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    g = np.full(valid.size, np.nan, dtype=np.float64)
    g[np.flatnonzero(valid.ravel())] = X[:, k]
    return g.reshape(valid.shape)


def main():
    ap = argparse.ArgumentParser(description="蜀道走廊合成图（H=tpi31/openness / V=高程 / S=覆盖）")
    ap.add_argument("--site", default="西坝村", choices=["西坝村", "杨家坝"])
    ap.add_argument("--h-channel", default="tpi31", choices=["tpi31", "openness"],
                    help="H（色相）通道：tpi31（像素级主通道，默认）/ openness（路径级主通道）")
    a = ap.parse_args()
    site = a.site

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
    from matplotlib.lines import Line2D
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(FIG, exist_ok=True)

    npz = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    dem = npz["dem_mean"].astype(np.float64)
    valid = npz["valid"] == 1
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    H, W = dem.shape

    x0, y1, wm, hm, gsd, _ = P.dem_window(site)
    print(f"[{site}] 窗口 x0={x0:.1f} y1={y1:.1f} {wm:.0f}×{hm:.0f} m @ {gsd} m")
    cover = project_covered(site, x0, y1, gsd, H, W, dilate=1)   # 地面点覆盖（3×3 膨胀）
    auc = load_auc(site)

    # H 通道（负向 → 取反得"走廊倾向"得分；tpi31 与 openness 都是"值越低越像走廊"）
    # ⚠ tpi31 必须在**填充后**的 DEM 上算：未填充时 valid 最外一圈含 −9999，
    #   `uniform_filter(mode="nearest")` 会把该圈 tpi31 推到极大正值（边界污染，
    #   D12 #13），使 AUC 虚高约 0.058（0.7285→0.6691 西 / 0.7711→0.7148 杨）。
    #   图中 H 通道与图例 AUC 必须同口径。
    dem_f = fill_nodata(dem, valid)
    if a.h_channel == "tpi31":
        chan = terrain_stack(dem_f, gsd=gsd)[CHANNELS.index("tpi31")]
        ch_auc_eff = auc["tpi31_auc"]
        ch_desc = "tpi31（跨站方向一致，像素级主通道）"
    else:
        chan = openness_grid(site)
        m = valid & np.isfinite(chan)
        raw = roc_auc(chan[m], (label & valid)[m])
        ch_auc_eff = max(raw, 1.0 - raw)
        chan = np.nan_to_num(chan, nan=0.0)
        ch_desc = "openness（路径提取最优代价，路径级主通道）"
    score = -chan
    s_norm = _rank_norm(score, valid)          # 分位映射：色相均匀铺开

    # HSV 合成
    hsv = np.zeros((H, W, 3))
    hsv[..., 0] = (1.0 - s_norm) * 0.66        # 0=红(像走廊) … 0.66=蓝(不像)
    hsv[..., 1] = cover.astype(np.float64)     # 饱和 = 置信度（地面点覆盖）
    hsv[..., 2] = 0.15 + 0.85 * _robust_norm(dem, valid)   # 明度 = 高程
    rgb = hsv_to_rgb(hsv)
    rgb[~valid] = 1.0                          # valid 之外留白

    # 色相图例（走廊倾向 → 颜色）
    cols = [hsv_to_rgb([(1.0 - k / 255) * 0.66, 1.0, 1.0]) for k in range(256)]
    cmap = LinearSegmentedColormap.from_list("tpi31_score", cols)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.15], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb)
    # 白色轮廓 = SHP 走廊真值
    if label.any():
        ax.contour(label.astype(float), levels=[0.5], colors="white", linewidths=0.9)
    # 细虚线 = 有测点覆盖区边界
    ax.contour(cover.astype(float), levels=[0.5], colors="#ffcc00",
               linewidths=0.6, linestyles="dashed")
    ax.set_title(f"蜀道走廊合成图（{site}，{wm:.0f}×{hm:.0f} m @ {gsd} m）\n"
                 f"H={a.h_channel} 走廊倾向（红=像走廊）  V=高程（亮=高）  S=地面点覆盖（灰=无测点不可信）",
                 fontsize=13)
    ax.axis("off")

    # 右侧图例/注释
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cax = ax2.inset_axes([0.06, 0.86, 0.86, 0.045])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(f"{a.h_channel} 走廊倾向（valid 内分位；左蓝=不像 → 右红=像）", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    handles = [
        Line2D([], [], color="white", lw=2, label="白色轮廓 = SHP 走廊真值"),
        Line2D([], [], color="#ffcc00", lw=1.2, ls="--", label="黄虚线 = 有测点覆盖区边界"),
        Line2D([], [], color="0.55", lw=6, label="灰色/去饱和 = 无地面点（不可信）"),
    ]
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 0.78),
               fontsize=10, frameon=False)

    if a.h_channel == "tpi31":
        txt = (
            "单因子 AUC（主通道 tpi31，**填充修正口径**）\n"
            f"  tpi31 = {auc['tpi31_auc']:.4f}   ← 本图主通道（跨站方向一致）\n"
            "  修正：原 0.7285/0.7711 含 nodata 边界污染虚高约 0.058\n\n"
            "组合 SSI = tpi31+slope+roughness\n"
            f"  aligned AUC = {auc['ssi_auc']:.4f}\n"
            f"  判定：{auc['verdict']}\n"
            "  → 组合仅在西坝村成立，杨站退化，故不作主通道\n\n"
            "多尺度 TPI 5/3/2、等权、2/6/2：杨站胜、西站负（不一致）→ 不换主通道\n\n"
            "冗余：tpi31 与其余因子 |ρ|<0.35，仅与 tpi15 冗余（ρ≈0.74）"
        )
    else:
        txt = (
            "H = openness（天空开放度；**路径级**主通道）\n"
            f"  有效 AUC = {ch_auc_eff:.4f}（走廊 vs 背景，valid 内，越低越像）\n"
            "  低开放度处是窄条状沟谷/切谷，与走廊同构\n\n"
            "为何路径提取选 openness 而非 tpi31（D12 #22）\n"
            "  路径 IoU：西坝村 0.126 vs 0.024（**5.1×**）\n"
            "  杨家坝 0.062 vs 0.065（相当）\n"
            "  像素可分性 ≠ 路径可分性：\n"
            "    tpi63 逐像素 AUC 最高，但 Dijkstra 借大盆地抄近道 → 路径最差\n"
            "    openness 低值区拓扑为细线，与走廊同构 → 路径最优\n\n"
            "分工：像素级主通道 = tpi31；路径级主通道 = openness"
        )
    ax2.text(0.0, 0.66, txt, fontsize=10, va="top", ha="left",
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.7"))
    ax2.text(0.0, 0.30,
             f"说明：H 由 {a.h_channel} 负向、valid 内分位映射；\n"
             "S=0（灰）仅表示该处无地面测点、\n"
             f"P2 输出不可信，不代表 {a.h_channel} 值本身。",
             fontsize=9, va="top", ha="left", color="0.35")

    tag = "" if a.h_channel == "tpi31" else f"_{a.h_channel}"
    out = os.path.join(FIG, f"shudao_composite{tag}_{site}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"写出 {os.path.relpath(out, ROOT)}")
    print(f"  H 通道={a.h_channel} 有效AUC={ch_auc_eff:.4f}"
          + (f"，组合 SSI AUC={auc['ssi_auc']:.4f}（{auc['verdict']}）"
             if a.h_channel == "tpi31" else ""))


if __name__ == "__main__":
    main()
