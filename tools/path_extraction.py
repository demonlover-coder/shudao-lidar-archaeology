# -*- coding: utf-8 -*-
"""第 3 步 · 迭代 1：以 tpi31 为代价的最朴素 Dijkstra 路径提取（无学习）。

思路
----
走廊是连续线状目标，逐像素二分类在跨站（全正塌缩）与同站（全负塌缩）都失败
（results/train_same_site_ss_base16.json）。改把走廊当"路径"：在 tpi31（唯一跨站
稳定因子，见 results/path_feasibility.json）代价场上求端点间最小代价路径。

域与端点
----
* 代价域 = `valid`（其余像元代价 = ∞，不可走）。
* 起终点默认应**由调用方给出坐标**（踏勘输入）。旧行为"取真值走廊骨架端点"属于
  SHP 派生先验，仅作 L3 泄漏对照保留，需显式开启。
* 搜索域默认全图，`roi_radius` 不生效；限制在真值走廊附近需显式 `allow_gt_roi=True`。
* 骨架端点 > 2 时先合并 10px 内近邻，再取欧氏距离最大的一对。

代价图
----
    neg  = max(0, -tpi31)                       # 越负（越像沟谷）代价越低
    cost = clip((neg - P5) / (P95 - P5), 0, 1) + eps
用负值部分的 **P5~P95 分位**归一化（不用 min-max，避免极值主导）；P50 作下限会把
一半负值压成 0 代价、形成零代价平台让 Dijkstra 乱走，故下限取 P5。
`--norm minmax` 可切回全量 min-max 对照。

`--cost` 选代价通道（见 COST_PRESETS）：默认 `tpi31`（迭代 1，可复现）；迭代 2 用
`tpi63` / `cliff` / `open` / `tpi31+cliff` / `combo`，通道选型由
`results/path_feasibility.json` 的 `cost_channel_rank`（骨架 vs 背景 |AUC−0.5|）决定。

Dijkstra
----
8 邻域栅格图，对角线 √2 加权（否则路径偏斜）；边权 = 步长 × 两端点代价均值，
用 `scipy.sparse.csgraph.dijkstra`（C 速度，1M 节点可控）。

评估
----
路径膨胀 `--dilate-radius` 像素成走廊 mask（默认 3px ≈ 6m 宽，对应走廊中位宽 6.95m），
与 SHP 真值对评：D10 口径 IoU@0.5 + 复用 eval_linear_metrics 的中心线指标（LR/LP/F1/
中心线距离/拓扑/端点）。

用法
----
    python tools/path_extraction.py                       # 两站，膨胀 3px
    python tools/path_extraction.py --dilate-radius 2
    python tools/path_extraction.py --roi-radius 60       # 限定在真值走廊周边 60px 内求路径
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label as cc_label, uniform_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from eval_linear_metrics import linear_metrics, window_mask       # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
SITES = ("杨家坝", "西坝村")
THRESH = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
TOL = 2.0
SQ2 = float(np.sqrt(2.0))
S8 = np.ones((3, 3), dtype=np.int32)
NB = np.ones((3, 3), dtype=np.int8)
DFS_GATE = 60.0          # 方向先验的置信门控分位（h1_deg ≥ 域内 P60）
# 8 邻域方向（罗盘序：N, NE, E, SE, S, SW, W, NW），供状态增广图的转向表使用
NB8 = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


# ------------------------------------------------------------ 几何工具 ----
def find_endpoints(skel):
    from scipy.ndimage import convolve
    nb = convolve(skel.astype(np.int8), NB, mode="constant") - skel.astype(np.int8)
    return skel & (nb == 1)


def merge_and_pick_pair(pts, merge_px=10.0):
    """pts: (N,2) int 数组。合并近邻后取欧氏距离最大的一对，返回 (start, end)。"""
    P = pts.astype(np.float64)
    keep, used = [], np.zeros(len(P), dtype=bool)
    for i in range(len(P)):
        if used[i]:
            continue
        grp = np.where(np.linalg.norm(P - P[i], axis=1) <= merge_px)[0]
        used[grp] = True
        keep.append(P[grp].mean(axis=0))
    K = np.array(keep if len(keep) > 1 else [P[0], P[-1]])
    if len(K) == 1:
        return K[0], K[0]
    d = np.linalg.norm(K[:, None, :] - K[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return K[i], K[j]


def snap_to_domain(pt, dom):
    """端点若不在代价域内，吸附到最近域内像元。"""
    y, x = int(round(pt[0])), int(round(pt[1]))
    if 0 <= y < dom.shape[0] and 0 <= x < dom.shape[1] and dom[y, x]:
        return y, x
    idx = distance_transform_edt(~dom, return_distances=False, return_indices=True)
    return int(idx[0][y, x]), int(idx[1][y, x])


# ------------------------------------------------------------ 代价图 ----
# --cost 预设：token 形如 "<factor>_low"（该通道值越低越像走廊，故代价越低）→ 权重。
# 通道来源：tpi31 取 p2_terrain_{site}.npy[7]（与迭代 1 完全同源，可复现）；
# 其余取 feature_matrix_{site}.npy 的 20 因子列。
# 选型依据 results/path_feasibility.json 的 cost_channel_rank（骨架 vs 背景 |AUC−0.5|）：
#   · tpi63      两站方向一致，杨家坝最强（AUC 0.171，纯度 98.9%）
#   · cliff_prox 两站方向一致，西坝村强（0.182，纯度 99.7%）
#   · openness   两站方向一致，西坝村最强（0.153，纯度 99.9%）
COST_PRESETS = {
    "tpi31":       [("tpi31_low", 1.0)],
    "tpi63":       [("tpi63_low", 1.0)],
    "cliff":       [("cliff_prox_low", 1.0)],
    "open":        [("openness_low", 1.0)],
    "tpi31+cliff": [("tpi31_low", 0.6), ("cliff_prox_low", 0.4)],
    "combo":       [("tpi63_low", 0.4), ("cliff_prox_low", 0.3), ("openness_low", 0.3)],
    "open+cliff":  [("openness_low", 0.5), ("cliff_prox_low", 0.5)],
    "open+tpi31":  [("openness_low", 0.5), ("tpi31_low", 0.5)],
    # 约束 1：DOM 彩度组合——两信号各自**秩归一化到 [0,1]** 后**等权**相加，再除最大值。
    # 无超参、可解释；若等权不行，说明信号互补性不足，而非权重没调好。
    "open+exg":    [("openness_low", 1.0), ("exg_high", 1.0)],
    "open+sat":    [("openness_low", 1.0), ("sat_high", 1.0)],
}
# 走"秩归一化 + 等权"路径的代价预设（其余预设沿用原 P5~P95 线性归一化，逐位不变）。
RANK_COSTS = {"open+exg", "open+sat"}

_CH_CACHE: dict = {}


def _dom_channel_grid(site, name):
    """从 data/dom_channels_{site}.npz 取 DOM 派生通道（exg/sat/...）并置 valid 外 NaN。"""
    key = ("dom", site, name)
    if key in _CH_CACHE:
        return _CH_CACHE[key]
    z = np.load(os.path.join(DATA, f"dom_channels_{site}.npz"))
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    g = np.asarray(z[name], dtype=np.float64)
    g = np.where(valid, g, np.nan)
    _CH_CACHE[key] = g
    return g


def _rank01(v, dom):
    """代价域内秩归一化到 [0,1]（并列取平均秩）；域外 = 1.0（最高代价，不可走）。"""
    out = np.ones(v.shape, dtype=np.float64)
    x = v[dom]
    m = np.isfinite(x)
    if int(m.sum()) > 1:
        rr = rankdata(x[m], method="average")
        col = np.ones(x.shape, dtype=np.float64)
        col[m] = (rr - 1.0) / (m.sum() - 1.0)
        out[dom] = col
    return out


def _channel_grid(site, name):
    """从 feature_matrix 取 20 因子中的一列并摊回栅格（valid 外 NaN）。"""
    key = (site, name)
    if key in _CH_CACHE:
        return _CH_CACHE[key]
    meta = json.load(open(os.path.join(DATA, f"feature_matrix_{site}.meta.json"),
                          encoding="utf-8"))
    k = meta["factors"].index(name)
    X = np.load(os.path.join(DATA, f"feature_matrix_{site}.npy"))
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    g = np.full(valid.size, np.nan, dtype=np.float64)
    g[np.flatnonzero(valid.ravel())] = X[:, k]
    g = g.reshape(valid.shape)
    _CH_CACHE[key] = g
    return g


def _norm01(v, dom, norm):
    """把 v 在代价域内线性映射到 [0,1]；返回 (归一化数组, (lo, hi))。"""
    x = v[dom]
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(v), (0.0, 1.0)
    if norm == "pct":
        lo, hi = np.percentile(x, 5.0), np.percentile(x, 95.0)
    else:
        lo, hi = float(x.min()), float(x.max())
    r = np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return np.nan_to_num(r, nan=1.0), (float(lo), float(hi))


def _component(site, token, tpi31_raw, dom, norm):
    """token 形如 '<factor>_low' → 该通道"低=像走廊"的代价分量。"""
    name = token[:-4]
    v = tpi31_raw if name == "tpi31" else _channel_grid(site, name)
    v = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0)
    if name.startswith("tpi"):                 # tpi* ：值越负越像走廊
        return _norm01(np.maximum(0.0, -v), dom, norm)
    return _norm01(v, dom, norm)               # cliff_prox / openness ：本身越低越像走廊


def _rank_component(site, token, tpi31_raw, dom):
    """秩归一化分量：token 中 `_high` → 值越高越像走廊（取 1−秩）。

    与 `_component`（P5~P95 线性归一化）的区别只在归一化口径：这里用域内**秩**，
    对量纲/尺度完全免疫，因此两个不同量纲的信号（openness 与 DOM 彩度）可以**等权**相加
    而不被一方主导（约束 1）。
    """
    name = token[:-5] if token.endswith("_high") else token[:-4]
    if name in ("exg", "sat"):
        v = _dom_channel_grid(site, name)
    elif name == "tpi31":
        v = tpi31_raw
    else:
        v = _channel_grid(site, name)
    v = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0)
    r = _rank01(v, dom)
    if token.endswith("_high"):
        r = 1.0 - r
    x = v[dom]
    x = x[np.isfinite(x)]
    lohi = (float(x.min()) if x.size else 0.0, float(x.max()) if x.size else 1.0)
    return r, lohi


def _cs_field(site, name):
    """CSSC / DRR 逐像元场（`tools/build_cs_fields.py` 产出，valid 外 NaN）。"""
    key = ("cs", site, name)
    if key in _CH_CACHE:
        return _CH_CACHE[key]
    z = np.load(os.path.join(RES, f"cs_fields_{site}.npz"))
    g = np.asarray(z[name], dtype=np.float64)
    _CH_CACHE[key] = g
    return g


def build_cost(site, tpi31_raw, dom, eps, norm, cost_spec, lam_cssc=0.0, lam_drr=0.0):
    """按 --cost 预设组装代价图；返回 (cost, {token: (lo, hi)})。

    `cost_spec ∈ RANK_COSTS`（DOM 彩度组合）走**秩归一化 + 等权 + 除最大值**（约束 1）：
        cost = rank_norm(open) + rank_norm(1 − exg);  cost /= cost.max()
    其余预设沿用原 P5~P95 线性加权（逐位不变，保证既有实验可复现）。

    `lam_cssc / lam_drr > 0` 时再叠加两个**剖面级惩罚**（约束：秩归一化、等权可加）：
        cost += lam_cssc · (1 − rank(CSSC)) + lam_drr · (1 − rank(DRR))
    即"剖面不像人工开凿断面（CSSC 低）/ 粗糙度各向异性弱（DRR 低）"→ 加价。
    用连续秩惩罚而非用户举例的 `CSSC<2` 硬门：硬门在 Dijkstra 里是阶跃，没有梯度，
    路径只会贴着阈值两侧抖动；秩惩罚在全域单调，对平移/尺度免疫。
    """
    parts = COST_PRESETS[cost_spec]
    tot = sum(w for _, w in parts)
    acc = np.zeros(dom.shape, dtype=np.float64)
    info = {}
    rank_mode = cost_spec in RANK_COSTS
    for token, w in parts:
        if rank_mode:
            r, lohi = _rank_component(site, token, tpi31_raw, dom)
        else:
            r, lohi = _component(site, token, tpi31_raw, dom, norm)
        acc += (w / tot) * r
        info[token] = lohi
    if rank_mode:
        acc = acc / max(float(acc.max()), 1e-9)
    if lam_cssc > 0 or lam_drr > 0:
        base = acc / max(float(acc.max()), 1e-9)
        for name, lam in (("cssc", lam_cssc), ("drr", lam_drr)):
            if lam <= 0:
                continue
            v = np.nan_to_num(_cs_field(site, name), nan=0.0)
            r = _rank01(v, dom)
            base = base + lam * (1.0 - r)
            x = v[dom]
            x = x[np.isfinite(x)]
            info[f"1-rank({name})"] = (float(x.min()), float(x.max()))
        acc = base / max(float(base.max()), 1e-9)
    return np.clip(acc, 0.0, 1.0) + eps, info


# ------------------------------------------------------------ 方向先验 ----
def smooth_phase_complex(phase, size=5):
    """复数域相位平滑（`exp(i·2φ)` 上做均值滤波，消除 180° 翻转歧义）。

    走廊法向是**轴**不是**向量**（φ 与 φ+180° 等价），直接平滑角度会在 0/π 边界
    出错；映射到 `2φ` 后两者同相，平滑才有意义。平滑后模长接近 0 表示邻域方向
    互相抵消（无统一走向），此时回退原值。
    """
    if size <= 1:
        return phase
    z = np.exp(1j * 2.0 * phase)
    zs = (uniform_filter(z.real, size=size)
          + 1j * uniform_filter(z.imag, size=size))
    mag = np.abs(zs)
    ok = mag > 1e-6
    out = phase.copy()
    out[ok] = np.angle(zs[ok]) / 2.0
    return out


def direction_field(site, gate_pct=60.0, smooth=5):
    """h1 相位 → 每像元的**走廊法向**单位向量 + 置信掩膜（缓存在 results/）。

    来源 `p2_terrain.horizon_harmonics`：8 向地平线角的 1 阶相位 φ 指向最大仰角
    方位，实测与走廊法向中位夹角 8.8°/13.4°（零假设 45°，D12 #32）。返回
    `(n_row, n_col, conf)`：`conf = h1_deg ≥ 域内 gate_pct 分位`——平坦/背景区
    的 φ 是噪声，必须门控掉（否则会把随机方向当约束）。
    罗盘约定：θ=0 为北(row−1)，θ=90 为东(col+1) → `n_row=−cosθ, n_col=+sinθ`。
    """
    p = os.path.join(RES, f"horizon_factors_{site}.npz")
    if not os.path.exists(p):
        from fix_tpi31_auc import fill_nodata
        from p2_terrain import horizon_harmonics
        d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
        v = d["valid"] == 1
        dem = fill_nodata(d["dem_mean"].astype(np.float64), v)
        h = horizon_harmonics(dem, gsd=1.0, max_dist_m=30.0)
        np.savez_compressed(p, h1_deg=h["h1_deg"].astype(np.float32),
                            h1_phase=h["h1_phase"].astype(np.float32),
                            h2_deg=h["h2_deg"].astype(np.float32))
    z = np.load(p)
    th = np.radians(z["h1_phase"].astype(np.float64))
    th = smooth_phase_complex(th, size=smooth)
    mag = z["h1_deg"].astype(np.float64)
    v = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    thr = float(np.percentile(mag[v], gate_pct))
    conf = mag >= thr
    return -np.cos(th), np.sin(th), conf, thr


# ------------------------------------------------------------ Dijkstra ----
def build_graph(cost, dom, n_row=None, n_col=None, conf=None, lam=0.0):
    """8 邻域栅格图（对角线 √2 加权），边权 = 步长 × 两端点代价均值。

    `lam > 0` 时叠加**方向先验**：`lam × |v̂·n̂|`，`v̂` 为移动方向、`n̂` 为走廊法向，
    **横穿走廊（与法向平行）受罚、沿走向（与法向垂直）不受罚**。法向取两端点
    各自的值再加权平均（两端都无置信 → 该项为 0，即背景区不受约束）。

    抽出来供多源/多候选路径脚本复用（`tools/path_multipath.py`）。
    """
    H, W = dom.shape
    idx = np.arange(H * W, dtype=np.int32).reshape(H, W)
    cf = cost.ravel()
    rows, cols, data = [], [], []
    use_dir = lam > 0 and n_row is not None
    for dy, dx, L in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQ2), (1, -1, SQ2)):
        if dx >= 0:
            A, B = idx[:H - dy, :W - dx], idx[dy:, dx:]
        else:
            A, B = idx[:H - dy, -dx:], idx[dy:, :W + dx]
        sa, sb = A.ravel(), B.ravel()
        ok = dom.ravel()[sa] & dom.ravel()[sb]
        sa, sb = sa[ok], sb[ok]
        w = L * 0.5 * (cf[sa] + cf[sb])
        if use_dir:
            uy, ux = dy / L, dx / L                   # 单位移动向量（row, col）
            cf_ = conf.ravel()
            cot = np.abs(uy * n_row.ravel()[sa] + ux * n_col.ravel()[sa])
            cbt = np.abs(uy * n_row.ravel()[sb] + ux * n_col.ravel()[sb])
            wa, wb = cf_[sa].astype(np.float64), cf_[sb].astype(np.float64)
            # 至少一端有置信才施加惩罚；两端都有则取平均，保持边权对称
            dot = (wa * cot + wb * cbt) / np.maximum(wa + wb, 1e-9)
            w = w + lam * dot
        rows.append(sa); cols.append(sb)
        data.append(w)
    return csr_matrix((np.concatenate(data),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(H * W, H * W))


def build_state_graph(cost, dom, n_row, n_col, conf, lam_dir=0.0, lam_turn=0.0,
                      turn_pow=2.0, turn_min=2, z=None, lam_z=0.0,
                      p_gate=None, max_grade=0.0, grade_lag=5,
                      lam_c=0.0, gmag=None, eps_c=0.05):
    """**状态增广图**：节点 = (像素, 入射方向)（8 个方向），边权再加转角惩罚。

    状态数 `8·N`，出边 `8` 条/状态。额外两项：
      - 方向先验：`lam_dir × |v̂·n̂|`（同 `build_graph`，取两端点置信均值）
      - **连续性惩罚**：`lam_turn × fold(|d_in−d_out|)^turn_pow`
        `fold` 把 8 个方向差折成 0…4（0=直行，2=90°，4=掉头）。

    `lam_z > 0` 时再加**沿程高差惩罚** `lam_z × |z_b − z_a|`（米）：只罚"上/下坡所走的
    高差"，不罚站在陡坡上，因此与"沿等高线/缓台绕行"的解法相容（逐像元坡度惩罚做不到
    这一点——真值本身就在 20° 坡面上）。

    `p_gate` 给定时，高差项乘 `(1 − P̄)`（两端概率均值）：高置信处免罚、低置信处全额罚，
    即"只在模型自己也不确定时才追究它爬坡"。

    `max_grade > 0` 时为**硬约束**：沿该方向 `±grade_lag` 像素（默认各 5 px，即 10 m 窗）
    的纵坡 `|z₊ − z₋| / (2·k·L)` 超过 `max_grade` 的边直接删掉（不可通行）。
    窗外（越界/域外）视为未知，不删。

    只惩罚 `fold > 1`（>45°）的转向，与 `fold ≤ 1` 的 45° 锯齿兼容——这是必需的，
    否则斜向行走无法成立（22.5° 的直线在 8 邻域里必须 0°/45° 交替）。

    返回 `(G, N, 8N, 8N+1)`：超源 = `8N`、超汇 = `8N+1`，所有方向自由起止。
    """
    H, W = dom.shape
    N = H * W
    domf = dom.ravel()
    cf = cost.ravel()
    cf_conf = conf.ravel().astype(np.float64)
    nr, nc = n_row.ravel(), n_col.ravel()
    idx = np.arange(N, dtype=np.int64).reshape(H, W)

    fold = np.abs(np.subtract.outer(np.arange(8), np.arange(8)))
    fold = np.minimum(fold, 8 - fold).astype(np.float64)     # 0…4
    turn = np.where(fold >= turn_min, fold ** turn_pow, 0.0)
    zf = None if z is None else np.asarray(z, dtype=np.float64).ravel()
    if lam_z > 0 and zf is None:
        raise ValueError("lam_z > 0 需要传入高程场 z")
    pgf = None if p_gate is None else np.asarray(p_gate, dtype=np.float64).ravel()
    if max_grade > 0 and zf is None:
        raise ValueError("max_grade > 0 需要传入高程场 z")
    gm = None if gmag is None else np.asarray(gmag, dtype=np.float64).ravel()
    if lam_c > 0 and (gm is None or zf is None):
        raise ValueError("lam_c > 0 需要传入高程场 z 与梯度模 gmag")
    n_block = 0

    rows, cols, data = [], [], []
    for d_in in range(8):
        for d_out, (dy, dx) in enumerate(NB8):
            L = 1.0 if dy == 0 or dx == 0 else SQ2
            r0, r1 = max(0, -dy), min(H, H - dy)
            c0, c1 = max(0, -dx), min(W, W - dx)
            if r0 >= r1 or c0 >= c1:
                continue
            A = idx[r0:r1, c0:c1]                  # 源像元
            B = idx[r0 + dy:r1 + dy, c0 + dx:c1 + dx]   # 目标像元
            sa, sb = A.ravel(), B.ravel()
            ok = domf[sa] & domf[sb]
            if max_grade > 0:
                k = int(grade_lag)
                ra, ca = np.divmod(sa, W)
                r1_, c1_ = ra + k * dy, ca + k * dx
                r2_, c2_ = ra - k * dy, ca - k * dx
                inside = ((r1_ >= 0) & (r1_ < H) & (c1_ >= 0) & (c1_ < W)
                          & (r2_ >= 0) & (r2_ < H) & (c2_ >= 0) & (c2_ < W))
                gr = np.zeros(sa.shape, dtype=np.float64)
                ii = inside & ok
                if ii.any():
                    p1 = r1_[ii] * W + c1_[ii]
                    p2 = r2_[ii] * W + c2_[ii]
                    gr[ii] = np.abs(zf[p1] - zf[p2]) / (2.0 * k * L)
                keep_g = gr <= max_grade
                n_block += int((ok & ~keep_g).sum())
                ok = ok & keep_g
            sa, sb = sa[ok], sb[ok]
            w = L * 0.5 * (cf[sa] + cf[sb])
            if lam_dir > 0:
                uy, ux = dy / L, dx / L
                dot = ((np.abs(uy * nr[sa] + ux * nc[sa]) * cf_conf[sa]
                        + np.abs(uy * nr[sb] + ux * nc[sb]) * cf_conf[sb])
                       / np.maximum(cf_conf[sa] + cf_conf[sb], 1e-9))
                w = w + lam_dir * dot
            w = w + lam_turn * turn[d_in, d_out]
            if lam_z > 0:
                dz = np.abs(zf[sb] - zf[sa])
                if pgf is not None:
                    dz = dz * (1.0 - 0.5 * (pgf[sa] + pgf[sb]))
                w = w + lam_z * dz
            if lam_c > 0:
                # 方向余弦惩罚：|Δz| / (|∇z|·L + ε) ≈ |cos θ|
                # 只罚"横切坡面"的运动分量，与地形陡缓无关（平地靠 ε 兜底），
                # 因此不会像 lam_z 那样在整条陡坡走廊上整体抬价。
                pen = np.abs(zf[sb] - zf[sa]) / (gm[sa] * L + eps_c)
                w = w + lam_c * np.clip(pen, 0.0, 1.0)
            rows.append(d_in * N + sa)
            cols.append(d_out * N + sb)
            data.append(w)

    if max_grade > 0:
        print(f"  [硬纵坡] 阈值 {max_grade * 100:.0f}%（{2 * int(grade_lag)} m 窗）"
              f"删除有向边 {n_block:,} 条")
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(data)), N


def dijkstra_state(cost, dom, start, end, n_row, n_col, conf,
                   lam_dir=0.0, lam_turn=0.0, turn_min=2):
    """状态增广 Dijkstra。返回 `(路径掩膜, 总代价)`。

    起终点用超源/超汇接入：把所有 `(start, d_in)` 与 `(end, d_in)` 视为 0 代价聚合。
    """
    H, W = dom.shape
    (rows, cols, data), N = build_state_graph(cost, dom, n_row, n_col, conf,
                                             lam_dir, lam_turn, turn_min=turn_min)
    ss, ee = 8 * N, 8 * N + 1
    s, e = int(start[0] * W + start[1]), int(end[0] * W + end[1])
    d_all = np.arange(8, dtype=np.int64) * N
    # 超源 → 起点的全部 8 个方向态；终点的全部 8 个方向态 → 超汇（均 0 代价）
    rows = np.concatenate([rows, np.full(8, ss, dtype=np.int64), d_all + e])
    cols = np.concatenate([cols, d_all + s, np.full(8, ee, dtype=np.int64)])
    data = np.concatenate([data, np.zeros(16)])

    G = csr_matrix((data, (rows, cols)), shape=(8 * N + 2, 8 * N + 2))
    dist, pred = dijkstra(G, directed=True, indices=ss, return_predecessors=True)
    if not np.isfinite(dist[ee]):
        return None, float("nan")
    path = np.zeros(N, dtype=bool)
    cur = ee
    while cur >= 0:
        if cur < 8 * N:
            path[cur % N] = True
        if cur == ss:
            break
        cur = int(pred[cur])
    return path.reshape(H, W), float(dist[ee])


def dijkstra_8(cost, dom, start, end, G=None):
    """8 邻域、对角线 √2 加权的最小代价路径；返回 bool 路径掩码与代价。"""
    H, W = dom.shape
    if G is None:
        G = build_graph(cost, dom)
    s, e = int(start[0] * W + start[1]), int(end[0] * W + end[1])
    dist, pred = dijkstra(G, directed=False, indices=s, return_predecessors=True)
    if not np.isfinite(dist[e]):
        return None, float("nan")
    path = np.zeros(H * W, dtype=bool)
    cur = e
    while cur != s and cur >= 0:
        path[cur] = True
        cur = int(pred[cur])
    path[s] = True
    return path.reshape(H, W), float(dist[e])


# ------------------------------------------------------------ 主 ----
def map_to_px(meta, x, y):
    """UTM → 栅格 (row, col)。栅格以像元中心建索引，故各减 0.5。"""
    gsd = float(meta["gsd_m"])
    col = (float(x) - float(meta["x0"])) / gsd - 0.5
    row = (float(meta["y1"]) - float(y)) / gsd - 0.5
    return row, col


def snap_report(pt, dom):
    """先把远在栅格外的点夹到栅格边界，再吸附到最近域内像元。返回 (吸附点, 偏移px)。"""
    r = min(max(float(pt[0]), 0.0), dom.shape[0] - 1.0)
    c = min(max(float(pt[1]), 0.0), dom.shape[1] - 1.0)
    pt = (r, c)
    s = snap_to_domain(pt, dom)
    return s, float(np.hypot(s[0] - pt[0], s[1] - pt[1]))


def extract_path(site, cost_spec, eps, norm, roi_radius, merge_px, log,
                 dir_lambda=0.0, dir_gate=None, dir_smooth=5, turn_lambda=0.0,
                 turn_min=2, lam_cssc=0.0, lam_drr=0.0, ep_pair=None,
                 allow_gt_roi=False):
    """跑一遍最小代价路径提取，返回**原始 1px 路径**与评估上下文（不膨胀、不评估）。

    抽出来供 `eval_path_adaptive.py` 复用：同一份提取逻辑既出 mask IoU，也出骨架 IoU
    与自适应膨胀 IoU，避免各脚本各跑一遍 Dijkstra。

    `dir_lambda > 0` 启用 h1 相位方向先验（见 `build_graph`）；`dir_gate` 为
    置信分位阈值（None 时用 `DFS_GATE` 默认值）。

    `roi_radius > 0` **默认不生效**：把搜索域限制在真值走廊附近等于让 SHP 参与流程
    （L3 泄漏）。仅在显式传 `allow_gt_roi=True` 时才启用，用于泄漏对照诊断。
    """
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    dom = valid.copy()
    if roi_radius > 0 and allow_gt_roi:
        # L3 诊断模式：搜索域限制在真值走廊 ±roi（默认关闭）
        dom &= binary_dilation(label, iterations=int(roi_radius))
    elif roi_radius > 0:
        log(f"  [警告] roi_radius={roi_radius} 已忽略：真值 ROI 需显式 "
            f"allow_gt_roi=True（L3 诊断模式），默认使用全域")
    cost, info = build_cost(site, tpi31, dom, eps, norm, cost_spec,
                            lam_cssc=lam_cssc, lam_drr=lam_drr)
    comp = "  ".join(f"{k}[{lo:.2f},{hi:.2f}]" for k, (lo, hi) in info.items())
    log(f"\n[{site}] 代价域 {int(dom.sum()):,} px  cost={cost_spec} 归一化({norm}) "
        f"eps={eps}  ROI={'全域' if (roi_radius == 0 or not allow_gt_roi) else f'{roi_radius}px(L3)'}  "
        f"λ_cssc={lam_cssc} λ_drr={lam_drr}")
    log(f"  分量 lo/hi: {comp}")

    G = None
    nr = nc = conf = None
    if dir_lambda > 0 or turn_lambda > 0:
        gate = DFS_GATE if dir_gate is None else float(dir_gate)
        nr, nc, conf, thr = direction_field(site, gate, smooth=dir_smooth)
        log(f"  方向先验：λ_dir={dir_lambda} 门控 P{gate:.0f}（h1_deg ≥ {thr:.3f}°，"
            f"置信像元 {100 * conf[dom].mean():.1f}% 域内）相位平滑窗 {dir_smooth}")
    if dir_lambda > 0 and turn_lambda <= 0:
        G = build_graph(cost, dom, nr, nc, conf, dir_lambda)
    if turn_lambda > 0:
        log(f"  状态增广：λ_turn={turn_lambda}（节点 = 像素 × 8 入射方向，"
            f"{8 * int(dom.sum()):,} 状态）")

    cc, n_cc = cc_label(label, structure=S8)
    segs = []
    if ep_pair is not None:
        # 全自动：起终点由外部（第三方）文件给出，脚本内不读任何真值几何
        meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
        (x0, y0), n0 = ep_pair[0]
        (x1, y1), n1 = ep_pair[1]
        s, ds = snap_report(map_to_px(meta, x0, y0), dom)
        e, de = snap_report(map_to_px(meta, x1, y1), dom)
        log(f"  外部端点 [{n0}]→[{n1}]：{n0} 像元{s}（吸附 {ds:.1f}px，"
            f"{np.hypot(x0 - (meta['x0'] + (s[1] + 0.5) * meta['gsd_m']), y0 - (meta['y1'] - (s[0] + 0.5) * meta['gsd_m'])):.0f} m 外）"
            f"；{n1} 像元{e}（吸附 {de:.1f}px）")
        path, pc = (dijkstra_state(cost, dom, s, e, nr, nc, conf, dir_lambda, turn_lambda,
                                   turn_min)
                    if turn_lambda > 0 else dijkstra_8(cost, dom, s, e, G=G))
        if path is None:
            log("  外部端点不可达，跳过")
        else:
            segs.append(path)
            log(f"  外部端点路径 {int(path.sum())}px 代价 {pc:.2f}")
    else:
        for cid in range(1, n_cc + 1):
            m = cc == cid
            ep = find_endpoints(skeletonize(m))
            pts = np.argwhere(ep)
            if len(pts) < 2:
                log(f"  域{cid}: 端点仅 {len(pts)} 个，跳过")
                continue
            s_raw, e_raw = merge_and_pick_pair(pts, merge_px)
            s = snap_to_domain(s_raw, dom)
            e = snap_to_domain(e_raw, dom)
            path, pc = (dijkstra_state(cost, dom, s, e, nr, nc, conf, dir_lambda, turn_lambda,
                                       turn_min)
                        if turn_lambda > 0 else dijkstra_8(cost, dom, s, e, G=G))
            if path is None:
                log(f"  域{cid}: 端点不可达（{s} → {e}），跳过")
                continue
            segs.append(path)
            log(f"  域{cid}: 真值 {int(m.sum())}px 端点 {len(pts)}→合并 起{s} 终{e} "
                f"路径 {int(path.sum())}px 代价 {pc:.2f}")

    path_all = np.logical_or.reduce(segs) if segs else None
    return dict(path=path_all, dom=dom, label=label, valid=valid,
                n_cc=int(n_cc), n_seg=len(segs), info=info)


def tag_of(a):
    """输出文件标签：代价名 + 方向先验强度（λ>0 时）+ 外部端点标记。"""
    t = a.cost.replace("+", "P")
    if a.dir_lambda and a.dir_lambda > 0:
        t += f"_dir{a.dir_lambda:g}"
        if a.dir_smooth and a.dir_smooth != 5:
            t += f"s{a.dir_smooth}"
    if a.turn_lambda and a.turn_lambda > 0:
        t += f"_turn{a.turn_lambda:g}"
        if a.turn_min != 2:
            t += f"m{a.turn_min}"
    if a.lam_cssc and a.lam_cssc > 0:
        t += f"_cssc{a.lam_cssc:g}"
    if a.lam_drr and a.lam_drr > 0:
        t += f"_drr{a.lam_drr:g}"
    if getattr(a, "endpoints", None):
        t += "_ext"
    return t


def load_endpoints(path, site):
    """读外部起止点文件 → [((x,y), name), ((x,y), name)]；缺站点返回 None。"""
    if not path:
        return None
    d = json.load(open(path, encoding="utf-8"))
    e = d.get("sites", d).get(site)
    if not e:
        return None
    return [((float(p["utm48"][0]), float(p["utm48"][1])), p.get("name", "?")) for p in e]


def run_site(site, a, log):
    r = extract_path(site, a.cost, a.eps, a.norm, a.roi_radius, a.merge_px, log,
                     dir_lambda=a.dir_lambda, dir_gate=a.dir_gate,
                     dir_smooth=a.dir_smooth, turn_lambda=a.turn_lambda,
                     turn_min=a.turn_min, lam_cssc=a.lam_cssc, lam_drr=a.lam_drr,
                     ep_pair=load_endpoints(a.endpoints, site))
    if r["path"] is None:
        log(f"  [FAIL] {site} 无可用路径")
        return None
    label, valid, n_cc, path_all = r["label"], r["valid"], r["n_cc"], r["path"]
    tag = tag_of(a)
    np.save(os.path.join(RES, f"path_raw_{tag}_{site}.npy"), path_all)
    pred = binary_dilation(path_all, iterations=a.dilate_radius) & r["dom"]

    win = window_mask(valid)
    inter = int((pred & label & win).sum())
    union = int(((pred | label) & win).sum())
    iou = float(inter / union) if union else 0.0
    rows = []
    for t in [0.5]:                       # 路径提取输出是二值，不扫阈值；保留 0.5 口径
        mm = linear_metrics(pred, label, win, TOL)
        mm["thresh"] = t
        rows.append(mm)
    best = rows[0]
    log(f"  → 预测 {int(pred.sum()):,}px  膨胀半径={a.dilate_radius}px  "
        f"IoU@0.5={iou:.4f} (i={inter},u={union})")
    log(f"     中心线F1={best['f1']:.4f} LR={best['lr']:.4f} LP={best['lp']:.4f} "
        f"中心线={best['centerline_px']:.2f}px 拓扑R/P={best['topo_recall']:.2f}/"
        f"{best['topo_precision']:.2f} 端点F1={best['ep_f1']:.2f}")

    np.save(os.path.join(RES, f"path_pred_{tag}_{site}.npy"), pred)
    return dict(site=site, n_cc=int(n_cc), n_segments=int(r["n_seg"]),
                path_px=int(path_all.sum()), pred_px=int(pred.sum()),
                dilate_radius=int(a.dilate_radius), roi_radius=int(a.roi_radius),
                eps=a.eps, norm=a.norm, cost=a.cost,
                dir_lambda=float(a.dir_lambda),
                turn_lambda=float(a.turn_lambda),
                lam_cssc=float(a.lam_cssc), lam_drr=float(a.lam_drr),
                dir_gate=float(a.dir_gate if a.dir_gate is not None else DFS_GATE),
                iou=iou, inter=inter, union=union,
                centerline_f1=float(best["f1"]), lr=float(best["lr"]), lp=float(best["lp"]),
                centerline_px=float(best["centerline_px"]),
                topo_recall=float(best["topo_recall"]),
                topo_precision=float(best["topo_precision"]),
                ep_f1=float(best["ep_f1"]), gt_px=int((label & win).sum()))


def figure(site, log, cost="tpi31", tag=None):
    """叠加图：tpi31 底图 + 真值轮廓 + 提取路径，直接看'像路但不是路'。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
    tag = tag or cost.replace("+", "P")
    pp = os.path.join(RES, f"path_pred_{tag}_{site}.npy")
    if not os.path.exists(pp):
        return None
    pred = np.load(pp)
    tpi31 = np.load(os.path.join(DATA, f"p2_terrain_{site}.npy"))[7]
    valid = np.load(os.path.join(DATA, f"{site}_1m.npz"))["valid"] == 1
    v = tpi31[valid]
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    bg = np.where(valid, np.clip((tpi31 - lo) / max(hi - lo, 1e-9), 0, 1), np.nan)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    ax.imshow(bg, cmap="terrain")
    ax.contour(label, levels=[0.5], colors="white", linewidths=1.8)
    ax.contour(pred, levels=[0.5], colors="red", linewidths=1.6, linestyles="--")
    ax.set_title(f"{site}: tpi31 底图 + 真值(白) vs 提取路径(红虚线, cost={cost}"
                 f"{'' if tag == cost.replace('+', 'P') else ', ' + tag})")
    ax.legend(handles=[Line2D([], [], color="white", lw=2, label="真值走廊(SHP)"),
                       Line2D([], [], color="red", lw=2, ls="--", label=f"最小代价路径({tag})")],
              loc="upper right", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    os.makedirs(FIG, exist_ok=True)
    fp = os.path.join(FIG, f"path_extraction_{tag}_{site}.png")
    fig.tight_layout(); fig.savefig(fp); plt.close(fig)
    log(f"  图: {os.path.relpath(fp, ROOT)}")
    return fp


def main():
    p = argparse.ArgumentParser(description="tpi31 最小代价路径提取（第 3 步 · 迭代 1）")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--dilate-radius", type=int, default=3, help="路径膨胀半径（px）")
    p.add_argument("--eps", type=float, default=0.05, help="防零代价的平台常数")
    p.add_argument("--norm", type=str, default="pct", choices=["pct", "minmax"])
    p.add_argument("--cost", type=str, default="tpi31", choices=list(COST_PRESETS),
                   help="代价通道预设（见 COST_PRESETS）")
    p.add_argument("--roi-radius", type=int, default=0, help="限定真值走廊周边 N px（0=全域）")
    p.add_argument("--merge-px", type=float, default=10.0, help="骨架端点合并距离（px）")
    p.add_argument("--dir-lambda", type=float, default=0.0,
                   help="h1 相位方向先验强度（0=关闭；建议 1~5，边权≈1 量级）")
    p.add_argument("--dir-gate", type=float, default=None,
                   help=f"方向置信门控分位（默认 P{DFS_GATE:g}，即 h1_deg 域内前 40%%）")
    p.add_argument("--dir-smooth", type=int, default=5,
                   help="相位复数域平滑窗（px；1=关闭。平滑 exp(i·2φ) 消除 180° 歧义）")
    p.add_argument("--turn-lambda", type=float, default=0.0,
                   help="状态增广转角惩罚强度（0=关闭；>0 时节点=像素×8入射方向，"
                        "只惩罚 >45° 的转向）")
    p.add_argument("--turn-min", type=int, default=2,
                   help="转角惩罚起效的最小折角（2=仅 >45°，即原方案；1=连 45° 也罚）")
    p.add_argument("--lam-cssc", type=float, default=0.0,
                   help="CSSC 横断面 S 形曲率惩罚强度（0=关闭；秩归一化后与基础代价同量级，"
                        "建议 0.1~1.0）")
    p.add_argument("--lam-drr", type=float, default=0.0,
                   help="DRR 方向性粗糙度比惩罚强度（0=关闭；建议 0.1~1.0）")
    p.add_argument("--endpoints", type=str, default=None,
                   help="外部起止点 JSON（全自动模式）：给定时不读真值骨架端点，"
                        "起终点全部来自该文件；缺站点则回退骨架端点")
    p.add_argument("--fig", action="store_true", default=True)
    p.add_argument("--no-fig", dest="fig", action="store_false")
    a = p.parse_args()
    tag = tag_of(a)

    log_path = os.path.join(RES, f"path_extraction_{tag}.log")
    lf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()

    log(f"第 3 步（提高廊道识别）· 代价={a.cost} dilate={a.dilate_radius}px eps={a.eps} "
        f"norm={a.norm} roi={a.roi_radius} 端点合并={a.merge_px}px "
        f"方向先验 λ={a.dir_lambda} gate={a.dir_gate or DFS_GATE} smooth={a.dir_smooth} "
        f"转角 λ_turn={a.turn_lambda}　CSSC λ={a.lam_cssc} DRR λ={a.lam_drr}")
    log(f"端点来源：{a.endpoints if a.endpoints else '真值 mask 骨架端点（SHP 参与输入）'}")
    out = []
    for site in a.sites:
        r = run_site(site, a, log)
        if r:
            out.append(r)
            if a.fig:
                r["figure"] = figure(site, log, a.cost, tag)

    log(f"\n{'=' * 76}\n汇总（真值 = SHP 走廊 mask，D10 口径：排除 85px 边界带 ∩ valid）")
    log(f"{'站点':>8s} {'域数':>4s} {'路径px':>8s} {'预测px':>9s} {'IoU@0.5':>9s} "
        f"{'中心线F1':>9s} {'LR':>7s} {'LP':>7s} {'中心线px':>9s}")
    log("-" * 84)
    for r in out:
        log(f"{r['site']:>8s} {r['n_cc']:>4d} {r['path_px']:>8d} {r['pred_px']:>9d} "
            f"{r['iou']:>9.4f} {r['centerline_f1']:>9.4f} {r['lr']:>7.4f} {r['lp']:>7.4f} "
            f"{r['centerline_px']:>9.2f}")
    log(f"\n对照：v6 跨站 IoU≈0.021 / 中心线F1≈0.021~0.251；"
        f"几何基线 IoU 0.011~0.028；同站自验 IoU 0.0000")

    if a.cost in RANK_COSTS:
        xb = next((r for r in out if r["site"] == "西坝村"), None)
        log(f"\n硬止损判据（约束 3，仅看西坝村 IoU）：")
        log("  ≥0.20 有实质增益，值得进一步调 | 0.15~0.20 弱增益，记录并停止路径提取深挖 "
            "| <0.15 无增益，停止路径提取深挖")
        if xb:
            iou = xb["iou"]
            verdict = ("有实质增益，值得进一步调" if iou >= 0.20
                       else ("弱增益，记录，停止路径提取深挖" if iou >= 0.15
                             else "无增益，停止路径提取深挖"))
            log(f"  西坝村 {a.cost}: IoU={iou:.4f}  →  {verdict}")
        else:
            log("  西坝村无结果")

    jp = os.path.join(RES, f"path_extraction_{tag}_d{a.dilate_radius}"
                           f"{'' if a.roi_radius == 0 else f'_roi{a.roi_radius}'}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(cost=a.cost, dilate_radius=a.dilate_radius, eps=a.eps, norm=a.norm,
                       roi_radius=a.roi_radius, merge_px=a.merge_px,
                       dir_lambda=a.dir_lambda,
                       dir_gate=(a.dir_gate if a.dir_gate is not None else DFS_GATE),
                       lam_cssc=a.lam_cssc, lam_drr=a.lam_drr,
                       sites=out),
                  f, ensure_ascii=False, indent=2)
    log(f"\n保存: {jp}")
    lf.close()


if __name__ == "__main__":
    main()
