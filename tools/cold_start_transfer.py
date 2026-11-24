# -*- coding: utf-8 -*-
"""冷启动迁移测试（cold_start_transfer.py）。

把目标站当成**从未见过的新站点**，只给两样东西：
    ① 原始 LAS 点云（项目内已栅格化成 `data/{site}_1m.npz`）
    ② 起止点坐标（可命令行给，也可由 CSSC 自动取）

然后现场跑完五步，最后才拿出 SHP 评分：

1. 数据降维与特征构建
   · 1 m DEM（默认 `dem_min`：像元最低高程，抑制植被残留）
   · Hillshade（315°/45°）
   · CSSC（横断面二阶导符号交替次数，剖面级特征）
   · 宏观场：openness（Yokoyama 天空开放度，R=30 m）+ h1_phase（8 向地平线角
     一阶相位 → 走廊法向）
2. 支路 1：宏观图论寻路
   cost = P5~P95 归一化的 openness（低开放度=更像走廊）+ h1 方向先验 + 转角惩罚，
   在状态增广图上跑 Dijkstra 得到 1px 路径；**膨胀 3px 作为伪标签**。
3. 支路 2：现场重训新模型
   输入只有 2 通道（Hillshade + CSSC，**不含**路径距离场，因此学不到"阈值复刻"的
   捷径），初始化全新微型 CNN（3×3 卷积 + BN + ReLU + 1×1 输出），从头训练；
   SHP 全程不参与训练。存 `results/checkpoint_{site}.pth`。
4. 双支路推理与融合
   整图推理 → 2D 走廊概率图 → 按网格契约反投影回 3D 点云 → 算 AGL（dem_min 基准）
   → D+ 高召回策略：核心 P≥0.5 全保留；C 的增量区（P≥0.1 ∪ 膨胀 10px 内首末回波）
   上对 AGL 跑一维 KMeans(2)，只留中心较低的簇（地面层）。
5. 交付
   输出保留点索引 + LAS（有 SHP 时顺带评分）。

用法
----
    python tools/cold_start_transfer.py                       # 杨家坝，端点自动
    python tools/cold_start_transfer.py --ep "119,344;460,186"
    python tools/cold_start_transfer.py --no-compare --no-las
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, distance_transform_edt,
                           label as cc_label, map_coordinates, uniform_filter1d)
from skimage.morphology import disk, skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import path_extraction as P                                          # noqa: E402
from build_cs_fields import alternation_count, detrend_rows          # noqa: E402
from build_feature_matrix import openness                            # noqa: E402
from fix_tpi31_auc import fill_nodata                                # noqa: E402
from fusion_train import diameter_pair, dijkstra_on, infer, train_net     # noqa: E402
from model_guided_3d_filter import bilinear                          # noqa: E402
from p2_terrain import horizon_harmonics                             # noqa: E402
from path_extraction import smooth_phase_complex                     # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")
CHUNK = 4_000_000

# 基于历史地理学文献（霍仁龙等，金牛道纵坡研究），古驿道极限纵坡<18%
# 因1m分辨率DEM包含微地形噪声，直接卡阈值会断图，故采用平滑惩罚系数
LAMBDA_Z = 2.0
# 方向余弦惩罚（古驿道沿等高线修建，横切坡面在工程上不可行）：
#   pen = |Δz| / (|∇z|·L + ε) ≈ |cosθ|，cost += LAMBDA_C · pen
# 定值依据：古驿道极限纵坡 15~18%（霍仁龙等，金牛道纵坡研究）；
# 能量等价换算（Naismith/Tobler 类徒步模型：1 m 爬升 ≈ 8~12 m 水平路程），
# 在 25~35° 走廊上折算为 λc ≈ 5~7，取中值 6；
# 实测该值使两站路径 10 m 窗纵坡 P95 落到 17~18%（<20%），视觉代价涨幅 <50%。
# 该常数不参与任何基于真值的扫描；--lam-c 0 可关闭。
LAMBDA_C = 6.0
# 视觉置信度熔断：地形项只是辅助视觉。若加地形后路径的平均视觉代价翻倍（>100%），
# 说明为了躲高差切断了视觉上最明显的古道痕迹，自动回退 λz=0。
# 判据只用模型自产的 openness 代价，不读 SHP。
BREAKER_DELTA = 1.0

# CSSC 剖面参数（与 build_cs_fields.py 的 B 口径一致）
HALF, STEP, SMOOTH_M, CURV_THR, CSSC_BLOCK = 15.5, 1.0, 2.0, 0.03, 96


# ================================================================ 第 1 步
def load_dem(site, base="min", log=print):
    """新站点的 1 m DEM：只读高程与 valid，不碰 npz 里的 label（那是 SHP 派生）。"""
    z = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    valid = z["valid"] == 1
    key = "dem_min" if base == "min" else "dem_mean"
    dem = fill_nodata(z[key].astype(np.float64), valid)
    log(f"  1 m DEM：{dem.shape}，valid {int(valid.sum()):,}px，基准 = {key}"
        f"（inv {int((~valid).sum()):,}px 用最近邻填洞）")
    return dem, valid


def hillshade_map(dem, valid, az=315.0, alt=45.0):
    """标准 Hillshade（无阴影方向先验之外的任何信息）。"""
    dzdx = np.gradient(dem, 1.0, axis=1)
    dzdy = -np.gradient(dem, 1.0, axis=0)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    a, h = np.radians(az), np.radians(alt)
    hs = np.sin(h) * np.cos(slope) + np.cos(h) * np.sin(slope) * np.cos(a - aspect)
    return np.where(valid, (np.clip(hs, -1.0, 1.0) + 1.0) / 2.0, 0.0)


def macro_fields(dem, valid, gate_pct=60.0, smooth=5, radii=None):
    """宏观场：openness（可多尺度融合）+ h1 相位法向 + 置信门控。"""
    t0 = time.time()
    rs = [float(v) for v in (radii or []) if str(v).strip()]
    if len(rs) > 1:
        # 多尺度 openness：各尺度按 P5~P95 归一化后等权平均，
        # 小半径贴合局部路槽、大半径稳定，用来校正单尺度谷底的横向偏移
        acc = []
        for r in rs:
            o = openness(dem, gsd=1.0, R=int(round(r)), positive=True)
            lo, hi = np.percentile(o[valid], [5, 95])
            acc.append(np.clip((o - lo) / max(hi - lo, 1e-6), 0, 1))
        op = np.mean(acc, axis=0)
    else:
        op = openness(dem, gsd=1.0, R=int(round(rs[0])) if rs else 30, positive=True)
    h = horizon_harmonics(dem, gsd=1.0, max_dist_m=30.0)
    th = np.radians(h["h1_phase"].astype(np.float64))
    th = smooth_phase_complex(th, size=smooth)
    n_row, n_col = -np.cos(th), np.sin(th)      # 罗盘约定：θ=0 指北
    thr = float(np.percentile(h["h1_deg"][valid], gate_pct))
    conf = h["h1_deg"] >= thr
    info = dict(open_range=[float(np.nanmin(op[valid])), float(np.nanmax(op[valid]))],
                h1_gate_deg=round(thr, 2), conf_px=int((conf & valid).sum()),
                seconds=round(time.time() - t0, 1))
    return op, n_row, n_col, conf, info


def cssc_map(dem, valid, n_row, n_col, log=print):
    """CSSC：沿走廊法向取 32 个剖面采样 → 去趋势 → 二阶导显著符号交替次数。

    与 `build_cs_fields.cs_maps` 同一口径，唯一区别是 DEM 由调用方传入（不读缓存、
    不固定用 dem_mean），保证"新站点现场算特征"。
    """
    H, W = dem.shape
    t_row, t_col = -n_col, n_row           # 切向 = 法向转 90°
    cssc = np.full((H, W), np.nan, dtype=np.float32)
    lags = np.arange(-HALF, HALF + STEP * 0.5, STEP)
    k_smooth = max(1, int(round(SMOOTH_M / STEP)))
    t0 = time.time()
    for r0 in range(0, H, CSSC_BLOCK):
        r1 = min(r0 + CSSC_BLOCK, H)
        rr0, cc = np.nonzero(valid[r0:r1])
        if rr0.size == 0:
            continue
        rr = rr0 + r0
        zn = np.empty((rr.size, lags.size), dtype=np.float64)
        inside = np.ones(rr.size, dtype=bool)
        for k, s in enumerate(lags):
            yr = rr + s * n_row[rr, cc]
            xr = cc + s * n_col[rr, cc]
            inside &= (yr >= 0) & (yr <= H - 1) & (xr >= 0) & (xr <= W - 1)
            zn[:, k] = map_coordinates(dem, [yr, xr], order=1, mode="nearest")
        if not inside.any():
            continue
        zn = zn[inside]
        zn = detrend_rows(zn)          # 去线性趋势：只留起伏，剔除整条坡面曲率
        zs = uniform_filter1d(zn, k_smooth, axis=1, mode="nearest") if k_smooth > 1 else zn
        d2 = np.gradient(np.gradient(zs, STEP, axis=1), STEP, axis=1)
        cssc[rr[inside], cc[inside]] = alternation_count(d2, CURV_THR).astype(np.float32)
    bad = valid & ~np.isfinite(cssc)
    if bad.any():
        cssc[bad] = np.nanmedian(cssc[valid])
    cssc[~valid] = np.nan
    v = cssc[valid]
    log(f"  CSSC（{time.time() - t0:.0f}s）：范围 [{np.nanmin(v):.0f}, {np.nanmax(v):.0f}]，"
        f"中位 {np.nanmedian(v):.0f}，P95 {np.nanpercentile(v, 95):.0f}")
    return cssc


# ================================================================ 第 2 步
def auto_endpoints(cssc, valid, pct=95.0, r=5, log=print):
    """CSS C≥P{pct} → 闭运算 → 膨胀 r → 最大连通域 → 骨架直径取两端。

    只用地形特征，不需要 SHP——这才是"新站点无真值"时的真实取点方式。
    """
    v = cssc[valid]
    v = v[np.isfinite(v)]
    thr = float(np.percentile(v, pct))
    raw = (cssc >= thr) & valid & np.isfinite(cssc)
    b = binary_dilation(binary_closing(raw, np.ones((3, 3), bool)), disk(r)) & valid
    lab, ncc = cc_label(b)
    comp = lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)
    skel = skeletonize(comp)
    p, q, gl = diameter_pair(skel)
    log(f"  端点（CSSC≥P{pct:g}={thr:.0f}，连通域 {int(comp.sum()):,}px，"
        f"骨架 {int(skel.sum()):,}px，测地 {gl:.0f}px）")
    log(f"    → ({p[0]:.0f},{p[1]:.0f}) ↔ ({q[0]:.0f},{q[1]:.0f})")
    return (p, q), dict(source="cs_sc_auto", thr=thr, cc_px=int(comp.sum()),
                        skel_px=int(skel.sum()), geo_len_px=round(gl, 1), n_cc=int(ncc))


def macro_path(op, valid, ep, n_row, n_col, conf, a, log=print, dem=None):
    """支路 1：openness 代价 + h1 方向先验 + 转角惩罚 → 状态增广 Dijkstra。

    `--slope-lambda > 0` 时在代价里再叠加**坡度惩罚**（P5~P95 归一化后的 slope_deg）：
        cost = norm(openness) + 0.05 + λ · norm(slope_deg)
    目的是压住"为抄近路而横切陡坡"的直线解：横切 35° 坡面每米都要多付 λ 的价，
    而沿等高线/缓台行走几乎不付。
    """
    lo, hi = np.percentile(op[valid], [5.0, 95.0])
    cost = np.clip((op - lo) / max(hi - lo, 1e-9), 0.0, 1.0) + 0.05
    lz = float(getattr(a, "z_lambda", LAMBDA_Z))      # 锁死参数，见文件头 LAMBDA_Z
    lc = float(getattr(a, "lam_c", 0.0))
    gmag = None
    if lc > 0:
        # 方向余弦惩罚：DEM 先高斯平滑（抑制微地貌噪声），3×3 中心差分取梯度模
        if dem is None:
            raise ValueError("--lam-c > 0 需要传入地形 dem")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from scipy.ndimage import gaussian_filter
        zs = gaussian_filter(np.asarray(dem, np.float64), 1.5)
        gy, gx = np.gradient(zs, 1.0, 1.0)
        gmag = np.hypot(gx, gy)
        lz = 0.0     # 方向项替代高差项，避免双重惩罚
        log(f"  方向余弦惩罚 λc={lc:g}（替代 λz；DEM 高斯 σ=1.5 m，ε=0.05）")
    sl_range = None
    if a.slope_lambda > 0:
        if dem is None:
            raise ValueError("--slope-lambda > 0 需要传入地形 dem")
        gy, gx = np.gradient(np.asarray(dem, dtype=np.float64), 1.0, 1.0)
        sl = np.degrees(np.arctan(np.hypot(gx, gy)))
        lo_s, hi_s = np.percentile(sl[valid], [5.0, 95.0])
        sn = np.clip((sl - lo_s) / max(hi_s - lo_s, 1e-9), 0.0, 1.0)
        cost = cost + a.slope_lambda * sn
        sl_range = (float(lo_s), float(hi_s))
        log(f"  坡度惩罚 λ={a.slope_lambda:g}：slope 归一化区间 "
            f"[{lo_s:.1f}, {hi_s:.1f}]°，代价域内均值 {sl[valid].mean():.1f}°")
    (rows, cols, data), N = P.build_state_graph(
        cost, valid, n_row, n_col, conf,
        lam_dir=a.dir_lambda, lam_turn=a.turn_lambda, turn_min=a.turn_min,
        z=dem, lam_z=lz,
        lam_c=lc, gmag=gmag,
        max_grade=getattr(a, "max_grade", 0.0), grade_lag=getattr(a, "grade_lag", 5))
    H, W = valid.shape
    s = int(round(ep[0][0]) * W + round(ep[0][1]))
    e = int(round(ep[1][0]) * W + round(ep[1][1]))
    t0 = time.time()
    path = dijkstra_on(rows, cols, data, N, s, e)
    if path is None:
        raise RuntimeError("起止点之间没有可达路径（检查端点是否落在 valid 内）")
    path = path.reshape(H, W)
    dist = distance_transform_edt(~path)
    log(f"  Dijkstra（{time.time() - t0:.0f}s）：路径 {int(path.sum())}px，"
        f"openness 归一化区间 [{lo:.1f}, {hi:.1f}]°，方向 λ={a.dir_lambda}"
        + (f"，坡度 λ={a.slope_lambda:g}" if a.slope_lambda > 0 else "")
        + f"，高差 λ={lz:g}" + ("（LAMBDA_Z）" if abs(lz - LAMBDA_Z) < 1e-9 else "")
        + (f"，方向余弦 λc={lc:g}" if lc > 0 else ""))
    return path, dist, cost, (lo, hi)


# ================================================================ 第 2.5 步（可选）
def refine_path_soft(op, valid, ep, n_row, n_col, conf, prob, dem, a, log=print):
    """**方案二：高差 × 概率软约束**（第二遍寻路，SHP 不参与）。

        cost        = (1 − P) + 0.05
        边权增量     = lam_base · |Δz| · (1 − P̄)

    `P̄` 为边两端概率均值。含义：模型很确定（P≈1）的地方，陡一点也放行；模型只有
    0.5 左右把握却要爬 40% 的坡，按"抄近路"重罚。`lam_base` 是常数，不由真值扫出。
    """
    pg = np.where(valid, np.asarray(prob, dtype=np.float64), 0.0)
    cost = np.clip(1.0 - pg, 0.0, 1.0) + 0.05
    (rows, cols, data), N = P.build_state_graph(
        cost, valid, n_row, n_col, conf,
        lam_dir=a.dir_lambda, lam_turn=a.turn_lambda, turn_min=a.turn_min,
        z=dem, lam_z=a.soft_z_lambda, p_gate=pg)
    H, W = valid.shape
    s = int(round(ep[0][0]) * W + round(ep[0][1]))
    e = int(round(ep[1][0]) * W + round(ep[1][1]))
    t0 = time.time()
    path = dijkstra_on(rows, cols, data, N, s, e)
    if path is None:
        log("  软约束寻路：起止点之间不可达，保留原路径")
        return None, None
    path = path.reshape(H, W)
    dist = distance_transform_edt(~path)
    log(f"  软约束重寻路（{time.time() - t0:.0f}s，λbase={a.soft_z_lambda:g}）："
        f"路径 {int(path.sum())}px")
    return path, dist


# ================================================================ 第 3 步
def build_inputs(hs, cssc, valid, dist, cost, in_ch, gate=3.0, log=print, tpi=None):
    """模型输入。

    `in_ch=2`：只有 Hillshade + CSSC（局部视觉，完全不含任何宏观走向信息）。
    `in_ch=4`：再加支路 1 的两个宏观通道——路径距离场 `clip(1−d/gate)` 与
               openness 走廊响应 `1−cost`（都是无标签的自监督量，不含 SHP）。
    2ch 版本在实测里会塌缩成常数场（局部特征不足以定位一条 400 m 长的全局路径），
    4ch 才具备"新站点可用"的判别力。
    """
    cmax = float(np.nanmax(cssc[valid]))
    cssc_n = np.where(valid, np.clip(cssc / max(cmax, 1e-9), 0.0, 1.0), 0.0)
    chans = [np.where(valid, hs, 0.0), cssc_n]
    names = ["Hillshade", f"CSSC÷{cmax:.0f}"]
    if in_ch >= 4:
        ch0 = np.where(valid, np.clip(1.0 - dist / float(gate), 0.0, 1.0), 0.0)
        ch1 = np.where(valid, np.clip(1.0 - (cost - 0.05), 0.0, 1.0), 0.0)
        chans += [ch0, ch1]
        names += ["路径距离场", "openness 走廊响应"]
    if tpi is not None:
        # 多尺度地形位置指数（30 m）：路堑/切坡 → TPI 显著为负
        v = tpi[valid]
        lo, hi = np.percentile(v, [2.0, 98.0])
        chans.append(np.where(valid, np.clip((tpi - lo) / max(hi - lo, 1e-9), 0, 1), 0.0))
        names.append("TPI30")
    X = np.stack(chans, axis=0)
    log(f"  模型输入 {X.shape}（{X.shape[0]}ch：" + " + ".join(names) + "）")
    return X, cmax


def cold_train(site, X, pseudo, valid, a, log=print):
    """现场重训：全新初始化（不载入任何旧权重）。"""
    import torch
    tag = f"{site}{a.tag}"
    net, hist = train_net(X, pseudo.astype(np.float32), valid, a.epochs, a.batch,
                          a.lr, a.patch, a.seed, a.device, log)
    npar = sum(p.numel() for p in net.parameters())
    p = os.path.join(RES, f"checkpoint_{tag}.pth")
    torch.save(dict(state_dict=net.state_dict(), site=site, in_ch=int(X.shape[0]),
                    params=int(npar), epochs=len(hist), pseudo_px=int(pseudo.sum()),
                    seed=a.seed, features=["hillshade", "cssc"]), p)
    log(f"  保存 checkpoint → {os.path.relpath(p, ROOT)}（{npar / 1e3:.2f}k 参数，"
        f"{len(hist)} epoch）")
    return net, hist, npar, p


# ================================================================ 第 4 步
def project_points(site, prob, prob_old, dem, meta, a, log=print):
    """概率图按网格契约反投影回 3D 点云，同时算 AGL 与首/末回波。"""
    H, W = prob.shape
    x0, y1, gsd = float(meta["x0"]), float(meta["y1"]), float(meta["gsd_m"])
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    ft = np.load(os.path.join(REAL, f"{site}_lasefeat.npy"), mmap_mode="r")
    n = xyz.shape[0]
    valid = np.isfinite(dem)

    g_core = (prob >= a.thr) & valid
    g_lo = (prob >= a.thr_lo) & valid
    g_dil = (binary_dilation(g_core, disk(a.dil_px)) & valid) if a.dil_px > 0 else g_core
    vmask = valid.astype(np.float64)
    demg = np.where(valid, dem, 0.0)

    pv = np.zeros(n, dtype=np.float32)
    pvo = np.zeros(n, dtype=np.float32) if prob_old is not None else None
    agl = np.full(n, 1e3, dtype=np.float32)
    echo = np.zeros(n, dtype=bool)
    hit_lo = np.zeros(n, dtype=bool)
    hit_dil = np.zeros(n, dtype=bool)

    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        p = np.asarray(xyz[s:e], dtype=np.float64) + np.asarray(o, dtype=np.float64)[None, :]
        c = (p[:, 0] - x0) / gsd - 0.5
        r = (y1 - p[:, 1]) / gsd - 0.5
        ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
        ii = np.flatnonzero(ins)
        if ii.size:
            v = bilinear(prob, r[ii], c[ii])
            vb = bilinear(vmask, r[ii], c[ii])
            inside = vb > 0.5
            pv[s + ii] = np.where(inside, v, 0.0)
            agl[s + ii] = np.where(inside, p[ii, 2] - bilinear(demg, r[ii], c[ii]), 1e3)
            rr, cc2 = np.rint(r[ii]).astype(np.int32), np.rint(c[ii]).astype(np.int32)
            hit_lo[s + ii] = g_lo[rr, cc2]
            hit_dil[s + ii] = g_dil[rr, cc2]
            if pvo is not None:
                vo = bilinear(prob_old, r[ii], c[ii])
                pvo[s + ii] = np.where(inside, vo, 0.0)
        f = np.asarray(ft[s:e])
        rn, nr = np.rint(f[:, 0]).astype(np.int32), np.rint(f[:, 1]).astype(np.int32)
        echo[s:e] = (rn == 1) | (rn == nr)
        if (s // CHUNK) % 4 == 0:
            log(f"    反投影 {e / 1e6:.0f}M / {n / 1e6:.1f}M（{time.time() - t0:.0f}s）")

    label = np.asarray(np.load(os.path.join(REAL, f"{site}_label.npy"), mmap_mode="r"),
                       dtype=bool)
    log(f"  反投影完成（{time.time() - t0:.0f}s）：P 分位 P50/P90/P99 = "
        + " / ".join(f"{v:.3f}" for v in np.percentile(pv, [50, 90, 99]))
        + f"；AGL 有效 {int((agl < 1e3).sum()) / n * 100:.1f}%；首末回波 {echo.mean() * 100:.1f}%")
    return dict(pv=pv, pv_old=pvo, agl=agl, echo=echo, label=label, n=n,
                hit_lo=hit_lo, hit_dil=hit_dil)


def metrics(keep, label):
    """点空间 P/R/F1/IoU（SHP 只在最后参与这里）。"""
    tp = int((keep & label).sum())
    fp = int((keep & ~label).sum())
    fn = int((~keep & label).sum())
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, keep=int(keep.sum()),
                keep_frac=float(keep.mean()), truth=int(label.sum()),
                precision=float(pr), recall=float(rc),
                f1=float(2 * pr * rc / max(pr + rc, 1e-9)),
                iou=float(tp / max(tp + fp + fn, 1)))


def dplus(pt, prob_key, a, log=print):
    """D+ 高召回：核心 P≥0.5 全留 + C 增量区按 AGL 一维 KMeans 只留地面簇。"""
    from balance_filter import kmeans_agl_split
    pv, agl = pt[prob_key], pt["agl"]
    base = pv >= a.thr
    band = (pv >= a.thr_lo) & (pv < a.thr)
    c_main = pt["hit_lo"] | (pt["echo"] & pt["hit_dil"])
    addon = c_main & ~base
    info = kmeans_agl_split(agl[addon], a.kmeans_seed, a.kmeans_fit, log, "D+ 增量区")
    ground = addon & (agl <= info["bound_m"])
    keep = base | ground
    return keep, info, dict(base=int(base.sum()), band=int(band.sum()),
                            addon=int(addon.sum()), ground=int(ground.sum()))


def pixel_metrics(mask, label, valid, tag, log=print):
    """栅格口径（与 fusion_train 的 IoU 同口径），仅作对照。"""
    inter = int((mask & label & valid).sum())
    union = int(((mask | label) & valid).sum())
    tp, fp = inter, int((mask & ~label & valid).sum())
    fn = int((~mask & label & valid).sum())
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    d = dict(iou=float(inter / max(union, 1)), f1=float(2 * pr * rc / max(pr + rc, 1e-9)),
             precision=float(pr), recall=float(rc), pred_px=int((mask & valid).sum()))
    log(f"  [{tag}] 栅格 IoU={d['iou']:.4f} F1={d['f1']:.4f} "
        f"P={d['precision']:.4f} R={d['recall']:.4f}（{d['pred_px']:,}px）")
    return d


def write_las(path, site, idx, log=print):
    """交付：把保留点写成 LAS（坐标回到绝对坐标系）。"""
    try:
        import laspy
    except ImportError:
        log("  未安装 laspy，跳过 LAS 输出（索引已保存）")
        return None
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    p = np.asarray(xyz[idx], dtype=np.float64) + np.asarray(o, dtype=np.float64)[None, :]
    hdr = laspy.LasHeader(point_format=0, version="1.4")
    off = p.min(axis=0)
    hdr.offsets = [float(v) for v in off]
    hdr.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = p[:, 0], p[:, 1], p[:, 2]
    las.write(path)
    log(f"  交付点云 {p.shape[0]:,} 点 → {os.path.relpath(path, ROOT)}"
        f"（{os.path.getsize(path) / 1e6:.1f} MB）")
    return path


# ================================================================ 主流程
def run_site(site, a, log):
    t0 = time.time()
    log(f"\n[{site}] 冷启动：只给 DEM（由原始点云栅格化）+ 起止点")
    dem, valid = load_dem(site, a.base, log)          # AGL 基准（默认 dem_min）
    if a.feat_base == a.base:
        dem_f = dem
    else:
        dem_f, _ = load_dem(site, a.feat_base, log)   # 地形形态特征基准（默认 dem_mean）
    meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
    hs = hillshade_map(dem_f, valid)
    log("  特征 1/3：Hillshade 完成")

    # ---- 第 1 步：宏观场 + 局部特征
    op, n_row, n_col, conf, minfo = macro_fields(
        dem_f, valid, radii=(a.op_r.split(",") if getattr(a, "op_r", "") else None))
    log(f"  特征 2/3：openness[{minfo['open_range'][0]:.1f},{minfo['open_range'][1]:.1f}]°，"
        f"h1 相位门控 P60={minfo['h1_gate_deg']}°（{minfo['conf_px']:,}px）"
        f"（{minfo['seconds']}s）")
    cssc = cssc_map(dem_f, valid, n_row, n_col, log)
    log("  特征 3/3：CSSC 完成")

    # ---- 第 2 步：支路 1 寻路 → 伪标签
    if a.ep:
        ep = [np.array([float(v) for v in q.split(",")]) for q in a.ep.split(";")]
        ep_info = dict(source="cli", ep0=ep[0].tolist(), ep1=ep[1].tolist())
    else:
        ep, ep_info = auto_endpoints(cssc, valid, a.pct, a.r, log)
    ep = [np.array(P.snap_to_domain(p, valid)) for p in ep]
    path, dist, cost, crange = macro_path(op, valid, ep, n_row, n_col, conf, a, log,
                                          dem=dem_f)

    # ---- 视觉置信度熔断（无 SHP）：地形项不得以牺牲视觉连贯性为代价
    bkinfo = dict(base_cost=None, lz_cost=None, delta=None, triggered=False,
                  lambda_z=float(getattr(a, "z_lambda", LAMBDA_Z)))
    if bkinfo["lambda_z"] > 0 and not getattr(a, "no_breaker", False):
        from argparse import Namespace as _NS
        a0 = _NS(**vars(a))
        a0.z_lambda = 0.0
        path0, dist0, _, _ = macro_path(op, valid, ep, n_row, n_col, conf, a0, log,
                                       dem=dem_f)
        c0 = float(cost[path0].mean())
        c1 = float(cost[path].mean())
        delta = (c1 - c0) / max(c0, 1e-9)
        bkinfo.update(base_cost=round(c0, 4), lz_cost=round(c1, 4),
                      delta=round(delta, 4))
        log(f"  熔断检查：λz=0 视觉代价 {c0:.3f} → λz={bkinfo['lambda_z']:g} "
            f"{c1:.3f}，膨胀率 {delta * 100:+.1f}%（阈值 {BREAKER_DELTA * 100:.0f}%）")
        if delta > BREAKER_DELTA:
            bkinfo["triggered"] = True
            path, dist = path0, dist0
            log("  **触发熔断**：地形项破坏了视觉连贯性 → 自动回退 λz=0")
    pseudo = (dist <= float(a.gate)) & valid
    log(f"  伪标签 = 路径膨胀 {a.gate:g}px：{int(pseudo.sum()):,}px"
        f"（{int(pseudo.sum()) / int(valid.sum()) * 100:.2f}% 的 valid）")

    # ---- A 多假设软标签：多个 λz 各跑一条路径，按高斯距离取共识（无 SHP）
    #      标签从"单条带 0/1"变成"被多少条假设支持"，带宽不再由单条 gate 决定
    if getattr(a, "multi_hyp", ""):
        import copy as _copy
        lams = [float(v) for v in a.multi_hyp.split(",") if v.strip()]
        ysum = np.zeros(valid.shape, np.float64)
        dmin = np.full(valid.shape, np.inf)
        votes = np.zeros(valid.shape, np.int32)
        import json as _json
        _meta = _json.load(open(os.path.join(ROOT, "data", f"meta_{site}.json"),
                                encoding="utf-8"))
        _gsd = float(_meta.get("gsd_m", 1.0))
        for lam in lams:
            a0 = _copy.copy(a)
            a0.z_lambda = lam
            a0.no_breaker = True          # 假设生成不受熔断影响，保证路径多样性
            pk, dk, _, _ = macro_path(op, valid, ep, n_row, n_col, conf, a0, log,
                                      dem=dem_f)
            d_m = dk * _gsd
            ysum += np.exp(-(d_m ** 2) / (2.0 * float(a.hyp_sigma) ** 2))
            dmin = np.minimum(dmin, d_m)
            votes += ((dk <= float(a.gate)) & valid).astype(np.int32)
            log(f"    假设 λz={lam:g}：路径 {int(pk.sum()):,}px，"
                f"带内 {int(((dk <= float(a.gate)) & valid).sum()):,}px")
        # 硬标签 = 多数假设共识（≥一半假设的带覆盖），避免软标签把整幅图变成弱正样本
        k_need = max(1, int(np.ceil(len(lams) / 2.0)))
        pseudo = (votes >= k_need) & valid
        log(f"  多假设共识标签：{len(lams)} 条路径，需 ≥{k_need} 票 → "
            f"{int(pseudo.sum()):,}px（并集 {int((votes > 0).sum()):,}px）")

    # ---- 第 3 步：支路 2 现场重训
    tpi = None
    if a.tpi:
        from scipy.ndimage import gaussian_filter as _gf, uniform_filter as _uf
        _zs = _gf(dem_f, 1.0)
        tpi = _zs - _uf(_zs, 31, mode="nearest")       # 30 m 窗口 TPI
        log(f"  TPI30 = 1m DEM − 30m 均值：{np.nanmin(tpi):.2f} ~ {np.nanmax(tpi):.2f} m")
    X, cmax = build_inputs(hs, cssc, valid, dist, cost, a.in_ch, a.gate, log, tpi=tpi)
    net, hist, npar, ckpt = cold_train(site, X, pseudo, valid, a, log)
    prob = infer(net, X, valid, a.device, log=log)

    # ---- B-2 自训练扩展：用模型自身 P≥st_prob 的微弱响应扩标签后重训
    #      距离上限 min(st_reach, st_half) 是硬约束，防止逐轮自激放大
    if getattr(a, "selftrain", 0) > 0:
        r_ext = min(float(a.st_reach), float(a.st_half))
        for it in range(1, a.selftrain + 1):
            ext = (prob >= float(a.st_prob)) & (dist <= r_ext) & valid
            new_pseudo = pseudo | ext
            log(f"  自训练 第{it}轮：P≥{a.st_prob:g} 且距轴≤{r_ext:g}m 新增 "
                f"{int((ext & ~pseudo).sum()):,}px → 标签 {int(new_pseudo.sum()):,}px")
            pseudo = new_pseudo
            net, hist, npar, ckpt = cold_train(site, X, pseudo, valid, a, log)
            prob = infer(net, X, valid, a.device, log=log)

    # ---- 第 2.5 步（可选）：高差 × 概率软约束的第二遍寻路
    if getattr(a, "soft_z_lambda", 0.0) > 0:
        p2, d2 = refine_path_soft(op, valid, ep, n_row, n_col, conf, prob, dem_f, a, log)
        if p2 is not None:
            path, dist = p2, d2

    # ---- 第 4 步：反投影 + D+
    prob_old = None
    if not a.no_compare:
        pz = os.path.join(RES, f"model_guided_prob_{site}.npz")
        if os.path.exists(pz):
            prob_old = np.load(pz)["prob"].astype(np.float64)
            log("  对照：载入同站域内 8ch 模型概率图（仅作对比，不参与冷启动推理）")
    pt = project_points(site, prob, prob_old, dem, meta, a, log)
    keep, kinfo, counts = dplus(pt, "pv", a, log)
    log(f"  D+ 结构：核心 {counts['base']:,} / 缓冲带(0.1≤P<0.5) {counts['band']:,} / "
        f"C 增量区 {counts['addon']:,} / 其中地面簇 {counts['ground']:,}")
    log(f"  **冷启动 D+ 自动地面/树冠分界 = {kinfo['bound_m']:.2f} m**")

    # ---- 第 5 步：评分 + 交付（SHP 此刻才读）
    label = pt["label"]
    rows = [dict(name="纯 2D P≥0.5（新模型）", grp="base", **metrics(pt["pv"] >= a.thr, label)),
            dict(name="纯 2D P≥0.1（新模型）", grp="S", **metrics(pt["pv"] >= a.thr_lo, label)),
            dict(name="C 膨胀并集（新模型）", grp="C",
                 **metrics(pt["hit_lo"] | (pt["echo"] & pt["hit_dil"]), label)),
            dict(name=f"D+ 冷启动（AGL≤{kinfo['bound_m']:.2f}m）", grp="D+", **metrics(keep, label))]
    if prob_old is not None:
        keep_o, kinfo_o, _ = dplus(pt, "pv_old", a, log)
        rows.append(dict(name="参照：域内 8ch 模型的 D+", grp="old", **metrics(keep_o, label)))
        rows.append(dict(name="参照：域内 8ch P≥0.5", grp="old",
                         **metrics(pt["pv_old"] >= a.thr, label)))

    for r in rows:
        log(f"  {r['name']:<26s} 保留 {r['keep']:>10,d}（{r['keep_frac'] * 100:5.2f}%）"
            f"  P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} IoU={r['iou']:.4f}")

    z = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    shp = z["label"].astype(bool) if "label" in z else None
    pix = dict(pseudo=pixel_metrics(pseudo >= 0.5, shp, valid, "伪标签（支路1）", log),
               prob05=pixel_metrics((prob >= a.thr) & valid, shp, valid, "新模型 prob≥0.5", log))
    log(f"  模型 vs 伪标签 IoU（栅格）= "
        f"{float(((prob >= a.thr) & (pseudo >= 0.5) & valid).sum() / max(((prob >= a.thr) | (pseudo >= 0.5)).sum(), 1)):.4f}")

    idx = np.flatnonzero(keep)
    fbase = f"cold_start_{site}{a.tag}"
    np.save(os.path.join(RES, f"{fbase}_keep_idx.npy"), idx.astype(np.int64))
    out = dict(site=site, tag=a.tag, in_ch=a.in_ch, params=vars(a).copy(),
               endpoints=ep_info, macro=minfo, pseudo_px=int(pseudo.sum()),
               breaker=bkinfo,
               model=dict(params=int(npar), epochs=len(hist), hist=[round(h, 6) for h in hist],
                          ckpt=os.path.relpath(ckpt, ROOT)),
               kmeans=kinfo, counts=counts,
               point_level=rows, pixel_level=pix,
               keep_idx_file=os.path.relpath(os.path.join(RES, f"{fbase}_keep_idx.npy"), ROOT),
               seconds=round(time.time() - t0, 1))
    np.savez_compressed(os.path.join(RES, f"{fbase}.npz"),
                        prob=prob.astype(np.float32), path=path.astype(np.uint8),
                        pseudo=np.asarray(pseudo, np.float32),
                        cssc=cssc.astype(np.float32),
                        openness=op.astype(np.float32))
    if not a.no_las:
        out["las"] = os.path.relpath(
            write_las(os.path.join(RES, f"{fbase}_corridor.las"), site, idx, log), ROOT)
    return out


def main():
    p = argparse.ArgumentParser(description="冷启动迁移：新站点五步走（无预训练权重）")
    p.add_argument("--site", default="杨家坝")
    p.add_argument("--base", choices=("min", "mean"), default="min",
                   help="AGL 用的基准地形（默认 dem_min）")
    p.add_argument("--feat-base", choices=("min", "mean"), default="mean",
                   help="Hillshade/CSSC/openness/h1 用的基准地形（默认 dem_mean："
                        "dem_min 的单点低值会在 1 m 格内造成尖刺，抖坏形态特征）")
    p.add_argument("--ep", type=str, default=None, help='起止点 "r,c;r,c"；缺省由 CSSC 自动取')
    p.add_argument("--pct", type=float, default=95.0, help="自动端点用的 CSSC 分位")
    p.add_argument("--r", type=int, default=5, help="自动端点的 CSSC 膨胀半径 px")
    p.add_argument("--gate", type=float, default=3.0, help="伪标签路径膨胀带宽 px")
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    p.add_argument("--slope-lambda", type=float, default=0.0,
                   help="坡度惩罚权重：cost += λ·norm(slope_deg, P5~P95)；0=关闭（默认，可复现）")
    p.add_argument("--tpi", action="store_true",
                   help="追加第 3 通道：30 m 窗口地形位置指数 TPI（路堑为负，无 SHP）")
    p.add_argument("--selftrain", type=int, default=0,
                   help="B-2 自训练扩展轮数（0=关闭）：用模型自身 P≥st_prob 的微弱响应扩标签后重训")
    p.add_argument("--multi-hyp", default="",
                   help="A 多假设软标签：逗号分隔的 λz 列表（如 0,1,2,4），"
                        "各跑一条路径，按高斯距离取共识作为软标签；空=关闭")
    p.add_argument("--op-r", default="",
                   help="openness 半径（米）。单个值=原行为；逗号分隔多个值=多尺度归一化融合")
    p.add_argument("--hyp-sigma", type=float, default=3.0,
                   help="多假设软标签的横向带宽 σ（米）")
    p.add_argument("--st-prob", type=float, default=0.3, help="自训练扩展的概率下限")
    p.add_argument("--st-reach", type=float, default=15.0, help="自训练扩展的距离上限（米）")
    p.add_argument("--st-half", type=float, default=7.0,
                   help="扩展后标签的最大半宽（米，硬上限，防自激振荡）")
    p.add_argument("--z-lambda", type=float, default=LAMBDA_Z,
                   help=f"沿程高差惩罚权重（默认锁死 LAMBDA_Z={LAMBDA_Z}）；"
                        f"仅作消融时显式覆盖，如 --z-lambda 0 复现旧基线")
    p.add_argument("--max-grade", type=float, default=0.0,
                   help="方案一：硬纵坡上限（如 0.20）。超过该纵坡的边直接删除；0=关闭")
    p.add_argument("--lam-c", type=float, default=LAMBDA_C,
                   help=f"方向余弦惩罚权重（默认锁死 LAMBDA_C={LAMBDA_C}）；"
                        f"pen=|Δz|/(|∇z|·L+ε)≈|cosθ|，只罚横切坡面、与地形陡缓无关；"
                        f"开启时自动替代 λz；0=关闭（回到 LAMBDA_Z 口径）")
    p.add_argument("--grade-lag", type=int, default=5,
                   help="硬纵坡的窗口半宽（px），2×lag 即窗口米数（默认 5 → 10 m）")
    p.add_argument("--soft-z-lambda", type=float, default=0.0,
                   help="方案二：高差×概率软约束 λbase，边权 += λ·|Δz|·(1−P)；0=关闭")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--thr", type=float, default=0.5, help="核心概率阈值")
    p.add_argument("--thr-lo", type=float, default=0.10, help="C 的低阈值")
    p.add_argument("--dil-px", type=int, default=10, help="C 的膨胀半径 px")
    p.add_argument("--kmeans-seed", type=int, default=0)
    p.add_argument("--kmeans-fit", type=int, default=2_000_000)
    p.add_argument("--no-compare", action="store_true", help="不加载域内 8ch 模型作对照")
    p.add_argument("--no-las", action="store_true", help="不写 LAS 交付文件")
    p.add_argument("--in-ch", type=int, choices=(2, 4), default=4,
                   help="模型输入通道数：2=Hillshade+CSSC（局部）；4=再加路径距离场与 openness")
    p.add_argument("--tag", type=str, default="",
                   help="输出文件后缀，用于同站多版本对照（如 --tag _2ch）")
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "cold_start_transfer.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True); lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"冷启动迁移测试 cold_start_transfer · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"站点 {a.site}：当作从未见过；输入 = 1m DEM（dem_{a.base}）+ 起止点坐标")
    log("流程：DEM/特征 → 宏观 Dijkstra 伪标签 → 现场重训 2ch 微型 CNN → 反投影 → D+ → SHP 评分")
    log("SHP 全程不参与训练与推理，只在最后评分")

    t0 = time.time()
    out = run_site(a.site, a, log)
    log("\n" + "-" * 78)
    for r in out["point_level"]:
        log(f"  {r['name']:<26s} P={r['precision']:.4f} R={r['recall']:.4f} "
            f"F1={r['f1']:.4f} IoU={r['iou']:.4f}")
    log(f"总耗时 {time.time() - t0:.0f}s")

    jp = os.path.join(RES, f"cold_start_{a.site}{a.tag}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
