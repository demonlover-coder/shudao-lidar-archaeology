# -*- coding: utf-8 -*-
"""扩充地形因子池：20 个因子 → `data/feature_matrix_{site}.npy`，形状 **[N_valid, 20]**。

动机
----
单因子走廊区分能力上限实测不存在有效 AUC > 0.78 的因子（tpi31 最高，**填充修正口径**
0.6691(西)/0.7148(杨)，原 0.7285/0.7711 含 nodata 边界污染虚高约 0.058，
见 `tools/fix_tpi31_auc.py`），走廊判定本质是**多因子上下文**问题。
本脚本把现有 + 可算的因子全部落成矩阵，
交给 `tools/sparse_pca.py` 学"超级特征"。

20 个因子（顺序固定，即矩阵列序）
----
   已有 8（来源 `p2_terrain.terrain_stack`，定义完全一致）：
     slope / sin_aspect / cos_aspect / curvature / roughness / profile_curv / tpi15 / tpi31
   新增 12：
     plan_curv     垂直坡向的二阶方向导（Hessian 在等高线方向的投影，1/m）
     relief_15     local_max − local_min，k=15（同 31）
     relief_31     local_max − local_min，k=31
     TRI           mean(|z − z_center|)，3×3（Riley 1999）
     tpi7          z − mean(z, 7px)
     tpi63         z − mean(z, 63px)
     openness      Yokoyama **正向（向上）**地平线开放度（度）；反向 = 对 −z 同式
     surface_ratio 真实表面积 / 平面投影面积 = √(1 + |∇z|²)
     cliff_prox    到 slope>60° 区域的欧氏距离（EDT，米）
     ridge_prox    到山脊线（局部极大 & tpi15>0）的 EDT（米）
     valley_prox   到沟谷线（局部极小 & tpi15<0）的 EDT（米）
     aspect_std    坡向在 15×15 窗内的**圆标准差**（度，√(−2 ln R)，非线性格差）

处理约定（重要）
----
1. **nodata 填充**：`dem_mean` 在 valid 之外是 **−9999**，直接进 3×3 卷积会污染
   valid 最外一圈（实测 slope 恒 90°、curvature 掉到 −6755，占 valid 1.26%/1.84%）。
   本脚本先按"最近 valid 像元"填充（`distance_transform_edt` 的 index 形式）再算因子，
   保证所有因子在 valid 内**无非物理离群值**。valid 内部取值与未填充版**完全一致**，
   仅最外一圈不同；AUC 口径因此与 `shudao_auc.py` 有极小差异（≤边界环占比）。
2. **行序**：矩阵行 = `np.flatnonzero(valid.ravel())`（C 序），下游脚本用同一表达式
   还原栅格，无需额外存索引。
3. **NaN**：若某因子在站内无有效值（如无 cliff 像元），该列以列中位数填补，
   并在 meta json 的 `n_nan` 中记录。
4. 因子均为**原始量纲**（度 / 米 / 1·m⁻¹ / 无量纲），标准化交给 `sparse_pca.py`。

用法
----
    python tools/build_feature_matrix.py                    # 两站
    python tools/build_feature_matrix.py --site 西坝村
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import (distance_transform_edt, maximum_filter,
                           minimum_filter, sobel, uniform_filter)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from p2_terrain import CHANNELS, terrain_stack   # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
GSD = 1.0

FACTORS = (
    # —— 已有 8（p2_terrain 口径）——
    "slope", "sin_aspect", "cos_aspect", "curvature",
    "roughness", "tpi15", "tpi31", "profile_curv",
    # —— 新增 12 ——
    "plan_curv", "relief_15", "relief_31", "TRI", "tpi7", "tpi63",
    "openness", "surface_ratio", "cliff_prox", "ridge_prox", "valley_prox",
    "aspect_std",
)
LEGACY = FACTORS[:8]
NEW = FACTORS[8:]

# 因子 → 物理含义短词（供 sparse_pca / factor_name_auc 拼"超级特征"名字）
NAME_HINT = {
    "slope": "坡度强度", "sin_aspect": "坡向(南北)", "cos_aspect": "坡向(东西)",
    "curvature": "曲面弯曲", "roughness": "粗糙度", "tpi15": "局部洼凸",
    "tpi31": "低洼度", "profile_curv": "坡面弯曲", "plan_curv": "等高线弯曲",
    "relief_15": "起伏量", "relief_31": "宏观起伏", "TRI": "地形起伏",
    "tpi7": "微地形洼凸", "tpi63": "宏观洼凸", "openness": "天空开放度",
    "surface_ratio": "表面积膨胀", "cliff_prox": "崖壁邻接",
    "ridge_prox": "山脊邻接", "valley_prox": "沟谷邻接", "aspect_std": "坡向混乱度",
}


# ------------------------------------------------------------ 基础算子 ----
def fill_nodata(dem: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    """把 valid 之外（含 nodata）用**最近 valid 像元**填充，返回 (filled, 填充像元数)。"""
    dem = np.asarray(dem, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return dem.copy(), 0
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return dem[tuple(idx)], int((~valid).sum())


def _slice(zp, H, W, R, dr, dc):
    """从已 pad R 的数组取"整幅平移 (dr, dc)"的视图（对应原图 [0:H, 0:W]）。"""
    return zp[R + dr: R + dr + H, R + dc: R + dc + W]


def tri(z: np.ndarray, size: int = 3) -> np.ndarray:
    """TRI = 邻域内 |z − z_center| 的均值（size×size，边缘 replicate）。"""
    R = size // 2
    H, W = z.shape
    zp = np.pad(z, R, mode="edge")
    acc = np.zeros_like(z)
    n = 0
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            if dr == 0 and dc == 0:
                continue
            acc += np.abs(_slice(zp, H, W, R, dr, dc) - z)
            n += 1
    return acc / n


def relief(z: np.ndarray, k: int) -> np.ndarray:
    """局部起伏 = k×k 窗内 max − min。"""
    return maximum_filter(z, size=k) - minimum_filter(z, size=k)


def openness(z: np.ndarray, gsd: float = 1.0, R: int = 30,
             positive: bool = True, n_az: int = 8) -> np.ndarray:
    """Yokoyama 地平线开放度（度）。

    `positive=True` → 向上开放度（对 z 本身）；`False` → 向下开放度（对 −z）。
    沿 8 个方位、R 米步长取"最大地平仰角" θ，开放度 = 90° − θ。
    纯球面／平面 → 90°；深沟 / 陡崖下方 → 趋近 0°。
    """
    zz = np.asarray(z, dtype=np.float64) if positive else -np.asarray(z, dtype=np.float64)
    H, W = zz.shape
    zp = np.pad(zz, R, mode="edge")
    dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    if n_az < 8:
        dirs = dirs[::max(1, 8 // n_az)]
    top = np.full((H, W), -np.pi / 2.0)
    for dr, dc in dirs:
        step = gsd * float(np.hypot(dr, dc))
        for r in range(1, R + 1):
            zn = _slice(zp, H, W, R, r * dr, r * dc)
            np.maximum(top, np.arctan2(zn - zz, r * step), out=top)
    return 90.0 - np.degrees(top)


def plan_curvature(z: np.ndarray, gsd: float) -> np.ndarray:
    """平面曲率 = Hessian 在**垂直于坡向**方向（等高线方向）的二阶方向导（1/m）。

    与 `profile_curv`（沿坡向）正交同族；符号约定：凸出等高线为正。
    """
    dzdx = sobel(z, axis=1) / (8 * gsd)
    dzdy = sobel(z, axis=0) / (8 * gsd)
    d2zdx2 = np.gradient(dzdx, gsd, axis=1)
    d2zdy2 = np.gradient(dzdy, gsd, axis=0)
    d2zdxy = np.gradient(dzdx, gsd, axis=0)
    mag = np.hypot(dzdx, dzdy) + 1e-9
    vx, vy = -dzdy / mag, dzdx / mag          # 垂直于坡向的单位向量
    return vx ** 2 * d2zdx2 + 2.0 * vx * vy * d2zdxy + vy ** 2 * d2zdy2


def aspect_circular_std(aspect_deg: np.ndarray, k: int = 15) -> np.ndarray:
    """坡向的圆标准差（度）：√(−2 ln R)，R = |窗口内单位向量合成长度|（[0,1]）。"""
    a = np.radians(np.asarray(aspect_deg, dtype=np.float64))
    ms = uniform_filter(np.sin(a), size=k, mode="nearest")
    mc = uniform_filter(np.cos(a), size=k, mode="nearest")
    Rlen = np.clip(np.hypot(ms, mc), 1e-6, 1.0)
    return np.degrees(np.sqrt(-2.0 * np.log(Rlen)))


def _edt_to(mask: np.ndarray, shape) -> np.ndarray:
    """到 mask=True 区域的欧氏距离（米）；mask 全 False 时返回全 NaN。"""
    if not mask.any():
        return np.full(shape, np.nan)
    return distance_transform_edt(~mask).astype(np.float64)


def build_factors(dem_filled: np.ndarray, valid: np.ndarray,
                  gsd: float = GSD, verbose: bool = True) -> dict:
    """算 20 个因子，返回 {因子名: [H, W]}（顺序同 FACTORS）。"""
    z = np.asarray(dem_filled, dtype=np.float64)
    t0 = time.time()
    stack = terrain_stack(z, gsd=gsd)            # [8, H, W]，CHANNELS 顺序
    legacy = {n: np.asarray(a, dtype=np.float64) for n, a in zip(CHANNELS, stack)}
    if verbose:
        print(f"    已有 8 通道（p2_terrain） {time.time() - t0:6.1f}s")

    t0 = time.time()
    dzdx = sobel(z, axis=1) / (8 * gsd)
    dzdy = sobel(z, axis=0) / (8 * gsd)
    tpi15 = legacy["tpi15"]
    out = dict(legacy)
    out["plan_curv"] = plan_curvature(z, gsd)
    out["relief_15"] = relief(z, 15)
    out["relief_31"] = relief(z, 31)
    out["TRI"] = tri(z, 3)
    out["tpi7"] = z - uniform_filter(z, size=7, mode="nearest")
    out["tpi63"] = z - uniform_filter(z, size=63, mode="nearest")
    out["surface_ratio"] = np.sqrt(1.0 + dzdx ** 2 + dzdy ** 2)
    # 坡向在 CHANNELS 里已被展开为 sin/cos，这里按 p2_terrain 同式重算裸角度
    aspect = np.degrees(np.arctan2(dzdy, -dzdx)) % 360.0
    out["aspect_std"] = aspect_circular_std(aspect, k=15)
    if verbose:
        print(f"    新增局部因子        {time.time() - t0:6.1f}s")

    t0 = time.time()
    out["openness"] = openness(z, gsd=gsd, R=30, positive=True)
    if verbose:
        print(f"    openness(R=30m, 8方位) {time.time() - t0:6.1f}s")

    t0 = time.time()
    slope = legacy["slope"]
    cliff = valid & (slope > 60.0)
    ridge = valid & (z >= maximum_filter(z, size=15)) & (tpi15 > 0)
    valley = valid & (z <= minimum_filter(z, size=15)) & (tpi15 < 0)
    out["cliff_prox"] = _edt_to(cliff, z.shape)
    out["ridge_prox"] = _edt_to(ridge, z.shape)
    out["valley_prox"] = _edt_to(valley, z.shape)
    if verbose:
        print(f"    崖/脊/谷 EDT         {time.time() - t0:6.1f}s"
              f"  崖={int(cliff.sum()):,} 脊={int(ridge.sum()):,} 谷={int(valley.sum()):,}")
    return {k: out[k] for k in FACTORS}


# ------------------------------------------------------------------ 主流程 ----
def run_site(site: str, verbose: bool = True) -> dict:
    npz = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    dem = npz["dem_mean"].astype(np.float64)
    valid = npz["valid"] == 1
    label = npz["label"].astype(bool)
    H, W = dem.shape
    n_val = int(valid.sum())
    print(f"\n[{site}] 栅格 {H}×{W}，valid {n_val:,}，走廊 {int((label & valid).sum()):,}")

    dem_f, n_fill = fill_nodata(dem, valid)
    print(f"  nodata 填充 {n_fill:,} 像元（最近 valid 邻居）")
    fac = build_factors(dem_f, valid, gsd=GSD, verbose=verbose)

    rows = np.flatnonzero(valid.ravel())
    cols, nans, stats = [], [], []
    for name in FACTORS:
        a = np.asarray(fac[name], dtype=np.float64).ravel()[rows]
        n_nan = int((~np.isfinite(a)).sum())
        if n_nan:
            a = np.where(np.isfinite(a), a, np.nanmedian(a) if np.isfinite(a).any() else 0.0)
        cols.append(a.astype(np.float32))
        nans.append(n_nan)
        stats.append((name, float(np.nanmin(a)), float(np.percentile(a, 1)),
                      float(np.median(a)), float(np.percentile(a, 99)), float(np.nanmax(a))))
    X = np.stack(cols, axis=1)                  # [N_valid, 20] float32

    print(f"  因子矩阵 {X.shape}（dtype={X.dtype}）")
    print(f"  {'因子':<15s}{'min':>11s}{'p1':>11s}{'median':>11s}{'p99':>11s}{'max':>11s}{'NaN':>6s}")
    for name, lo, p1, md, p99, hi in stats:
        print(f"  {name:<15s}{lo:>11.4f}{p1:>11.4f}{md:>11.4f}{p99:>11.4f}{hi:>11.4f}"
              f"{nans[FACTORS.index(name)]:>6d}")
    print("  列间量级比（|median| 最大/最小 非零）："
          f"{max(abs(s[3]) for s in stats) / max(min(abs(s[3]) for s in stats if abs(s[3]) > 0), 1e-12):.1f}×")

    os.makedirs(DATA, exist_ok=True)
    npy = os.path.join(DATA, f"feature_matrix_{site}.npy")
    np.save(npy, X)
    meta = dict(site=site, shape=list(X.shape), dtype=str(X.dtype), gsd=GSD,
                factors=list(FACTORS), legacy=list(LEGACY), new=list(NEW),
                name_hint=NAME_HINT, row_order="np.flatnonzero((npz['valid']==1).ravel())",
                n_valid=n_val, n_corridor=int((label & valid).sum()),
                n_filled_pixels=n_fill, n_nan=dict(zip(FACTORS, nans)),
                stats={n: dict(min=lo, p1=p1, median=md, p99=p99, max=hi)
                       for n, lo, p1, md, p99, hi in stats},
                note="因子为原始量纲；nodata 已按最近 valid 邻居填充后再算因子")
    with open(os.path.join(DATA, f"feature_matrix_{site}.meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  写出 {os.path.relpath(npy, ROOT)} + .meta.json")
    return meta


def main():
    ap = argparse.ArgumentParser(description="扩充地形因子池 → feature_matrix_{site}.npy")
    ap.add_argument("--site", default=None, choices=["西坝村", "杨家坝"],
                    help="默认两站")
    a = ap.parse_args()
    for site in ([a.site] if a.site else ["西坝村", "杨家坝"]):
        run_site(site)


if __name__ == "__main__":
    main()
