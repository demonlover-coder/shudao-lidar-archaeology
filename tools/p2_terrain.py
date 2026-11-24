# -*- coding: utf-8 -*-
"""从 P2 输出的 DEM 算地形因子，拼成多通道张量。

    DEM 一个通道 [H, W]
          ↓ 逐像元用相邻像元差
    坡度 = arctan(√((dz/dx)² + (dz/dy)²))   → [H, W]
    坡向 = arctan2(dz/dy, -dz/dx)            → [H, W]
    曲率 = -∇²z                                → [H, W]
          ↓
    拼成多通道张量 [C, H, W]（C = 通道数）

栅格约定（与 `p2_dem.pack_dem` / `eval_p2.grid_axes` 一致）：
    第 0 行 = 北侧，第 0 列 = 西侧，像元中心 `(i+0.5)·gsd`，`gsd` 单位 米/像元。
    ⇒ 行号 i 增大 = 向南，列号 j 增大 = 向东。
    ⇒ `sobel(dem, axis=0)` 得到的是 **∂z/∂(行号)**，等于 **−∂z/∂(北向)**。

⚠ 坡向缺省口径 = **raw（与 npz 对齐，当前默认 `aspect_standard=False`）**：
    - `False`（**默认**）得到"原文 raw 公式" `degrees(arctan2(dzdy,-dzdx))%360`，
      实测与 `data/{site}_1m.npz` 的 `aspect_deg` 一致（MAE 0.017° / 0.021°）。
      `tools/build_dataset.py:132` 用的就是这条公式。**本项目坡向 = raw，不是 GDAL 标准。**
    - `True` 得到 GDAL/ESRI 标准罗盘坡向（0=北 / 90=东 / 180=南 / 270=西），
      实测 `标准 = (90 − raw) % 360`，与 npz 的 `aspect_deg` 相差 MAE 106.2° / 87.1°。
    - 两个目标只能选一个。要与既有 npz / 训练管线可比 → 保持默认 `False`；
      要与 GDAL/ESRI 对齐 → 传 `True`，但须先改 `build_dataset.terrain()` 补上
      `90 − raw` 并**重生成**两站 npz，否则项目内并存两套坡向。
    - `slope_deg` 两口径一致（实测 MAE 0.00059° / 0.00085°），选哪个都对得上。

⚠ **`data/{site}_1m.npz` 的地形通道在 nodata 边界被污染**（实测）：
   `build_dataset.terrain()` 的输入 `ch["mean"]` 里 nodata 是 **−9999**，直接进了 3×3 核。
   后果：有效像元里 3×3 窗触及 nodata 的占 **1.84%（杨）/ 1.26%（西）**，这些像元
   `slope_deg` 恒为 **90.0°**、`curvature` 掉到 **−6755 ~ −1339**（西 −7005 ~ −1370）。
   npz `curvature` 的 −6755 全来自这里，**未触及 nodata 的像元范围只有 −22.6 ~ 22.2**
   （p99|·| 0.61）。用这两个通道做统计/训练前必须先按"3×3 不触 nodata"掩膜剔除。

⚠ 各通道量纲差 ~1.7 个数量级（实测见 `python tools/p2_terrain.py`）：
   `slope` 度（std 18.58）/ `sin,cos` [−1,1]（std 0.60/0.57）/
   `curvature` 1·m⁻¹（std 0.397）/ `roughness` 无量纲（std 0.743）。
   直接当特征喂网络时 `curvature` 会被 `slope` 淹没（约 47×），需要先做逐通道标准化。
   另注：npz 里的 `curvature` 与本模块**不是同一个量**（npz 是"中心 − 8 邻域均值"、
   米制；本模块是 −∇²z、1/m），两者定义不同，不要互相参照。

⚠ 标准化接口（决策 2 骨架，本轮**不落盘**）：robust = **P1/P99 截断 + 中位数/IQR**。
   统计量**只从训练集汇总**、存 `data/terrain_stats_{site}.json`、推理时读；
   **不得**从单幅推断结果现算（否则同图不同 tile 的归一化不一致）。
   本模块提供 `compute_robust_stats` / `load_stats` / `robust_normalize` /
   `robust_denormalize`、`terrain_stack(..., stats=)` 与 `--stats` 入口；
   本轮**不生成统计文件、不接训练管线**，需要时再补统计脚本。

用法：
    python tools/p2_terrain.py            # 合成面自检 + 两站真实 DEM 对照
    python tools/p2_terrain.py --stats data/terrain_stats_杨家坝.json   # 叠加标准化
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import sobel, uniform_filter

CHANNELS = ("slope", "sin_aspect", "cos_aspect", "curvature", "roughness",
            "profile_curv", "tpi15", "tpi31")
HS_CHANNELS = ("hs_max", "hs_var")     # 多方向 Hillshade 导出的两个判别因子


def multidir_hillshade(dem, gsd=1.0, num_dirs=8, alt_deg=45.0):
    """多方向 Hillshade（固定高度角，8 个等间距方位角）。

    shade_i = sin(alt)·cos(slope) + cos(alt)·sin(slope)·cos(az_i − aspect)
    返回 dict：hs_stack [num_dirs,H,W] / hs_max / hs_min / hs_var / hs_range。

    ⚠ **理论性质（实测见 `_selfcheck`）**：8 个方位等间距时，对**局部平面**
      有 `Var_i cos(az_i − aspect) = 1/2`（与 aspect 无关），故
      `hs_var = ½·cos²(alt)·sin²(slope)`——它等价于坡度的单调重参数化，
      只携带坡度信息；`hs_max` 额外带一个 45° 周期的坡向混叠项。
      因此两者与 slope 的相关系数应接近 1 / 较高，不要指望独立信息。
    """
    dem = np.asarray(dem, dtype=np.float64)
    dzdx = sobel(dem, axis=1) / (8.0 * gsd)
    dzdy = sobel(dem, axis=0) / (8.0 * gsd)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)                 # raw 口径，同 terrain_factors
    alt = np.radians(alt_deg)
    az = np.radians(360.0 - np.arange(num_dirs) * (360.0 / num_dirs) + 90.0)
    hs = (np.sin(alt) * np.cos(slope)[None, :, :]
          + np.cos(alt) * np.sin(slope)[None, :, :]
          * np.cos(az[:, None, None] - aspect[None, :, :]))
    return {
        "hs_stack": hs,
        "hs_max": hs.max(axis=0),
        "hs_min": hs.min(axis=0),
        "hs_var": hs.var(axis=0),
        "hs_range": hs.max(axis=0) - hs.min(axis=0),
    }


def hillshade_factors(dem, gsd=1.0, num_dirs=8, alt_deg=45.0):
    """[H, W] → {hs_max, hs_var}（`HS_CHANNELS`，AUC 评估用）。"""
    hs = multidir_hillshade(dem, gsd, num_dirs, alt_deg)
    return {c: hs[c] for c in HS_CHANNELS}


def hs_harmonics(dem, gsd=1.0, num_dirs=8, alt_deg=45.0):
    """8 向 Hillshade 沿**方位轴**做圆周 FFT，返回逐阶幅值 {hs_h0 … hs_h4}。

    归一化：纯余弦 `D·cos(az−aspect)` 的 k=1 幅值恰为 D（Nyquist 与直流不乘 2）。

    ⚠ **恒等式（本函数存在的意义就是证明它）**：
      `multidir_hillshade` 内每像元的 slope/aspect 是**单一值**，故
      `shade_i = sin(alt)cos(slope) + cos(alt)sin(slope)·cos(az_i − aspect)`
      ——沿方位轴是**单一余弦**。于是
        hs_h0 = sin(alt)·cos(slope)
        hs_h1 = cos(alt)·sin(slope)
        hs_h2 = hs_h3 = hs_h4 ≡ 0（只剩浮点残差）
      也就是"8 向 Hillshade 的全部信息"= (slope, aspect) 的重新编码，
      谐波幅值**不可能**比坡度多出任何东西。高阶幅值只在照明模型**非局部**时
      才非零（投射阴影/地平线遮挡），那不是本函数所做的事。
    """
    hs = multidir_hillshade(dem, gsd, num_dirs, alt_deg)["hs_stack"]
    n = hs.shape[0]
    x = np.fft.rfft(hs, axis=0) / n
    amp = 2.0 * np.abs(x)
    amp[0] = np.abs(x[0])                       # 直流不乘 2
    if n % 2 == 0:
        amp[n // 2] = np.abs(x[n // 2])         # Nyquist 不乘 2
    return {f"hs_h{k}": amp[k] for k in range(n // 2 + 1)}


# 罗盘方位 k*45°（0=N, 1=NE, 2=E … 顺时针）→ (drow, dcol)；北为 -row、东为 +col
_AZ_VEC = ((-1, 0), (-1, 1), (0, 1), (1, 1),
           (1, 0), (1, -1), (0, -1), (-1, -1))


def horizon_angles(dem, gsd=1.0, max_dist_m=30.0, num_dirs=8):
    """8 个罗盘方位的**地平线角**（非局部）：沿射线取 `max_dist_m` 内最大仰角。

    返回 [num_dirs, H, W]（**弧度**，第 k 轴 = 方位 k·45°）：
        hz[k][i,j] = max_s arctan((z[i+dy·s, j+dx·s] − z[i,j]) / (s·gsd·‖(dy,dx)‖))
    平地 8 向全为 0；山顶点全为负；V 型谷底两侧为正、沿谷向接近 0。

    实现：**逐方向、逐步做真实切片平移**（不用 `np.roll`，避免每步环绕把对侧地形
    假借进来），对角线步长按 √2 计。8×30 次切片 ≈ 1–2 s/M 像元。
    """
    dem = np.asarray(dem, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError("dem 必须是 2D")
    if num_dirs != len(_AZ_VEC):
        raise ValueError(f"num_dirs 仅支持 8（当前 {num_dirs}）")
    H, W = dem.shape
    steps = max(1, int(round(max_dist_m / gsd)))
    out = np.empty((len(_AZ_VEC), H, W), dtype=np.float32)
    for k, (dy, dx) in enumerate(_AZ_VEC):
        m = np.full((H, W), -np.inf, dtype=np.float32)
        for s in range(1, steps + 1):
            r0, r1 = max(0, -dy * s), min(H, H - dy * s)
            c0, c1 = max(0, -dx * s), min(W, W - dx * s)
            if r0 >= r1 or c0 >= c1:
                continue
            src = dem[r0 + dy * s:r1 + dy * s, c0 + dx * s:c1 + dx * s]
            dst = dem[r0:r1, c0:c1]
            ang = np.arctan((src - dst) / (s * gsd * np.hypot(dy, dx)))
            m[r0:r1, c0:c1] = np.maximum(m[r0:r1, c0:c1], ang)
        out[k] = np.where(np.isfinite(m), m, 0.0)
    return out


def horizon_harmonics(dem, gsd=1.0, max_dist_m=30.0, num_dirs=8):
    """地平线角剖面 → 圆周谐波 {h{n}_deg, h{n}_cos, h{n}_sin}（n=1,2）+ h0_deg。

    相位约定：`h{n}_deg = A`，剖面为 `A·cos(n(θ − φ))`，则
      `h{n}_cos = A·cosφ`、`h{n}_sin = A·sinφ`，**φ 为最大仰角方位**（罗盘度）。

    ⚠ **物理**：对称 V 形谷（两壁等高）在 θ₀ 与 θ₀+180° 同时最大，
    剖面 180° 周期 → **k=2 主导**，`h2_phase = φ` 即走廊**法向**（切向需 +90°）；
    k=1 只在两壁高差明显时非零，编码**不对称性**。二者都输出，用相位对齐实测裁决。
    """
    hz = horizon_angles(dem, gsd, max_dist_m, num_dirs)
    n = hz.shape[0]
    X = np.fft.rfft(hz, axis=0) / n
    out = {"h0_deg": np.degrees(X[0].real)}      # 8 向平均仰角（度）
    for k in (1, 2):
        A = np.degrees(2.0 * np.abs(X[k]))
        phi = -np.angle(X[k])
        out[f"h{k}_deg"] = A
        out[f"h{k}_cos"] = A * np.cos(phi)
        out[f"h{k}_sin"] = A * np.sin(phi)
        out[f"h{k}_phase"] = np.degrees(phi % (2.0 * np.pi))
    return out


def tpi(z, k=15):
    """TPI（Topographic Position Index）= z − 邻域均值：正=山脊，负=沟谷。

    `k` = `uniform_filter` 窗口（**像素**）；1 m 网格下 k=15/31 即 15/31 m 尺度。
    """
    z = np.asarray(z, dtype=np.float64)
    return z - uniform_filter(z, size=k, mode="nearest")


def profile_curvature(z, gsd):
    """剖面曲率 = 沿**坡向**方向的二阶方向导数（各向异性，1/m）。

    与既有 `curvature`（=−∇²z，各向同性 Laplacian）**不是同一个量**，不要互相参照。
    """
    z = np.asarray(z, dtype=np.float64)
    dzdx = sobel(z, axis=1) / (8 * gsd)
    dzdy = sobel(z, axis=0) / (8 * gsd)
    d2zdx2 = np.gradient(dzdx, gsd, axis=1)
    d2zdy2 = np.gradient(dzdy, gsd, axis=0)
    d2zdxy = np.gradient(dzdx, gsd, axis=0)
    slope_mag = np.hypot(dzdx, dzdy) + 1e-9
    ux = dzdx / slope_mag
    uy = dzdy / slope_mag
    return ux ** 2 * d2zdx2 + 2 * ux * uy * d2zdxy + uy ** 2 * d2zdy2


def terrain_factors(dem, gsd=1.0, aspect_standard=False, tpi_ks=(15, 31)):
    """输入: dem [H, W] 米制（第 0 行 = 北）

    输出: dict，每项 [H, W]（`aspect` 始终是**裸角度**，不展开）
        slope     坡度（度，0~90）
        aspect    坡向（度，0~360）；`aspect_standard=False`（**默认**）为本项目 raw
                  公式（与 npz `aspect_deg` 一致）；True 为 GDAL/ESRI 标准
                  （0=北/90=东/180=南/270=西）
        curvature 曲率 −∇²z（1/m，各向同性）
        roughness |dz/dx| + |dz/dy|（无量纲）
        profile_curv 剖面曲率（沿坡向的二阶方向导数，1/m，各向异性）
        tpi15/tpi31  TPI = z − 15/31 px 邻域均值（米；正=山脊，负=沟谷）
    """
    dem = np.asarray(dem, dtype=np.float64)

    # Horn 3x3 梯度（= Sobel 核，正权重和 4、负权重和 −4、离线斜率响应 8）
    dzdx = sobel(dem, axis=1) / (8 * gsd)
    dzdy = sobel(dem, axis=0) / (8 * gsd)

    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))       # 坡度
    aspect = np.degrees(np.arctan2(dzdy, -dzdx)) % 360.0      # 坡向
    if aspect_standard:
        aspect = (90.0 - aspect) % 360.0
    curv = -np.gradient(np.gradient(dem, gsd, axis=0), gsd, axis=0) \
           -np.gradient(np.gradient(dem, gsd, axis=1), gsd, axis=1)  # 曲率(=-∇²z)

    return {
        "slope": slope,
        "aspect": aspect,
        "curvature": curv,
        "roughness": np.abs(dzdx) + np.abs(dzdy),
        "profile_curv": profile_curvature(dem, gsd),
        "tpi15": tpi(dem, tpi_ks[0]),
        "tpi31": tpi(dem, tpi_ks[1]),
    }


def terrain_stack(dem, gsd=1.0, aspect_standard=False, stats=None, tpi_ks=(15, 31)):
    """[H, W] → **[8, H, W]**。通道顺序 = `CHANNELS`。

    坡向按环形量展开为 `sin/cos` 两通道，保证在 0°/360° 处连续。
    `aspect_standard` 默认 `False`（raw 口径，与 npz `aspect_deg` 对齐），
    语义与默认值同 `terrain_factors`，两者必须保持一致。

    `stats`（[决策 2 骨架]）为 `compute_robust_stats` 产出的逐通道 robust 统计量；
    为 None 时返回**原始量纲**，非 None 时按 `robust_normalize` 标准化后返回。
    骨架阶段不在本函数内生成统计量（统计量须由训练集汇总，见模块 docstring）。
    """
    f = terrain_factors(dem, gsd, aspect_standard=aspect_standard, tpi_ks=tpi_ks)
    a = np.radians(f["aspect"])
    stack = np.stack([
        f["slope"],
        np.sin(a),
        np.cos(a),
        f["curvature"],
        f["roughness"],
        f["profile_curv"],
        f["tpi15"],
        f["tpi31"],
    ], axis=0)   # [8, H, W]
    if stats is not None:
        stack = robust_normalize(stack, stats)
    return stack


# ------------------------------------------------- 标准化（决策 2 接口骨架）----
# robust = P1/P99 截断 + 中位数/IQR。统计量只从训练集汇总、存
# `data/terrain_stats_{site}.json`、推理时读；不得从单幅推断结果现算。
def compute_robust_stats(stack, valid=None, q=(1.0, 99.0)) -> dict:
    """[C,H,W] → {通道: {"p_lo","p_hi","median","iqr"}}（逐通道 robust 统计）。

    `valid`（[H,W] bool）用于排除 nodata 像元，None 表示全用。
    汇总**多张训练集样本**后再写文件；单幅图统计不满足"训练集统计"要求。
    """
    stack = np.asarray(stack, dtype=np.float64)
    out: dict = {}
    for c, arr in zip(CHANNELS, stack):
        a = arr[valid] if valid is not None else arr.ravel()
        lo, hi = np.percentile(a, q)
        q25, q75 = np.percentile(a, (25.0, 75.0))
        out[c] = {"p_lo": float(lo), "p_hi": float(hi),
                  "median": float(np.median(a)), "iqr": float(q75 - q25)}
    return out


def robust_normalize(stack, stats):
    """[C,H,W] → [C,H,W]：逐通道先 P1/P99 截断，再 (x − median) / IQR。"""
    stack = np.asarray(stack, dtype=np.float64)
    if stats is None:
        return stack
    out = []
    for c, arr in zip(CHANNELS, stack):
        s = stats[c]
        iqr = s["iqr"] if abs(s["iqr"]) > 1e-12 else 1.0
        out.append((np.clip(arr, s["p_lo"], s["p_hi"]) - s["median"]) / iqr)
    return np.stack(out, axis=0)


def robust_denormalize(z, stats):
    """`robust_normalize` 的逆；截断不可逆，这里只还原线性部分。"""
    z = np.asarray(z, dtype=np.float64)
    if stats is None:
        return z
    out = []
    for c, arr in zip(CHANNELS, z):
        s = stats[c]
        iqr = s["iqr"] if abs(s["iqr"]) > 1e-12 else 1.0
        out.append(arr * iqr + s["median"])
    return np.stack(out, axis=0)


def stats_path(site: str, root: str | None = None) -> str:
    """`data/terrain_stats_{site}.json` 路径（本轮不生成，仅预留）。"""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "data", f"terrain_stats_{site}.json")


def load_stats(path: str | None) -> dict | None:
    """读 terrain_stats json；路径为空或文件不存在时返回 None（= 不标准化）。"""
    if not path or not os.path.exists(path):
        return None
    st = json.load(open(path, encoding="utf-8"))
    st = {c: st[c] for c in CHANNELS if c in st}
    return st or None


# ------------------------------------------------------------------ 自检 ----
def _plane(H, W, gsd, g_east, g_north):
    """构造 z = g_east·x + g_north·y 的平面（x 向东、y 向北，米）。"""
    j = np.arange(W) * gsd                       # 向东
    i = np.arange(H) * gsd                       # 向南
    return (g_east * j)[None, :] - (g_north * i)[:, None]


def _selfcheck():
    H = W = 32
    gsd = 1.0
    # (名称, g_east, g_north, 期望标准坡向)；下坡方向 = 高程递减方向
    #   朝北下坡 → g_north = −1；朝东下坡 → g_east = −1；以此类推
    cases = [
        ("朝北下坡", 0.0, -1.0, 0.0),
        ("朝东下坡", -1.0, 0.0, 90.0),
        ("朝南下坡", 0.0, 1.0, 180.0),
        ("朝西下坡", 1.0, 0.0, 270.0),
        ("东北下坡", -1.0, -1.0, 45.0),
    ]
    print("合成面自检（H=W=32, gsd=1m；标准坡向 = 0 北 / 90 东 / 180 南 / 270 西）")
    print(f"  {'坡面':<10}{'期望坡度':>10}{'实测坡度':>10}{'标准坡向':>10}"
          f"{'期望':>8}{'原文坡向':>10}")
    ok = True
    for name, ge, gn, exp_std in cases:
        dem = _plane(H, W, gsd, ge, gn)
        f_raw = terrain_factors(dem, gsd, aspect_standard=False)
        f_std = terrain_factors(dem, gsd, aspect_standard=True)
        exp_slope = np.degrees(np.arctan(np.hypot(ge, gn)))
        exp_raw = (90.0 - exp_std) % 360.0
        s = f_raw["slope"][8:-8, 8:-8]
        a_raw = f_raw["aspect"][8:-8, 8:-8]
        a_std = f_std["aspect"][8:-8, 8:-8]
        if not (np.allclose(s, exp_slope, atol=1e-6) and
                np.allclose(a_raw, exp_raw, atol=1e-6) and
                np.allclose(a_std, exp_std, atol=1e-6)):
            ok = False
        print(f"  {name:<10}{exp_slope:>10.2f}{s.mean():>10.2f}"
              f"{a_std.mean():>10.2f}{exp_std:>8.0f}{a_raw.mean():>10.2f}")
    print(f"  → 坡度严格成立；标准坡向 = GDAL/ESRI 罗盘坡向；"
          f"原文坡向 = (90 − 标准) % 360（{'全部通过' if ok else '有失败项'}）")

    # 曲率 / 粗糙度：抛物面 z = k(x²+y²) → ∇²z = 4k
    k = 0.01
    j = (np.arange(64) - 31.5) * gsd
    i = (np.arange(64) - 31.5) * gsd
    zz = k * (j[None, :] ** 2 + i[:, None] ** 2)
    f = terrain_factors(zz, gsd)
    c = f["curvature"][16:-16, 16:-16]
    print(f"  曲率自检（z=0.01(x²+y²)，真值 −∇²z = {-(4 * k):.4f}）："
          f"实测均值 {c.mean():.6f}")

    # TPI / 剖面曲率自检
    ii, jj = np.mgrid[0:64, 0:64].astype(float)
    hill = 5.0 * np.exp(-((ii - 32) ** 2 + (jj - 32) ** 2) / (2 * 5.0 ** 2))
    tp_hill = tpi(hill, 15)
    tp_plane = tpi(_plane(32, 32, gsd, 1.0, 1.0), 15)
    pc = profile_curvature(k * (ii ** 2 + jj ** 2), gsd)
    print(f"  剖面曲率自检（z=0.01(x²+y²)，沿坡向真值 2k={2 * k:.4f}）："
          f"内部实测 {pc[16:-16, 16:-16].mean():.5f}")
    print(f"  TPI 自检（高斯丘 k=15）：峰顶 {tp_hill[32, 32]:+.3f} m（应>0）；"
          f"平面 |TPI|max={np.abs(tp_plane[8:-8, 8:-8]).max():.2e}（应≈0）")

    # 多方向 Hillshade 自检：平面（坡度 45°）上 8 向方差应 = ½cos²(alt)sin²(slope)
    hs = multidir_hillshade(_plane(32, 32, gsd, -1.0, 0.0), gsd, num_dirs=8, alt_deg=45.0)
    exp = 0.5 * np.cos(np.radians(45.0)) ** 2 * np.sin(np.radians(45.0)) ** 2
    got = float(hs["hs_var"][8:-8, 8:-8].mean())
    print(f"  Hillshade 自检（坡面 45°、高度角 45°、8 向）：hs_var 实测 {got:.5f}，"
          f"解析值 ½cos²·sin² = {exp:.5f}"
          f"（{'一致' if abs(got - exp) < 5e-3 else '不一致'}）；"
          f"hs_max {float(hs['hs_max'][8:-8, 8:-8].mean()):.4f}")
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser(description="P2 地形因子通道（标准化接口骨架）")
    ap.add_argument("--stats", default=None,
                    help="terrain_stats json 路径；缺省则不标准化（骨架阶段不生成该文件）")
    ap.add_argument("--site", default="杨家坝", help="对照中标示标准化统计的站点")
    args = ap.parse_args()

    _selfcheck()
    stats = load_stats(args.stats)
    if args.stats and stats is None:
        print(f"\n⚠ 未找到 stats 文件 {args.stats} → 按骨架跳过标准化")
    elif stats is None:
        print(f"\n（未提供 --stats，按骨架返回原始量纲；统计量应存 "
              f"{stats_path(args.site)}，须由训练集汇总）")

    print("\n真实 DEM 对照（P2 模型 L0 vs DEM 栅格自带 slope_deg/aspect_deg，只用 valid 像元）")
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import p2_dem as P

    print(f"  {'站点':<8}{'坡度差 RMSE':>12}{'坡度差 MAE':>12}{'坡向差 MAE':>12}"
          f"{'坡向差(raw)':>13}")
    for site in ("杨家坝", "西坝村"):
        x0, y1, wm, hm, gsd, d = P.dem_window(site)
        packed = P.pointcloud_to_dem(site, x0, y1, wm, hm, gsd, n_levels=1)
        z = P.to_utm(site, packed[0]).cpu().numpy().astype(np.float64)
        v = d["valid"] == 1
        f_std = terrain_factors(z, gsd, aspect_standard=True)
        f_raw = terrain_factors(z, gsd, aspect_standard=False)
        gt_s = np.asarray(d["slope_deg"], dtype=np.float64)
        gt_a = np.asarray(d["aspect_deg"], dtype=np.float64)
        es = f_std["slope"] - gt_s
        ea = np.abs((f_std["aspect"] - gt_a + 180) % 360 - 180)
        ea_raw = np.abs((f_raw["aspect"] - gt_a + 180) % 360 - 180)
        print(f"  {site:<8}{np.sqrt((es[v] ** 2).mean()):>12.3f}"
              f"{np.abs(es[v]).mean():>12.3f}{ea[v].mean():>12.3f}{ea_raw[v].mean():>13.3f}")

        stack = terrain_stack(z, gsd)
        if site == "杨家坝":
            print(f"\n  形状: L0 {z.shape} → terrain_stack {stack.shape}"
                  f"（通道 {CHANNELS}）")
            print(f"  {'通道':<12}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}{'p99-abs':>10}")
            for c, arr in zip(CHANNELS, stack):
                a = arr[v]
                print(f"  {c:<12}{a.min():>9.4f}{a.max():>9.4f}{a.mean():>9.4f}"
                      f"{a.std():>9.4f}{np.percentile(np.abs(a), 99):>10.4f}")
            rng = [float(np.abs(arr[v]).max()) / max(float(np.abs(stack[0][v]).std()), 1e-12)
                   for arr in stack]
            print(f"  → 各通道 |最大值| / slope 标准差 = "
                  + ", ".join(f"{c}:{r:.1f}" for c, r in zip(CHANNELS, rng)))
            print("  → 比值跨 %.1f 个数量级；直接拼接喂网络时 curvature 会被淹没，"
                  % (np.log10(max(rng)) - np.log10(min(rng))))
            print("     建议先做逐通道标准化（z-score 或 robust）再拼。")
            if stats is not None:
                nstack = robust_normalize(stack, stats)
                print(f"  → 已按 {args.stats} 做 robust 标准化（决策 2 骨架演示）：")
                print(f"  {'通道':<12}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}")
                for c, arr in zip(CHANNELS, nstack):
                    a = arr[v]
                    print(f"  {c:<12}{a.min():>9.4f}{a.max():>9.4f}"
                          f"{a.mean():>9.4f}{a.std():>9.4f}")


if __name__ == "__main__":
    main()
