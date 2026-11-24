# -*- coding: utf-8 -*-
"""任务 (a) DEM 空洞填补 + 任务 (b) 走廊高差残差：基线指标表。

协议 D10：
- 跨站双向（S1 杨训西验，S2 西训杨验）
- 同 mask、同评估脚本、排除边界带（距图像边界 ≥85 px）、冻结分层边界
- 分层维度：地形复杂度（坡度 P20/40/60/80）、走廊密度、地理距离、边界/内部

任务 (a) DEM 空洞填补：
  在验证站的有效区（valid=1）内，人为制造空洞（与真实空洞率匹配），
  用四档插值基线（常数/最近邻/双线性/IDW）填补，评估 RMSE/MAE。
  真实空洞 = valid=0 区域（外部 DEM 航带缺失），但无法评估（无真值）。
  因此采用"合成空洞"协议：从 valid=1 区域随机掩盖像元，用插值恢复。

任务 (b) 走廊高差残差：
  在验证站的走廊像元（label=1）上，计算 dem_mean - 局部地形均值（5×5 窗口）作为真值残差，
  用零残差和训练站平均残差作为基线。
"""
import os
import json
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

SITES = ["杨家坝", "西坝村"]
BORDER = 85  # 排除边界带


def load_site(site):
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    return {k: d[k] for k in d.files}


def interior_mask(shape, border):
    """距图像边界 ≥border 的内部区域。"""
    m = np.zeros(shape, dtype=bool)
    m[border:-border, border:-border] = True
    return m


def eval_mask(site_data, border=BORDER):
    """评估域 = 有效区 ∩ 内部区域（排除边界带）。"""
    v = site_data["valid"] == 1
    inner = interior_mask(v.shape, border)
    return v & inner


# ------------------------------------------------------------- 任务 (a) ----
def make_synthetic_holes(dem, valid, hole_rate, seed=42):
    """在 valid=1 区域内随机制造空洞（与真实空洞率匹配）。

    返回：dem_masked（空洞处设为 -9999）、hole_mask（被掩盖的像元）
    """
    rng = np.random.default_rng(seed)
    v = valid == 1
    n_valid = v.sum()
    n_holes = int(n_valid * hole_rate)
    # 随机选像元
    idx = np.flatnonzero(v)
    chosen = rng.choice(idx, n_holes, replace=False)
    hole_mask = np.zeros(v.shape, dtype=bool)
    hole_mask.flat[chosen] = True
    dem_masked = dem.copy()
    dem_masked[hole_mask] = -9999
    return dem_masked, hole_mask


def fill_constant(dem_masked, hole_mask, train_dem, train_valid):
    """常数基线：用训练站有效区 DEM 均值填补。"""
    v = train_valid == 1
    mean_val = train_dem[v].mean()
    filled = dem_masked.copy()
    filled[hole_mask] = mean_val
    return filled


def fill_nearest(dem_masked, hole_mask):
    """最近邻基线：用最近的非空洞像元值填补。"""
    filled = dem_masked.copy()
    # 非空洞像元的距离变换
    has_data = dem_masked > -9999
    if not has_data.any():
        return filled
    # distance_transform_edt 返回最近非零像元的索引
    ind = distance_transform_edt(~has_data, return_distances=False, return_indices=True)
    filled[hole_mask] = dem_masked[ind[0][hole_mask], ind[1][hole_mask]]
    return filled


def fill_bilinear(dem_masked, hole_mask):
    """双线性基线：用 4 近邻加权平均（距离反比权重）。"""
    filled = dem_masked.copy()
    has_data = (dem_masked > -9999).astype(np.float32)
    # 用距离变换找最近 4 个像元
    ind = distance_transform_edt(~has_data.astype(bool), return_distances=True, return_indices=True)
    dist = ind[0]
    # 对空洞像元，用 3×3 邻域内有数据的像元加权平均
    rows, cols = np.where(hole_mask)
    for r, c in zip(rows, cols):
        r0, r1 = max(0, r - 2), min(dem_masked.shape[0], r + 3)
        c0, c1 = max(0, c - 2), min(dem_masked.shape[1], c + 3)
        patch = dem_masked[r0:r1, c0:c1]
        valid_patch = patch > -9999
        if valid_patch.any():
            # 距离权重
            pr, pc = np.mgrid[r0:r1, c0:c1]
            d = np.sqrt((pr - r) ** 2 + (pc - c) ** 2)
            d[~valid_patch] = np.inf
            w = 1.0 / (d + 1e-6)
            w[~valid_patch] = 0
            filled[r, c] = (patch * w).sum() / w.sum()
        else:
            # 用最近邻
            filled[r, c] = dem_masked[ind[1][0, r, c], ind[1][1, r, c]]
    return filled


def fill_idw(dem_masked, hole_mask, power=2):
    """IDW 基线：距离反比加权（5×5 邻域）。"""
    filled = dem_masked.copy()
    rows, cols = np.where(hole_mask)
    for r, c in zip(rows, cols):
        r0, r1 = max(0, r - 3), min(dem_masked.shape[0], r + 4)
        c0, c1 = max(0, c - 3), min(dem_masked.shape[1], c + 4)
        patch = dem_masked[r0:r1, c0:c1]
        valid_patch = patch > -9999
        if valid_patch.any():
            pr, pc = np.mgrid[r0:r1, c0:c1]
            d = np.sqrt((pr - r) ** 2 + (pc - c) ** 2)
            d[~valid_patch] = np.inf
            w = 1.0 / (d + 1e-6) ** power
            w[~valid_patch] = 0
            filled[r, c] = (patch * w).sum() / w.sum()
        else:
            # 扩大到 7×7
            r0, r1 = max(0, r - 5), min(dem_masked.shape[0], r + 6)
            c0, c1 = max(0, c - 5), min(dem_masked.shape[1], c + 6)
            patch = dem_masked[r0:r1, c0:c1]
            valid_patch = patch > -9999
            if valid_patch.any():
                pr, pc = np.mgrid[r0:r1, c0:c1]
                d = np.sqrt((pr - r) ** 2 + (pc - c) ** 2)
                d[~valid_patch] = np.inf
                w = 1.0 / (d + 1e-6) ** power
                w[~valid_patch] = 0
                filled[r, c] = (patch * w).sum() / w.sum()
    return filled


def eval_fill(dem_true, dem_filled, hole_mask, em):
    """在评估域 ∩ 空洞上计算 RMSE/MAE。"""
    eval_region = hole_mask & em
    if eval_region.sum() == 0:
        return np.nan, np.nan, 0
    true_vals = dem_true[eval_region]
    pred_vals = dem_filled[eval_region]
    rmse = np.sqrt(((true_vals - pred_vals) ** 2).mean())
    mae = np.abs(true_vals - pred_vals).mean()
    return rmse, mae, eval_region.sum()


# ------------------------------------------------------------- 任务 (b) ----
def corridor_residual(dem, label, valid, win=5):
    """走廊高差残差 = dem - 局部地形均值（win×win 窗口，仅在 valid 区内）。"""
    v = valid == 1
    # 局部均值（只用 valid 像元）
    v_f = v.astype(np.float32)
    local_mean = uniform_filter(dem * v_f, size=win, mode="constant") / \
                 np.maximum(uniform_filter(v_f, size=win, mode="constant"), 1e-6)
    residual = dem - local_mean
    # 只在走廊像元上有意义
    return residual


def eval_residual(res_true, res_pred, label, em):
    """在评估域 ∩ 走廊上计算 RMSE/MAE。"""
    eval_region = (label == 1) & em
    if eval_region.sum() == 0:
        return np.nan, np.nan, 0
    true_vals = res_true[eval_region]
    pred_vals = res_pred[eval_region]
    rmse = np.sqrt(((true_vals - pred_vals) ** 2).mean())
    mae = np.abs(true_vals - pred_vals).mean()
    return rmse, mae, eval_region.sum()


# ------------------------------------------------------------- 主流程 ----
def run_baseline(train_site, val_site, hole_rate=0.72):
    """跨站双向基线。"""
    tr = load_site(train_site)
    va = load_site(val_site)
    em = eval_mask(va)
    print(f"\n{'='*60}")
    print(f"S: {train_site} → {val_site}")
    print(f"评估域像元: {em.sum():,}, 域内走廊: {va['label'][em].sum():,}, "
          f"正样本率: {va['label'][em].mean():.2%}")

    # ---- 任务 (a) DEM 空洞填补 ----
    print(f"\n--- 任务 (a) DEM 空洞填补（合成空洞率 {hole_rate:.0%}） ---")
    dem_va = va["dem_mean"].copy()
    valid_va = va["valid"]
    dem_masked, hole_mask = make_synthetic_holes(dem_va, valid_va, hole_rate)
    n_holes = hole_mask.sum()
    print(f"合成空洞: {n_holes:,} 像元")

    methods = {
        "常数": lambda: fill_constant(dem_masked, hole_mask, tr["dem_mean"], tr["valid"]),
        "最近邻": lambda: fill_nearest(dem_masked, hole_mask),
        "双线性": lambda: fill_bilinear(dem_masked, hole_mask),
        "IDW": lambda: fill_idw(dem_masked, hole_mask),
    }
    results_a = {}
    for name, fn in methods.items():
        filled = fn()
        rmse, mae, n = eval_fill(dem_va, filled, hole_mask, em)
        results_a[name] = dict(rmse=float(rmse), mae=float(mae), n_eval=int(n))
        print(f"  {name}: RMSE={rmse:.3f} m, MAE={mae:.3f} m (n={n:,})")

    # ---- 任务 (b) 走廊高差残差 ----
    print(f"\n--- 任务 (b) 走廊高差残差 ---")
    res_va = corridor_residual(va["dem_mean"], va["label"], va["valid"])
    res_tr = corridor_residual(tr["dem_mean"], tr["label"], tr["valid"])

    # 零残差基线
    rmse_z, mae_z, n_b = eval_residual(res_va, np.zeros_like(res_va), va["label"], em)
    print(f"  零残差: RMSE={rmse_z:.3f} m, MAE={mae_z:.3f} m (n={n_b:,})")

    # 训练站平均残差基线
    tr_corridor = tr["label"] == 1
    tr_em = eval_mask(tr)
    mean_res_tr = res_tr[tr_corridor & tr_em].mean()
    pred_avg = np.full_like(res_va, mean_res_tr)
    rmse_a, mae_a, _ = eval_residual(res_va, pred_avg, va["label"], em)
    print(f"  训练站平均残差 ({mean_res_tr:.3f} m): RMSE={rmse_a:.3f} m, MAE={mae_a:.3f} m")

    # 走廊真值残差统计
    va_corridor = va["label"] == 1
    true_res_vals = res_va[va_corridor & em]
    print(f"  真值残差统计: mean={true_res_vals.mean():.3f}, std={true_res_vals.std():.3f}, "
          f"min={true_res_vals.min():.3f}, max={true_res_vals.max():.3f}")

    results_b = dict(
        zero=dict(rmse=float(rmse_z), mae=float(mae_z), n_eval=int(n_b)),
        train_avg=dict(rmse=float(rmse_a), mae=float(mae_a), mean_res=float(mean_res_tr)),
        true_stats=dict(mean=float(true_res_vals.mean()), std=float(true_res_vals.std()),
                        min=float(true_res_vals.min()), max=float(true_res_vals.max())),
    )

    # ---- 分层评估（任务 a，坡度分层）----
    print(f"\n--- 任务 (a) 坡度分层（验证站） ---")
    slope = va["slope_deg"]
    p20, p40, p60, p80 = np.percentile(slope[em], [20, 40, 60, 80])
    layers = [
        ("P20", 0, p20), ("P40", p20, p40), ("P60", p40, p60),
        ("P80", p60, p80), ("P100", p80, 90),
    ]
    stratified = {}
    for lname, lo, hi in layers:
        layer_mask = em & (slope >= lo) & (slope < hi)
        layer_holes = hole_mask & layer_mask
        if layer_holes.sum() == 0:
            continue
        row = dict(n_pixels=int(layer_mask.sum()), n_holes=int(layer_holes.sum()))
        for name, fn in methods.items():
            filled = fn()
            rmse, mae, _ = eval_fill(dem_va, filled, layer_holes, em)
            row[name] = dict(rmse=float(rmse), mae=float(mae))
        stratified[lname] = row
        print(f"  {lname} [{lo:.1f},{hi:.1f}°): {layer_mask.sum():,} 像元, "
              f"{layer_holes.sum():,} 空洞, IDW RMSE={row['IDW']['rmse']:.3f}")

    return dict(
        direction=f"{train_site}→{val_site}",
        task_a=results_a,
        task_b=results_b,
        stratified=stratified,
        eval_domain=dict(n_pixels=int(em.sum()), n_corridor=int(va["label"][em].sum()),
                         pos_rate=float(va["label"][em].mean())),
    )


def main():
    all_results = []
    for tr_site, va_site in [("杨家坝", "西坝村"), ("西坝村", "杨家坝")]:
        r = run_baseline(tr_site, va_site)
        all_results.append(r)

    # 汇总表
    print(f"\n{'='*60}")
    print("汇总表")
    print(f"{'='*60}")
    for r in all_results:
        d = r["direction"]
        print(f"\n### {d}")
        print("任务 (a) DEM 空洞填补:")
        print(f"  {'方法':<10} {'RMSE(m)':>10} {'MAE(m)':>10}")
        for name in ["常数", "最近邻", "双线性", "IDW"]:
            a = r["task_a"][name]
            print(f"  {name:<10} {a['rmse']:>10.3f} {a['mae']:>10.3f}")
        print("任务 (b) 走廊高差残差:")
        b = r["task_b"]
        print(f"  零残差: RMSE={b['zero']['rmse']:.3f}, MAE={b['zero']['mae']:.3f}")
        print(f"  训练站平均残差 ({b['train_avg']['mean_res']:.3f} m): "
              f"RMSE={b['train_avg']['rmse']:.3f}, MAE={b['train_avg']['mae']:.3f}")
        print(f"  真值残差: mean={b['true_stats']['mean']:.3f}, std={b['true_stats']['std']:.3f}")

    # 保存
    out = os.path.join(RES, "baseline_regression.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
