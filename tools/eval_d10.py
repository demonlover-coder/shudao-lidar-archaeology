# -*- coding: utf-8 -*-
"""D10 分层评估：全图推理 + 逐坡度层 IoU（仅 v2 多任务 U-Net）。

协议（与 results/contract.json::stratified_eval 一致）：
- 同 mask、同评估脚本、排除 85px 边界带、冻结分层边界。
- 分层维度：地形复杂度（坡度 P20/40/60/80 切 5 层）。
- 两个权重档，由 --stage 选择：
    ft  = 按验证站全图 IoU 选出的微调权重（分割指标用）
    pre = 按训练站合成丢块 RMSE 选出的预训练权重（补洞指标用）
"""
import argparse
import os
import sys
import json
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")

BORDER = 85          # 评估排除边界带（与 results/contract.json 同口径）


def load_site(site):
    """加载站点 npz（v2 的特征拼装在 train_mt_v2.build_channels，这里只要元数据）。"""
    return np.load(os.path.join(DATA, f"{site}_1m.npz"))


def chamfer_distance(pred_bin, gt_bin, mask):
    """预测边界 vs 真值边界的 Chamfer 距离（像素）。只在 mask 内计算。"""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    p = pred_bin & mask
    g = gt_bin & mask
    pe = p ^ binary_erosion(p)
    ge = g ^ binary_erosion(g)
    if pe.sum() == 0 or ge.sum() == 0:
        return float("nan")
    dt_g = distance_transform_edt(~ge)
    dt_p = distance_transform_edt(~pe)
    return float((dt_g[pe].mean() + dt_p[ge].mean()) / 2.0)


def compute_iou(pred_bin, label, mask):
    """IoU。"""
    inter = (pred_bin & label & mask).sum()
    union = ((pred_bin | label) & mask).sum()
    if union == 0:
        return 0.0, int(inter), int(union)
    return float(inter / union), int(inter), int(union)


def pool_bool(a, k):
    """bool 图的 k×k 存在性下采样（any），与训练侧 max_pool 口径一致。"""
    H, W = a.shape
    H2, W2 = H // k * k, W // k * k
    return a[:H2, :W2].reshape(H2 // k, k, W2 // k, k).any(axis=(1, 3))


def eval_direction(train_site, val_site, base, dev, log, prefix="", stage="ft",
                   multi_scale=False):
    """评估一个方向。stage='ft' 用按 IoU 选出的微调权重；'pre' 用按补洞 RMSE 选出的预训练权重。"""
    ckpt_path = os.path.join(CKPT, f"{prefix}{stage}_{train_site}_{val_site}_base{base}.pth")
    if not os.path.exists(ckpt_path):
        log(f"  [SKIP] 检查点不存在: {ckpt_path}")
        return None

    d = load_site(val_site)
    label = d["label"].astype(bool)
    valid = d["valid"].astype(bool)
    slope = d["slope_deg"]
    dem = d["dem_mean"].astype(np.float32)

    from train_mt_v2 import (EVAL_BS, EVAL_FRAC, MultiTaskUNetV2, block_drop_mask,
                             compute_stats, full_infer)
    from scipy.ndimage import distance_transform_edt

    st = compute_stats(d)
    sd = torch.load(ckpt_path, map_location=dev)
    in_ch = int(sd["enc.0.body.0.weight"].shape[1])   # 10=无坐标, 12=含绝对坐标, +N_TERRAIN 地形
    # v3/v6/v7（tools/train_mt_v3.py）：in_ch 比 v2 多 N_TERRAIN 个已标准化地形通道。
    # in_ch<=12 走原 full_infer，行为与 v2/v4 完全一致（保证历史结果可复现）。
    terrain = in_ch > 12
    dom = "lum"                                       # 地形档（v3/v6/v7）DOM 沿用灰度
    if terrain:
        from train_mt_v3 import N_TERRAIN
        coord = (in_ch - N_TERRAIN == 12)
    else:
        # 非地形档：in_ch=10/12 → lum（v4 布局）；in_ch=11/13 → exg_sat（v8 真彩色彩度替换灰度）
        dom = "exg_sat" if in_ch in (11, 13) else "lum"
        coord = (in_ch == (11 if dom == "exg_sat" else 10) + 2)
    has_ms = any(k.startswith("seg_heads.") for k in sd)
    if multi_scale and not has_ms:
        log(f"  [SKIP] 该检查点无多档头（seg_heads.*）：{ckpt_path}")
        return None
    model = MultiTaskUNetV2(in_ch=in_ch, base=base, norm="none",
                            multi_scale=multi_scale).to(dev)
    model.load_state_dict(sd)
    if terrain:
        from train_mt_v3 import full_infer_v3, load_terrain_norm
        _terrain = load_terrain_norm(val_site)

        def _infer(model_, site_, dev_, coord=True, drop=None):
            return full_infer_v3(model_, site_, dev_, _terrain, coord=coord, drop=drop)
    else:
        def _infer(model_, site_, dev_, coord=True, drop=None):
            return full_infer(model_, site_, dev_, coord=coord, drop=drop, dom=dom)
    log(f"  检查点: in_ch={in_ch} coord={coord} 地形通道={terrain} dom={dom} "
        f"multi_scale={multi_scale} (含多档头={has_ms})")

    r = _infer(model, val_site, dev, coord=coord)
    seg_prob = r["seg_prob"]

    # 合成丢块补洞（与训练同口径）：模型 vs 站点常数 vs 最近邻填充
    drop = block_drop_mask(valid, np.random.default_rng(777), *EVAL_FRAC, *EVAL_BS)
    r2 = _infer(model, val_site, dev, coord=coord, drop=drop)
    m = drop
    dem_r = r2["dem_m"]
    idx = distance_transform_edt(~(valid & (~drop)), return_distances=False,
                                 return_indices=True)
    nn_fill = dem[tuple(idx)]
    fill_model = float(np.sqrt(((r2["dem_pred_m"][m] - dem_r[m]) ** 2).mean()))
    fill_const = float(np.sqrt(((st["dem_mean"][0] - dem_r[m]) ** 2).mean()))
    fill_nn = float(np.sqrt(((nn_fill[m] - dem_r[m]) ** 2).mean()))
    fill_extra = dict(fill_model_rmse=fill_model, fill_const_rmse=fill_const,
                      fill_nn_rmse=fill_nn, fill_drop_px=int(m.sum()))
    log(f"  合成丢块补洞 RMSE（撤块 2–4%）：模型={fill_model:.3f} m，"
        f"站点常数={fill_const:.3f} m，最近邻填充={fill_nn:.3f} m")

    seg_bin = seg_prob > 0.5

    # D10 评估 mask：排除 85px 边界 + 有效区
    em = np.zeros(slope.shape, dtype=bool)
    em[BORDER:-BORDER, BORDER:-BORDER] = True
    em_valid = em & valid

    # ---- 整体指标 ----
    iou_all, inter_all, union_all = compute_iou(seg_bin, label, em_valid)
    chamfer_all = chamfer_distance(seg_bin, label, em_valid)

    log(f"\n{'='*60}")
    log(f"方向: {train_site} → {val_site}")
    log(f"  整体 IoU = {iou_all:.4f}  (inter={inter_all}, union={union_all})")
    log(f"  Chamfer 距离 = {chamfer_all:.2f} px")
    log(f"  合成丢块补洞 RMSE = {fill_model:.4f} m "
        f"(常数={fill_const:.4f}, 最近邻={fill_nn:.4f})")

    # ---- 多档分割 IoU（1/1 与整体 IoU 同值，1/2、1/4 供深监督对照）----
    ms_iou = {}
    if multi_scale:
        for t in ("1", "2", "4"):
            key = f"seg_prob_{t}"
            if key not in r:
                continue
            k = int(t)
            lbl_i = label if k == 1 else pool_bool(label, k)
            val_i = valid if k == 1 else pool_bool(valid, k)
            b = max(BORDER // k, 1)
            em_i = np.zeros(lbl_i.shape, dtype=bool)
            if b < min(lbl_i.shape):
                em_i[b:-b, b:-b] = True
            else:
                em_i[:] = True
            iou_i, i_i, u_i = compute_iou(r[key] > 0.5, lbl_i, em_i & val_i)
            ms_iou[t] = dict(iou=float(iou_i), inter=int(i_i), union=int(u_i),
                             n_label=int(lbl_i.sum()))
        log("  多档 IoU（同一权重）: " + "  ".join(
            f"1/{t}={ms_iou[t]['iou']:.4f}(GT={ms_iou[t]['n_label']})"
            for t in ("1", "2", "4") if t in ms_iou))

    # ---- 坡度分层 ----
    p20, p40, p60, p80 = np.percentile(slope[em_valid], [20, 40, 60, 80])
    layers = [
        ("P20", 0, p20),
        ("P40", p20, p40),
        ("P60", p40, p60),
        ("P80", p60, p80),
        ("P100", p80, 90),
    ]

    log(f"\n  坡度分层 (边界: P20={p20:.1f}, P40={p40:.1f}, P60={p60:.1f}, P80={p80:.1f}):")
    log(f"  {'层':>6s} {'坡度范围':>12s} {'像元数':>10s} {'走廊数':>8s} "
        f"{'IoU':>8s} {'inter':>8s} {'union':>8s}")
    log(f"  {'-'*66}")

    stratified = {}
    for lname, lo, hi in layers:
        layer_mask = em_valid & (slope >= lo) & (slope < hi)
        n_px = int(layer_mask.sum())
        n_cor = int((label & layer_mask).sum())
        if n_px == 0:
            log(f"  {lname:>6s} [{lo:5.1f},{hi:5.1f}°) {n_px:>10,d} {n_cor:>8,d}  (空)")
            stratified[lname] = dict(n_pixels=0, n_corridor=0, iou=0.0)
            continue
        iou, inter, union = compute_iou(seg_bin, label, layer_mask)
        log(f"  {lname:>6s} [{lo:5.1f},{hi:5.1f}°) {n_px:>10,d} {n_cor:>8,d} "
            f"{iou:>8.4f} {inter:>8,d} {union:>8,d}")
        stratified[lname] = dict(
            n_pixels=n_px, n_corridor=n_cor,
            iou=float(iou), inter=int(inter), union=int(union),
            slope_lo=float(lo), slope_hi=float(hi),
        )

    return dict(
        direction=f"{train_site}→{val_site}",
        stage=stage,
        overall_iou=float(iou_all),
        chamfer_px=float(chamfer_all),
        inter=int(inter_all), union=int(union_all),
        slope_boundaries=dict(p20=float(p20), p40=float(p40),
                              p60=float(p60), p80=float(p80)),
        seg_stratified=stratified,
        multi_scale=bool(multi_scale),
        seg_multiscale_iou=ms_iou,
        **fill_extra,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="v2", choices=["v2"], help="仅保留 v2（v1 已废弃）")
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--prefix", default="", help="检查点前缀，如 'v4_'；空=不带前缀")
    ap.add_argument("--stage", default="ft", choices=["ft", "pre"],
                    help="ft=按 IoU 选出的微调权重；pre=按补洞 RMSE 选出的预训练权重")
    ap.add_argument("--multi-scale", action="store_true",
                    help="按多档模型加载权重，并额外报告 1/1、1/2、1/4 三档 IoU")
    a = ap.parse_args()
    base = a.base
    prefix = a.prefix

    tag = "_v2"
    ptag = prefix.rstrip("_")
    if ptag:
        tag += f"_{ptag}"          # 把版本前缀带进输出文件名，避免不同 tag 互相覆盖
    if a.stage != "ft":
        tag += f"_{a.stage}"
    if a.multi_scale:
        tag += "_ms"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = os.path.join(RES, f"eval_d10{tag}_base{base}.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"D10 分层评估: arch={a.arch} base={base}, prefix={prefix!r}, stage={a.stage}, "
        f"multi_scale={a.multi_scale}, 排除 {BORDER}px 边界带")
    log(f"设备: {dev}")

    all_results = []
    for tr_site, va_site in [("杨家坝", "西坝村"), ("西坝村", "杨家坝")]:
        r = eval_direction(tr_site, va_site, base, dev, log,
                           prefix=prefix, stage=a.stage,
                           multi_scale=a.multi_scale)
        if r:
            all_results.append(r)

    # 汇总
    log(f"\n{'='*60}")
    log("D10 汇总表")
    log(f"{'='*60}")
    header = (f"{'方向':>16s} {'IoU':>8s} {'补洞RMSE':>10s} {'Chamfer':>9s} "
              f"{'P20':>6s} {'P40':>6s} {'P60':>6s} {'P80':>6s} {'P100':>6s}")
    log(header)
    log("-" * len(header))
    for r in all_results:
        layers = r["seg_stratified"]
        row = (f"{r['direction']:>16s} {r['overall_iou']:>8.4f} "
               f"{r['fill_model_rmse']:>10.4f} {r['chamfer_px']:>9.2f} ")
        for ln in ["P20", "P40", "P60", "P80", "P100"]:
            iou = layers.get(ln, {}).get("iou", 0.0)
            row += f" {iou:>6.4f}"
        log(row)

    out = os.path.join(RES, f"eval_d10{tag}_base{base}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log(f"\n保存: {out}")
    log_file.close()


if __name__ == "__main__":
    main()
