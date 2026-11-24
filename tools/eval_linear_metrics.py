# -*- coding: utf-8 -*-
"""线性（中心线）评估指标：换掉对细长走廊不友好的 IoU。

背景
----
走廊在 1 m 网格上只占 ~1% 像元、是**细长条带**，逐像素 IoU 天然偏低
（v6 跨站 IoU ≈ 0.021，几何基线 0.011~0.028）。可能不是模型差，而是指标不友好。
本脚本用 **v6 分割输出**重算一组对线状目标更公平的指标：

    · 长度召回率  LR = 真值中心线被预测中心线覆盖的比例
    · 长度精确率  LP = 预测中心线落在真值中心线容差内的比例
    · 中心线 F1   = LR/LP 的调和平均（**主线判据**）
    · 中心线距离  = 真值↔预测中心线的对称平均距离（px）
    · 拓扑连通性  = 真值骨架连通分量被"走到"的比例（topology recall）
                    与预测骨架连通分量命中真值的比例（topology precision）
    · 端点匹配    = 骨架端点按容差匹配后的 recall / precision / F1

判据（用户给定）
----
    中心线 F1 > 0.3  → 模型其实不差，是 IoU 不友好 → 可直接用
    中心线 F1 < 0.1  → 确实差 → 继续第 2 步（同站自验）

口径
----
与 D10 完全一致：同 mask（npz label）、排除 85 px 边界带、只在 valid 内评估。
分割阈值默认在 {0.5,0.6,0.7,0.8,0.9,0.95,0.99} 上扫（0.5 = IoU 口径），
取 F1 最高的一档作为"可达线性性能"，同时保留 0.5 档以便与 IoU 对照。

⚠ 通道口径修正（相对 `eval_d10.py`）
----
`train_mt_v3.N_TERRAIN` 已由 5 改为 8（v7 扩通道），但 v6 检查点 in_ch=15
（= 10 基础 + **5** 地形）。若沿用 eval_d10 的 `load_terrain_norm(site)`（返回 8 通道）
会给 15 通道模型喂 18 通道输入 → 形状不匹配。本脚本按**检查点自身的 in_ch**
反推地形通道数并切片（v4=0 / v6=5 / v7=8），保证 v6 结果与 D12 #16 口径一致。

用法
----
    python tools/eval_linear_metrics.py                 # 两站都跑（v6）
    python tools/eval_linear_metrics.py --prefix v6_ --stage pre
    python tools/eval_linear_metrics.py --tol 3 --thresh 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from scipy.ndimage import distance_transform_edt, label as ndlabel
from skimage.morphology import skeletonize

DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")

BORDER = 85          # 与 eval_d10 / contract.json 同口径
CONN8 = np.ones((3, 3), dtype=np.int32)
NEIGH_K = np.ones((3, 3), dtype=np.int8)


# ------------------------------------------------------------ 推理 ----
def infer(prefix, stage, base, dev, train_site, val_site, log):
    """按与 eval_d10 相同的口径做全图推理，返回 seg_prob 与掩膜（修正地形通道数）。"""
    ckpt = os.path.join(CKPT, f"{prefix}{stage}_{train_site}_{val_site}_base{base}.pth")
    if not os.path.exists(ckpt):
        log(f"  [SKIP] 检查点不存在: {ckpt}")
        return None

    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd["state_dict"]
    in_ch = int(sd["enc.0.body.0.weight"].shape[1])

    # 基础通道：10 = 无坐标，12 = 含绝对坐标；其余为已标准化地形通道
    if in_ch <= 12:
        n_ter, coord = 0, (in_ch == 12)
    else:
        n_ter, coord = in_ch - 10, False      # 跨站场景一律不含绝对坐标（D12 #6）

    from train_mt_v2 import MultiTaskUNetV2, full_infer
    model = MultiTaskUNetV2(in_ch=in_ch, base=base, norm="none", multi_scale=False).to(dev)
    model.load_state_dict(sd)
    log(f"  检查点: {os.path.basename(ckpt)}  in_ch={in_ch}  地形通道={n_ter}  coord={coord}")

    if n_ter:
        from train_mt_v3 import full_infer_v3, load_terrain_norm
        terrain = load_terrain_norm(val_site)[:n_ter]     # ← 按 in_ch 切片（关键修正）
        r = full_infer_v3(model, val_site, dev, terrain, coord=coord)
    else:
        r = full_infer(model, val_site, dev, coord=coord)
    return r


# ------------------------------------------------------------ 几何工具 ----
def window_mask(valid: np.ndarray) -> np.ndarray:
    """D10 评估窗：排除 85 px 边界带 ∩ valid。"""
    em = np.zeros(valid.shape, dtype=bool)
    em[BORDER:-BORDER, BORDER:-BORDER] = True
    return em & valid


def endpoints(skel: np.ndarray) -> np.ndarray:
    """8 邻域内恰有 1 个骨架邻居的像元 = 端点。"""
    if skel.sum() == 0:
        return np.zeros_like(skel)
    from scipy.ndimage import convolve
    nb = convolve(skel.astype(np.int8), NEIGH_K, mode="constant") - skel.astype(np.int8)
    return skel & (nb == 1)


def comp_stats(skel: np.ndarray):
    lab, n = ndlabel(skel, structure=CONN8)
    return lab, int(n)


# ------------------------------------------------------------ 指标 ----
def linear_metrics(pred_bin, gt_bin, win, tol):
    """在评估窗内计算线性指标。pred_bin / gt_bin 为全图 bool。"""
    p = (pred_bin & win)
    g = (gt_bin & win)

    ps, gs = skeletonize(p), skeletonize(g)
    ps, gs = ps & win, gs & win
    n_p, n_g = int(ps.sum()), int(gs.sum())
    out = dict(pred_px=int(p.sum()), gt_px=int(g.sum()),
               pred_skel_px=n_p, gt_skel_px=n_g)
    if n_p == 0 or n_g == 0:
        out.update(lr=0.0, lp=0.0, f1=0.0, centerline_px=float("nan"),
                   topo_recall=0.0, topo_precision=0.0,
                   gt_comps=0, pred_comps=0, ep_recall=0.0, ep_precision=0.0,
                   ep_f1=0.0, n_ep_gt=0, n_ep_pred=0)
        return out

    dt_p = distance_transform_edt(~ps)        # 到最近"预测中心线"的距离
    dt_g = distance_transform_edt(~gs)

    g_cov = gs & (dt_p <= tol)                # 真值中心线被预测覆盖
    p_cov = ps & (dt_g <= tol)                # 预测中心线落在真值容差内
    lr = float(g_cov.sum() / n_g)
    lp = float(p_cov.sum() / n_p)
    f1 = float(2 * lr * lp / (lr + lp)) if (lr + lp) > 0 else 0.0

    cl = float((dt_p[gs].mean() + dt_g[ps].mean()) / 2.0)

    # 拓扑：真值骨架的每个连通分量是否"被走到"；预测骨架的分量是否命中真值
    lab_g, n_cg = comp_stats(gs)
    lab_p, n_cp = comp_stats(ps)
    reached = set(np.unique(lab_g[g_cov]).tolist()) - {0}
    hit = set(np.unique(lab_p[p_cov]).tolist()) - {0}
    topo_rec = float(len(reached) / n_cg) if n_cg else 0.0
    topo_pre = float(len(hit) / n_cp) if n_cp else 0.0

    # 端点匹配
    ep_g, ep_p = endpoints(gs), endpoints(ps)
    n_eg, n_ep = int(ep_g.sum()), int(ep_p.sum())
    ep_rec = float((dt_p[ep_g] <= tol).sum() / n_eg) if n_eg else float("nan")
    ep_pre = float((dt_g[ep_p] <= tol).sum() / n_ep) if n_ep else float("nan")
    if np.isfinite(ep_rec) and np.isfinite(ep_pre) and (ep_rec + ep_pre) > 0:
        ep_f1 = float(2 * ep_rec * ep_pre / (ep_rec + ep_pre))
    else:
        ep_f1 = 0.0

    out.update(lr=lr, lp=lp, f1=f1, centerline_px=cl,
               topo_recall=topo_rec, topo_precision=topo_pre,
               gt_comps=n_cg, pred_comps=n_cp,
               ep_recall=ep_rec, ep_precision=ep_pre, ep_f1=ep_f1,
               n_ep_gt=n_eg, n_ep_pred=n_ep)
    return out


def iou(pred_bin, gt_bin, win):
    inter = int((pred_bin & gt_bin & win).sum())
    union = int(((pred_bin | gt_bin) & win).sum())
    return (float(inter / union) if union else 0.0), inter, union


# ------------------------------------------------------------ 单方向 ----
def run_direction(prefix, stage, base, dev, train_site, val_site, tol, thresh_list, log):
    r = infer(prefix, stage, base, dev, train_site, val_site, log)
    if r is None:
        return None

    seg_prob = r["seg_prob"]
    gt = r["label"].astype(bool)
    win = window_mask(r["valid"].astype(bool))
    log(f"  评估窗像元 {int(win.sum()):,}，真值走廊（窗内）{int((gt & win).sum()):,}")

    rows = []
    for t in thresh_list:
        m = linear_metrics(seg_prob > t, gt, win, tol)
        i, inter, union = iou(seg_prob > t, gt, win)
        m["thresh"] = float(t)
        m["iou"] = i
        m["inter"] = inter
        m["union"] = union
        rows.append(m)
        log(f"  thr={t:<5.2f} LR={m['lr']:.4f} LP={m['lp']:.4f} F1={m['f1']:.4f} "
            f"中心线={m['centerline_px']:.2f}px 拓扑R/P={m['topo_recall']:.2f}/"
            f"{m['topo_precision']:.2f} 端点R={m['ep_recall']:.2f} "
            f"(IoU={i:.4f}, 预测像元={m['pred_px']:,})")

    best = max(rows, key=lambda m: m["f1"])
    base05 = next((m for m in rows if abs(m["thresh"] - 0.5) < 1e-9), rows[0])
    return dict(direction=f"{train_site}→{val_site}", prefix=prefix, stage=stage,
                base=base, tol_px=tol, win_px=int(win.sum()),
                gt_px=int((gt & win).sum()),
                best=best, at_0p5=base05, sweep=rows)


def main():
    ap = argparse.ArgumentParser(description="线性（中心线）评估指标 —— 替代对细长走廊不友好的 IoU")
    ap.add_argument("--prefix", default="v6_", help="检查点前缀（默认 v6_）")
    ap.add_argument("--stage", default="pre", choices=["pre", "ft"],
                    help="pre=按补洞 RMSE 选权重（默认，与 D12 #16 同口径）")
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--tol", type=float, default=2.0, help="中心线/端点匹配容差（px）")
    ap.add_argument("--thresh", type=float, nargs="*", default=None,
                    help="分割阈值列表；缺省则扫 0.5..0.99")
    a = ap.parse_args()

    thresh_list = a.thresh if a.thresh else [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = f"{a.prefix.rstrip('_') or 'raw'}_{a.stage}"
    log_path = os.path.join(RES, f"eval_linear_metrics_{tag}.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"线性评估指标: prefix={a.prefix!r} stage={a.stage} base={a.base} "
        f"tol={a.tol}px 阈值扫={thresh_list}  设备={dev}")
    log(f"口径：D10（同 mask / 排除 {BORDER}px 边界带 / 仅 valid）")

    results = []
    for tr, va in [("杨家坝", "西坝村"), ("西坝村", "杨家坝")]:
        log(f"\n{'='*78}\n方向: {tr} → {va}")
        out = run_direction(a.prefix, a.stage, a.base, dev, tr, va, a.tol, thresh_list, log)
        if out:
            results.append(out)

    if results:
        log(f"\n{'='*78}\n汇总（主线判据 = 中心线 F1，取阈值扫描中的最优档）")
        log(f"{'方向':>16s} {'F1':>7s} {'LR':>7s} {'LP':>7s} {'中心线px':>9s} "
            f"{'拓扑R':>7s} {'拓扑P':>7s} {'端点F1':>7s} {'IoU@0.5':>8s} {'最优thr':>8s}")
        log("-" * 96)
        for r in results:
            b, z = r["best"], r["at_0p5"]
            log(f"{r['direction']:>16s} {b['f1']:>7.4f} {b['lr']:>7.4f} {b['lp']:>7.4f} "
                f"{b['centerline_px']:>9.2f} {b['topo_recall']:>7.4f} {b['topo_precision']:>7.4f} "
                f"{b['ep_f1']:>7.4f} {z['iou']:>8.4f} {b['thresh']:>8.2f}")

        f1s = [r["best"]["f1"] for r in results]
        mean_f1 = float(np.mean(f1s))
        if mean_f1 > 0.3:
            verdict = f"模型其实不差，是 IoU 不友好 → 可直接用（中心线 F1 均值 {mean_f1:.4f} > 0.3）"
        elif mean_f1 < 0.1:
            verdict = f"确实差 → 继续第 2 步（中心线 F1 均值 {mean_f1:.4f} < 0.1）"
        else:
            verdict = (f"中间地带 → 保守继续第 2 步（中心线 F1 均值 {mean_f1:.4f} "
                       f"落在 [0.1, 0.3]）")
        log(f"\n判据：{verdict}")

        summary = dict(prefix=a.prefix, stage=a.stage, base=a.base, tol_px=a.tol,
                       thresholds=thresh_list, mean_best_f1=mean_f1,
                       verdict=verdict, sites=results)
    else:
        summary = dict(prefix=a.prefix, stage=a.stage, base=a.base, tol_px=a.tol,
                       thresholds=thresh_list, sites=[])

    out_json = os.path.join(RES, f"eval_linear_metrics_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n保存: {out_json}")
    log_file.close()


if __name__ == "__main__":
    main()
