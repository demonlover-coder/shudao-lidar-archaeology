# -*- coding: utf-8 -*-
"""同站自验（提高廊道识别 · 第 2 步）：站内切 2×2，3 训 1 验，块间 85px 缓冲带。

问题
----
跨站双向 IoU 只有 0.021 / 0.023（与几何基线 0.011~0.028 同量级），且 v6 分割头在
验证站**饱和成全正**（预测像元≈整窗）。这到底是"跨站泛化失败"还是"走廊逐像素分割
任务本身不可分"？本脚本在**站内**做留一象限验证：训练与验证真值同分布，唯一变量是
"没见过的空间位置"，因此给出模型在本任务上的**站内上限**。

协议
----
* 每站按 y=H//2、x=W//2 切 2×2 象限，轮流留 1 个象限做验证（3 训 1 验），4 折全跑。
* 缓冲带 85 px = RF(170 px) 的一半：训练像元到验证区的最近距离 > 85 px，
  感受野恰好看不到验证区 → 无标签泄漏。
* 验证窗 = 留出象限 ∩ (到训练区距离 > 85px) ∩ D10 内缩 85px 边界带 ∩ valid，
  即沿用 `eval_d10.py` 的 BORDER=85 口径。
* 监督只在训练区：把数据集 `obs_nat`/`label` 乘训练区掩码 → 验证区像元不进损失，
  但输入特征仍取真实值（与跨站训练/推理口径一致：只屏蔽监督，不屏蔽特征）。
* 归一化 / 模型 / 两阶段 / 损失权重全部沿用 v2（in_ch=10、coord=False），
  只换数据集与验证口径 —— 与跨站 v4 同架构，可直接比。

判据（用户给定）
----
    站内 IoU > 0.3 → 模型有能力，跨站是泛化问题 → 做第 4 步
    站内 IoU < 0.1 → 任务定义问题 → 做第 3 步
细长走廊对 IoU 不友好，故同报中心线 F1 / 长度召回（与第 1 步
`eval_linear_metrics.py` 完全同口径，便于交叉印证）。

用法
----
    python tools/train_same_site.py                       # 两站 4 折
    python tools/train_same_site.py --sites 杨家坝 --folds 2
    python tools/train_same_site.py --pretrain_epochs 12 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import train_mt_v2 as V2                                            # noqa: E402
from train_mt_v2 import (BORDER, EVAL_BS, EVAL_FRAC, CorridorDatasetV2,  # noqa: E402
                         MultiTaskUNetV2, block_drop_mask, full_infer, load_site)
from eval_linear_metrics import linear_metrics                       # noqa: E402

DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")
SITES = ("杨家坝", "西坝村")
BUF = 85              # 训练/验证缓冲带（px）= RF(170) / 2
THRESH = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
TOL = 2.0             # 中心线/端点匹配容差（px）


# ------------------------------------------------------------ 象限划分 ----
def quad_id(shape) -> np.ndarray:
    """0=左上 1=右上 2=左下 3=右下（切点 H//2、W//2）。"""
    H, W = shape
    hy = (np.arange(H) >= H // 2).astype(np.int8)
    wx = (np.arange(W) >= W // 2).astype(np.int8)
    return hy[:, None] * 2 + wx[None, :]


def region_masks(site, val_q, buf=BUF):
    """返回 (train_ok, val_ok)。均为 bool 全图掩码。"""
    d = load_site(site)
    valid = d["valid"] == 1
    q = quad_id(valid.shape)
    val_region = (q == val_q)
    train_region = ~val_region
    # 训练像元到验证区的最近距离必须 > buf（感受野看不到验证区）
    d_val = distance_transform_edt(~val_region)
    # 验证像元到训练区的最近距离必须 > buf（边界处预测不受训练区上下文偏置）
    d_tr = distance_transform_edt(~train_region)
    train_ok = train_region & (d_val > buf) & valid
    em = np.zeros(valid.shape, dtype=bool)
    em[BORDER:-BORDER, BORDER:-BORDER] = True      # D10 内缩边界带
    val_ok = val_region & (d_tr > buf) & em & valid
    return train_ok, val_ok


# ------------------------------------------------------------ 数据集 ----
class SameSiteDataset(CorridorDatasetV2):
    """v2 数据集 + 训练区掩码：验证区不参与采样与监督（输入特征保持真实）。

    基类 `__getitem__` 只读 `self.obs_nat` / `self.label`，因此把这两个成员替换成
    训练区受限版本即可，无需覆写 `__getitem__`：
      * `obs_nat` → 训练区自然有效掩码（合成丢块也只落在训练区）
      * `label`   → 训练区标签（验证区标签置 0，不进分割损失）
      * 采样池     → 仅训练区（正样本优先策略不变）
    """

    def __init__(self, site, train_ok, **kw):
        super().__init__(site, **kw)
        self.train_ok = train_ok
        self.obs_nat = self.obs_nat & train_ok
        self.label = (self.label * train_ok).astype(np.float32)
        self.pos_coords = np.argwhere(self.label == 1)
        self.valid_coords = np.argwhere(self.obs_nat)


# ------------------------------------------------------------ 评估 ----
@torch.no_grad()
def eval_fill_rmse_region(model, site, region, dev, coord, seed=777):
    """合成丢块补洞 RMSE（m），丢块只在 `region` 内 → 预训练选点不看验证区。"""
    d = load_site(site)
    obs = (d["valid"] == 1) & region
    drop = block_drop_mask(obs, np.random.default_rng(seed), *EVAL_FRAC, *EVAL_BS)
    if int(drop.sum()) == 0:
        return float("nan"), 0
    r = full_infer(model, site, dev, coord=coord, drop=drop)
    m = r["drop"]
    return float(np.sqrt(((r["dem_pred_m"][m] - r["dem_m"][m]) ** 2).mean())), int(m.sum())


@torch.no_grad()
def eval_val_region(model, site, val_ok, dev, coord, tol=TOL):
    """验证象限评估：IoU@0.5（主判据）+ 中心线 F1 阈值扫描（细长走廊友好口径）。"""
    r = full_infer(model, site, dev, coord=coord)
    seg = r["seg_prob"]
    label = r["label"].astype(bool)
    iou, inter, union = V2._iou(seg > 0.5, label, val_ok)
    rows = []
    for t in THRESH:
        m = linear_metrics(seg > t, label, val_ok, tol)
        m["thresh"] = float(t)
        rows.append(m)
    best = max(rows, key=lambda m: m["f1"])
    return dict(iou=float(iou), inter=int(inter), union=int(union),
                pred_px_at05=int(((seg > 0.5) & val_ok).sum()),
                val_px=int(val_ok.sum()),
                best=best, sweep=rows)


# ------------------------------------------------------------ 单折训练 ----
def train_fold(site, val_q, a, log, dev):
    d = load_site(site)
    label = d["label"].astype(bool)
    train_ok, val_ok = region_masks(site, val_q, BUF)
    n_tr_pos = int((label & train_ok).sum())
    n_va_pos = int((label & val_ok).sum())
    H, W = label.shape
    log(f"\n{'-' * 68}")
    log(f"折: {site} 验证象限 q{val_q}  切点 y={H // 2} x={W // 2}  缓冲带={BUF}px(=RF/2)")
    log(f"  训练区: 像元={int(train_ok.sum()):,}  走廊={n_tr_pos:,}")
    log(f"  验证区: 像元={int(val_ok.sum()):,}  走廊={n_va_pos:,} "
        f"(占验证区 {n_va_pos / max(int(val_ok.sum()), 1):.4f})")
    if n_va_pos == 0:
        log(f"  [SKIP] 验证区无走廊真值（被缓冲带/D10 边界带剔除），本折不成立")
        return None
    if n_tr_pos < 200:
        log(f"  [警告] 训练区走廊仅 {n_tr_pos} 像元，样本极少")

    ds = SameSiteDataset(site, train_ok, crop=a.crop, seed=a.seed, epoch_len=a.epoch_len,
                         coord=a.coord, drop_frac_lo=a.drop_lo, drop_frac_hi=a.drop_hi,
                         fill_radius=a.fill_radius)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=0)
    log(f"  训练集采样池: 有效={len(ds.valid_coords):,} 走廊={len(ds.pos_coords):,} "
        f"(正样本优先 pos_frac={ds.pos_frac})  走廊/有效="
        f"{float(ds.label.mean()) / max(float(ds.obs_nat.mean()), 1e-9):.4f}")
    log(f"  站点统计 mu={ds.mu:.2f} m sd={ds.sd:.2f} m  in_ch={ds.in_ch} coord={a.coord}")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = MultiTaskUNetV2(in_ch=ds.in_ch, base=a.base, down=4, norm=a.norm,
                            alpha=a.alpha, feather=a.feather,
                            multi_scale=a.multi_scale).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    log(f"  模型: MultiTaskUNetV2 base={a.base} 参数量={npar:,} ({npar / 1e6:.2f}M) "
        f"norm={a.norm} multi_scale={a.multi_scale}")

    pre_ckpt = os.path.join(CKPT, f"{a.tag}_pre_{site}_q{val_q}_base{a.base}.pth")
    ft_ckpt = os.path.join(CKPT, f"{a.tag}_ft_{site}_q{val_q}_base{a.base}.pth")

    # ---- 阶段 1：空洞预训练（选点=训练区合成丢块 RMSE） ----
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.pretrain_epochs, 1))
    log(f"\n  --- 阶段 1：空洞预训练（{a.pretrain_epochs} ep） ---")
    best_fill, no_imp = float("inf"), 0
    for ep in range(1, a.pretrain_epochs + 1):
        t0 = time.time()
        st = V2.train_epoch(model, loader, opt, dev, "pretrain", a)
        sched.step()
        tf, npx = eval_fill_rmse_region(model, site, train_ok, dev, a.coord)
        log(f"  [pre {ep:02d}] loss={st['loss']:.4f} fill={st['fill']:.4f} "
            f"res={st['res']:.4f} hole={st['hole']:.4f} | train_fill_rmse={tf:.4f} m "
            f"(n={npx:,}) alpha={torch.sigmoid(model.alpha).item():.4f} "
            f"({time.time() - t0:.0f}s)")
        if tf < best_fill - 1e-6:
            best_fill, no_imp = tf, 0
            torch.save(model.state_dict(), pre_ckpt)
        else:
            no_imp += 1
            if no_imp >= a.patience:
                log(f"  早停（patience={a.patience}）")
                break
    log(f"  预训练最优 train_fill_rmse={best_fill:.4f} m")

    # ---- 阶段 2：微调（选点=验证象限 IoU@0.5） ----
    if os.path.exists(pre_ckpt):
        model.load_state_dict(torch.load(pre_ckpt, map_location=dev))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))
    log(f"\n  --- 阶段 2：微调（{a.epochs} ep；早停=验证象限 IoU@0.5） ---")
    best_iou, best_ep, no_imp = -1.0, -1, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        st = V2.train_epoch(model, loader, opt, dev, "finetune", a)
        sched.step()
        ev = eval_val_region(model, site, val_ok, dev, a.coord)
        log(f"  [ft {ep:02d}] loss={st['loss']:.4f} seg={st['seg']:.4f} "
            f"grad={st['grad']:.4f} fill={st['fill']:.4f} res={st['res']:.4f} "
            f"hole={st['hole']:.4f} | IoU@0.5={ev['iou']:.4f} (i={ev['inter']},"
            f"u={ev['union']}) 中心线F1={ev['best']['f1']:.4f} "
            f"LR={ev['best']['lr']:.4f} 预测像元={ev['pred_px_at05']:,}/"
            f"{ev['val_px']:,} ({time.time() - t0:.0f}s)")
        if ev["iou"] > best_iou + 1e-6:
            best_iou, best_ep, no_imp = ev["iou"], ep, 0
            torch.save(model.state_dict(), ft_ckpt)
        else:
            no_imp += 1
            if no_imp >= a.patience:
                log(f"  早停（patience={a.patience}）")
                break

    # ---- 末评（用最优 IoU 权重） ----
    if os.path.exists(ft_ckpt):
        model.load_state_dict(torch.load(ft_ckpt, map_location=dev))
    ev = eval_val_region(model, site, val_ok, dev, a.coord)
    vf, _ = eval_fill_rmse_region(model, site, val_ok, dev, a.coord)
    b = ev["best"]
    log(f"  【{site} q{val_q}】最优 IoU@0.5={ev['iou']:.4f} (ep {best_ep}) | "
        f"中心线F1={b['f1']:.4f} LR={b['lr']:.4f} LP={b['lp']:.4f} "
        f"中心线={b['centerline_px']:.2f}px 拓扑R/P={b['topo_recall']:.2f}/"
        f"{b['topo_precision']:.2f} 端点F1={b['ep_f1']:.2f} (最优thr={b['thresh']}) | "
        f"验证区补洞RMSE={vf:.4f} m")
    return dict(site=site, val_q=int(val_q), n_params=int(npar),
                train_pos=n_tr_pos, val_pos=n_va_pos, val_px=int(val_ok.sum()),
                iou=ev["iou"], best_ep=int(best_ep), val_fill_rmse=float(vf),
                centerline_f1=float(b["f1"]), lr=float(b["lr"]), lp=float(b["lp"]),
                centerline_px=float(b["centerline_px"]),
                topo_recall=float(b["topo_recall"]),
                topo_precision=float(b["topo_precision"]),
                ep_f1=float(b["ep_f1"]), best_thresh=float(b["thresh"]),
                pred_px_at05=int(ev["pred_px_at05"]),
                sweep=ev["sweep"])


# ------------------------------------------------------------ 主 ----
def main():
    p = argparse.ArgumentParser(description="同站自验：站内 2×2 切块，3 训 1 验，85px 缓冲带")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3])
    p.add_argument("--base", type=int, default=16, choices=[16, 24])
    p.add_argument("--norm", type=str, default="none", choices=["none", "bn"])
    p.add_argument("--coord", action="store_true", default=False,
                   help="绝对坐标通道；跨站消融显示它显著恶化泛化，默认关闭")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--pretrain_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--epoch_len", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--feather", type=int, default=3)
    p.add_argument("--multi_scale", action="store_true", help="开启 1/1、1/2、1/4 深监督")
    p.add_argument("--drop_lo", type=float, default=0.15)
    p.add_argument("--drop_hi", type=float, default=0.40)
    p.add_argument("--fill_radius", type=int, default=8)
    p.add_argument("--w_seg", type=float, default=1.0)
    p.add_argument("--w_grad", type=float, default=0.3)
    p.add_argument("--w_hole", type=float, default=0.5)
    p.add_argument("--w_res", type=float, default=0.2)
    p.add_argument("--w_prior", type=float, default=0.05)
    p.add_argument("--w_holetype", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="ss")
    a = p.parse_args()

    os.makedirs(CKPT, exist_ok=True)
    log_path = os.path.join(RES, f"train_same_site_{a.tag}_base{a.base}.log")
    lf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n")
        lf.flush()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"同站自验（提高廊道识别 · 第 2 步） tag={a.tag} base={a.base} norm={a.norm} "
        f"coord={a.coord} crop={a.crop} bs={a.batch_size} "
        f"pre_ep={a.pretrain_epochs} ft_ep={a.epochs} patience={a.patience}")
    log(f"协议：站内 2×2 切块，3 训 1 验；缓冲带 {BUF}px(=RF/2)；验证窗 = 留出象限 ∩ "
        f"距离训练区>{BUF}px ∩ D10 内缩 {BORDER}px ∩ valid")
    log(f"判据：站内 IoU@0.5 > 0.3 → 模型有能力/跨站是泛化问题；< 0.1 → 任务定义问题")
    log(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    out = []
    for site in a.sites:
        log(f"\n{'=' * 78}\n站点: {site}")
        for q in a.folds:
            r = train_fold(site, q, a, log, dev)
            if r:
                out.append(r)

    # ---- 汇总（按验证区走廊像元加权） ----
    log(f"\n{'=' * 78}\n汇总（同站自验）")
    log(f"{'站点':>8s} {'折':>3s} {'验证走廊':>9s} {'IoU@0.5':>9s} {'中心线F1':>9s} "
        f"{'LR':>7s} {'LP':>7s} {'中心线px':>9s} {'补洞RMSE':>9s}")
    log("-" * 88)
    for r in out:
        log(f"{r['site']:>8s} {r['val_q']:>3d} {r['val_pos']:>9d} {r['iou']:>9.4f} "
            f"{r['centerline_f1']:>9.4f} {r['lr']:>7.4f} {r['lp']:>7.4f} "
            f"{r['centerline_px']:>9.2f} {r['val_fill_rmse']:>9.4f}")

    per_site = {}
    for site in a.sites:
        rs = [r for r in out if r["site"] == site]
        if not rs:
            continue
        w = np.array([max(r["val_pos"], 1) for r in rs], dtype=np.float64)
        w = w / w.sum()
        iou_w = float(np.sum([r["iou"] * wi for r, wi in zip(rs, w)]))
        f1_w = float(np.sum([r["centerline_f1"] * wi for r, wi in zip(rs, w)]))
        iou_max = float(max(r["iou"] for r in rs))
        f1_max = float(max(r["centerline_f1"] for r in rs))
        per_site[site] = dict(n_folds=len(rs), iou_weighted=iou_w, f1_weighted=f1_w,
                              iou_best_fold=iou_max, f1_best_fold=f1_max)
        log(f"  {site}: {len(rs)} 折  加权 IoU@0.5={iou_w:.4f}  加权 中心线F1={f1_w:.4f}  "
            f"最优折 IoU={iou_max:.4f}  最优折 F1={f1_max:.4f}")

    if per_site:
        iou_bar = float(np.mean([v["iou_weighted"] for v in per_site.values()]))
        f1_bar = float(np.mean([v["f1_weighted"] for v in per_site.values()]))
        if iou_bar > 0.3:
            verdict = (f"站内 IoU@{0.5} 加权均值 {iou_bar:.4f} > 0.3 → 模型有能力，"
                       f"跨站是泛化问题 → 做第 4 步")
        elif iou_bar < 0.1:
            verdict = (f"站内 IoU@0.5 加权均值 {iou_bar:.4f} < 0.1 → 任务定义问题 → 做第 3 步")
        else:
            verdict = (f"站内 IoU@0.5 加权均值 {iou_bar:.4f} 落在 [0.1, 0.3] → 中间地带："
                       f"结合中心线 F1={f1_bar:.4f} 判断；若 F1 亦 < 0.1 则按任务定义问题"
                       f"继续第 3 步")
        log(f"\n判据：{verdict}")

    jp = os.path.join(RES, f"train_same_site_{a.tag}_base{a.base}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(tag=a.tag, base=a.base, buf_px=BUF, border=BORDER,
                       sites=list(a.sites), folds=list(a.folds),
                       per_site=per_site, folds_detail=out,
                       verdict=verdict if per_site else "无有效折"),
                  f, ensure_ascii=False, indent=2)
    log(f"\n保存: {jp}")
    lf.close()


if __name__ == "__main__":
    main()
