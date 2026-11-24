# -*- coding: utf-8 -*-
"""v6 多任务 U-Net：在 v2 输入上追加 5 个 P2 地形因子通道，从 v4 权重**热启动**。

相对 v2（tools/train_mt_v2.py）的唯一数据差异：
  输入通道由 10（9 数据/掩码 + 0 坐标，coord=False）扩到 15，尾部追加 5 个已
  robust 标准化的 P2 地形因子 (slope, sin_aspect, cos_aspect, curvature, roughness)，
  读 `data/p2_terrain_{site}_norm.npy`（由 tools/p2_terrain_norm.py 生成，
  统计量只取自训练站并两站共用，避免训练/推理归一化不一致）。

热启动（不走全新预训练）：
  从 `checkpoints/v4_pre_{train}_{val}_base{base}.pth`（in_ch=10）载入；
  凡以 in_ch 为输入维的 4D 权重（首层 `enc.0.body.0` 与残差短接 `enc.0.short.0`）
  按 dim=1 扩展——前 10 通道复制 v4、后 8 通道置 0。零初值使初始输出与 v4 逐位一致，
  地形通道从"零贡献"开始学习。

复用 v2 的数据集骨架/损失/训练循环（只换数据集与全图推理口径），不改任何现有文件。
相对对照基线 = v4（`results/train_v4_base16.json`：杨→西 IoU 0.0，西→杨 IoU 0.0393）。

用法：
    python tools/train_mt_v3.py                  # tag=v6，两方向，热启动
    python tools/train_mt_v3.py --no-warm-start  # 消融：随机初始化
    python tools/train_mt_v3.py --tag v6a
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
from train_mt_v2 import (BORDER, DATA_IDX, FILL_RADIUS, HOLE_IDX,   # noqa: E402
                         CorridorDatasetV2, build_channels, compute_stats,
                         eval_drop_mask, idw_fill, load_site, train_epoch)
from unet_mt import MultiTaskUNetV2                                 # noqa: E402

DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")
SITES = ("杨家坝", "西坝村")
N_TERRAIN = 8


# ------------------------------------------------------------ 地形因子 ----
def load_terrain_norm(site: str) -> np.ndarray:
    """读 robust 标准化后的 8 通道地形因子 `(8,H,W)` float32。"""
    p = os.path.join(DATA, f"p2_terrain_{site}_norm.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"缺少 {p}；请先运行 `python tools/p2_terrain_norm.py` 生成标准化地形因子")
    t = np.load(p).astype(np.float32)
    if t.ndim != 3 or t.shape[0] != N_TERRAIN:
        raise ValueError(f"{p} 形状应为 (8,H,W)，收到 {t.shape}")
    return t


class CorridorDatasetV3(CorridorDatasetV2):
    """v2 数据集 + 尾部 8 个已标准化地形因子通道（静态量，不随丢块置零）。"""

    def __init__(self, site: str, terrain: np.ndarray, **kw):
        super().__init__(site, **kw)
        if terrain.shape[-2:] != self.obs_nat.shape:
            raise ValueError(
                f"地形因子 {terrain.shape[-2:]} 与站点网格 {self.obs_nat.shape} 不一致")
        # v2 的 __getitem__ 只对 0..8（数据）与 9（空洞 mask）动手，地形追加在尾部不受影响
        self.x = np.concatenate([self.x, terrain], axis=0)
        self.in_ch = int(self.x.shape[0])


# ------------------------------------------------------------ 热启动 ----
def expand_in_ch(sd_old: dict, sd_new: dict, log) -> dict:
    """in_ch 10 → 18：以 in_ch 为输入维的 4D 权重前 10 通道复制、后 8 置 0。"""
    out, n_exp = {}, 0
    for k, v_new in sd_new.items():
        if k not in sd_old:
            out[k] = v_new
            continue
        v_old = sd_old[k]
        if tuple(v_old.shape) == tuple(v_new.shape):
            out[k] = v_old
            continue
        if not (v_old.dim() == v_new.dim() == 4
                and v_old.shape[0] == v_new.shape[0]
                and v_old.shape[2:] == v_new.shape[2:]
                and v_new.shape[1] - v_old.shape[1] == N_TERRAIN):
            raise ValueError(f"权重 {k} 形状无法扩展：{tuple(v_old.shape)} → {tuple(v_new.shape)}")
        merged = torch.zeros_like(v_new)
        merged[:, :v_old.shape[1]] = v_old
        out[k] = merged
        n_exp += 1
        log(f"    扩展 {k}: {tuple(v_old.shape)} → {tuple(v_new.shape)}"
            f"（前 {v_old.shape[1]} 通道复制 v4，后 {N_TERRAIN} 通道置 0）")
    log(f"  共扩展 {n_exp} 处 in_ch 相关权重")
    return out


def warm_start_from_v4(model, train_site: str, val_site: str, base: int, log) -> bool:
    ws = os.path.join(CKPT, f"v4_pre_{train_site}_{val_site}_base{base}.pth")
    if not os.path.exists(ws):
        log(f"  [警告] 未找到 {ws}，改为随机初始化")
        return False
    sd = torch.load(ws, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd["state_dict"]
    log(f"  热启动：{os.path.basename(ws)}（v4 in_ch=10 → v3 in_ch={model.cfg['in_ch']}）")
    model.load_state_dict(expand_in_ch(sd, model.state_dict(), log), strict=True)
    return True


# ------------------------------------------------------------ 全图推理 ----
@torch.no_grad()
def full_infer_v3(model, site, dev, terrain, coord=True, drop=None):
    """v2.full_infer 的 v3 版：唯一差异是输入尾部拼 8 通道地形因子。"""
    d = load_site(site)
    st = compute_stats(d)
    mu, sd = st["dem_mean"]
    obs_nat = d["valid"] == 1
    if drop is None:
        drop = np.zeros_like(obs_nat)
    obs = obs_nat & (~drop)
    dem_m = d["dem_mean"].astype(np.float32)
    fill_m = idw_fill(dem_m, obs, radius=FILL_RADIUS)
    x = build_channels(d, st, coord=coord, fill=fill_m)
    x = np.concatenate([x, terrain], axis=0)                  # ← 唯一差异
    if drop.any():
        x[:len(DATA_IDX)][:, drop] = 0.0
        x[HOLE_IDX][drop] = 1.0
    x[0] = (fill_m - mu) / sd
    model.eval()
    xt = torch.from_numpy(x[None]).to(dev)
    hm = torch.from_numpy((~obs)[None, None].astype(np.float32)).to(dev)
    dm = torch.from_numpy(fill_m[None, None]).to(dev)
    out = model(xt, dem_fill=dm, hole_mask=hm)
    r = dict(seg_prob=torch.sigmoid(out["seg"])[0, 0].cpu().numpy(),
             dem_pred_m=out["dem_pred"][0, 0].cpu().numpy(),
             hole_prob=torch.sigmoid(out["hole_logit"])[0, 0].cpu().numpy(),
             obs=obs, drop=drop, valid=obs_nat, dem_m=dem_m, fill_m=fill_m,
             label=d["label"].astype(bool), slope=d["slope_deg"], stats=st)
    if getattr(model, "multi_scale", False):
        for t in ("1", "2", "4"):
            r[f"seg_prob_{t}"] = torch.sigmoid(out[f"seg_{t}"])[0, 0].cpu().numpy()
    return r


@torch.no_grad()
def eval_fill_rmse_v3(model, site, terrain, dev, coord=True, seed=777):
    """合成丢块补洞 RMSE（m）+ 同 mask 的常数 / 最近邻基线（口径同 v2）。"""
    drop = eval_drop_mask(site, seed)
    r = full_infer_v3(model, site, dev, terrain, coord=coord, drop=drop)
    m = r["drop"]
    if int(m.sum()) == 0:
        return float("nan"), float("nan"), float("nan")
    dem = r["dem_m"]
    mu = r["stats"]["dem_mean"][0]
    model_rmse = float(np.sqrt(((r["dem_pred_m"][m] - dem[m]) ** 2).mean()))
    const_rmse = float(np.sqrt(((mu - dem[m]) ** 2).mean()))
    nn_idx = distance_transform_edt(~r["obs"], return_distances=False, return_indices=True)
    nn_fill = dem[tuple(nn_idx)]
    nn_rmse = float(np.sqrt(((nn_fill[m] - dem[m]) ** 2).mean()))
    return model_rmse, const_rmse, nn_rmse


@torch.no_grad()
def eval_val_iou_v3(model, val_site, val_terrain, dev, coord=True):
    """验证站全图 IoU（排除 BORDER 边界带；早停只比 1/1 档，口径同 v2）。"""
    r = full_infer_v3(model, val_site, dev, val_terrain, coord=coord)
    em = np.zeros(r["valid"].shape, dtype=bool)
    em[BORDER:-BORDER, BORDER:-BORDER] = True
    seg = r["seg_prob_1"] if getattr(model, "multi_scale", False) else r["seg_prob"]
    return V2._iou(seg > 0.5, r["label"], em & r["valid"])


# ------------------------------------------------------------ 训练 ----
def train_direction_v3(train_site, val_site, a, log):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    log(f"\n{'=' * 68}")
    log(f"方向: {train_site} → {val_site}   设备={dev}")

    tr_terrain = load_terrain_norm(train_site)
    va_terrain = load_terrain_norm(val_site)

    tr_ds = CorridorDatasetV3(train_site, tr_terrain, crop=a.crop, seed=a.seed,
                              epoch_len=a.epoch_len, coord=a.coord,
                              drop_frac_lo=a.drop_lo, drop_frac_hi=a.drop_hi,
                              fill_radius=a.fill_radius)
    tr_loader = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, num_workers=0)
    lab = tr_ds.label
    vf = float(tr_ds.obs_nat.mean())
    log(f"  训练站: {train_site} 网格={tr_ds.obs_nat.shape} 有效占比={vf:.3f} "
        f"走廊像元={int(lab.sum())} 走廊/有效={float(lab.mean()) / max(vf, 1e-9):.4f}")
    log(f"  站点统计 mu={tr_ds.mu:.2f} m sd={tr_ds.sd:.2f} m（站点级常数，禁 tile 级）")
    base_ch = 12 if a.coord else 10
    log(f"  输入通道 in_ch={tr_ds.in_ch}（基础 {base_ch} + 地形 {N_TERRAIN}），"
        f"坐标通道={'开' if a.coord else '关'}")

    torch.manual_seed(a.seed)
    model = MultiTaskUNetV2(in_ch=tr_ds.in_ch, base=a.base, down=4, norm=a.norm,
                            alpha=a.alpha, feather=a.feather,
                            multi_scale=a.multi_scale).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    log(f"  模型: MultiTaskUNetV2 base={a.base} 参数量={npar:,} ({npar / 1e6:.2f}M) "
        f"norm={a.norm} in_ch={tr_ds.in_ch} multi_scale={a.multi_scale}"
        + (f" 档权重={V2.W_MS}" if a.multi_scale else ""))
    if a.warm_start:
        warm_start_from_v4(model, train_site, val_site, a.base, log)
    else:
        log("  消融：--no-warm-start，随机初始化（不走 v4 热启动）")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))

    # ---- 阶段 1：空洞预训练 ----
    log(f"\n  --- 阶段 1：空洞预训练（{a.pretrain_epochs} ep，空洞 L1） ---")
    pre_ckpt = os.path.join(CKPT, f"{a.tag}_pre_{train_site}_{val_site}_base{a.base}.pth")
    best_fill, no_imp = float("inf"), 0
    for ep in range(1, a.pretrain_epochs + 1):
        t0 = time.time()
        st = train_epoch(model, tr_loader, opt, dev, "pretrain", a)
        sched.step()
        tf, tc, tn = eval_fill_rmse_v3(model, train_site, tr_terrain, dev, coord=a.coord)
        log(f"  [pre {ep:02d}] loss={st['loss']:.4f} fill={st['fill']:.4f} "
            f"res={st['res']:.4f} hole={st['hole']:.4f} | train_fill_rmse={tf:.4f} m "
            f"(常数 {tc:.4f}, 最近邻 {tn:.4f}) ({time.time() - t0:.0f}s) "
            f"alpha={torch.sigmoid(model.alpha).item():.4f}")
        if tf < best_fill - 1e-6:
            best_fill, no_imp = tf, 0
            torch.save(model.state_dict(), pre_ckpt)
        else:
            no_imp += 1
            if no_imp >= a.patience:
                log(f"  早停（patience={a.patience}）")
                break
    log(f"  预训练最优 train_fill_rmse={best_fill:.4f} m")

    # ---- 阶段 2：微调 ----
    log(f"\n  --- 阶段 2：微调（分割+补洞，{a.epochs} ep；早停=全图 IoU） ---")
    if os.path.exists(pre_ckpt):
        model.load_state_dict(torch.load(pre_ckpt, map_location=dev))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))
    ft_ckpt = os.path.join(CKPT, f"{a.tag}_ft_{train_site}_{val_site}_base{a.base}.pth")
    best_iou, best_ep, no_imp = -1.0, -1, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        st = train_epoch(model, tr_loader, opt, dev, "finetune", a)
        sched.step()
        iou, inter, union = eval_val_iou_v3(model, val_site, va_terrain, dev, coord=a.coord)
        vf_, vc, vn = eval_fill_rmse_v3(model, val_site, va_terrain, dev, coord=a.coord)
        log(f"  [ft {ep:02d}] loss={st['loss']:.4f} seg={st['seg']:.4f} "
            f"grad={st['grad']:.4f} fill={st['fill']:.4f} res={st['res']:.4f} "
            f"hole={st['hole']:.4f} | "
            f"IoU={iou:.4f} (i={inter},u={union}) val_fill={vf_:.4f} m "
            f"(常数 {vc:.4f}, 最近邻 {vn:.4f}) "
            f"alpha={torch.sigmoid(model.alpha).item():.4f} ({time.time() - t0:.0f}s)")
        if iou > best_iou + 1e-6:
            best_iou, best_ep, no_imp = iou, ep, 0
            torch.save(model.state_dict(), ft_ckpt)
        else:
            no_imp += 1
            if no_imp >= a.patience:
                log(f"  早停（patience={a.patience}）")
                break

    log(f"  微调最优全图 IoU={best_iou:.4f} (epoch {best_ep})")
    return dict(direction=f"{train_site}→{val_site}", best_iou=float(best_iou),
                best_ep=int(best_ep), n_params=int(npar),
                pretrain_fill_rmse=float(best_fill),
                stats=dict(mu=tr_ds.mu, sd=tr_ds.sd),
                cfg={k: v for k, v in model.cfg.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=int, default=16, choices=[16, 24])
    p.add_argument("--norm", type=str, default="none", choices=["none", "bn"])
    p.add_argument("--coord", action="store_true", default=False,
                   help="启用绝对坐标通道；跨站消融显示它显著恶化跨站泛化，故默认关闭")
    p.add_argument("--no-coord", dest="coord", action="store_false")
    p.add_argument("--warm_start", action="store_true", default=True,
                   help="从 v4 权重热启动（首层 in_ch 扩展：前 10 复制、后 8 置 0）")
    p.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--pretrain_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--crop", type=int, default=V2.CROP)
    p.add_argument("--epoch_len", type=int, default=200)
    p.add_argument("--w_seg", type=float, default=1.0)
    p.add_argument("--w_grad", type=float, default=0.3)
    p.add_argument("--w_hole", type=float, default=0.5)
    p.add_argument("--w_res", type=float, default=0.2,
                   help="观测区门控残差→0 的权重；这是 alpha 唯一能收到梯度的路径")
    p.add_argument("--w_prior", type=float, default=0.05)
    p.add_argument("--w_holetype", type=float, default=0.1)
    p.add_argument("--fill_radius", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--feather", type=int, default=3)
    p.add_argument("--multi_scale", action="store_true",
                   help="开启 1/1、1/2、1/4 三档深监督")
    p.add_argument("--drop_lo", type=float, default=0.15)
    p.add_argument("--drop_hi", type=float, default=0.40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="v7")
    a = p.parse_args()

    os.makedirs(CKPT, exist_ok=True)
    log_path = os.path.join(RES, f"train_{a.tag}_base{a.base}.log")
    lf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n")
        lf.flush()

    log(f"多任务 U-Net（v2 + {N_TERRAIN} 通道地形因子，v4 热启动） tag={a.tag} base={a.base} "
        f"norm={a.norm} coord={a.coord} warm_start={a.warm_start} "
        f"multi_scale={a.multi_scale} crop={a.crop} bs={a.batch_size} ep={a.epochs}")
    log(f"地形输入：{N_TERRAIN} 通道（slope, sin_aspect, cos_aspect, curvature, roughness, "
        f"profile_curv, tpi15, tpi31），robust 标准化自 data/p2_terrain_{{site}}_norm.npy")
    log(f"补洞 baseline：IDW/最近邻填充面（radius={a.fill_radius}px，米制残差）")
    log(f"损失权重 seg={a.w_seg} grad={a.w_grad} hole_l1={a.w_hole} "
        f"res={a.w_res} prior={a.w_prior} holetype={a.w_holetype}")
    log(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"对照基线 v4：杨→西 IoU 0.0，西→杨 IoU 0.0393（results/train_v4_base16.json）")

    out = []
    for tr, va in [("杨家坝", "西坝村"), ("西坝村", "杨家坝")]:
        out.append(train_direction_v3(tr, va, a, log))

    log(f"\n{'=' * 68}\n汇总（对照 v4：杨→西 0.0，西→杨 0.0393）")
    for r in out:
        log(f"  {r['direction']}: 全图 IoU={r['best_iou']:.4f} (best ep={r['best_ep']}), "
            f"预训练补洞 RMSE={r['pretrain_fill_rmse']:.4f} m, params={r['n_params']:,}")
    jp = os.path.join(RES, f"train_{a.tag}_base{a.base}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"保存: {jp}")
    lf.close()


if __name__ == "__main__":
    main()
