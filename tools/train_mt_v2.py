# -*- coding: utf-8 -*-
"""v2 多任务 U-Net 训练（契约内合规改造，最小合规子集）。

相对 v1（tools/train_multitask.py）的改动，逐条对应：
  1. 高程按**站点全局** mu/std 标准化为 z 空间（绝不用 tile 级统计，否则破坏滑窗一致性）；
     未观测像元喂 0（=站点均值），不再喂 -9999 → 消除"空洞损失量级压垮分割"的缺陷
  2. 输入 12ch：9 数据通道 + 空洞 mask + 绝对坐标 (xx, yy)
  3. 空洞监督：真值空洞无 GT，故训练时在有效区**随机丢块**制造合成空洞（域泛化），
     空洞损失高权重、有效区低权重（去噪）；自然空洞只做平滑约束
  4. 门控残差：dem_pred = dem_fill + ΔZ·gate（空洞 ≈ 全幅补值）
  5. 地形损失：加权 MSE + Sobel 梯度 + 法向 + 空洞二阶平滑
  6. 分割头加"未观测"加性偏置（hole_type → seg logits，可学，初值 0）
  7. 骨干 ResBlock(3×3,3×3)、解码器 ResBlock(3×3,1×1) → RF 严格保持 170 px
  8. 两阶段：空洞预训练 → 分割+补洞微调；早停用**全图** IoU（v1 用随机 crop，方差大）

协议（results/contract.json）：
  * 归一化 none/bn（BN 推理 eval）；禁止 gn/in
  * 分层边界由训练集统计并冻结；评估排除 85 px 边界带
  * 采样：80% 以走廊像元为中心（sampling_policy 规则 1）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, uniform_filter
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from unet_mt import MultiTaskUNetV2  # noqa: E402

DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")

BORDER = 85          # 评估排除边界带
CROP = 512
DEM_CH = ["dem_mean", "dem_min", "dem_max", "dem_std"]
# DOM 输入模式：'lum' 保持 v4/v6/v7 逐位复现（灰度亮度，走廊 AUC≈随机）；
# 'exg_sat' 用真彩色彩度双通道（超绿 2G−R−B + 饱和度）替换灰度，两站方向一致的强判别因子
DOM_MODES = {"lum": ["dom_lum"], "exg_sat": ["dom_exg", "dom_sat"]}
AUX_CH = ["slope_deg", "curvature", "dom_lum", "dom_exg", "dom_sat"]
N_PRE_DOM = 8                 # DOM 之前的 8 个数据通道：填充面 + min/max/std + slope + sin/cos + 曲率
DATA_IDX = list(range(9))     # 0..8 数据通道（lum 模式，未观测处置 0）
HOLE_IDX = 9                  # 空洞 mask（lum 模式）
COORD_IDX = [10, 11]          # 绝对坐标（lum 模式）
IN_CH = 12                    # lum + coord


def dom_layout(dom: str) -> tuple[int, int]:
    """返回 (n_data, hole_idx)：数据通道数（含 DOM）与空洞 mask 索引。

    lum     → (9, 9)   布局 [0..7 数据, 8 dom_lum, 9 hole]（= v4，coord 关时 in_ch=10）
    exg_sat → (10, 10) 布局 [0..7 数据, 8 exg, 9 sat, 10 hole]（coord 关时 in_ch=11）
    """
    n_data = N_PRE_DOM + len(DOM_MODES[dom])
    return n_data, n_data



# ---------------------------------------------------------------- 数据 ----
def load_site(site: str):
    return np.load(os.path.join(DATA, f"{site}_1m.npz"))


def compute_stats(d) -> dict:
    """站点级统计量（仅观测像元）——必须是站点常数，不能用 tile 统计。"""
    obs = d["valid"] == 1
    st = {}
    for k in DEM_CH + AUX_CH:
        a = d[k].astype(np.float64)[obs]
        st[k] = [float(a.mean()), float(a.std()) or 1.0]
    return st


def idw_fill(dem: np.ndarray, obs: np.ndarray, radius: int = 8,
             iters: int = 2) -> np.ndarray:
    """对 obs==0 的像元做 IDW 式填充（最近邻 + 局部均值平滑），返回**米**。

    这是门控残差的 baseline：填充面越接近真值，模型要学的残差越小、越可跨站泛化。
    站点常数填充必然跨站崩溃（杨 749 m / 西 1046 m），必须换成局部插值面。
    """
    filled = dem.astype(np.float32).copy()
    miss = ~obs
    if not miss.any():
        return filled
    idx = distance_transform_edt(miss, return_distances=False, return_indices=True)
    nearest = dem.astype(np.float32)[idx[0], idx[1]]
    filled[miss] = nearest[miss]
    if iters > 0:
        k = 2 * radius + 1
        for _ in range(iters):
            sm = uniform_filter(filled, size=k, mode="nearest")
            filled[miss] = sm[miss]
    return filled


def build_channels(d, st: dict, coord: bool = True,
                   fill: np.ndarray | None = None, dom: str = "lum") -> np.ndarray:
    """拼 (C,H,W) 输入。

    channel 0 在未观测处不再是 0（=站点均值），而是**填充面** fill 的标准化值：
    这样模型看到的是一个连续地表，空洞处由局部插值给出近似高程，残差目标变小。
    其余数据通道在未观测处仍置 0（无信息），空洞 mask 与坐标通道保留。

    dom: 'lum' → 第 8 通道 = dom_lum（v4/v6/v7 布局，逐位复现）；
         'exg_sat' → 第 8、9 通道 = dom_exg、dom_sat（真彩色彩度替换灰度）。
    """
    obs = d["valid"] == 1

    def norm_obs(k):
        mu, sd = st[k]
        return np.where(obs, (d[k].astype(np.float32) - mu) / sd, 0.0).astype(np.float32)

    mu0, sd0 = st["dem_mean"]
    base0 = d["dem_mean"].astype(np.float32) if fill is None else fill
    ch = [((base0 - mu0) / sd0).astype(np.float32)]          # 0: 填充面
    ch += [norm_obs(k) for k in DEM_CH[1:]]                  # 1..3: min/max/std
    ch.append(norm_obs("slope_deg"))                         # 4
    asp = np.radians(d["aspect_deg"].astype(np.float32))
    ch.append(np.where(obs, np.sin(asp), 0.0).astype(np.float32))   # 5
    ch.append(np.where(obs, np.cos(asp), 0.0).astype(np.float32))   # 6
    ch.append(norm_obs("curvature"))                         # 7
    ch += [norm_obs(k) for k in DOM_MODES[dom]]              # 8 (lum) 或 8,9 (exg,sat)
    n_data, _ = dom_layout(dom)
    ch.append(np.where(obs, 0.0, 1.0).astype(np.float32))    # hole mask（索引 n_data）
    if coord:
        H, W = obs.shape
        yy = np.repeat(np.linspace(-1, 1, H, dtype=np.float32)[:, None], W, axis=1)
        xx = np.repeat(np.linspace(-1, 1, W, dtype=np.float32)[None, :], H, axis=0)
        ch += [xx, yy]                                       # n_data+1, n_data+2
    x = np.stack(ch, 0)
    assert x.shape[0] == n_data + 1 + (2 if coord else 0), (x.shape, n_data)
    return x


def block_drop_mask(obs: np.ndarray, rng: np.random.Generator,
                    lo: float, hi: float, bs_lo: int = 16, bs_hi: int = 64) -> np.ndarray:
    """在观测像元上随机丢块，模拟航带 nodata 空洞。

    真值空洞（nodata）没有 GT，无法直接监督补洞；因此在**观测区**人为挖洞制造 GT。
    训练与验证用同一函数、同一块尺寸区间，保证口径一致。
    块尺寸必须与图像尺寸解耦：若按图像边长取比例，全图评估时会生成远超 RF(170 px)
    的巨块，块心无任何上下文，RMSE 失去意义。
    """
    frac = rng.uniform(lo, hi)
    budget = frac * float(obs.sum())
    hole = np.zeros_like(obs, dtype=bool)
    got = 0.0
    H, W = obs.shape
    for _ in range(256):
        if got >= budget:
            break
        bh = int(rng.integers(bs_lo, bs_hi + 1))
        bw = int(rng.integers(bs_lo, bs_hi + 1))
        y = int(rng.integers(0, max(1, H - bh)))
        x = int(rng.integers(0, max(1, W - bw)))
        new = obs[y:y + bh, x:x + bw] & (~hole[y:y + bh, x:x + bw])
        got += float(new.sum())
        hole[y:y + bh, x:x + bw] |= new
    return hole


def scatter_drop_mask(obs: np.ndarray, rng: np.random.Generator, frac: float) -> np.ndarray:
    return obs & (rng.random(obs.shape) < frac)


class CorridorDatasetV2(Dataset):
    """站点级标准化 + 随机丢块合成空洞 + 正样本优先采样。"""

    def __init__(self, site, crop=CROP, pos_frac=0.8, seed=42, epoch_len=200,
                 coord=True, drop_frac_lo=0.15, drop_frac_hi=0.40, fill_radius=8,
                 dom="lum"):
        d = load_site(site)
        self.site = site
        self.obs_nat = d["valid"] == 1
        self.label = d["label"].astype(np.float32)
        self.st = compute_stats(d)
        self.mu, self.sd = self.st["dem_mean"]
        self.dem_m = d["dem_mean"].astype(np.float32)          # 米（处理量）
        self.fill_radius = fill_radius
        # 自然空洞的站点级填充面（推理时用；训练时按 crop 重算以消除合成丢块泄漏）
        self.fill_global = idw_fill(self.dem_m, self.obs_nat, radius=fill_radius)
        self.x = build_channels(d, self.st, coord=coord, fill=self.fill_global, dom=dom)
        self.in_ch = self.x.shape[0]
        self.dom = dom
        self.n_data, self.hole = dom_layout(dom)               # 需置 0 / 置 1 的动态索引
        self.crop, self.pos_frac, self.epoch_len = crop, pos_frac, epoch_len
        self.drop_lo, self.drop_hi = drop_frac_lo, drop_frac_hi
        self.rng = np.random.default_rng(seed)
        self.pos_coords = np.argwhere(self.label == 1)
        self.valid_coords = np.argwhere(self.obs_nat)

    def __len__(self):
        return self.epoch_len

    def _sample_center(self):
        if self.rng.random() < self.pos_frac and len(self.pos_coords) > 0:
            return self.pos_coords[self.rng.integers(len(self.pos_coords))]
        return self.valid_coords[self.rng.integers(len(self.valid_coords))]

    def _drop_blocks(self, obs_c: np.ndarray) -> np.ndarray:
        return block_drop_mask(obs_c, self.rng, self.drop_lo, self.drop_hi)

    def __getitem__(self, idx):
        H, W = self.obs_nat.shape
        c = self.crop
        cy, cx = self._sample_center()
        y0 = int(np.clip(cy - c // 2, 0, H - c))
        x0 = int(np.clip(cx - c // 2, 0, W - c))
        sl = (slice(y0, y0 + c), slice(x0, x0 + c))

        valid = self.obs_nat[sl].copy()      # 真值可用的像元（自然有效区）
        obs = valid.copy()                   # 观测掩码（可再叠加合成空洞）
        drop = np.zeros_like(valid)
        if self.drop_lo > 0:
            drop = self._drop_blocks(obs)
            obs = obs & (~drop)

        # 填充面必须用**挖洞后**的观测掩码重算：否则被挖掉的像元会把自己的真实值
        # 泄漏进 baseline，残差恒为 0，模型学不到任何东西。
        dem_c = self.dem_m[sl]
        fill_m = idw_fill(dem_c, obs, radius=self.fill_radius)

        x = self.x[:, sl[0], sl[1]].copy()
        if drop.any():
            x[:self.n_data][:, drop] = 0.0       # 数据通道视图，赋值生效
            x[self.hole][drop] = 1.0
        x[0] = (fill_m - self.mu) / self.sd      # 0 通道 = 填充面（标准化）

        # 真值残差（米）：只在自然有效像元上有定义
        residual = np.where(valid, dem_c - fill_m, 0.0).astype(np.float32)
        return (torch.from_numpy(x),
                torch.from_numpy(self.label[sl]),
                torch.from_numpy(obs.astype(np.float32)),
                torch.from_numpy(valid.astype(np.float32)),
                torch.from_numpy(fill_m),
                torch.from_numpy(residual))


# ---------------------------------------------------------------- 损失 ----
def hole_l1_loss(dz, res_gt, drop_known):
    """空洞区 L1（米）：预测残差 vs 真值残差，只在"挖掉且真值已知"的像元上。

    原生空洞没有 GT，补洞监督只能来自在观测区人为挖块。res_gt = dem_raw − 填充面，
    单位米，跨站可直接比较（这正是把 baseline 从站点常数换成填充面后损失要改成米制的原因）。
    """
    m = drop_known
    if float(m.sum()) == 0:
        return torch.zeros((), device=dz.device, dtype=dz.dtype)
    return ((dz - res_gt).abs() * m).sum() / m.sum().clamp(min=1.0)


def trust_fill_loss(dz, unk):
    """无真值区（原生空洞）：轻推残差→0，即信任填充面，避免输出乱飞。"""
    if float(unk.sum()) == 0:
        return torch.zeros((), device=dz.device, dtype=dz.dtype)
    return (dz.abs() * unk).sum() / unk.sum().clamp(min=1.0)


def edge_grad_loss(logit, label, valid, dilate_r=2):
    """走廊边界带的梯度损失：比较预测概率的梯度与真值标签的梯度。

    Dice 对边界不敏感——预测一个模糊大区域就能让 loss 停在低值；走廊是长条形、
    边界梯度大，这一项强迫模型把边界走向学出来，正是坡度阈值几何基线做不到的。
    """
    pred = torch.sigmoid(logit.squeeze(1))            # [B,H,W]
    lab = label
    k = 2 * dilate_r + 1
    dil = F.max_pool2d(lab.unsqueeze(1), k, 1, dilate_r).squeeze(1)
    ero = -F.max_pool2d(-lab.unsqueeze(1), k, 1, dilate_r).squeeze(1)
    edge = (dil - ero) * valid
    gx_p = pred[:, 1:, :] - pred[:, :-1, :]
    gy_p = pred[:, :, 1:] - pred[:, :, :-1]
    gx_t = lab[:, 1:, :] - lab[:, :-1, :]
    gy_t = lab[:, :, 1:] - lab[:, :, :-1]
    ex = edge[:, 1:, :]
    ey = edge[:, :, 1:]
    lx = ((gx_p - gx_t).abs() * ex).sum() / ex.sum().clamp(min=1.0)
    ly = ((gy_p - gy_t).abs() * ey).sum() / ey.sum().clamp(min=1.0)
    return lx + ly


def seg_loss(logit, target, mask):
    """BCE + Dice，仅在真值可用像元上（契约：禁止纯 CE）。"""
    bce = F.binary_cross_entropy_with_logits(logit.squeeze(1), target, reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp(min=1.0)
    p = torch.sigmoid(logit.squeeze(1)) * mask
    t = target * mask
    dice = 1.0 - (2.0 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
    return bce + dice


def hole_type_loss(logit, hole, pos_weight=2.0):
    pw = torch.as_tensor(pos_weight, dtype=logit.dtype, device=logit.device)
    return F.binary_cross_entropy_with_logits(logit.squeeze(1), hole, pos_weight=pw)


# ---------------------------------------------------------------- 全图推理 ----
@torch.no_grad()
def full_infer(model, site, dev, coord=True, drop=None, dom="lum"):
    """全图推理。返回 seg_prob / dem_pred(m) / 各 mask（H,W）。

    drop: 预先算好的合成空洞 mask（bool，只在观测区内）；None 表示不挖洞。
    dom:  DOM 输入模式，'lum'（v4 布局）/ 'exg_sat'（真彩色彩度双通道）。
    """
    d = load_site(site)
    st = compute_stats(d)
    mu, sd = st["dem_mean"]
    obs_nat = d["valid"] == 1
    if drop is None:
        drop = np.zeros_like(obs_nat)
    obs = obs_nat & (~drop)
    dem_m = d["dem_mean"].astype(np.float32)
    # 填充面（米）：用挖洞后的观测掩码重算，与训练口径一致 → 门控残差只需学小残差
    fill_m = idw_fill(dem_m, obs, radius=FILL_RADIUS)
    x = build_channels(d, st, coord=coord, fill=fill_m, dom=dom)
    n_data, hole = dom_layout(dom)
    if drop.any():
        x[:n_data][:, drop] = 0.0
        x[hole][drop] = 1.0
    x[0] = (fill_m - mu) / sd
    model.eval()
    xt = torch.from_numpy(x[None]).to(dev)
    hm = torch.from_numpy((~obs)[None, None].astype(np.float32)).to(dev)
    dm = torch.from_numpy(fill_m[None, None]).to(dev)     # 米
    out = model(xt, dem_fill=dm, hole_mask=hm)
    seg = torch.sigmoid(out["seg"])[0, 0].cpu().numpy()
    dem_pred_m = out["dem_pred"][0, 0].cpu().numpy()
    hole_prob = torch.sigmoid(out["hole_logit"])[0, 0].cpu().numpy()
    r = dict(seg_prob=seg, dem_pred_m=dem_pred_m, hole_prob=hole_prob,
             obs=obs, drop=drop, valid=obs_nat, dem_m=dem_m, fill_m=fill_m,
             label=d["label"].astype(bool), slope=d["slope_deg"], stats=st)
    if getattr(model, "multi_scale", False):
        # seg_1 与 seg 同值；另给 1/2、1/4 档概率，供分档评估
        for t in ("1", "2", "4"):
            r[f"seg_prob_{t}"] = torch.sigmoid(out[f"seg_{t}"])[0, 0].cpu().numpy()
    return r


def _iou(pred_bin, label, mask):
    inter = int((pred_bin & label & mask).sum())
    union = int(((pred_bin | label) & mask).sum())
    return (0.0 if union == 0 else inter / union), inter, union


# 合成丢块评估协议（固定，训练/评估/报告三处共用）
EVAL_BS = (16, 48)      # 块边长区间（px），远小于 RF=170 → 块心仍有上下文
EVAL_FRAC = (0.02, 0.04)  # 占观测像元比例
FILL_RADIUS = 8         # 填充面搜索半径（px），训练/推理/评估三处必须一致


def eval_drop_mask(site, seed=777):
    d = load_site(site)
    obs = d["valid"] == 1
    return block_drop_mask(obs, np.random.default_rng(seed), *EVAL_FRAC, *EVAL_BS)


@torch.no_grad()
def eval_fill_rmse(model, site, dev, coord=True, seed=777, dom="lum"):
    """合成丢块补洞 RMSE（m）+ 同 mask 的常数 / 最近邻基线。"""
    drop = eval_drop_mask(site, seed)
    r = full_infer(model, site, dev, coord=coord, drop=drop, dom=dom)
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
def eval_val_iou(model, val_site, dev, coord=True, dom="lum"):
    r = full_infer(model, val_site, dev, coord=coord, dom=dom)
    em = np.zeros(r["valid"].shape, dtype=bool)
    em[BORDER:-BORDER, BORDER:-BORDER] = True
    # 早停只比 1/1 档（= seg_prob），与多档解耦，保证与单档基线同口径
    seg = r["seg_prob_1"] if getattr(model, "multi_scale", False) else r["seg_prob"]
    iou, inter, union = _iou(seg > 0.5, r["label"], em & r["valid"])
    return iou, inter, union


# ---------------------------------------------------------------- 多档监督目标 ----
def downsample_trio(label, valid, hole, res):
    """把 label/valid/hole/res 降到 1/2、1/4 档，供多档（深监督）用。

    下采样方式与「朴素 avg>0.5」不同，理由：
      * label / valid 用 **max_pool**（存在性）。走廊仅占 ~0.6% 像元，2×2 平均后
        单像元命中率 0.25、4×4 仅 0.0625，若按 avg>0.5 取阈值，**所有**粗尺度标签
        都会被抹成 0 → 粗档退化成全负监督（正是历史 S2 崩溃的形态）。
      * hole 也用 max_pool，与门控/监督语义（该处是否被挖掉）一致。
      * res 用「被挖像元上的均值」（分子分母同池化），避免未挖像元的 0 稀释目标。
    """
    out = {"1": dict(label=label, valid=valid, hole=hole, res=res)}
    for tag, k in (("2", 2), ("4", 4)):
        lp = F.max_pool2d(label.unsqueeze(1), k).squeeze(1)
        vp = F.max_pool2d(valid.unsqueeze(1), k).squeeze(1)
        hp = F.max_pool2d(hole.unsqueeze(1), k).squeeze(1)
        h1 = hole.unsqueeze(1)
        num = F.avg_pool2d((res.unsqueeze(1) * h1), k)
        den = F.avg_pool2d(h1, k).clamp(min=1e-6)
        out[tag] = dict(label=lp, valid=vp, hole=hp, res=(num / den).squeeze(1))
    return out


# 三档损失权重（1/1、1/2、1/4）
W_MS = {"1": 0.5, "2": 0.3, "4": 0.2}


# ---------------------------------------------------------------- 训练 ----
def train_epoch(model, loader, opt, dev, phase, a, log_sampling=False):
    model.train()
    agg = dict(loss=0.0, seg=0.0, grad=0.0, fill=0.0, res=0.0, hole=0.0)
    nb = 0
    for x, label, obs, valid, fill_m, res_gt in loader:
        x, label, obs, valid, fill_m, res_gt = (
            t.to(dev) for t in (x, label, obs, valid, fill_m, res_gt))
        # 补齐到 4D，避免 [B,C,H,W] 与 [B,H,W] 广播成 [B,B,H,W]
        hm4 = (1.0 - obs).unsqueeze(1)
        fill4 = fill_m.unsqueeze(1)
        valid4 = valid.unsqueeze(1)
        res4 = res_gt.unsqueeze(1)
        out = model(x, dem_fill=fill4, hole_mask=hm4)
        # 补洞监督只在"被挖掉且真值已知"的像元上（原生空洞无 GT）
        drop_known_2d = (1.0 - obs) * valid
        l_hole = hole_type_loss(out["hole_logit"], hm4.squeeze(1))
        if not getattr(model, "multi_scale", False):
            # 门控后的残差（米）= dem_pred − 填充面 = dz·gate。必须用它算损失：
            # 直接用 dz 会把 gate（含可学习 alpha）旁路掉，alpha 永远收不到梯度。
            res_pred = (out["dem_pred"] - fill4) if out["dem_pred"] is not None else out["dz"]
            l_fill = hole_l1_loss(res_pred, res4, drop_known_2d.unsqueeze(1))
            # 观测区去噪：门控在该处 ≈ alpha，真值残差为 0 → 唯一能让 alpha 学到东西的项
            # （空洞内部 feather(mask)=1 使 gate≡1，与 alpha 无关）
            l_res = trust_fill_loss(res_pred, obs.unsqueeze(1))
            # 无真值区（原生空洞）：轻推残差→0，即信任填充面，避免输出乱飞
            l_prior = trust_fill_loss(res_pred, ((1.0 - obs) * (1.0 - valid)).unsqueeze(1))
            if phase == "pretrain":
                l_seg = torch.zeros((), device=dev)
                l_grad = torch.zeros((), device=dev)
                loss = (l_fill + a.w_res * l_res
                        + a.w_prior * l_prior + a.w_holetype * l_hole)
            else:
                l_seg = seg_loss(out["seg"], label, valid)
                l_grad = edge_grad_loss(out["seg"], label, valid)
                loss = (a.w_seg * l_seg + a.w_grad * l_grad + a.w_hole * l_fill
                        + a.w_res * l_res + a.w_prior * l_prior + a.w_holetype * l_hole)
        else:
            # ---- 多档深监督：补洞 + 分割在 1/1、1/2、1/4 三档各算一遍 ----
            tgt = downsample_trio(label, valid, drop_known_2d, res_gt)
            l_fill = l_res = l_seg = l_grad = torch.zeros((), device=dev)
            for tag in ("1", "2", "4"):
                t = tgt[tag]
                lbl_i, val_i, hole_i, res_t = t["label"], t["valid"], t["hole"], t["res"]
                size = lbl_i.shape[-2:]
                if tag == "1":
                    fill_i = fill4                              # 1/1 档不插值，与单档逐位一致
                else:
                    fill_i = F.interpolate(fill4, size=size, mode="bilinear",
                                           align_corners=False)
                obs_i = F.interpolate(obs.unsqueeze(1), size=size, mode="nearest")
                res_i = out[f"dem_{tag}"] - fill_i
                l_fill = l_fill + W_MS[tag] * hole_l1_loss(res_i, res_t.unsqueeze(1),
                                                           hole_i.unsqueeze(1))
                l_res = l_res + W_MS[tag] * trust_fill_loss(res_i, obs_i)
                if phase != "pretrain":
                    l_seg = l_seg + W_MS[tag] * seg_loss(out[f"seg_{tag}"], lbl_i, val_i)
                    l_grad = l_grad + W_MS[tag] * edge_grad_loss(out[f"seg_{tag}"],
                                                                 lbl_i, val_i)
            # 原生空洞先验 + 空洞类型只在 1/1 档算（与单档口径一致）
            res_1 = out["dem_1"] - fill4
            l_prior = trust_fill_loss(res_1, ((1.0 - obs) * (1.0 - valid)).unsqueeze(1))
            if phase == "pretrain":
                loss = (l_fill + a.w_res * l_res
                        + a.w_prior * l_prior + a.w_holetype * l_hole)
            else:
                loss = (a.w_seg * l_seg + a.w_grad * l_grad + a.w_hole * l_fill
                        + a.w_res * l_res + a.w_prior * l_prior + a.w_holetype * l_hole)
        opt.zero_grad()
        loss.backward()
        opt.step()
        agg["loss"] += float(loss.detach())
        agg["seg"] += float(l_seg.detach())
        agg["grad"] += float(l_grad.detach())
        agg["fill"] += float(l_fill.detach())
        agg["res"] += float(l_res.detach())
        agg["hole"] += float(l_hole.detach())
        nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


def warm_start_v4_dom(model, train_site, val_site, base, log) -> bool:
    """v4(in_ch=10, dom_lum) → v8(in_ch=11, exg+sat) 热启动（扩展 1 通道）。

    布局：
      v4  [0..7 数据, 8 dom_lum, 9 hole]
      v8  [0..7 数据, 8 exg,     9 sat, 10 hole]
    重映射：新[0:8]←旧[0:8]（8 个前置数据通道原样复制）；新[10]←旧[9]（hole 后移一格）；
    新[8:10]（exg/sat）置 0。零初值使初始前向输出与 v4 逐位一致，exg/sat 从"零贡献"开始学习
    （与 v3 地形通道热启动同理）。
    """
    ws = os.path.join(CKPT, f"v4_pre_{train_site}_{val_site}_base{base}.pth")
    if not os.path.exists(ws):
        log(f"  [警告] 未找到 {ws}，改为随机初始化")
        return False
    sd_old = torch.load(ws, map_location="cpu", weights_only=False)
    if hasattr(sd_old, "state_dict"):
        sd_old = sd_old["state_dict"]
    sd_new = model.state_dict()
    merged, n_map = {}, 0
    for k, v_new in sd_new.items():
        if k not in sd_old:
            merged[k] = v_new
            continue
        v_old = sd_old[k]
        if tuple(v_old.shape) == tuple(v_new.shape):
            merged[k] = v_old
            continue
        if not (v_old.dim() == v_new.dim() == 4
                and v_old.shape[0] == v_new.shape[0]
                and v_old.shape[2:] == v_new.shape[2:]
                and v_old.shape[1] == 10 and v_new.shape[1] == 11):
            raise ValueError(f"权重 {k} 形状无法重映射：{tuple(v_old.shape)} → {tuple(v_new.shape)}")
        m = torch.zeros_like(v_new)
        m[:, :N_PRE_DOM] = v_old[:, :N_PRE_DOM]     # 8 个前置数据通道
        m[:, 10] = v_old[:, 9]                      # hole mask 后移一格
        merged[k] = m
        n_map += 1
        log(f"    重映射 {k}: {tuple(v_old.shape)} → {tuple(v_new.shape)}"
            f"（[0:8]复制, [10]←旧[9]hole, [8:10]置 0 供 exg/sat）")
    log(f"  共重映射 {n_map} 处 in_ch 相关权重")
    model.load_state_dict(merged, strict=True)
    return True


def train_direction(train_site, val_site, a, log):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    log(f"\n{'=' * 68}")
    log(f"方向: {train_site} → {val_site}   设备={dev}")

    tr_ds = CorridorDatasetV2(train_site, crop=a.crop, seed=a.seed,
                              epoch_len=a.epoch_len, coord=a.coord,
                              drop_frac_lo=a.drop_lo, drop_frac_hi=a.drop_hi,
                              fill_radius=a.fill_radius, dom=a.dom)
    tr_loader = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, num_workers=0)
    # 采样协议记录（sampling_policy：必须记录命中率）
    lab = tr_ds.label
    vf = float(tr_ds.obs_nat.mean())
    log(f"  训练站: {train_site} 网格={tr_ds.obs_nat.shape} 有效占比={vf:.3f} "
        f"走廊像元={int(lab.sum())} 走廊/有效={float(lab.mean()) / max(vf, 1e-9):.4f}")
    log(f"  站点统计 mu={tr_ds.mu:.2f} m sd={tr_ds.sd:.2f} m（站点级常数，禁 tile 级）")
    log(f"  DOM 输入: {a.dom}（通道 {DOM_MODES[a.dom]}）")
    log(f"  坐标通道: {'绝对坐标 xx,yy∈[-1,1]（站点网格归一化，滑窗不变）' if a.coord else '关闭'}")

    torch.manual_seed(a.seed)
    model = MultiTaskUNetV2(in_ch=tr_ds.in_ch, base=a.base, down=4, norm=a.norm,
                            alpha=a.alpha, feather=a.feather,
                            multi_scale=a.multi_scale).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    log(f"  模型: MultiTaskUNetV2 base={a.base} 参数量={npar:,} ({npar / 1e6:.2f}M) "
        f"norm={a.norm} in_ch={tr_ds.in_ch} multi_scale={a.multi_scale}"
        + (f" 档权重={W_MS}" if a.multi_scale else ""))
    if a.warm_start:
        warm_start_v4_dom(model, train_site, val_site, a.base, log)
    else:
        log("  消融：--no-warm-start，随机初始化（不走 v4 热启动）")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))

    # ---- 阶段 1：空洞预训练 ----
    log(f"\n  --- 阶段 1：空洞预训练（{a.pretrain_epochs} ep，空洞 L1） ---")
    pre_ckpt = os.path.join(CKPT, f"{a.tag}_pre_{train_site}_{val_site}_base{a.base}.pth")
    best_fill = float("inf")
    no_imp = 0
    for ep in range(1, a.pretrain_epochs + 1):
        t0 = time.time()
        st = train_epoch(model, tr_loader, opt, dev, "pretrain", a)
        sched.step()
        # 预训练选中在**训练站**的合成丢块 RMSE：跨站指标在未收敛时方差过大，
        # 会误选 epoch 1（见 v2 首轮日志）。跨站指标只在阶段 2 报告。
        tf, tc, tn = eval_fill_rmse(model, train_site, dev, coord=a.coord, dom=a.dom)
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
        iou, inter, union = eval_val_iou(model, val_site, dev, coord=a.coord, dom=a.dom)
        vf, vc, vn = eval_fill_rmse(model, val_site, dev, coord=a.coord, dom=a.dom)
        log(f"  [ft {ep:02d}] loss={st['loss']:.4f} seg={st['seg']:.4f} "
            f"grad={st['grad']:.4f} fill={st['fill']:.4f} res={st['res']:.4f} "
            f"hole={st['hole']:.4f} | "
            f"IoU={iou:.4f} (i={inter},u={union}) val_fill={vf:.4f} m "
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
                   help="启用绝对坐标通道；跨站消融显示它显著恶化跨站泛化"
                        "（S1 IoU 0.0454→0.0067，补洞 RMSE 12→132 m），故默认关闭")
    p.add_argument("--no-coord", dest="coord", action="store_false")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--pretrain_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--crop", type=int, default=CROP)
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
                   help="开启 1/1、1/2、1/4 三档深监督（分割与 ΔZ 各挂一个 1×1 头）")
    p.add_argument("--drop_lo", type=float, default=0.15)
    p.add_argument("--drop_hi", type=float, default=0.40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dom", type=str, default="lum", choices=["lum", "exg_sat"],
                   help="DOM 输入通道：lum（灰度，兼容 v4/v6/v7 逐位复现）"
                        " / exg_sat（真彩色彩度双通道 exg+sat，替换灰度）")
    p.add_argument("--warm_start", action="store_true", default=False,
                   help="从 v4 权重热启动（in_ch 10→11：前 8 复制、hole 后移、exg/sat 置 0）")
    p.add_argument("--tag", type=str, default="v2")
    a = p.parse_args()

    os.makedirs(CKPT, exist_ok=True)
    log_path = os.path.join(RES, f"train_{a.tag}_base{a.base}.log")
    lf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n")
        lf.flush()

    log(f"v2 多任务 U-Net 训练  tag={a.tag} base={a.base} norm={a.norm} "
        f"coord={a.coord} dom={a.dom} warm_start={a.warm_start} "
        f"multi_scale={a.multi_scale} crop={a.crop} bs={a.batch_size} ep={a.epochs}")
    log(f"DOM 输入通道: {a.dom} → {DOM_MODES[a.dom]}（in_ch 基础 = "
        f"{dom_layout(a.dom)[0] + 1 + (2 if a.coord else 0)}）")
    log(f"补洞 baseline：IDW/最近邻填充面（radius={a.fill_radius}px，米制残差）")
    log(f"损失权重 seg={a.w_seg} grad={a.w_grad} hole_l1={a.w_hole} "
        f"res={a.w_res} prior={a.w_prior} holetype={a.w_holetype}")
    log(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    out = []
    for tr, va in [("杨家坝", "西坝村"), ("西坝村", "杨家坝")]:
        out.append(train_direction(tr, va, a, log))

    log(f"\n{'=' * 68}\n汇总")
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
