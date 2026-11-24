# -*- coding: utf-8 -*-
"""模型容量对照（GPU）：同一 4 通道输入 / 同一伪标签 / 同一预算，比较四类分割网络。

对照对象
--------
    cnn           现役基线（fusion_train.build_net：3×3 conv → 1×1，32 宽）
    unet          小型 U-Net（16/32/64/128）
    deeplabv3plus DeepLabV3+（torchvision MobileNetV3-Large 骨干，**无预训练权重**）
    lineformer    线状提取网络：条带卷积（1×k / k×1）+ 瓶颈轴向自注意力

公平性
------
* 输入完全一致：直接复用 `cold_start_transfer.build_inputs` 的 4 通道
  （Hillshade、CSSC、路径距离场、openness 走廊响应）；
* 标签完全一致：读 `cold_start_{site}_lc6.npz` 里的 `pseudo`（Dijkstra 路径膨胀 gate）；
* 训练预算完全一致：同 seed / patch / batch / lr / 最大 epoch / 每轮 patch 数，
  早停判据相同（连续 10 轮无改善且 ≥20 轮）；网络结构是唯一变量。

输出
----
    results/segcmp_{site}_{net}.npz     概率图（prob）
    results/segcmp_{site}_{net}.json    参数量 / 训练耗时 / 曲线
随后由 `eval_freeaxis.py` 统一评分（骨架轴线 + D+ 交付），汇总成对比表与图。

用法
----
    python tools/train_seg_compare.py --device cuda
    python tools/train_seg_compare.py --site 西坝村 --nets cnn,deeplabv3plus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import cold_start_transfer as C                                       # noqa: E402
from fusion_train import build_net                                    # noqa: E402

RES = os.path.join(ROOT, "results")
GATES = {"杨家坝": 3.0, "西坝村": 5.0}          # 与 lc6 交付一致
NPZ = {"杨家坝": "cold_start_杨家坝_lc6.npz", "西坝村": "cold_start_西坝村_lc6.npz"}
EVAL_ARGS = {
    "杨家坝": ["--truth", "old", "--dist3", "3.5"],
    "西坝村": ["--truth", "new", "--truth-main", "--dist3", "3.5"],
}


# ================================================================ 数据
def load_case(site, log=print, npz=None):
    """4 通道输入 + 伪标签 + 轴线，全部来自既有 lc6 产物（逐位一致）。"""
    dem, valid = C.load_dem(site, "mean", log)
    hs = C.hillshade_map(dem, valid)
    z = np.load(os.path.join(RES, npz or NPZ[site]))
    cssc = z["cssc"].astype(np.float64)
    op = z["openness"].astype(np.float64)
    path = z["path"].astype(bool)
    pseudo = z["pseudo"].astype(np.float32)
    gate = GATES[site]
    dist = distance_transform_edt(~path)
    lo, hi = np.percentile(op[valid], [5.0, 95.0])
    cost = np.clip((op - lo) / max(hi - lo, 1e-9), 0.0, 1.0) + 0.05
    X, _ = C.build_inputs(hs, cssc, valid, dist, cost, 4, gate, log)
    # 校验：重算伪标签必须与存档一致（保证"同一标签"）
    rec = (dist <= gate) & valid
    same = float((rec == (pseudo >= 0.5)).mean())
    log(f"  伪标签校验：与存档一致率 {same * 100:.3f}%（{int(rec.sum()):,}px）")
    if same < 0.999:
        raise RuntimeError("重算伪标签与存档不一致，输入口径已漂移，停止")
    return dict(X=X, Y=pseudo, valid=valid, path=path, dem=dem)


# ================================================================ 网络
class ConvBNAct(nn.Module):
    def __init__(self, ci, co, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(ci, co), ConvBNAct(co, co))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """标准 U-Net（3 次下采样），只改输入通道数。"""

    def __init__(self, in_ch=4, w=(16, 32, 64, 128)):
        super().__init__()
        self.inc = DoubleConv(in_ch, w[0])
        self.d1 = DoubleConv(w[0], w[1])
        self.d2 = DoubleConv(w[1], w[2])
        self.d3 = DoubleConv(w[2], w[3])
        self.u3 = DoubleConv(w[3] + w[2], w[2])
        self.u2 = DoubleConv(w[2] + w[1], w[1])
        self.u1 = DoubleConv(w[1] + w[0], w[0])
        self.head = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        f0 = self.inc(x)
        f1 = self.d1(F.max_pool2d(f0, 2))
        f2 = self.d2(F.max_pool2d(f1, 2))
        f3 = self.d3(F.max_pool2d(f2, 2))
        y = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.u3(torch.cat([y, f2], 1))
        y = F.interpolate(y, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.u2(torch.cat([y, f1], 1))
        y = F.interpolate(y, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.u1(torch.cat([y, f0], 1))
        return self.head(y)


class StripBlock(nn.Module):
    """条带卷积：沿行/列各一条长核，专门吃线状连续性，再接残差。"""

    def __init__(self, c):
        super().__init__()
        self.conv = ConvBNAct(c, c)
        self.sh = nn.Conv2d(c, c, (1, 9), padding=(0, 4), bias=False)
        self.sv = nn.Conv2d(c, c, (9, 1), padding=(4, 0), bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        y = self.conv(x)
        return x + F.relu(self.bn(self.sh(y) + self.sv(y)))


class AxialAttention(nn.Module):
    """轴向自注意力：先在每行内、再在每列内做全图长度的 self-attention。"""

    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.mw = nn.MultiheadAttention(c, heads, batch_first=True)
        self.mh = nn.MultiheadAttention(c, heads, batch_first=True)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, Cc, H, W = x.shape
        y = self.norm(x)
        yw = y.permute(0, 2, 1, 3).reshape(B * H, W, Cc)
        aw, _ = self.mw(yw, yw, yw, need_weights=False)
        aw = aw.reshape(B, H, W, Cc).permute(0, 3, 1, 2)
        yh = y.permute(0, 3, 2, 1).reshape(B * W, H, Cc)
        ah, _ = self.mh(yh, yh, yh, need_weights=False)
        ah = ah.reshape(B, W, H, Cc).permute(0, 3, 2, 1)
        return x + self.proj(aw + ah)


class LineNet(nn.Module):
    """轻量线状提取网络：U 形骨架 + 条带卷积 + 瓶颈轴向注意力。"""

    def __init__(self, in_ch=4, w=24):
        super().__init__()
        self.e0 = nn.Sequential(ConvBNAct(in_ch, w), StripBlock(w))
        self.e1 = nn.Sequential(ConvBNAct(w, w * 2, s=2), StripBlock(w * 2))
        self.e2 = nn.Sequential(ConvBNAct(w * 2, w * 4, s=2), StripBlock(w * 4),
                                AxialAttention(w * 4))
        self.u1 = ConvBNAct(w * 4 + w * 2, w * 2)
        self.b1 = StripBlock(w * 2)
        self.u2 = ConvBNAct(w * 2 + w, w)
        self.b2 = StripBlock(w)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        f0 = self.e0(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        y = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.b1(self.u1(torch.cat([y, f1], 1)))
        y = F.interpolate(y, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.b2(self.u2(torch.cat([y, f0], 1)))
        return self.head(y)


def build_model(name, in_ch):
    if name == "cnn":
        return build_net(in_ch)
    if name == "unet":
        return UNet(in_ch)
    if name == "lineformer":
        return LineNet(in_ch)
    if name == "deeplabv3plus":
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        m = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None,
                                        num_classes=1, aux_loss=False)
        m.backbone["0"][0] = nn.Conv2d(in_ch, 16, 3, 2, 1, bias=False)
        return m
    raise ValueError(name)


def logits_of(net, x):
    y = net(x)
    return y["out"] if isinstance(y, dict) else y


# ================================================================ 训练
def train_one(net, X, Y, valid, a, log):
    """在 GPU 上按统一预算训练（同 seed / patch / batch / lr / epoch）。"""
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True

    Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(dev)
    Vt = torch.from_numpy(valid.astype(bool)).to(dev)

    npos = int((Y[valid] >= 0.5).sum())
    nneg = int((Y[valid] < 0.5).sum())
    pw = torch.tensor([nneg / max(npos, 1)], dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    H, W = Y.shape
    m = a.patch // 2
    cand = np.argwhere(valid[m:H - m, m:W - m]) + m
    rng = np.random.default_rng(a.seed)
    n_patch = min(a.patch_per_epoch, len(cand))

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    hist, best, bad, t0 = [], float("inf"), 0, time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        pick = rng.choice(len(cand), size=n_patch, replace=False)
        tot = nb = 0
        for i in range(0, n_patch, a.batch):
            rows = cand[pick[i:i + a.batch]]
            xb = torch.stack([Xt[:, r - m:r + m, c - m:c + m] for r, c in rows])
            yb = torch.stack([Yt[r - m:r + m, c - m:c + m] for r, c in rows])
            vb = torch.stack([Vt[r - m:r + m, c - m:c + m] for r, c in rows])
            lg = logits_of(net, xb)[:, 0]
            loss = lossf(lg[vb], yb[vb])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        hist.append(tot / max(nb, 1))
        if ep == 1 or ep % 10 == 0 or ep == a.epochs:
            log(f"    epoch {ep:3d}/{a.epochs}  BCE {hist[-1]:.5f}  "
                f"({time.time() - t0:.0f}s, {torch.cuda.max_memory_allocated() / 2**20:.0f}MB)"
                if dev.type == "cuda" else
                f"    epoch {ep:3d}/{a.epochs}  BCE {hist[-1]:.5f}  ({time.time() - t0:.0f}s)")
        if hist[-1] < best - 1e-4:
            best, bad = hist[-1], 0
        else:
            bad += 1
            if bad >= a.patience and ep >= a.min_epochs:
                log(f"    早停：连续 {a.patience} 轮无改善（best {best:.5f}），停在 {ep}")
                break
    sec = time.time() - t0
    return hist, sec


@torch.no_grad()
def recalibrate_bn(net, X, valid, a, log=print, n_batch=40):
    """BN 重标定：用累计平均重建 running stats（各网络统一执行）。

    深层网络（尤其 MobileNetV3 骨干，BN momentum=0.01）在小样本短训练下，
    running stats 会严重滞后于训练期 batch 统计，导致 eval 模式输出塌缩成常数。
    这里只重算统计量，不改任何权重。
    """
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    mom = [b.momentum for b in bns]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None                     # 累计平均
    dev = torch.device(a.device)
    Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
    H, W = valid.shape
    m = a.patch // 2
    cand = np.argwhere(valid[m:H - m, m:W - m]) + m
    rng = np.random.default_rng(a.seed + 1)
    net.train()
    seen = 0
    for _ in range(n_batch):
        rows = cand[rng.choice(len(cand), size=min(a.batch, len(cand)), replace=False)]
        xb = torch.stack([Xt[:, r - m:r + m, c - m:c + m] for r, c in rows])
        logits_of(net, xb)
        seen += len(rows)
    net.eval()
    for b, mm in zip(bns, mom):
        b.momentum = mm
    log(f"    BN 重标定：{len(bns)} 层 × {n_batch} 批（{seen:,} patch，仅统计量）")


@torch.no_grad()
def infer(net, X, valid, device, tile=384, ov=32, log=print):
    """分块推理（与 fusion_train.infer 同口径），兼容 dict 输出。"""
    H, W = valid.shape
    net.eval()
    prob = np.zeros((H, W), dtype=np.float64)
    Xt = torch.from_numpy(X.astype(np.float32))[None]
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            r1, c1 = min(r0 + tile, H), min(c0 + tile, W)
            a0, b0 = max(0, r0 - ov), max(0, c0 - ov)
            a1, b1 = min(H, r1 + ov), min(W, c1 + ov)
            p = torch.sigmoid(logits_of(net, Xt[:, :, a0:a1, b0:b1].to(device)))[0, 0]
            prob[r0:r1, c0:c1] = p[r0 - a0:r1 - a0, c0 - b0:c1 - b0].cpu().numpy()
    prob = np.where(valid, prob, 0.0)
    log(f"  推理完成：prob∈[{prob[valid].min():.4f},{prob[valid].max():.4f}]")
    return prob


# ================================================================ 主
def run_site(site, a, log):
    log(f"\n[{site}] 载入同一口径的 4ch 输入与伪标签")
    d = load_case(site, log, a.npz)
    X, Y, valid = d["X"], d["Y"], d["valid"]
    log(f"  网格 {X.shape[1:]}，X {tuple(X.shape)}，伪标签 "
        f"{int((Y >= 0.5).sum()):,}px")

    rows = []
    for name in [s.strip() for s in a.nets.split(",") if s.strip()]:
        log(f"\n  ── {name} ──")
        torch.manual_seed(a.seed)
        net = build_model(name, X.shape[0])
        npar = sum(p.numel() for p in net.parameters())
        log(f"  参数量 {npar / 1e3:.1f}k")
        net = net.to(a.device)
        hist, sec = train_one(net, X, Y, valid, a, log)
        recalibrate_bn(net, X, valid, a, log)
        prob = infer(net, X, valid, a.device, log=log)
        prefix = f"segcmp_{site}_{name}{a.tag}"
        np.savez_compressed(os.path.join(RES, f"{prefix}.npz"),
                            prob=prob.astype(np.float32))
        meta = dict(site=site, net=name, params=int(npar), epochs=len(hist),
                    seconds=round(sec, 1), hist=[round(h, 6) for h in hist],
                    seed=a.seed, patch=a.patch, batch=a.batch, lr=a.lr,
                    patch_per_epoch=a.patch_per_epoch, in_ch=int(X.shape[0]),
                    pseudo_px=int((Y >= 0.5).sum()), device=str(a.device),
                    gpu=torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu")
        json.dump(meta, open(os.path.join(RES, f"{prefix}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        rows.append(meta)
        log(f"  写出 results/{prefix}.npz（{sec:.0f}s，{len(hist)} epoch）")
    return rows


def main():
    p = argparse.ArgumentParser(description="四类分割网络同口径容量对照（GPU）")
    p.add_argument("--site", default="杨家坝")
    p.add_argument("--nets", default="cnn,unet,deeplabv3plus,lineformer")
    p.add_argument("--tag", default="", help="输出文件名后缀（如 _p256），不影响训练口径")
    p.add_argument("--npz", default="", help="伪标签/输入来源 npz（默认 lc6）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patch-per-epoch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "train_seg_compare.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n")
        lf.flush()

    log("\n" + "=" * 78)
    log(f"模型容量对照 · {time.strftime('%Y-%m-%d %H:%M:%S')} · device={a.device}")
    t0 = time.time()
    rows = run_site(a.site, a, log)
    log(f"\n[{a.site}] 完成，用时 {time.time() - t0:.0f}s")
    for r in rows:
        log(f"  {r['net']:<14s} {r['params'] / 1e3:8.1f}k 参数  "
            f"{r['epochs']:3d} epoch  {r['seconds']:6.0f}s")


if __name__ == "__main__":
    main()
