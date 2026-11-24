# -*- coding: utf-8 -*-
"""在 20 个地形因子上训练自编码器，学 8 维深度学习特征。

动机
----
`tools/sparse_pca.py` 用**线性**稀疏 PCA 把 20 因子压成 5 个可命名分量
（重建 R² ≈ 0.815）。本脚本换**非线性**自编码器（MLP），问：
    同样的 20 因子，用 AE 能否学到**重建更好**、且**跨站稳定**的低维特征？

输入 / 输出
----
    data/feature_matrix_{site}.npy   [N_valid, 20] float32（原始量纲）
        ↓ P1/P99 截断 + 单位方差（= winsor_z，与 sparse_pca 默认口径一致）
        ↓ MLP AE（20→64→32→latent→32→64→20），MSE + 可选 L1(z) 稀疏
    data/ae_latent_{site}_L{latent}.npy   [N_valid, latent] float32（瓶颈 Z）
    checkpoints/ae_{site}_L{latent}.pth   state_dict + 标准化统计量 + cfg
    results/ae_meta_{site}_L{latent}.json 重建 R² / 参数量 / 配置

用法
----
    python tools/train_ae.py --site 西坝村 --latent 8
    python tools/train_ae.py --site 杨家坝 --latent 8

下游：`tools/ae_interpret.py` 给每个瓶颈维命名（与 20 因子的相关度 top-3）、
算走廊 AUC、做跨站载荷余弦匹配。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class TerrainAE(nn.Module):
    def __init__(self, in_dim: int = 20, latent: int = 8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, in_dim))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def main():
    ap = argparse.ArgumentParser(description="20 地形因子上的自编码器")
    ap.add_argument("--site", required=True, choices=["杨家坝", "西坝村"])
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sparse", type=float, default=0.01,
                    help="L1 稀疏权重，0 则关闭")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # 1) 载入因子矩阵
    X = np.load(os.path.join(DATA, f"feature_matrix_{a.site}.npy"))
    print(f"[{a.site}] X.shape = {X.shape}  device = {DEV}")

    # 2) 标准化：P1/P99 截断 + 单位方差（与 sparse_pca 同口径 winsor_z）
    lo = np.percentile(X, 1, axis=0)
    hi = np.percentile(X, 99, axis=0)
    Xc = np.clip(X, lo, hi)
    mu = Xc.mean(0)
    sd = Xc.std(0) + 1e-6
    Xs = ((Xc - mu) / sd).astype(np.float32)

    # 3) 训练
    ds = TensorDataset(torch.from_numpy(Xs))
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=0)

    model = TerrainAE(in_dim=X.shape[1], latent=a.latent).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_par:,}")

    for ep in range(1, a.epochs + 1):
        tot = 0.0
        for (xb,) in dl:
            xb = xb.to(DEV)
            recon, z = model(xb)
            loss = nn.functional.mse_loss(recon, xb)
            if a.sparse > 0:
                loss = loss + a.sparse * z.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        if ep % 20 == 0 or ep == 1:
            print(f"  ep {ep:3d}  loss = {tot / len(ds):.5f}")

    # 4) 重建 R²
    with torch.no_grad():
        recon, Z = model(torch.from_numpy(Xs).to(DEV))
        recon = recon.cpu().numpy()
    Z = Z.cpu().numpy()
    ss_res = ((Xs - recon) ** 2).sum()
    ss_tot = ((Xs - Xs.mean(0)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"重建 R² = {r2:.4f}")

    # 5) 落盘
    os.makedirs(CKPT, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "cfg": dict(in_dim=X.shape[1], latent=a.latent),
                "mu": mu.tolist(), "sd": sd.tolist(),
                "lo": lo.tolist(), "hi": hi.tolist(),
                "site": a.site, "seed": a.seed},
               os.path.join(CKPT, f"ae_{a.site}_L{a.latent}.pth"))
    np.save(os.path.join(DATA, f"ae_latent_{a.site}_L{a.latent}.npy"), Z)

    meta = dict(site=a.site, latent=a.latent, r2=float(r2),
                n_samples=int(X.shape[0]), n_factors=int(X.shape[1]),
                sparse=a.sparse, seed=a.seed, epochs=a.epochs,
                n_params=int(n_par), device=DEV)
    with open(os.path.join(RES, f"ae_meta_{a.site}_L{a.latent}.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"写出 checkpoints/ae_{a.site}_L{a.latent}.pth")
    print(f"写出 data/ae_latent_{a.site}_L{a.latent}.npy  shape={Z.shape}")


if __name__ == "__main__":
    main()
