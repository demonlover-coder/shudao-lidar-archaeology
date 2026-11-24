# -*- coding: utf-8 -*-
"""P2 任务 3：纯坐标 HashGrid+MLP 的 DEM 表征训练。

每个 epoch 随机采 `epoch_len` 个 batch，每 batch `batch_size` 个点，
损失 = L1(pred_z, true_z)（归一化 [0,1] 空间，与任务书一致），
同时打印米制 RMSE 便于判读。

数据默认用**地面点**（cls==2）：DEM 是地面曲面，用含树冠的全点云训练，
目标本身不是 (x,y) 的单值函数，1m RMSE 下界就被植被偏差锁死在 ~5 m。
全点口径可用 `--data full` 复现作对照。

用法：
    python tools/train_p2.py --site 杨家坝
    python tools/train_p2.py --sites 杨家坝 西坝村 --epochs 50
    python tools/train_p2.py --site 杨家坝 --data full --tag full
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hashgrid import HashGridDEM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")


def pick_device() -> str:
    """NPU > CUDA > CPU；torch_npu 必须先 import 再判 torch.npu。"""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    print("[警告] 未检测到 NPU/GPU，回退 CPU，训练会非常慢")
    return "cpu"


class Tee:
    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s: str) -> None:
        self.stdout.write(s)
        self.f.write(s)

    def flush(self) -> None:
        self.stdout.flush()
        self.f.flush()


def load_points(site: str, which: str, seed: int, val_frac: float):
    suffix = "_ground" if which == "ground" else ""
    path = os.path.join(DATA, f"p2_{site}_points{suffix}.npz")
    d = np.load(path)
    xyz = d["xyz"]
    bbox = d["bbox"]                       # [2,3] UTM min/max
    n = len(xyz)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    tri, vai = perm[n_val:], perm[:n_val]
    return xyz, bbox, tri, vai, path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=["杨家坝"])
    ap.add_argument("--site", default=None)
    ap.add_argument("--data", choices=["ground", "full"], default="ground")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--epoch-len", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--fpl", type=int, default=2)
    ap.add_argument("--base-res", type=int, default=16)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--hash-size", type=int, default=2 ** 15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--init-scale", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    sites = [a.site] if a.site else a.sites
    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    dev = pick_device()
    tag = a.tag or ("" if a.data == "ground" else "full")

    print(f"设备: {dev}  数据口径: {a.data}  站点: {sites}")
    for site in sites:
        torch.manual_seed(a.seed)
        t0 = time.perf_counter()
        log_path = os.path.join(RES, f"train_p2_{site}{('_' + tag) if tag else ''}.log")
        tee = Tee(log_path)
        sys.stdout = tee
        try:
            xyz, bbox, tri, vai, dpath = load_points(site, a.data, a.seed, a.val_frac)
            zmin, zmax = float(bbox[0, 2]), float(bbox[1, 2])
            zrange = zmax - zmin
            print(f"\n{'='*72}\n[{site}] {dpath}")
            print(f"  点 {len(xyz):,}  训练 {len(tri):,}  验证 {len(vai):,}  "
                  f"z∈[{zmin:.2f},{zmax:.2f}] 跨度 {zrange:.2f} m")

            model = HashGridDEM(a.levels, a.fpl, a.base_res, a.scale,
                                a.hash_size, a.hidden).to(dev)
            model.hashgrid.table.data.mul_(a.init_scale / 1e-4)   # 允许改初值尺度
            n_par = sum(p.numel() for p in model.parameters())
            print(f"  模型 总参数 {n_par:,} ({n_par/1e6:.3f} M)  "
                  f"{'OK <2M' if n_par < 2e6 else '⚠ >2M 违反判据'}  设备 {next(model.parameters()).device}")
            opt = torch.optim.Adam(model.parameters(), lr=a.lr)
            l1 = nn.L1Loss()

            def to_t(idx):
                x = torch.from_numpy(np.ascontiguousarray(xyz[idx, :2])).float().to(dev)
                z = torch.from_numpy(np.ascontiguousarray(xyz[idx, 2:3])).float().to(dev)
                return x, z

            vx, vz = to_t(vai[:200000])
            hist = []
            for ep in range(1, a.epochs + 1):
                model.train()
                tot = 0.0
                for _ in range(a.epoch_len):
                    sel = tri[np.random.randint(0, len(tri), a.batch)]
                    x, z = to_t(sel)
                    opt.zero_grad(set_to_none=True)
                    loss = l1(model(x), z)
                    loss.backward()
                    opt.step()
                    tot += loss.item()
                tr_norm = tot / a.epoch_len
                model.eval()
                with torch.no_grad():
                    vr = []
                    for i in range(0, len(vx), 100000):
                        vr.append(((model(vx[i:i + 100000]) - vz[i:i + 100000]) ** 2))
                    vr = torch.cat(vr)
                    val_rmse_m = float(vr.mean().sqrt()) * zrange
                hist.append((ep, tr_norm, val_rmse_m))
                if ep == 1 or ep % 5 == 0 or ep == a.epochs:
                    print(f"  [ep {ep:3d}] train_L1={tr_norm:.3e} "
                          f"({tr_norm*zrange:.3f} m)  val_RMSE={val_rmse_m:.3f} m  "
                          f"({time.perf_counter()-t0:.0f}s)")

            name = f"p2_{site}{('_' + tag) if tag else ''}.pth"
            torch.save({"state_dict": model.state_dict(),
                        "cfg": model.cfg,
                        "bbox": bbox.tolist(),
                        "site": site, "data": a.data,
                        "train_args": vars(a), "hist": hist}, os.path.join(CKPT, name))
            # 单调性判定（用 5 epoch 平滑序列，避免随机采样噪声误判）
            sm = [h[2] for h in hist]
            k = 5
            sm = [float(np.mean(sm[max(0, i - k + 1):i + 1])) for i in range(len(sm))]
            mono = all(sm[i] <= sm[i - 1] * 1.02 for i in range(1, len(sm)))
            print(f"  最优 val_RMSE={min(h[2] for h in hist):.3f} m "
                  f"(ep {min(hist, key=lambda h: h[2])[0]})  平滑单调={mono}")
            print(f"  保存 checkpoints/{name}  日志 results/{os.path.basename(log_path)}")
        finally:
            sys.stdout = tee.stdout
            tee.f.flush()
            tee.f.close()
    print(f"\n总耗时 {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
