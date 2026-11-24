# -*- coding: utf-8 -*-
"""UTM 窗口 → [L,H,W] 多分辨率 DEM 张量（与 DEM 栅格逐像元配准）。

P2 训练出的 HashGrid+MLP 是一个**连续坐标场** `z = f(u, v)`（u,v ∈ [0,1]²，按点云
bbox 归一化），因此查询分辨率与训练分辨率解耦：给定任意 UTM 窗口 + 任意 gsd，
都能直接出配准好的 DEM，不需要重训、也不需要改网络。

坐标约定（已逐像元实测确认）：
  - 像元中心取 `(i+0.5)`，与 `eval_p2.grid_axes` 和 GeoTIFF tiepoint 一致；
  - 第 0 行 = **北侧**（`utm_y1`），与 `data/{site}_1m.npz` 的 `dem_mean` 一致；
  - `pack_dem` 内部把 UTM 映射回点云 bbox 归一化坐标，所以窗口可以大于点云 bbox，
    落在 bbox 之外的像元是外推（无监督），结果不可信 —— 与 `eval_p2` 的
    `in_box` 口径一致。

`gsd_m` 来自 `data/meta_{site}.json`；`x0/y1` 也在 meta 里（`{site}_1m.npz` 内没有）。

用法：
    python tools/p2_dem.py          # 两站 1m，与 dem_mean 逐像元比 RMSE
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hashgrid import HashGridDEM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
DATA = os.path.join(ROOT, "data")


def pick_device() -> str:
    """设备回退顺序 NPU > CUDA > CPU（项目硬约束：torch_npu 必须先 import）。"""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    print("[警告] 未找到 NPU/GPU，回退 CPU")
    return "cpu"


DEV = pick_device()

_CACHE: dict = {}


def load_model(site: str, ckpt_tag: str | None = None) -> HashGridDEM:
    """载入 `checkpoints/p2_{site}.pth`（给了 tag 则 `p2_{tag}_{site}.pth`）。

    权重文件是 dict（含 `state_dict` / `cfg` / `bbox`），按 `cfg` 建模型再灌权重，
    因此超参不硬编码。`pack_dem` 需要的 `bbox`（4 元 UTM 元组）与 `zmin/zmax`
    都由这里从 ckpt 写进模型。
    """
    key = (site, ckpt_tag)
    if key in _CACHE:
        return _CACHE[key]
    name = f"p2_{ckpt_tag}_{site}.pth" if ckpt_tag else f"p2_{site}.pth"
    ckpt = torch.load(os.path.join(CKPT, name), map_location="cpu", weights_only=False)
    m = HashGridDEM(**ckpt["cfg"]).to(DEV)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    lo, hi = np.asarray(ckpt["bbox"], dtype=np.float64)
    m.bbox = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))  # bx0,by0,bx1,by1
    m.zmin, m.zmax = float(lo[2]), float(hi[2])
    _CACHE[key] = m
    return m


def pointcloud_to_dem(site: str, utm_x0: float, utm_y1: float, width_m: float,
                      height_m: float, gsd: float, n_levels: int = 4,
                      ckpt_tag: str | None = None) -> torch.Tensor:
    """输入: UTM 窗口参数

    输出: [L, H, W] 配准的 DEM 张量（归一化 z ∈ [0,1]，米制用 `to_utm`）
          H = round(height_m / gsd)，W = round(width_m / gsd)
          L0 = 1/1 最细，L1 = 1/2，L2 = 1/4，L3 = 1/8，全部上采样回 [H, W]
          第 0 行 = 北侧
    """
    model = load_model(site, ckpt_tag)
    return model.pack_dem(utm_x0, utm_y1, width_m, height_m, gsd,
                          n_levels=n_levels, device=DEV)


def to_utm(site: str, z_norm: torch.Tensor, ckpt_tag: str | None = None) -> torch.Tensor:
    """归一化 z → UTM 高程（米）。zmin/zmax 取自训练站的 bbox。"""
    m = load_model(site, ckpt_tag)
    return (m.zmin + z_norm.double() * (m.zmax - m.zmin)).to(z_norm.dtype)


def dem_window(site: str):
    """读 `data/{site}_1m.npz` + `data/meta_{site}.json`，返回 1m 窗口参数。"""
    meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    H, W = d["dem_mean"].shape
    gsd = float(meta["gsd_m"])
    return float(meta["x0"]), float(meta["y1"]), W * gsd, H * gsd, gsd, d


if __name__ == "__main__":
    for site in ("杨家坝", "西坝村"):
        x0, y1, width_m, height_m, gsd, d = dem_window(site)
        packed = pointcloud_to_dem(site, x0, y1, width_m, height_m, gsd, n_levels=4)
        print(f"{site}: 窗口 {width_m:.0f}×{height_m:.0f} m @ {gsd} m "
              f"→ {tuple(packed.shape)}")

        gt = d["dem_mean"]
        valid = d["valid"] == 1
        print(f"  {'档':<4}{'RMSE(m)':>10}{'MAE(m)':>10}{'bias(m)':>10}")
        for l in range(4):
            z = to_utm(site, packed[l]).cpu().numpy()
            err = z - gt
            rmse = float(np.sqrt((err[valid] ** 2).mean()))
            print(f"  L{l} 1/{2**l:<2}{rmse:>9.3f}{np.abs(err[valid]).mean():>10.3f}"
                  f"{err[valid].mean():>+10.3f}")
