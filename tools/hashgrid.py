# -*- coding: utf-8 -*-
"""P2 任务 2：多分辨率哈希网格 + MLP（Instant-NGP 风格的 2D 变体）。

想法：把 (x,y) ∈ [0,1]² 送进 L 级、分辨率递增的网格；每级网格用哈希表存
`features_per_level` 维特征，四角三线性插值后拼接，再过一个 MLP 回归高程 z。
低分辨率级负责大尺度地形趋势，高分辨率级负责局部细节——这正是 DEM 表征需要的
"多尺度 + 局部"结构，而纯 MLP 对坐标的高频拟合要慢得多。

两个必须说清的偏离（原文参数与判据自相矛盾）：
  1. `hash_table_size` 默认 **2^15**，不是原文的 2^19。
     16 级 × 2^19 × 2 维 = 16.78 M 参数，**远超"参数量 < 2M"判据**；
     2^15 时哈希表 1.05 M + MLP 6.3 k ≈ 1.06 M，满足判据。
     两个要求无法同时成立，此处按判据取小，并暴露 `--hash-size` 供放开。
  2. 网格分辨率超过哈希表容量后必然冲突（第 15 级 7006²=49.1 M 格 ≫ 表容量），
     这是哈希而非直接网格的固有代价；靠低层级承担主要拟合。

用法（自检）：
    python tools/hashgrid.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

_PRIME_X = 73856093
_PRIME_Y = 19349663
_PRIME_L = 83492791


class HashGrid(nn.Module):
    """多分辨率二维哈希网格，输出 [B, levels*features_per_level]。"""

    def __init__(self, levels: int = 16, features_per_level: int = 2,
                 base_resolution: int = 16, per_level_scale: float = 1.5,
                 hash_table_size: int = 2 ** 15, init_scale: float = 1e-4):
        super().__init__()
        self.levels = levels
        self.features_per_level = features_per_level
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.hash_table_size = hash_table_size
        res = [int((base_resolution * per_level_scale ** l) + 1e-9) + 1 for l in range(levels)]
        self.register_buffer("resolutions", torch.tensor(res, dtype=torch.long))
        # 四角偏移（顺序固定，插值权重与之对应）
        self.register_buffer("corners", torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long))
        self.table = nn.Parameter(torch.randn(levels, hash_table_size,
                                              features_per_level) * init_scale)

    def _hash(self, ix: torch.Tensor, iy: torch.Tensor, level: int) -> torch.Tensor:
        h = (ix * _PRIME_X) ^ (iy * _PRIME_Y) ^ (level * _PRIME_L)
        h = h & 0x7FFFFFFF
        h = h ^ (h >> 15)
        return h % self.hash_table_size

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """xy: [B, 2]，取值应在 [0,1]。返回 [B, levels*features_per_level]。"""
        out = []
        for l in range(self.levels):
            res = int(self.resolutions[l])
            p = xy * res
            i0 = torch.floor(p).long()                       # [B,2]
            f = p - i0.float()                               # [B,2] 小数部分
            corner = i0.unsqueeze(1) + self.corners          # [B,4,2]
            h = self._hash(corner[..., 0], corner[..., 1], l)  # [B,4]
            v = self.table[l][h]                             # [B,4,fpl]
            # 四角插值权重，与 self.corners 顺序 [ (0,0),(0,1),(1,0),(1,1) ] 一致
            fx, fy = f[:, 0], f[:, 1]
            wc = torch.stack([(1 - fx) * (1 - fy), (1 - fx) * fy,
                              fx * (1 - fy), fx * fy], dim=1)   # [B,4]
            out.append((v * wc.unsqueeze(-1)).sum(dim=1))    # [B,fpl]
        return torch.cat(out, dim=-1)


class HashGridDEM(nn.Module):
    """HashGrid → MLP(32→64→64→1)，回归归一化高程 z ∈ [0,1]。"""

    def __init__(self, levels: int = 16, features_per_level: int = 2,
                 base_resolution: int = 16, per_level_scale: float = 1.5,
                 hash_table_size: int = 2 ** 15, hidden: int = 64):
        super().__init__()
        self.hashgrid = HashGrid(levels, features_per_level, base_resolution,
                                 per_level_scale, hash_table_size)
        d = levels * features_per_level
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(inplace=False),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=False),
            nn.Linear(hidden, 1),
        )
        self.cfg = dict(levels=levels, features_per_level=features_per_level,
                        base_resolution=base_resolution, per_level_scale=per_level_scale,
                        hash_table_size=hash_table_size, hidden=hidden)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:  # [B,2] → [B,1]
        return self.mlp(self.hashgrid(xy))

    @torch.no_grad()
    def predict_utm_z(self, xy: torch.Tensor, zmin: float, zmax: float) -> torch.Tensor:
        """归一化输出 → UTM 高程（米）。"""
        return zmin + self(xy) * (zmax - zmin)

    @torch.no_grad()
    def pack_dem(self, utm_x0, utm_y1, width_m, height_m, gsd,
                 n_levels: int = 4, device: str = "cuda"):
        """输入:
            utm_x0, utm_y1: 目标 UTM 窗口左上角坐标（东、北）
            width_m, height_m: 窗口宽高（米）
            gsd: 目标分辨率（米/像素），比如 1.0
        输出:
            [L, H, W] 栅格序 DEM 张量
            H = round(height_m / gsd), W = round(width_m / gsd)
            L0 = 1/1，L1 = 1/2，...
            第 0 行 = 北侧（与 DEM 栅格一致）

        像元中心取 `(i+0.5)`，与 `eval_p2.grid_axes` / GeoTIFF tiepoint 约定一致，
        因此 L0 可与 `data/{site}_1m.npz` 的 `dem_mean` 逐像元直接比。

        需要 `self.bbox = (bx0, by0, bx1, by1)`（点云 bbox，UTM 米），
        由 `p2_dem.load_model` 从 ckpt 写入。
        """
        import torch.nn.functional as F
        self.eval()

        # 目标栅格尺寸
        H = int(round(height_m / gsd))
        W = int(round(width_m / gsd))

        # 需要 bbox（从 ckpt 里拿，见 p2_dem.py）
        bx0, by0, bx1, by1 = self.bbox   # 点云 bbox：xmin, ymin, xmax, ymax

        outs = []
        for l in range(n_levels):
            k = 2 ** l
            h, w = max(H // k, 1), max(W // k, 1)
            # 在 UTM 窗口内生成 h×w 网格
            xs_utm = utm_x0 + (torch.arange(w, device=device) + 0.5) * (width_m / w)
            ys_utm = utm_y1 - (torch.arange(h, device=device) + 0.5) * (height_m / h)
            yy, xx = torch.meshgrid(ys_utm, xs_utm, indexing="ij")
            # 转成 bbox 归一化坐标
            u = (xx - bx0) / (bx1 - bx0)
            v = (yy - by0) / (by1 - by0)
            xy = torch.stack([u.reshape(-1), v.reshape(-1)], dim=1)
            z = self.forward(xy).reshape(1, 1, h, w)
            # 上采样到目标尺寸
            z_up = F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(z_up.squeeze())

        packed = torch.stack(outs, dim=0)   # [L, H, W]
        return packed


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for hs in (2 ** 15, 2 ** 19):
        m = HashGridDEM(hash_table_size=hs).to(dev)
        n_hash = m.hashgrid.table.numel()
        n_mlp = sum(p.numel() for p in m.mlp.parameters())
        total = sum(p.numel() for p in m.parameters())
        print(f"hash_table_size=2^{hs.bit_length()-1}  哈希表 {n_hash:,} + MLP {n_mlp:,} "
              f"= 总参数 {total:,}  ({total/1e6:.3f} M)  "
              f"{'<2M OK' if total < 2e6 else '>2M 违反判据'}")
        if hs != 2 ** 15:
            continue
        xy = torch.rand(4096, 2, device=dev)
        z = m(xy)
        print(f"  前向 [4096,2] → {tuple(z.shape)}  范围 [{z.min():.4f}, {z.max():.4f}]")
        loss = (z ** 2).mean()
        loss.backward()
        g = m.hashgrid.table.grad
        print(f"  反传 OK  loss={loss.item():.3e}  哈希表梯度非零占比 "
              f"{(g.abs().sum(dim=-1) > 0).float().mean():.3%}")
        # 边界取值不应崩
        edge = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], device=dev)
        print(f"  边界 [0/1]² 前向 {m(edge).flatten().tolist()}")


if __name__ == "__main__":
    main()
