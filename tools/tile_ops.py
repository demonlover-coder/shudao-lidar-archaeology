"""滑窗切片与拼接（推理侧的唯一实现）。

把 T3 抓到的那个 bug 的防护固化进管线本身：
    图像最外圈 halo 宽度内没有任何 tile 的核心区覆盖 → 拼接结果在那里是 0。
做法：先四周 pad halo（edge）并补到 (n-1)*step + tile 的网格 → 切整块 tile → 只搬运核心区
→ 裁回 → **断言全覆盖**。补到网格是为了让每块都是整块、起点同在 step 网格上（同一下采样相位）。

任何滑窗推理都必须走这里的 iter_tiles()/stitch()，不要在别处重复实现。
"""
from __future__ import annotations

import numpy as np


def tile_starts(size: int, tile: int, step: int) -> list[int]:
    """返回各 tile 的起始坐标；末块锚定在 size-tile，保证覆盖到边界。"""
    if size <= tile:
        return [0]
    s = list(range(0, size - tile + 1, step))
    if s[-1] + tile < size:
        s.append(size - tile)
    return s


def iter_tiles(h: int, w: int, tile: int, overlap: int):
    """产出 (y0, x0, y1, x1) —— 注意坐标是 **pad 后** 的坐标系。"""
    step = tile - overlap
    if step <= 0:
        raise ValueError(f"step={step} <= 0：overlap({overlap}) 必须小于 tile({tile})")
    for y0 in tile_starts(h, tile, step):
        for x0 in tile_starts(w, tile, step):
            yield y0, x0, min(y0 + tile, h), min(x0 + tile, w)


def core_slice(y0: int, x0: int, tile: int, halo: int, hp: int, wp: int,
               h: int, w: int):
    """核心区在 pad 坐标与 crop 坐标下的切片。"""
    cy0, cx0 = y0 + halo, x0 + halo
    cy1, cx1 = min(y0 + tile - halo, hp), min(x0 + tile - halo, wp)
    return (slice(cy0, cy1), slice(cx0, cx1)), (slice(cy0 - halo, cy1 - halo),
                                                slice(cx0 - halo, cx1 - halo))


def _grid_extra(size: int, tile: int, step: int, halo: int) -> int:
    """在两侧 halo 之外再补多少行/列，使 pad 后尺寸 = (n-1)*step + tile。

    为什么必须补到网格：只有 pad 后尺寸恰好是 (n-1)*step + tile，才能同时保证
      ① 每个 tile 都是整块 tile×tile（核心区规则才成立）；
      ② 所有 tile 起点都是 step 的倍数（step 是 align 的倍数），与整图共享同一下采样相位。
    旧实现不做这步补，改用「末块锚定 size-tile」兜底 → 末块起点不在 step 网格上，
    与邻块核心区大幅重叠，重叠处把两个**不同相位**的预测平均，实测内部相对差 2.6e-2
    （判据 1e-3）。
    """
    target = size + 2 * halo
    if target <= tile:
        return tile - target
    n = -(-(target - tile) // step) + 1          # 覆盖 target 所需块数
    return (n - 1) * step + tile - target


def stitch(predict_fn, x: np.ndarray, tile: int, overlap: int, *args, **kwargs) -> np.ndarray:
    """滑窗推理 + 核心区拼接。支持 (H,W) 或 (C,H,W)，任意尺寸输入 → 同尺寸输出。

    predict_fn(sub) -> 与 sub 同形状（除通道）的预测。
    三条不变量（T3 滑窗一致性依赖）：
      1. pad 到 (n-1)*step + tile 的网格 → 每个 tile 都是整块；
      2. 起点全是 step 的倍数 → 与整图推理同一下采样相位；
      3. 核心区宽 = tile - 2*halo = step → 相邻核心区首尾相接、互不重叠，计数恒为 1。
    """
    h, w = x.shape[-2:]
    halo = overlap // 2
    step = tile - overlap
    if step <= 0:
        raise ValueError(f"step={step} <= 0：overlap({overlap}) 必须小于 tile({tile})")
    eh = _grid_extra(h, tile, step, halo)
    ew = _grid_extra(w, tile, step, halo)
    pad = [(0, 0)] * (x.ndim - 2) + [(halo, halo + eh), (halo, halo + ew)]
    xp = np.pad(x, pad, mode="edge")
    hp, wp = xp.shape[-2:]
    out = np.zeros_like(xp, dtype=np.float64)
    cnt = np.zeros_like(xp, dtype=np.float64)

    for y0, x0, y1, x1 in iter_tiles(hp, wp, tile, overlap):
        assert (y1 - y0) == tile and (x1 - x0) == tile, "补到网格后不应出现非整块 tile"
        sub = xp[..., y0:y1, x0:x1]
        pred = np.asarray(predict_fn(sub, *args, **kwargs), dtype=np.float64)
        # 核心区 = 预测图里**去掉 halo** 的那一块（不是左上角！）
        out[..., y0 + halo:y1 - halo, x0 + halo:x1 - halo] += \
            pred[..., halo:tile - halo, halo:tile - halo]
        cnt[..., y0 + halo:y1 - halo, x0 + halo:x1 - halo] += 1.0

    core = cnt[..., halo:halo + h, halo:halo + w]
    # 常规断言：任何像素都必须被至少一个 tile 的核心区覆盖
    assert core.min() > 0, (
        f"存在未被任何 tile 核心区覆盖的像素：{int((core == 0).sum())} 个 "
        f"(tile={tile} overlap={overlap} h={h} w={w})")
    res = out / np.maximum(cnt, 1.0)
    return res[..., halo:halo + h, halo:halo + w]


def whole_or_tiled(predict_fn, x: np.ndarray, tile: int | None, overlap: int = 0,
                   max_whole: int = 4096) -> np.ndarray:
    """L1 尺寸自适应推理入口：任意 (H,W) 或 (C,H,W) 输入 → 同尺寸输出。

    小图（装得下一个 tile）走整图推理，大图走滑窗拼接；两条路径都返回与输入
    相同的空间尺寸。模型侧的动态 padding 由 predict_fn 内部负责（见 unet.py）。
    """
    h, w = x.shape[-2:]
    if tile is None or (h <= max_whole and w <= max_whole and h <= tile and w <= tile):
        return np.asarray(predict_fn(x), dtype=np.float64)
    return stitch(predict_fn, x, tile, overlap)
