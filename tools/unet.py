"""与契约表一致的 U-Net（torch）+ 动态 padding。

结构严格对应 tools/plan_infer.py::unet_spec：
  * 每级 enc 层 3x3 卷积（默认 2 层），级间 MaxPool2x2
  * bottleneck 再 enc 层
  * 解码每级：最近邻上采样 + dec 层 3x3（默认 1 层）
  * 动态 padding：内部把输入补到 2**down 的整数倍（replicate），输出裁回原尺寸

归一化（norm 参数）是本文件最重要的开关，直接决定"整图推理 == 滑窗推理"能否成立：
  'gn'   GroupNorm —— 在 (C_group, H, W) 上统计，**逐窗统计量不同 → 滑窗会改变预测**
  'bn'   BatchNorm —— 推理用 running stats（全局固定），**与窗口无关**
  'none' 不归一化 —— 与窗口无关
T3 会对三种各测一遍（tests/test_t3_model.py）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_norm_relu(cin: int, cout: int, norm: str = "gn", groups: int = 8) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(cin, cout, 3, padding=1, bias=(norm == "none"))]
    if norm == "gn":
        g = groups if cout % groups == 0 else 1
        layers.append(nn.GroupNorm(g, cout))
    elif norm == "bn":
        layers.append(nn.BatchNorm2d(cout))
    elif norm != "none":
        raise ValueError(f"未知 norm={norm}")
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 1, base: int = 48,
                 down: int = 4, enc: int = 2, dec: int = 1, norm: str = "gn"):
        super().__init__()
        self.cfg = dict(in_ch=in_ch, out_ch=out_ch, base=base, down=down,
                        enc=enc, dec=dec, norm=norm)
        chs = [base * (2 ** min(i, 3)) for i in range(down + 1)]

        self.enc = nn.ModuleList()
        cin = in_ch
        for i in range(down):
            layers = [conv_norm_relu(cin if j == 0 else chs[i], chs[i], norm) for j in range(enc)]
            self.enc.append(nn.Sequential(*layers))
            cin = chs[i]
        self.bottleneck = nn.Sequential(
            *[conv_norm_relu(cin if j == 0 else chs[down], chs[down], norm) for j in range(enc)])
        self.dec = nn.ModuleList()
        for i in range(down - 1, -1, -1):
            layers = [conv_norm_relu(chs[i + 1] + chs[i] if j == 0 else chs[i], chs[i], norm)
                      for j in range(dec)]
            self.dec.append(nn.Sequential(*layers))
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for blk in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, skips.pop()], dim=1)
            x = blk(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = 2 ** self.cfg["down"]
        h, w = x.shape[-2:]
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")   # 动态 padding
        y = self._forward(x)
        return y[..., :h, :w]


def build_model(**kw) -> UNet:
    return UNet(**kw)
