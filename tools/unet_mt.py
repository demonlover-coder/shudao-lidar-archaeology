# -*- coding: utf-8 -*-
"""v2 多任务 U-Net（契约内合规改造，最小合规子集）。

与 results/contract.json 的一致性约束（改任何一处都要重跑 tests/test_t3_mt.py）：
  * RF 严格保持 170 px —— 编码器 ResBlock(3×3,3×3)，解码器 ResBlock(3×3,1×1)；
    1×1 不增加 RF，故解码器等价 dec=1（见 tools/plan_infer.py::unet_spec）。
  * 归一化只允许 'none' / 'bn'（BN 推理必须 eval，用 running stats）；
    禁止 gn/in —— 空间统计归一化逐窗不同，破坏滑窗一致性（contract.norm_policy）。
  * 禁止一切全局池化（SE / non-local / GlobalFiLM）—— 同为逐窗统计，破坏 T3。
  * CoordConv 使用**预计算绝对坐标通道**（随像素走，站点级常数归一化），
    不在网络内按窗口尺寸生成 linspace，否则滑窗推理会改变坐标值 → T3 FAIL。
  * 回归头输出 ΔZ（**米**，相对填充面的残差），门控残差 dem_pred = dem_fill + ΔZ·gate：
    空洞 gate≈1（全额补值），有效区 gate≈α（低幅去噪微调）。
    dem_fill 必须是 IDW/最近邻填充面（不再是站点常数），否则跨站会因海拔基准不同而崩溃。

输入通道约定（in_ch=12，由数据侧拼好；见 tools/train_mt_v2.py::build_channels）：
  [0]      dem_mean 的**填充面**（未观测处 = IDW/最近邻插值，不再是 0）
  [1..3]   dem_min/max/std（按站点统计标准化；未观测处置 0）
  [4..8]   slope, sin(aspect), cos(aspect), curvature, dom_lum（同上）
  [9]      hole_mask（1=未观测）
  [10..11] 绝对坐标 xx, yy（站点网格归一化到 [-1,1]）

输出（dict）：
  seg        分割 logits        [B,1,H,W]
  dz         ΔZ（米，残差）      [B,1,H,W]
  hole_logit 未观测 logits       [B,1,H,W]
  dem_pred   门控残差后的高程（米）[B,1,H,W]（未传 dem_fill 时为 None）
  feat       解码器末端特征       [B,C,H,W]（可视化用）

multi_scale=True 时另加各档键（深监督，不改变 RF 与门禁；1/1 档与上面同名键**同值**，
故 full_infer / 评估侧无需分支）：
  seg_{1,2,4}  1/1、1/2、1/4 三档分割 logits（1/1 = seg）
  dz_{1,2,4}   三档 ΔZ（1/1 = dz）
  dem_{1,2,4}  三档门控残差高程（1/1 = dem_pred）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_ALLOWED = ("none", "bn")


def _norm_layer(c: int, norm: str) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(c)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"归一化只允许 {NORM_ALLOWED}（契约 norm_policy）；收到 {norm!r}")


class ResBlock(nn.Module):
    """残差块。ks 决定 RF 贡献：编码器 (3,3) → 4·jump；解码器 (3,1) → 2·jump（≡dec=1）。"""

    def __init__(self, cin: int, cout: int, norm: str = "none", ks=(3, 3)):
        super().__init__()
        layers: list[nn.Module] = []
        c = cin
        for k in ks:
            layers.append(nn.Conv2d(c, cout, k, padding=k // 2, bias=(norm == "none")))
            layers.append(_norm_layer(cout, norm))
            layers.append(nn.SiLU())
            c = cout
        self.body = nn.Sequential(*layers)
        if cin == cout:
            self.short: nn.Module = nn.Identity()
        else:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, bias=(norm == "none")), _norm_layer(cout, norm))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.short(x))


class MultiTaskUNetV2(nn.Module):
    def __init__(self, in_ch: int = 12, base: int = 16, down: int = 4,
                 norm: str = "none", alpha: float = 0.1, feather: int = 3,
                 fuse: bool = True, multi_scale: bool = False):
        super().__init__()
        if norm not in NORM_ALLOWED:
            raise ValueError(f"归一化只允许 {NORM_ALLOWED}；收到 {norm!r}")
        self.cfg = dict(in_ch=in_ch, base=base, down=down, norm=norm,
                        alpha=alpha, feather=feather, fuse=fuse, coord="abs",
                        multi_scale=multi_scale)
        chs = [base * (2 ** min(i, 3)) for i in range(down + 1)]

        self.enc = nn.ModuleList()
        cin = in_ch
        for i in range(down):
            self.enc.append(ResBlock(cin, chs[i], norm, ks=(3, 3)))
            cin = chs[i]
        self.bottleneck = ResBlock(cin, chs[down], norm, ks=(3, 3))

        self.dec = nn.ModuleList()
        for i in range(down - 1, -1, -1):
            self.dec.append(ResBlock(chs[i + 1] + chs[i], chs[i], norm, ks=(3, 1)))

        self.seg_head = nn.Conv2d(chs[0], 1, 1)
        self.dz_head = nn.Conv2d(chs[0], 1, 1)
        self.hole_head = nn.Conv2d(chs[0], 1, 1)
        # gate = a + (1-a)·feather(hole_mask)，a = sigmoid(self.alpha) ∈ (0,1)
        import math
        self.alpha = nn.Parameter(torch.tensor(math.log(alpha / (1.0 - alpha))))
        # 把"未观测"作为分割 logits 的加性偏置（可学，初值 0 → 初始无影响）
        self.fuse_w = nn.Parameter(torch.zeros(1)) if fuse else None

        # 深监督：解码器各档再挂一个 1×1 头（3 个 1×1 conv，参数量可忽略，RF 不变）
        self.multi_scale = multi_scale
        if multi_scale:
            if down < 3:
                raise ValueError(f"multi_scale 需要 down>=3（要 H、H/2、H/4 三级）；收到 {down}")
            self.seg_heads = nn.ModuleDict()
            self.dz_heads = nn.ModuleDict()
            # dec 输出顺序 H/8, H/4, H/2, H → dec_feats[1:][::-1] = [H, H/2, H/4]
            # 通道对应 chs[0](H), chs[1](H/2), chs[2](H/4)
            for tag, c in zip(("1", "2", "4"), (chs[0], chs[1], chs[2])):
                self.seg_heads[tag] = nn.Conv2d(c, 1, 1)
                self.dz_heads[tag] = nn.Conv2d(c, 1, 1)

    # ---------------------------------------------------------------- 门控
    def gate(self, hole_mask: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.alpha)
        f = int(self.cfg["feather"])
        m = hole_mask
        if f > 0:
            m = F.avg_pool2d(hole_mask, 2 * f + 1, 1, f)
        return a + (1.0 - a) * m

    # ---------------------------------------------------------------- 前向
    def forward(self, x: torch.Tensor, dem_fill: torch.Tensor | None = None,
                hole_mask: torch.Tensor | None = None) -> dict:
        m = 2 ** self.cfg["down"]
        h, w = x.shape[-2:]
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")   # 动态 padding（契约）
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        dec_feats: list[torch.Tensor] = []
        for blk in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, skips.pop()], dim=1)
            x = blk(x)
            dec_feats.append(x)                      # [H/8, H/4, H/2, H]（down=4）

        if not self.multi_scale:
            feat = dec_feats[-1][..., :h, :w]
            seg = self.seg_head(feat)
            dz = self.dz_head(feat)
            hole_logit = self.hole_head(feat)
            dem_pred = None
            if dem_fill is not None and hole_mask is not None:
                dem_pred = dem_fill + dz * self.gate(hole_mask)
            if self.fuse_w is not None:
                seg = seg + self.fuse_w * hole_logit
            return dict(seg=seg, dz=dz, hole_logit=hole_logit,
                        dem_pred=dem_pred, feat=feat)

        # ---- 多档（深监督）：1/1 档与单档同名键同值，下游无需分支 ----
        out: dict[str, torch.Tensor | None] = {}
        for tag, feat_i in zip(("1", "2", "4"), dec_feats[1:][::-1]):
            k = int(tag)
            # 裁到与 max_pool2d(label, k) 同尺寸：H//k（不是 h_pad//k），保证与监督目标逐像素对齐
            h_i, w_i = h // k, w // k
            fi = feat_i[..., :h_i, :w_i]
            seg_i = self.seg_heads[tag](fi)
            dz_i = self.dz_heads[tag](fi)
            out[f"seg_{tag}"] = seg_i
            out[f"dz_{tag}"] = dz_i
            if dem_fill is not None and hole_mask is not None:
                if k == 1:
                    fill_i, mask_i = dem_fill, hole_mask   # 1/1 档不插值，与单档逐位一致
                else:
                    fill_i = F.interpolate(dem_fill, size=(h_i, w_i),
                                           mode="bilinear", align_corners=False)
                    mask_i = F.interpolate(hole_mask, size=(h_i, w_i), mode="nearest")
                out[f"dem_{tag}"] = fill_i + dz_i * self.gate(mask_i)

        feat = dec_feats[-1][..., :h, :w]
        hole_logit = self.hole_head(feat)
        out["hole_logit"] = hole_logit
        out["feat"] = feat
        seg1 = out["seg_1"]
        if self.fuse_w is not None:
            seg1 = seg1 + self.fuse_w * hole_logit
            out["seg_1"] = seg1
        out["seg"] = seg1
        out["dz"] = out["dz_1"]
        out["dem_pred"] = out.get("dem_1")
        return out


def build_model(**kw) -> MultiTaskUNetV2:
    return MultiTaskUNetV2(**kw)


# ---------------------------------------------------------------- 自检
def _spec(down: int = 4) -> list[tuple[int, int, int]]:
    """与 MultiTaskUNetV2 对应的 RF spec。"""
    spec: list[tuple[int, int, int]] = []
    for _ in range(down):
        spec += [(3, 1, 1), (3, 1, 1), (2, 2, 1)]        # ResBlock(3,3) + maxpool
    spec += [(3, 1, 1), (3, 1, 1)]                        # bottleneck ResBlock(3,3)
    for _ in range(down):
        spec += [(1, 0.5, 1), (3, 1, 1), (1, 1, 1)]       # up + ResBlock(3,1)
    return spec


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plan_infer import receptive_field

    rf, jump = receptive_field(_spec(4))
    n = sum(p.numel() for p in build_model().parameters())
    print(f"RF={rf:.0f} px  jump={jump}  (契约要求 RF=170)")
    print(f"参数量={n:,} ({n / 1e6:.2f}M)")
    m = build_model()
    y = m(torch.zeros(1, 12, 256, 256))
    print("输出形状:", {k: tuple(v.shape) for k, v in y.items() if v is not None})
