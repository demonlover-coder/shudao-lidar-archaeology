# -*- coding: utf-8 -*-
"""P2 L0 与 v4 补洞输出的对照（西坝村）。

v4 权重方向：train=西坝村, val=杨家坝 → `v4_pre_西坝村_杨家坝_base16.pth`。
权重不存在时打印 [SKIP] 并正常退出（不算失败）。

用法：
    python tools/p2_vs_v4.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P                        # noqa: E402
from train_mt_v2 import full_infer        # noqa: E402
from unet_mt import MultiTaskUNetV2       # noqa: E402
from p2_export_all import project_covered  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
FIG = os.path.join(ROOT, "figures")
SITE = "西坝村"
V4_NAME = "v4_pre_西坝村_杨家坝_base16.pth"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    path = os.path.join(CKPT, V4_NAME)
    if not os.path.exists(path):
        print(f"[SKIP] 未找到 v4 权重 {V4_NAME}，跳过对照")
        return

    # 1) P2 L0（米）
    x0, y1, wm, hm, gsd, d = P.dem_window(SITE)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    m = P.load_model(SITE)
    packed = m.pack_dem(x0, y1, wm, hm, gsd, n_levels=1, device=P.DEV)
    p2 = P.to_utm(SITE, packed[0]).cpu().numpy().astype(np.float64)
    print(f"[{SITE}] P2 L0 {p2.shape}，设备 {P.DEV}")

    # 2) v4 补洞输出（米）
    sd = torch.load(path, map_location="cpu", weights_only=False)
    in_ch = next(v.shape[1] for v in sd.values() if v.ndim == 4 and v.shape[1] in (10, 12))
    model = MultiTaskUNetV2(in_ch=in_ch, base=16, down=4, norm="none",
                            alpha=0.1, feather=3, multi_scale=False).to(P.DEV)
    model.load_state_dict(sd)
    model.eval()
    print(f"  载入 v4 权重 {V4_NAME}（in_ch={in_ch}）")
    r = full_infer(model, SITE, P.DEV, coord=False)
    v4 = r["dem_pred_m"].astype(np.float64)

    # 3) 共同像元：DEM valid ∩ 地面点覆盖（P2 可信域）∩ 点云 bbox
    gt = d["dem_mean"].astype(np.float64)
    cov = project_covered(SITE, x0, y1, gsd, H, W)
    bx0, by0, bx1, by1 = m.bbox
    cols = x0 + (np.arange(W) + 0.5) * gsd
    rows = y1 - (np.arange(H) + 0.5) * gsd
    in_box = ((rows >= by0) & (rows <= by1))[:, None] & ((cols >= bx0) & (cols <= bx1))[None, :]
    common = valid & cov & in_box
    print(f"  共同像元 {int(common.sum()):,}（valid {int(valid.sum()):,}，"
          f"地面点覆盖 {int(cov.sum()):,}，bbox 内 {int(in_box.sum()):,}）")

    def _rmse(a, b):
        e = (a - b)[common]
        return float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean())

    r_p2, m_p2 = _rmse(p2, gt)
    r_v4, m_v4 = _rmse(v4, gt)
    r_pv, m_pv = _rmse(p2, v4)
    print(f"  各自 vs 真值 dem_mean（共同像元 {int(common.sum()):,}）：")
    print(f"    P2_L0 : RMSE={r_p2:.3f} m  MAE={m_p2:.3f} m")
    print(f"    v4    : RMSE={r_v4:.3f} m  MAE={m_v4:.3f} m")
    print(f"    差值  : P2 − v4 = {r_p2 - r_v4:+.3f} m（正 = P2 更差）")
    print(f"    （参考）P2 vs v4：RMSE={r_pv:.3f} m  MAE={m_pv:.3f} m")
    print("    注：dem_mean 含非地面回波，未覆盖地面点的像元已由 cov 排除，"
          "故仅在此口径下 dem_mean ≈ 地面真值。")

    # 4) 四联图
    diff = p2 - v4
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    panels = [(f"P2 L0（RMSE {r_p2:.2f} m）", p2, "terrain"),
              (f"v4 补洞（RMSE {r_v4:.2f} m）", v4, "terrain"),
              (f"diff = P2 − v4（RMSE {r_pv:.2f} m）", diff, "coolwarm"),
              ("|diff|", np.abs(diff), "magma")]
    for ax, (t, arr, cm) in zip(axes, panels):
        if cm == "coolwarm":
            vmax = float(np.percentile(np.abs(arr[common]), 99)) or 1.0
            im = ax.imshow(arr, cmap=cm, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(arr, cmap=cm)
        ax.set_title(t)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{SITE} P2 vs v4 vs 真值（共同像元：valid ∩ 地面点覆盖 ∩ bbox，"
                 f"P2 RMSE {r_p2:.2f} m / v4 RMSE {r_v4:.2f} m）")
    fig.tight_layout()
    out = os.path.join(FIG, f"p2_vs_v4_{SITE}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  写出 {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
