# -*- coding: utf-8 -*-
"""真彩色 DOM 的通道级走廊区分 AUC 筛选（改管线前的 30 分钟门禁）。

背景
----
杨家坝 openness 法向剖面诊断为 **(a) 代价图与真值不对齐**
（`results/openness_profile.json`：− 侧走廊外更深 4.8 @d≈19px，已超半宽 2.2px）。
西坝村为 (b) 算法问题。 → 杨站需换信息源。

关键事实（本次实测量测，修正"DOM 是灰度 1 通道"的前提）
----
源 DOM 实为**真彩色 RGBA**，是数据管线的 `dom_luminance()`（`build_dataset.py`）
用 `convert("RGB") @ [0.299,0.587,0.114]` 把它压成了单通道 `dom_lum`：
  西坝村 |G−B| 均值 45.4（G 120 vs B 74.8，植被强绿）；杨家坝 |G−B| 均值 13.3。
因此"彩度/颜色"与"高分辨率(2.4~3.3cm)纹理"两层信息都被丢掉了。

本脚本做什么
----
在**与 `dom_lum` 完全相同的降采样映射**（同一 `dem_grid` + 整幅 DOM linspace 到网格）下，
从源 DOM 一次读出并同时聚合出下列 1 m 通道，逐一算走廊区分 AUC：

  r / g / b            三通道各自 box 均值
  lum                  0.299R+0.587G+0.114B（对照，≈现有 dom_lum）
  exg = 2G−R−B         超绿指数（植被 vs 石砌/裸地）
  sat = (max−min)/max  饱和度
  tex_std              cell 内亮度标准差（高分辨率纹理被 1m box 平均抹掉的部分）
  grad                 cell 内亮度梯度幅值均值

判据（沿用 D10：valid 内、AUC 越大越能区分；有效 AUC = max(auc, 1−auc)；方向可翻）
  有效 AUC ≥ 0.60 → 该通道在杨站确有信号，值得进管线
  有效 AUC ≈ 0.5  → 与随机无异，别指望

产出：results/dom_channel_auc.json / .log

用法
----
    python tools/dom_channel_auc.py                 # 两站
    python tools/dom_channel_auc.py --sites 杨家坝
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from audit_data import BASE                     # noqa: E402
from build_dataset import SITE_FILES, dem_grid   # noqa: E402
from shudao_auc import roc_auc                  # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")


def dom_channels(path: str, grid: dict, band_out_rows: int = 8) -> dict:
    """把源 DOM 一次读出，同块聚合出 r/g/b/lum/exg/sat/tex_std/grad 各 [H,W] 通道。

    降采样映射与 `build_dataset.dom_luminance()` **逐位一致**（整数边界 box 平均），
    保证与现有 `dom_lum` 通道同口径可比。
    """
    with Image.open(path) as im:
        rw, rh = im.width, im.height
        H, W = grid["h"], grid["w"]
        rb = np.linspace(0, rh, H + 1).round().astype(int)
        cb = np.linspace(0, rw, W + 1).round().astype(int)
        cw = np.maximum(cb[1:] - cb[:-1], 1)
        keys = ("r", "g", "b", "lum", "exg", "sat", "grad", "lum2")
        out = {k: np.zeros((H, W), np.float32) for k in keys}

        def _box(field_rows: np.ndarray) -> np.ndarray:
            cs = np.concatenate([[0.0], np.cumsum(field_rows, dtype=np.float64)])
            return ((cs[cb[1:]] - cs[cb[:-1]]) / cw).astype(np.float32)

        for r in range(0, H, band_out_rows):
            r1 = min(r + band_out_rows, H)
            band = np.asarray(im.crop((0, int(rb[r]), rw, int(rb[r1]))).convert("RGB"),
                              dtype=np.float32)
            R, G, B = band[..., 0], band[..., 1], band[..., 2]
            lum = 0.299 * R + 0.587 * G + 0.114 * B
            exg = 2.0 * G - R - B
            mx = band.max(2)
            mn = band.min(2)
            sat = (mx - mn) / np.maximum(mx, 1.0)
            gx = np.zeros_like(lum)
            gy = np.zeros_like(lum)
            gy[1:-1] = 0.5 * (lum[2:] - lum[:-2])
            gx[:, 1:-1] = 0.5 * (lum[:, 2:] - lum[:, :-2])
            grad = np.hypot(gx, gy)
            fields = {"r": R, "g": G, "b": B, "lum": lum, "exg": exg,
                      "sat": sat, "grad": grad, "lum2": lum * lum}
            del band, mx, mn, gx, gy
            for rr in range(r, r1):
                s = slice(int(rb[rr]) - int(rb[r]), int(rb[rr + 1]) - int(rb[r]))
                for k, f in fields.items():
                    out[k][rr] = _box(f[s].mean(axis=0))
            del fields, lum, exg, sat, grad
    out["tex_std"] = np.sqrt(np.maximum(out.pop("lum2") - out["lum"] ** 2, 0.0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="真彩色 DOM 通道级走廊 AUC 筛选")
    ap.add_argument("--sites", nargs="*", default=list(SITE_FILES))
    ap.add_argument("--save", action="store_true",
                    help="把各通道写入 data/dom_channels_{site}.npz（供代价图读取）")
    a = ap.parse_args()

    lf = open(os.path.join(RES, "dom_channel_auc.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()

    log("真彩色 DOM 通道级走廊区分 AUC（映射与现有 dom_lum 逐位一致；有效AUC≥0.60 视为有信号）")
    report = {}
    for site in a.sites:
        f = SITE_FILES[site]
        grid = dem_grid(os.path.join(BASE, f["dem"]))
        npz = np.load(os.path.join(DATA, f"{site}_1m.npz"))
        valid = npz["valid"] == 1
        label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)
        log(f"\n[{site}] 网格 {grid['w']}×{grid['h']}  有效 {int(valid.sum()):,}  "
            f"走廊 {int((label & valid).sum()):,}（基准率 {float((label & valid).sum()) / int(valid.sum()):.4f}）")

        ch = dom_channels(os.path.join(BASE, f["dom"]), grid)
        ch["dom_lum_npz"] = npz["dom_lum"].astype(np.float64)   # 现有通道，作对照
        if a.save:
            keep = ("r", "g", "b", "lum", "exg", "sat", "tex_std", "grad")
            sp = os.path.join(DATA, f"dom_channels_{site}.npz")
            np.savez_compressed(sp, **{k: np.asarray(ch[k], np.float32) for k in keep})
            log(f"  写出: {os.path.relpath(sp, ROOT)}")

        rows = []
        log(f"  {'通道':<12s}{'AUC':>8s}{'有效AUC':>9s}{'方向':>7s}"
            f"{'走廊中位':>11s}{'背景中位':>11s}  备注")
        note = {
            "r": "红", "g": "绿", "b": "蓝", "lum": "亮度(≈现有dom_lum)",
            "dom_lum_npz": "现有npz通道(对照)", "exg": "超绿=2G−R−B(植被)",
            "sat": "饱和度", "tex_std": "cell内亮度std(高分辨率纹理)", "grad": "cell内梯度幅值",
        }
        for name in ("r", "g", "b", "lum", "dom_lum_npz", "exg", "sat", "tex_std", "grad"):
            arr = np.asarray(ch[name], dtype=np.float64)
            m = valid & np.isfinite(arr)
            pos = label & m
            auc = roc_auc(arr[m], pos[m])
            eff = max(auc, 1.0 - auc)
            direction = "+" if auc >= 0.5 else "−"
            mp = float(np.median(arr[pos])) if int(pos.sum()) else float("nan")
            mn = float(np.median(arr[m & ~label])) if int((m & ~label).sum()) else float("nan")
            rows.append(dict(name=name, auc=float(auc), auc_eff=float(eff),
                             direction=direction, median_pos=mp, median_neg=mn,
                             note=note[name]))
            log(f"  {name:<12s}{auc:>8.4f}{eff:>9.4f}{direction:>7s}{mp:>11.4f}{mn:>11.4f}  {note[name]}")

        rows.sort(key=lambda r: r["auc_eff"], reverse=True)
        log("  按有效AUC排序： " + "  ".join(
            f"{r['name']}={r['auc_eff']:.3f}" for r in rows))
        best = rows[0]
        log(f"  → 最优通道 {best['name']}（有效AUC {best['auc_eff']:.4f}，{best['direction']}）"
            + ("  ✅有信号(≥0.60)" if best["auc_eff"] >= 0.60 else "  ❌≈随机(<0.60)"))
        report[site] = dict(grid=[grid["w"], grid["h"]],
                            n_valid=int(valid.sum()), n_corridor=int((label & valid).sum()),
                            channels=rows)

    out = os.path.join(RES, "dom_channel_auc.json")
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"\n保存: {os.path.relpath(out, ROOT)}")
    lf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
