# -*- coding: utf-8 -*-
"""线状地物 2D 评估：SHP 面 → 骨架中轴线 → 2 px 等宽带真值 → 预测掩膜 Buffer IoU。

流程：
    1) 读 SHP 面要素，栅格化到 1 m 网格
    2) skeletonize 提取中轴线（1 px 宽）
    3) 中轴线膨胀 `--gt-dilate` 像元（默认 2 px = 2 m）→ 宽度统一的 GT_Line
    4) 收窄交付点云（results/narrow_{site}_corridor.las，即 P≥0.5 ∧ 3D 轴距≤3 m）
       投影到同一 1 m 栅格 → 预测掩膜
    5) Buffer IoU：双方各膨胀 `--tol` 像元（默认 1 px）后求交并
    6) 打印中心线 IoU

输出：
    results/eval_line_iou_{site}.json
    figures/eval_line_iou_{site}.png（--fig 时）

用法：
    python tools/eval_line_iou.py --site 杨家坝
    python tools/eval_line_iou.py --site 杨家坝 --pred p05 --tol 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import RES, ROOT, SHP_BASE, SHP_NEW, parts_of, poly_mask, read_shp_any  # noqa: E402

FIG = os.path.join(ROOT, "figures")


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(2 * p * r / max(p + r, 1e-9)), 4),
            "iou": round(float(tp / max(tp + fp + fn, 1)), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def pred_from_las(path, x0, y1, gsd, H, W):
    """点云 → 1 m 占用掩膜。"""
    import laspy
    occ = np.zeros((H, W), dtype=bool)
    n = 0
    with laspy.open(path) as f:
        for pts in f.chunk_iterator(3_000_000):
            X = np.asarray(pts.x, dtype=np.float64)
            Y = np.asarray(pts.y, dtype=np.float64)
            c = np.floor((X - x0) / gsd).astype(np.int64)
            r = np.floor((y1 - Y) / gsd).astype(np.int64)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            occ[r[ok], c[ok]] = True
            n += X.size
    return occ, n


def pred_from_prob(path, H, W):
    z = np.load(path)
    p = z["prob"]
    v = (z["valid"] == 1) if "valid" in z.files else (p > 0)
    return (p >= 0.5) & v


def poly_mask_hr(shapes, x0, y1, gsd, H, W, ss):
    """面要素 → 高分辨率布尔栅格（ss 倍），用于骨架化。"""
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    img = Image.new("1", (W * ss, H * ss), 0)
    dr = ImageDraw.Draw(img)
    for sh in shapes:
        for xy in parts_of(sh):
            cols = (xy[:, 0] - x0) / gsd * ss
            rows = (y1 - xy[:, 1]) / gsd * ss
            dr.polygon([(float(c), float(r)) for c, r in zip(cols, rows)], fill=1)
    # 注意：skimage 的 skeletonize 在本机对 bool 大数组会段错误，这里返回 uint8 连续数组
    return np.ascontiguousarray(np.array(img, dtype=np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="杨家坝")
    ap.add_argument("--pred", default="narrow", choices=["narrow", "p05"])
    ap.add_argument("--gt-dilate", type=int, default=2, help="GT 中轴线膨胀像元数（1 px = 1 m）")
    ap.add_argument("--tol", type=int, default=1, help="Buffer IoU 的互膨胀容差（像元）")
    ap.add_argument("--skel-ss", type=int, default=4,
                    help="骨架化超采样倍率（4 → 0.25 m 上提骨架，再降到 1 m；1 = 直接 1 m）")
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()
    site = a.site

    from scipy.ndimage import binary_dilation, label
    from skimage.morphology import skeletonize

    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1

    # ---- 1/2/3) SHP 面 → 骨架 → 等宽带真值
    _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
    pmask = poly_mask(polys, x0, y1, gsd, H, W) & valid
    ss = max(int(a.skel_ss), 1)
    if ss == 1:
        skel_raw = skeletonize(pmask)
    else:
        hr = poly_mask_hr(polys, x0, y1, gsd, H, W, ss)
        skel_raw = skeletonize(hr)[:H * ss, :W * ss].reshape(H, ss, W, ss).any(axis=(1, 3))
        skel_raw &= valid
    skel = skel_raw
    gt = binary_dilation(skel, iterations=a.gt_dilate) if a.gt_dilate else skel
    lab, ncc = label(skel)
    comp = [(int((lab == k).sum())) for k in range(1, ncc + 1)]
    print(f"[{site}] 网格 {H}×{W} @ {gsd} m，SHP 面 {len(polys)} 个要素")
    print(f"  GT：面 {int(pmask.sum()):,} px → 骨架（{gsd/ss:g} m 上提）{int(skel.sum()):,} px，"
          f"{ncc} 段，最长 {max(comp) if comp else 0} px → 膨胀 {a.gt_dilate} px "
          f"带宽 {int(gt.sum()):,} px")

    # ---- 4) 预测掩膜
    if a.pred == "narrow":
        las = os.path.join(RES, f"narrow_{site}_corridor.las")
        if not os.path.exists(las):
            print(f"  [错误] 缺少 {las}，先跑 tools/line_metrics_narrow.py --site {site}")
            return 2
        pred, npt = pred_from_las(las, x0, y1, gsd, H, W)
        print(f"  PRED：{npt:,} 点 → 占用 {int(pred.sum()):,} px（{las.split(os.sep)[-1]}）")
    else:
        pred = pred_from_prob(os.path.join(RES, f"cold_start_{site}_ep.npz"), H, W)
        print(f"  PRED：P≥0.5 栅格 {int(pred.sum()):,} px")

    # ---- 5) Buffer IoU
    pep = binary_dilation(pred, iterations=a.tol) if a.tol else pred
    gep = binary_dilation(gt, iterations=a.tol) if a.tol else gt
    raw = prf(int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum()))
    buf = prf(int((pep & gep).sum()), int((pep & ~gep).sum()), int((~pep & gep).sum()))
    print(f"  原始 IoU：{raw['iou']:.4f}（P {raw['precision']:.4f} / R {raw['recall']:.4f} / "
          f"F1 {raw['f1']:.4f}）")
    print(f"  中心线 Buffer IoU（互膨胀 {a.tol} px）：{buf['iou']:.4f}"
          f"（P {buf['precision']:.4f} / R {buf['recall']:.4f} / F1 {buf['f1']:.4f}）")
    print(f"  覆盖率：GT 带被预测覆盖 {raw['recall']:.1%}；"
          f"预测落在 GT 带内 {raw['precision']:.1%}")

    res = {"site": site, "grid": [H, W], "gsd_m": gsd, "pred": a.pred,
           "gt": {"poly_px": int(pmask.sum()), "skeleton_px": int(skel.sum()),
                  "components": ncc, "component_px": sorted(comp, reverse=True),
                  "gt_dilate_px": a.gt_dilate, "band_px": int(gt.sum())},
           "pred_px": int(pred.sum()), "tol_px": a.tol,
           "raw": raw, "buffer_iou": buf,
           "centerline_iou": buf["iou"]}
    out = os.path.join(RES, f"eval_line_iou_{site}_{a.pred}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 {os.path.relpath(out, ROOT)}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        ys, xs = np.where(pred | gt)
        if ys.size:
            y0, y1_ = max(ys.min() - 20, 0), min(ys.max() + 20, H)
            c0, c1 = max(xs.min() - 20, 0), min(xs.max() + 20, W)
        else:
            y0, y1_, c0, c1 = 0, H, 0, W
        sl = (slice(y0, y1_), slice(c0, c1))
        rgb = np.zeros((y1_ - y0, c1 - c0, 3), dtype=np.float32)
        rgb[..., 0] = pred[sl] * 1.0          # 红 = 预测
        rgb[..., 1] = gt[sl] * 1.0            # 绿 = GT 带
        rgb[..., 2] = skel[sl] * 0.8          # 蓝 = 真值骨架
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{site}：红=预测掩膜  绿=GT 带（骨架 ±{a.gt_dilate} px）  蓝=骨架\n"
                     f"Buffer IoU({a.tol} px) = {buf['iou']:.4f}")
        ax.axis("off")
        fig.tight_layout()
        p = os.path.join(FIG, f"eval_line_iou_{site}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  写出 figures/{os.path.basename(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
