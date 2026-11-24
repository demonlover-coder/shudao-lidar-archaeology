# -*- coding: utf-8 -*-
"""P2 任务 4/5/6：在规则网格上评估 HashGrid+MLP 的 DEM 表征。

主判据（经用户确认）：**同站自验**（杨训杨验 / 西训西验）——纯坐标网络只有 2 个
输入，跨 35 km、高差 300 m 的两站之间不存在可迁移的地形信息，跨站只作泛化对照。

评估域 = 源 DEM 的 valid 像元 ∩ 落在**训练站**点云包围盒内的像元
（归一化坐标只在训练站 bbox 内有意义，域外是纯外推）。
另报 "域内且被地面点覆盖" 的子集，用于区分"模型误差"与"该处本来就没点"。

用法：
    python tools/eval_p2.py --train-site 杨家坝 --eval-site 杨家坝
    python tools/eval_p2.py --all                       # 自验两站 + 跨站双向
    python tools/eval_p2.py --all --multi-res           # 加 0.5m/0.25m 与三联图
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_data import BASE  # noqa: E402
from build_dataset import SITE_FILES  # noqa: E402
from hashgrid import HashGridDEM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# IDW 基线（results/baseline_regression.json，合成空洞率 72%，1m 网格）
IDW_BASELINE = {"杨家坝→西坝村": 0.577, "西坝村→杨家坝": 0.435}


def pick_device() -> str:
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


# ------------------------------------------------------------------ 网格

def grid_from_1m(site: str):
    """现有 1m 产物：dem_mean + valid（primary 判据口径）。"""
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
    return d["dem_mean"].astype(np.float32), d["valid"].astype(np.uint8), \
        float(meta["x0"]), float(meta["y1"]), 1.0


def grid_from_source(site: str, res: float):
    """从源 DEM 用 BOX（面积平均）重采样到 res，同时把 valid 掩膜一起降采样。

    用 PIL 的 BOX 滤波器做真面积平均，避免 bilinear/cubic 的混叠
    （契约 D3）；对 0.1m→0.25m 这种非整数倍率也成立。
    """
    f = SITE_FILES[site]["dem"]
    path = os.path.join(BASE, f)
    with Image.open(path) as im:
        tags = dict(im.tag_v2)
        src_gsd = float(tags[33550][0])
        tp = tags[33922]
        x0, y1 = float(tp[3]), float(tp[4])
        nod = float(str(tags.get(42113, "-9999")).strip("\x00") or -9999)
        ow = int(round(im.width * src_gsd / res))
        oh = int(round(im.height * src_gsd / res))
        arr = np.asarray(im, dtype=np.float32)
    val = (arr != nod).astype(np.float32)
    arr = np.where(arr == nod, np.float32(0.0), arr)
    a = np.asarray(Image.fromarray(arr).resize((ow, oh), Image.BOX), dtype=np.float32)
    v = np.asarray(Image.fromarray(val).resize((ow, oh), Image.BOX), dtype=np.float32)
    valid = (v >= 0.5).astype(np.uint8)
    mean = np.where(valid == 1, a / np.maximum(v, 1e-6), 0.0).astype(np.float32)
    return mean, valid, x0, y1, res


def grid_axes(x0: float, y1: float, res: float, h: int, w: int):
    """像元中心 UTM 坐标（x 向右、y 向下，与 GeoTIFF tiepoint 约定一致）。"""
    xs = x0 + (np.arange(w, dtype=np.float64) + 0.5) * res
    ys = y1 - (np.arange(h, dtype=np.float64) + 0.5) * res
    return xs, ys


# ------------------------------------------------------------------ 推理

@torch.no_grad()
def predict_utm(model, bbox, X: np.ndarray, Y: np.ndarray, dev: str,
                batch: int = 1 << 20) -> np.ndarray:
    """UTM (X,Y) → 预测高程（米）。bbox 来自**训练站**（归一化空间的定义域）。"""
    lo, hi = bbox[0], bbox[1]
    nx = (X - lo[0]) / (hi[0] - lo[0])
    ny = (Y - lo[1]) / (hi[1] - lo[1])
    zmin, zmax = lo[2], hi[2]
    out = np.empty(X.shape, dtype=np.float32)
    flat_x, flat_y, flat_o = nx.ravel(), ny.ravel(), out.ravel()
    model.eval()
    for i in range(0, len(flat_x), batch):
        t = torch.from_numpy(np.stack([flat_x[i:i + batch], flat_y[i:i + batch]], 1)).float().to(dev)
        flat_o[i:i + batch] = (zmin + model(t)[:, 0] * (zmax - zmin)).float().cpu().numpy()
    return out


_SITE_CONST_CACHE: dict = {}


def site_const(site: str) -> float:
    """训练站地面点的高程均值（UTM 米），作为"常数预测"基线。"""
    if site in _SITE_CONST_CACHE:
        return _SITE_CONST_CACHE[site]
    d = np.load(os.path.join(DATA, f"p2_{site}_points_ground.npz"))
    xyz, bb = d["xyz"], d["bbox"].astype(np.float64)
    z = bb[0, 2] + xyz[:, 2].astype(np.float64) * (bb[1, 2] - bb[0, 2])
    c = float(z.mean())
    _SITE_CONST_CACHE[site] = c
    return c


def covered_mask(site: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """网格像元是否被该站的**地面点**覆盖（用于区分模型误差与无点区）。"""
    d = np.load(os.path.join(DATA, f"p2_{site}_points_ground.npz"))
    xyz, bb = d["xyz"], d["bbox"]
    X = bb[0, 0] + xyz[:, 0] * (bb[1, 0] - bb[0, 0])
    Y = bb[0, 1] + xyz[:, 1] * (bb[1, 1] - bb[0, 1])
    res = float(xs[1] - xs[0])
    ci = np.round((X - xs[0]) / res).astype(np.int64)
    ri = np.round((ys[0] - Y) / res).astype(np.int64)
    ok = (ci >= 0) & (ci < len(xs)) & (ri >= 0) & (ri < len(ys))
    m = np.zeros((len(ys), len(xs)), dtype=bool)
    m[ri[ok], ci[ok]] = True
    return m


# ------------------------------------------------------------------ 评估

def eval_one(train_site: str, eval_site: str, res: float, dev: str,
             ckpt: dict, verbose: bool = True) -> dict:
    model = HashGridDEM(**ckpt["cfg"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    bbox = np.asarray(ckpt["bbox"], dtype=np.float64)

    gt, valid, x0, y1, res = (grid_from_1m(eval_site) if res == 1.0
                              else grid_from_source(eval_site, res))
    h, w = gt.shape
    xs, ys = grid_axes(x0, y1, res, h, w)
    XX, YY = np.meshgrid(xs, ys)
    pred = predict_utm(model, bbox, XX, YY, dev)

    in_box = ((XX >= bbox[0, 0]) & (XX <= bbox[1, 0]) &
              (YY >= bbox[0, 1]) & (YY <= bbox[1, 1]))
    dom_box = (valid == 1) & in_box                       # 主评估域
    # 跨站时两站 bbox 相距 35 km，交集为空 → 退化为整幅 valid 区（纯外推），
    # 必须显式标注，不能当作"同口径"数字使用。
    extrap = not dom_box.any()
    dom = (valid == 1) if extrap else dom_box
    if not dom.any():
        return {"train_site": train_site, "eval_site": eval_site, "res_m": res,
                "rmse_m": float("nan"), "n": 0, "note": "评估域为空"}

    err = pred - gt
    rmse = float(np.sqrt((err[dom] ** 2).mean()))
    mae = float(np.abs(err[dom]).mean())
    bias = float(err[dom].mean())

    # 子集：域内且被地面点覆盖（区分模型误差与"该处本来就没点"）
    cov = covered_mask(eval_site, xs, ys)
    sub = dom & cov
    rmse_c = float(np.sqrt((err[sub] ** 2).mean())) if sub.any() else float("nan")

    # 对照：用训练站的站点常数预测（跨站失败的天然参照物）
    c = site_const(train_site)
    const_rmse = float(np.sqrt(((c - gt[dom]) ** 2).mean()))

    out = {"train_site": train_site, "eval_site": eval_site, "res_m": res,
           "extrapolation": bool(extrap),
           "n": int(dom.sum()), "rmse_m": rmse, "mae_m": mae, "bias_m": bias,
           "n_covered": int(sub.sum()),
           "rmse_covered_m": rmse_c,
           "rmse_covered_pct": round(100 * rmse_c / rmse, 1) if np.isfinite(rmse_c) else None,
           "const_baseline_rmse_m": const_rmse,
           "const_value_m": c}
    if verbose:
        tag = "同站自验" if train_site == eval_site else "跨站"
        extra = "  [外推：两站 bbox 无交集]" if extrap else ""
        print(f"  {tag} {train_site}→{eval_site} @{res}m: "
              f"RMSE={rmse:.3f} m  MAE={mae:.3f}  bias={bias:+.3f}  "
              f"n={dom.sum():,}  覆盖子集 RMSE={rmse_c:.3f} m ({sub.sum():,})  "
              f"站点常数基线={const_rmse:.3f} m{extra}")
    return out


def stratified(train_site: str, eval_site: str, dev: str, ckpt: dict) -> list:
    """1m 网格上的坡度分层 RMSE（沿用 D10 的分层边界口径）。"""
    model = HashGridDEM(**ckpt["cfg"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    bbox = np.asarray(ckpt["bbox"], dtype=np.float64)
    gt, valid, x0, y1, res = grid_from_1m(eval_site)
    h, w = gt.shape
    xs, ys = grid_axes(x0, y1, res, h, w)
    XX, YY = np.meshgrid(xs, ys)
    pred = predict_utm(model, bbox, XX, YY, dev)
    d = np.load(os.path.join(DATA, f"{eval_site}_1m.npz"))
    slope = d["slope_deg"]
    dom = (valid == 1) & (XX >= bbox[0, 0]) & (XX <= bbox[1, 0]) & \
          (YY >= bbox[0, 1]) & (YY <= bbox[1, 1])
    p = np.percentile(slope[dom], [20, 40, 60, 80])
    edges = [0, p[0], p[1], p[2], p[3], 90]
    rows = []
    for i in range(5):
        m = dom & (slope >= edges[i]) & (slope < edges[i + 1])
        if m.sum() == 0:
            continue
        rows.append({"layer": f"P{20*(i+1)}", "slope": [round(float(edges[i]), 1),
                                                        round(float(edges[i + 1]), 1)],
                     "n": int(m.sum()),
                     "rmse_m": round(float(np.sqrt(((pred - gt)[m] ** 2).mean())), 3)})
    return rows


# ------------------------------------------------------------------ 图

def multires_figure(site: str, ckpt: dict, dev: str, out_png: str,
                    win_m: int = 160) -> dict:
    """三联图：同一窗口的 1m / 0.5m / 0.25m 预测。用于伪影目检。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bbox = np.asarray(ckpt["bbox"], dtype=np.float64)
    model = HashGridDEM(**ckpt["cfg"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    gt1, val1, x0, y1, _ = grid_from_1m(site)
    # 找走廊最密的位置作为窗口中心
    lab = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    yy, xx = np.where(lab["label"] == 1) if "label" in lab.files else (np.array([0]), np.array([0]))
    cr = int(np.median(yy)) if len(yy) else gt1.shape[0] // 2
    cc = int(np.median(xx)) if len(xx) else gt1.shape[1] // 2
    cx = x0 + (cc + 0.5)
    cy = y1 - (cr + 0.5)

    panels = []
    for res in (1.0, 0.5, 0.25):
        if res == 1.0:
            gt, valid = gt1, val1
        else:
            gt, valid, _, _, _ = grid_from_source(site, res)
        h, w = gt.shape
        xs, ys = grid_axes(x0, y1, res, h, w)
        # 窗口
        ci = np.where((xs >= cx - win_m / 2) & (xs <= cx + win_m / 2))[0]
        ri = np.where((ys >= cy - win_m / 2) & (ys <= cy + win_m / 2))[0]
        if len(ci) == 0 or len(ri) == 0:
            ci, ri = np.arange(min(w, int(win_m / res))), np.arange(min(h, int(win_m / res)))
        XX, YY = np.meshgrid(xs[ci], ys[ri])
        pred = predict_utm(model, bbox, XX, YY, dev)
        xmin, xmax = bbox[0, 0], bbox[1, 0]
        ymin, ymax = bbox[0, 1], bbox[1, 1]
        inb = (XX >= xmin) & (XX <= xmax) & (YY >= ymin) & (YY <= ymax)
        dom = inb & (valid[ri][:, ci] == 1)
        if dom.sum() > 0:
            e = pred - gt[ri][:, ci]
            r = float(np.sqrt((e[dom] ** 2).mean()))
        else:
            r = float("nan")
        panels.append((res, pred, r))
        print(f"    窗口 {res}m: {pred.shape}  窗口内 RMSE={r:.3f} m")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    vmin = min(np.nanpercentile(p[1], 1) for p in panels)
    vmax = max(np.nanpercentile(p[1], 99) for p in panels)
    for ax, (res, pred, r) in zip(axes, panels):
        im = ax.imshow(pred, cmap="terrain", vmin=vmin, vmax=vmax)
        ax.set_title(f"{site} 预测 @ {res} m  ({pred.shape[0]}×{pred.shape[1]})\n"
                     f"窗口 RMSE={r:.3f} m", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"P2 HashGrid+MLP 多分辨率输出（同一 {win_m}m 窗口）", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return {f"{p[0]}m": p[2] for p in panels}


# ------------------------------------------------------------------ CLI

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-site", default=None)
    ap.add_argument("--eval-site", default=None)
    ap.add_argument("--all", action="store_true", help="自验两站 + 跨站双向")
    ap.add_argument("--multi-res", action="store_true", help="跑 0.5m/0.25m 并存三联图")
    ap.add_argument("--res", type=float, nargs="*", default=None)
    a = ap.parse_args()
    dev = pick_device()
    print(f"设备: {dev}")

    if a.all:
        pairs = [("杨家坝", "杨家坝"), ("西坝村", "西坝村"),
                 ("杨家坝", "西坝村"), ("西坝村", "杨家坝")]
    else:
        pairs = [(a.train_site, a.eval_site)]

    ckpts = {}
    for tr, _ in pairs:
        p = os.path.join(CKPT, f"p2_{tr}.pth")
        if not os.path.exists(p):
            raise SystemExit(f"缺少权重 {p}，先跑 tools/train_p2.py --site {tr}")
        if tr not in ckpts:
            ckpts[tr] = torch.load(p, map_location="cpu", weights_only=False)

    results = {"self": {}, "cross": {}, "stratified": {}, "figures": {}}
    res_list = ([1.0] + (a.res if a.res else [0.5, 0.25])) if a.multi_res else \
               (a.res if a.res else [1.0])

    def key_of(tr: str, ev: str) -> str:
        return "S1" if (tr, ev) == ("杨家坝", "西坝村") else \
               "S2" if (tr, ev) == ("西坝村", "杨家坝") else f"self_{tr}"

    for tr, ev in pairs:
        key = key_of(tr, ev)
        bucket = results["self"] if tr == ev else results["cross"]
        print(f"\n{'='*72}\n{key}: {tr} → {ev}")
        bucket[key] = {}
        for res in res_list:
            tag = "1m" if res == 1.0 else f"{res}m"
            bucket[key][tag] = eval_one(tr, ev, res, dev, ckpts[tr])
        if tr == ev:
            results["stratified"][key] = stratified(tr, ev, dev, ckpts[tr])
            print("    坡度分层 RMSE（1m）:")
            for row in results["stratified"][key]:
                print(f"      {row['layer']:>4} [{row['slope'][0]:>5},{row['slope'][1]:>5}°): "
                      f"RMSE={row['rmse_m']:.3f} m  n={row['n']:,}")
            if a.multi_res:
                png = os.path.join(FIG, f"p2_multires_{ev}.png")
                results["figures"][ev] = multires_figure(tr, ckpts[tr], dev, png)
                print(f"    三联图 figures/{os.path.basename(png)}")

    out = os.path.join(RES, "p2_cross_site.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n{'='*72}")
    print("P2 结果汇总（主判据 = 同站自验；IDW 基线 0.577 m(杨) / 0.435 m(西)）")
    print(f"{'方向':<14}{'1m':>9}{'0.5m':>9}{'0.25m':>9}   判定")
    for bucket, label in ((results["self"], "自验"), (results["cross"], "跨站对照")):
        for k, v in bucket.items():
            cells = "".join(f"{v.get(t, {}).get('rmse_m', float('nan')):>9.3f}"
                            for t in ("1m", "0.5m", "0.25m"))
            r1 = v.get("1m", {}).get("rmse_m", float("nan"))
            if bucket is results["self"]:
                vs = "≤1.0 → 继续" if r1 <= 1.0 else (">1.5 → 停" if r1 > 1.5 else "1.0~1.5 待定")
            else:
                vs = "泛化对照"
            print(f"{k:<14}{cells}   {vs}")
    print(f"\n保存 results/{os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
