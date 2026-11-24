# -*- coding: utf-8 -*-
"""无监督双分支残差融合：图论拓扑先验 + 预训练 CNN 视觉特征 → Otsu 二值化。

严格约束
--------
* **SHP 真值只在最后算 IoU/F1 时读取**，不参与端点、代价、融合、阈值任何环节。
* 起止点来自外部文件 `data/shu_endpoints.json`（第三方蜀道起止点），不读 mask。
* 分支 2 的 ResNet18 权重冻结、1×1 卷积权重固定为通道均值（等价于 1×1 卷积但无训练），
  因此全流程零梯度、零标签。

流程
----
1. 输入：由 dem_mean 生成的 Hillshade（az=315°, alt=45°），valid 外置 0。
2. 分支 1（宏观拓扑）：cost = 1 − hillshade（阴影处代价低），8 邻域 + 转角惩罚
   的状态增广 Dijkstra，端点取外部文件；路径膨胀 + 高斯 → `F_topo ∈ [0,1]`。
3. 分支 2（微观纹理）：ResNet18(layer2, layer3) 冻结特征，拼接到 1/16 分辨率，
   1×1 卷积压成单通道，上采样回原尺寸 → `F_visual ∈ [0,1]`。
4. 残差融合：`F = F_topo + F_visual`，高斯平滑 + 形态学闭运算。
5. Otsu（仅在 valid 内统计）二值化 → mask；最后读 SHP 算 IoU / F1。

用法
----
    python tools/unsup_two_branch.py
    python tools/unsup_two_branch.py --sites 杨家坝 --dilate 6 --sigma 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, binary_closing, gaussian_filter, label as cc_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import path_extraction as P                                     # noqa: E402
from eval_linear_metrics import window_mask                      # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
SITES = ("杨家坝", "西坝村")
EP_FILE = os.path.join(DATA, "shu_endpoints.json")


# ------------------------------------------------------------ 输入 ----
def load_hillshade(site, az_deg=315.0, alt_deg=45.0, gsd=1.0):
    """由 dem_mean 生成单方向 Hillshade（阴影=值低），返回 (hs, valid)。"""
    d = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    valid = d["valid"] == 1
    dem = d["dem_mean"].astype(np.float64)
    dem = np.where(valid, dem, np.nan)
    # 邻域填充，避免 NaN 扩散到梯度
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    dem = dem[tuple(idx)]
    dzdx = np.gradient(dem, gsd, axis=1)          # 东为正
    dzdy = -np.gradient(dem, gsd, axis=0)         # 北为正（行向下）
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)              # ESRI 约定，自北顺时针
    az = np.radians(az_deg)
    alt = np.radians(alt_deg)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = np.clip(hs, -1.0, 1.0)
    hs = (hs + 1.0) / 2.0
    return np.where(valid, hs, 0.0), valid


# ------------------------------------------------------------ 分支 1 ----
def dijkstra_on(rows, cols, data, N, s, e):
    ss, ee = 8 * N, 8 * N + 1
    d = np.arange(8, dtype=np.int64) * N
    r = np.concatenate([rows, np.full(8, ss, dtype=np.int64), d + int(e)])
    c = np.concatenate([cols, d + int(s), np.full(8, ee, dtype=np.int64)])
    w = np.concatenate([data, np.zeros(16)])
    G = csr_matrix((w, (r, c)), shape=(8 * N + 2, 8 * N + 2))
    dist, pred = dijkstra(G, directed=True, indices=ss, return_predecessors=True)
    if not np.isfinite(dist[ee]):
        return None
    path = np.zeros(N, dtype=bool)
    cur = ee
    while cur >= 0:
        if cur < 8 * N:
            path[cur % N] = True
        if cur == ss:
            break
        cur = int(pred[cur])
    return path


def branch_topo(site, hs, valid, endpoints, dilate=6, turn_lambda=0.2, turn_min=1,
                dir_lambda=0.0, log=print):
    """图论拓扑分支：阴影图上最小代价路径 → 膨胀 → 高斯 → 归一化。"""
    dom = valid.copy()
    H, W = dom.shape
    N = H * W
    cost = np.where(dom, 1.0 - hs, 1.0).astype(np.float64) + 0.05
    nr = np.zeros(dom.shape)
    nc = np.zeros(dom.shape)
    conf = np.zeros(dom.shape, dtype=bool)
    if dir_lambda > 0:
        nr, nc, conf, _ = P.direction_field(site, P.DFS_GATE, smooth=5)
    (rows, cols, data), Nn = P.build_state_graph(cost, dom, nr, nc, conf,
                                                 lam_dir=dir_lambda, lam_turn=turn_lambda,
                                                 turn_min=turn_min)
    meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
    (x0, y0), n0 = endpoints[0]
    (x1, y1), n1 = endpoints[1]
    s, ds = P.snap_report(P.map_to_px(meta, x0, y0), dom)
    e, de = P.snap_report(P.map_to_px(meta, x1, y1), dom)
    path = dijkstra_on(rows, cols, data, Nn, int(s[0] * W + s[1]), int(e[0] * W + e[1]))
    log(f"  分支1 端点 [{n0}]→[{n1}] 吸附 {ds:.0f}/{de:.0f}px；"
        f"路径 {0 if path is None else int(path.sum())} px")
    if path is None:
        return None
    band = binary_dilation(path.reshape(H, W), iterations=dilate) & dom
    f = gaussian_filter(band.astype(np.float64), sigma=3.0)
    f = np.where(dom, f, 0.0)
    mx = f.max()
    return f / mx if mx > 0 else f


# ------------------------------------------------------------ 分支 2 ----
def branch_visual(hs, valid, log=print):
    """ResNet18 冻结特征（layer2+layer3）→ 1×1 固定卷积 → 单通道视觉特征图。"""
    import torch
    import torch.nn.functional as Fn
    from torchvision.models import ResNet18_Weights, resnet18

    H, W = hs.shape
    ph, pw = (-H) % 32, (-W) % 32
    img = np.pad(hs, ((0, ph), (0, pw)), mode="reflect")
    rgb = np.repeat(img[None].astype(np.float32), 3, axis=0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = torch.from_numpy((rgb - mean) / std)[None]

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    feats = {}
    h2 = model.layer2.register_forward_hook(lambda m, i, o: feats.__setitem__("l2", o))
    h3 = model.layer3.register_forward_hook(lambda m, i, o: feats.__setitem__("l3", o))
    with torch.no_grad():
        model(x)
    h2.remove(); h3.remove()

    f2, f3 = feats["l2"], feats["l3"]          # (1,128,H/8,W/8), (1,256,H/16,W/16)
    f2 = Fn.interpolate(f2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
    cat = torch.cat([f2, f3], dim=1)           # (1,384,H/16,W/16)
    c = cat.shape[1]
    w = torch.full((1, c, 1, 1), 1.0 / c)      # 固定 1×1 卷积（通道均值），无训练
    with torch.no_grad():
        g = Fn.conv2d(cat, w, bias=None)
    g = Fn.interpolate(g, size=(H + ph, W + pw), mode="bilinear", align_corners=False)
    g = g[0, 0, :H, :W].cpu().numpy().astype(np.float64)

    g = np.where(valid, g, np.nan)
    lo, hi = np.nanpercentile(g, 1), np.nanpercentile(g, 99)
    g = np.clip((g - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    log(f"  分支2 layer2={tuple(feats['l2'].shape)} layer3={tuple(feats['l3'].shape)} "
        f"→ 1ch（固定 1×1），有效区 P1~P99=[{lo:.3f},{hi:.3f}]")
    return np.nan_to_num(g, nan=0.0)


# ------------------------------------------------------------ 主 ----
def run_site(site, a, log):
    hs, valid = load_hillshade(site)
    log(f"\n[{site}] hillshade {hs.shape} 有效 {int(valid.sum()):,}px "
        f"范围 [{hs[valid].min():.3f},{hs[valid].max():.3f}]")

    eps = P.load_endpoints(a.endpoints, site)
    if eps is None:
        log("  [FAIL] 端点文件缺该站点")
        return None
    ft = branch_topo(site, hs, valid, eps, dilate=a.dilate,
                     turn_lambda=a.turn_lambda, turn_min=a.turn_min,
                     dir_lambda=a.dir_lambda, log=log)
    if ft is None:
        log("  [FAIL] 分支1 无路径")
        return None
    fv = branch_visual(hs, valid, log=log)

    fused = ft + fv                                           # 残差融合
    if a.fuse_norm == "zscore":
        # 诊断模式：两分支在 valid 内 z-score 后再相加，消除"稀疏 vs 稠密"的量级失衡
        def _z(x):
            m = x[valid]
            return np.where(valid, (x - m.mean()) / (m.std() + 1e-9), 0.0)
        ft, fv = _z(ft), _z(fv)
        fused = ft + fv
    fused = gaussian_filter(fused, sigma=a.sigma)
    fused = np.where(valid, fused, 0.0)

    thr = float(threshold_otsu(fused[valid]))                 # Otsu，仅 valid 内统计
    mask = (fused >= thr) & valid
    if a.close > 0:
        mask = binary_closing(mask, iterations=a.close) & valid   # 形态学闭运算去噪

    label = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool)  # 仅此处读真值
    win = window_mask(valid)

    def sc(m, sel):
        inter = int((m & label & sel).sum())
        union = int(((m | label) & sel).sum())
        tp = inter
        fp = int((m & ~label & sel).sum())
        fn = int((~m & label & sel).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return dict(iou=float(inter / union) if union else 0.0, f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
                    precision=float(prec), recall=float(rec), pred_px=int((m & sel).sum()))

    raw = sc(mask, valid)
    d10 = sc(mask, valid & win)
    log(f"  分支1 均值 {ft[valid].mean():.3f} | 分支2 均值 {fv[valid].mean():.3f} | "
        f"融合 Otsu 阈值 {thr:.3f}")
    log(f"  → 预测 {raw['pred_px']:,}px  IoU(全域)={raw['iou']:.4f} F1={raw['f1']:.4f} | "
        f"IoU(D10)={d10['iou']:.4f} F1={d10['f1']:.4f}  "
        f"P={d10['precision']:.3f} R={d10['recall']:.3f}")
    log(f"     对照：仅分支1 IoU≈0.00~0.01（路线级端点）；SHP 端点图论上限 0.2791/0.2231")

    os.makedirs(FIG, exist_ok=True)
    np.save(os.path.join(RES, f"unsup_fused_{site}.npy"), fused.astype(np.float32))
    np.save(os.path.join(RES, f"unsup_mask_{site}.npy"), mask)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4), dpi=130)
    ax[0].imshow(np.where(valid, hs, np.nan), cmap="gray"); ax[0].set_title("Hillshade")
    ax[1].imshow(np.where(valid, ft, np.nan), cmap="viridis"); ax[1].set_title("F_topo")
    ax[2].imshow(np.where(valid, fv, np.nan), cmap="viridis"); ax[2].set_title("F_visual")
    ax[3].imshow(np.where(valid, fused, np.nan), cmap="magma")
    ax[3].contour(label, levels=[0.5], colors="cyan", linewidths=1.2)
    ax[3].contour(mask.astype(float), levels=[0.5], colors="red", linewidths=1.2,
                  linestyles="--")
    ax[3].set_title(f"F_fused + Otsu (IoU={d10['iou']:.3f})")
    ax[3].legend(handles=[Line2D([], [], color="cyan", lw=1.6, label="SHP 真值"),
                          Line2D([], [], color="red", lw=1.6, ls="--", label="预测掩膜")],
                 loc="lower right", fontsize=7)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{site} · 无监督双分支残差融合"
                 f"（端点来自 {os.path.basename(a.endpoints)}，SHP 仅用于评分）", fontsize=10)
    fig.tight_layout()
    fp = os.path.join(FIG, f"unsup_two_branch_{site}.png")
    fig.savefig(fp); plt.close(fig)
    log(f"  图: {os.path.relpath(fp, ROOT)}")

    return dict(site=site, otsu=thr, dilate=a.dilate, sigma=a.sigma,
                turn_lambda=a.turn_lambda, endpoints=a.endpoints,
                raw=raw, d10=d10, ft_mean=float(ft[valid].mean()),
                fv_mean=float(fv[valid].mean()), figure=os.path.relpath(fp, ROOT))


def main():
    p = argparse.ArgumentParser(description="无监督双分支残差融合（SHP 仅用于评分）")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--endpoints", type=str, default=EP_FILE)
    p.add_argument("--dilate", type=int, default=6, help="拓扑分支膨胀半径 px")
    p.add_argument("--sigma", type=float, default=2.0, help="融合后高斯平滑 σ")
    p.add_argument("--close", type=int, default=1, help="形态学闭运算迭代次数（0=关）")
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    p.add_argument("--dir-lambda", type=float, default=0.0, help="方向先验（默认关）")
    p.add_argument("--fuse-norm", type=str, default="01", choices=["01", "zscore"],
                   help="融合前归一化：01=各分支归一到[0,1]（原设定）；"
                        "zscore=valid 内 z-score（诊断量级失衡）")
    a = p.parse_args()

    lp = os.path.join(RES, "unsup_two_branch.log")
    lf = open(lp, "w", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n"); lf.flush()

    log("无监督双分支残差融合 · SHP 仅用于最终 IoU/F1")
    log(f"端点文件：{a.endpoints}（第三方蜀道起止点）  dilate={a.dilate}px σ={a.sigma} "
        f"close={a.close} λ_turn={a.turn_lambda} m={a.turn_min}")
    out = []
    for s in a.sites:
        r = run_site(s, a, log)
        if r:
            out.append(r)

    log("\n" + "=" * 74)
    log(f"{'站点':>8s} {'Otsu':>7s} {'预测px':>9s} {'IoU':>8s} {'F1':>8s} {'P':>7s} {'R':>7s}")
    log("-" * 62)
    for r in out:
        d = r["d10"]
        log(f"{r['site']:>8s} {r['otsu']:>7.3f} {d['pred_px']:>9,d} "
            f"{d['iou']:>8.4f} {d['f1']:>8.4f} {d['precision']:>7.3f} {d['recall']:>7.3f}")

    jp = os.path.join(RES, "unsup_two_branch.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="无监督双分支残差融合（图论拓扑 + ResNet18 视觉，Otsu）",
                       endpoints=a.endpoints, dilate=a.dilate, sigma=a.sigma,
                       close=a.close, turn_lambda=a.turn_lambda, turn_min=a.turn_min,
                       dir_lambda=a.dir_lambda, sites=out), f,
                  ensure_ascii=False, indent=2)
    log(f"\n保存: {jp}")
    lf.close()


if __name__ == "__main__":
    main()
