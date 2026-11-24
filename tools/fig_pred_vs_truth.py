# -*- coding: utf-8 -*-
"""预测 vs 真值 可视化诊断图。

每站出一张 2×3 大图，直接把"差在哪"画出来：
  (a) 全局叠加：真值面（灰）+ 真值中轴（绿）+ 模型轴（青）+ 交付骨架（红）
  (b) 概率场 P + 真值中轴 + 模型轴 —— 看概率覆盖是否偏离
  (c) FP 点（保留但不在真值内的地面点）—— 横向漂移证据
  (d) FN 点（真值地面点但未保留）
  (e)(f) 放大窗：FP / FN 最密集处的 80 m 窗口细节

用法：
    python tools/fig_pred_vs_truth.py --site 西坝村 --truth new --truth-main
    python tools/fig_pred_vs_truth.py --site 杨家坝 --truth old
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, uniform_filter
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_new_shp import LAS, RES, SHP_BASE, SHP_NEW  # noqa: E402
from eval_freeaxis import axis_field, fill_nodata, poly_mask_hr  # noqa: E402
from p2_adaptive_gsd import FIG  # noqa: E402

CH = 4_000_000
GROUND = 0.3


def load_axis(npz, prob, valid, H, W):
    z = np.load(npz)
    if "path" in z.files:
        a = z["path"]
        if a.ndim == 2:
            return a.astype(bool)
        ax = np.zeros((H, W), bool)      # N×2 点列 → 栅格化
        ax[a[:, 0].astype(int), a[:, 1].astype(int)] = True
        return ax
    return skeletonize(prob >= 0.5)


def order_path(mask, min_len=20):
    """把 1 px 宽的轴线掩膜排成有序折线（双 BFS 取直径端点后回溯）。"""
    from collections import deque
    pts = np.argwhere(mask)
    if pts.shape[0] < min_len:
        return None
    S = set(map(tuple, pts))

    def nbrs(p):
        r, c = p
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and (r + dr, c + dc) in S]

    def bfs(s):
        prev = {s: None}
        q = deque([s])
        last = s
        while q:
            u = q.popleft()
            last = u
            for v in nbrs(u):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        return last, prev

    a_, _ = bfs(next(iter(S)))
    b_, prev = bfs(a_)
    out = [b_]
    while prev[out[-1]] is not None:
        out.append(prev[out[-1]])
    return np.array(out[::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--npz", default="")
    ap.add_argument("--truth", default="new", choices=("new", "old"))
    ap.add_argument("--truth-main", action="store_true")
    ap.add_argument("--dist3", type=float, default=3.5)
    ap.add_argument("--agl", type=float, default=2.0)
    ap.add_argument("--zoom", type=int, default=80, help="放大窗边长（米）")
    a = ap.parse_args()

    site = a.site
    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem = fill_nodata(d["dem_min"], valid)
    npz = a.npz or f"cold_start_{site}_lc6.npz"
    if not os.path.isabs(npz):
        npz = os.path.join(RES, npz)
    prob = np.where(valid, np.load(npz)["prob"].astype(np.float64), 0.0)

    if a.truth == "old":
        gt = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
    else:
        from eval_new_shp import read_shp_any
        SS = 4
        _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        mh = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
        gt = mh.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5
        gt &= valid
    if a.truth_main:
        from scipy.ndimage import label as ccl
        lb, n = ccl(gt)
        if n > 1:
            sz = np.bincount(lb.ravel())
            sz[0] = 0
            gt = lb == sz.argmax()
    gt_skel = skeletonize(gt)
    axis = load_axis(npz, prob, valid, H, W)
    d2, zax = axis_field(site, prob, valid, H, W, gsd, x0, y1, None, dem, "path", npz)

    # ---- 单次流式扫描：统计每像元的地面点，并收集 FP/FN 坐标 ----
    import laspy
    gt_g = np.zeros((H, W), np.int32)      # 真值地面点
    gt_g_keep = np.zeros((H, W), np.int32)  # 真值地面点且被保留
    fp_c = np.zeros((H, W), np.int32)       # 非真值地面点却被保留
    occ = np.zeros((H, W), bool)
    FP, FN, FN_LOW = [], [], []
    low_p_x, low_p_y = [], []
    with laspy.open(LAS[site]) as f:
        for pts in f.chunk_iterator(CH):
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            Z = np.asarray(pts.z, np.float64)
            c = np.floor((X - x0) / gsd).astype(np.int64)
            r = np.floor((y1 - Y) / gsd).astype(np.int64)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            if not ok.any():
                continue
            X, Y, Z, c, r = X[ok], Y[ok], Z[ok], c[ok], r[ok]
            pv = prob[r, c]
            agl_all = Z - dem[r, c]
            tg_all = gt[r, c] & (agl_all <= GROUND)
            # 真值地面点但像元 P<0.5（未进候选）—— 主要漏检来源
            ml = tg_all & (pv < 0.5)
            if ml.any():
                low_p_x.append(X[ml])
                low_p_y.append(Y[ml])
            cand = pv >= 0.5
            if not cand.any():
                continue
            X, Y, Z, c, r, pv = X[cand], Y[cand], Z[cand], c[cand], r[cand], pv[cand]
            agl = Z - dem[r, c]
            d3 = np.hypot(d2[r, c], Z - zax[r, c])
            keep = (d3 <= a.dist3) & (agl <= a.agl)
            ground = agl <= GROUND
            tr = gt[r, c]
            # 每像元计数
            np.add.at(gt_g, (r, c), (tr & ground).astype(np.int32))
            np.add.at(gt_g_keep, (r, c), (tr & ground & keep).astype(np.int32))
            np.add.at(fp_c, (r, c), (~tr & ground & keep).astype(np.int32))
            occ[r[keep], c[keep]] = True
            if (keep & ground & ~tr).any():
                m = keep & ground & ~tr
                FP.append(np.stack([X[m], Y[m]], 1))
            if (tr & ground & ~keep).any():
                m = tr & ground & ~keep
                FN.append(np.stack([X[m], Y[m]], 1))

    fp_n, fn_n = int(fp_c.sum()), int((gt_g - gt_g_keep).sum())
    print(f"[{site}] FP {fp_n:,}  FN {fn_n:,}")
    FP = np.concatenate(FP) if FP else np.zeros((0, 2))
    FN = np.concatenate(FN) if FN else np.zeros((0, 2))
    sk_del = skeletonize(binary_closing(occ, np.ones((3, 3), bool)))
    FNc = gt_g - gt_g_keep
    LO = (np.concatenate([np.stack([x, y], 1) for x, y in zip(low_p_x, low_p_y)])
          if low_p_x else np.zeros((0, 2)))
    n_low = int(LO.shape[0])
    FNALL = np.vstack([FN, LO]) if (FN.size or LO.size) else np.zeros((0, 2))
    print(f"  其中 P<0.5 未进候选的真值地面点 {n_low:,}")

    # ---- 出图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    ext = [x0, x0 + W * gsd, y1 - H * gsd, y1]
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))

    def base(ax, title, show_prob=False):
        if show_prob:
            im = ax.imshow(np.where(valid, prob, np.nan), cmap="magma",
                           vmin=0, vmax=1, extent=ext)
            plt.colorbar(im, ax=ax, fraction=0.032, label="P")
        else:
            ax.imshow(np.where(gt, np.nan, np.nan), extent=ext)   # 占位定范围
        ax.set_title(title, fontsize=11)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    # (a) 全局叠加（真值面作底，只叠两条轴线，避免互相遮盖）
    ax = axes[0, 0]
    base(ax, "(a) 真值面（灰）/ 模型轴（青）/ 交付骨架（红）")
    # 真值面用 RGBA 覆盖层（散点在宽走廊/大图幅下会细到不可见）
    rgba = np.zeros((*gt.shape, 4), np.float32)
    rgba[gt] = (0.62, 0.62, 0.62, 0.95)
    ax.imshow(rgba, extent=ext, interpolation="nearest",
              label=f"真值面 {int(gt.sum()):,}px")
    for m, col, lb, ms, al in ((axis, "#00b7cc", "模型轴", 0.9, 1.0),
                               (sk_del, "#e3001b", "交付骨架", 0.7, 0.75)):
        rr, cc = np.where(m)
        ax.plot(x0 + (cc + 0.5) * gsd, y1 - (rr + 0.5) * gsd, ".", color=col,
                ms=ms, label=lb, ls="none", alpha=al)
    ax.legend(markerscale=6, fontsize=8, loc="upper right")

    # (b) 概率场
    ax = axes[0, 1]
    base(ax, "(b) 模型概率 P（绿=真值中轴，青=模型轴）", show_prob=True)
    for m, col in ((gt_skel, "#00ff66"), (axis, "#00e5ff")):
        rr, cc = np.where(m)
        ax.plot(x0 + (cc + 0.5) * gsd, y1 - (rr + 0.5) * gsd, ".", color=col,
                ms=0.5, ls="none")

    # (c) FP 密度
    ax = axes[0, 2]
    base(ax, f"(c) FP 密度：保留但不在真值内的地面点（{fp_n:,}）")
    im = ax.imshow(np.where(fp_c > 0, np.log1p(fp_c), np.nan), cmap="Reds",
                   extent=ext, vmin=0,
                   vmax=float(np.percentile(np.log1p(fp_c[fp_c > 0]), 99))
                   if (fp_c > 0).any() else 1.0)
    plt.colorbar(im, ax=ax, fraction=0.032, label="log(1+点数)")
    for m, col in ((gt_skel, "#00cc44"), (axis, "#00e5ff")):
        rr, cc = np.where(m)
        ax.plot(x0 + (cc + 0.5) * gsd, y1 - (rr + 0.5) * gsd, ".", color=col,
                ms=0.5, ls="none")

    # (d) FN 密度 + 低概率漏检
    ax = axes[1, 0]
    base(ax, f"(d) FN 密度（候选内 {fn_n:,}）+ P<0.5 漏检 {n_low:,}（浅蓝点）")
    im = ax.imshow(np.where(FNc > 0, np.log1p(FNc), np.nan), cmap="Blues",
                   extent=ext, vmin=0,
                   vmax=float(np.percentile(np.log1p(FNc[FNc > 0]), 99))
                   if (FNc > 0).any() else 1.0)
    plt.colorbar(im, ax=ax, fraction=0.032, label="log(1+点数)")
    if n_low:
        idx = np.arange(0, n_low, max(1, n_low // 40000))
        ax.scatter(LO[idx, 0], LO[idx, 1], s=0.25, c="#5ba6ff",
                   linewidths=0, alpha=0.55)
    for m, col in ((gt_skel, "#00cc44"), (sk_del, "#ff7f0e")):
        rr, cc = np.where(m)
        ax.plot(x0 + (cc + 0.5) * gsd, y1 - (rr + 0.5) * gsd, ".", color=col,
                ms=0.5, ls="none")

    # (e)(f) 放大窗：FP / FN 最密集处
    K = int(round(a.zoom / gsd))
    for ax, pts, title, col in ((axes[1, 1], FP, "FP 最密集窗口", "#e3001b"),
                                (axes[1, 2], FNALL, "FN 最密集窗口", "#1f4fd8")):
        if pts.size == 0:
            ax.set_title(f"({title}) 无")
            ax.axis("off")
            continue
        if pts.shape[0] > 60000:                      # 控制散点密度
            pts = pts[:: max(1, pts.shape[0] // 60000)]
        cc0 = np.floor((pts[:, 0] - x0) / gsd).astype(int)
        rr0 = np.floor((y1 - pts[:, 1]) / gsd).astype(int)
        hh = np.zeros((H, W), np.int32)
        np.add.at(hh, (rr0, cc0), 1)
        hs = uniform_filter(hh.astype(float), K)
        k = np.unravel_index(int(np.argmax(hs)), hs.shape)
        r0 = int(np.clip(k[0] - K // 2, 0, H - K))
        c0 = int(np.clip(k[1] - K // 2, 0, W - K))
        sub = (slice(r0, r0 + K), slice(c0, c0 + K))
        ax.imshow(np.where(gt[sub], 1.0, np.nan), cmap="Greys", vmin=0, vmax=2,
                  extent=[x0 + c0 * gsd, x0 + (c0 + K) * gsd,
                          y1 - (r0 + K) * gsd, y1 - r0 * gsd], alpha=0.55)
        ax.imshow(np.where(valid[sub], prob[sub], np.nan), cmap="magma",
                  vmin=0, vmax=1, alpha=0.65,
                  extent=[x0 + c0 * gsd, x0 + (c0 + K) * gsd,
                          y1 - (r0 + K) * gsd, y1 - r0 * gsd])
        for m, cl in ((gt_skel, "#00ff66"), (axis, "#00e5ff"), (sk_del, "#ff7f0e")):
            rr, cc = np.where(m[sub])
            ax.plot(x0 + (c0 + cc + 0.5) * gsd, y1 - (r0 + rr + 0.5) * gsd, ".",
                    color=cl, ms=1.1, ls="none")
        m = (pts[:, 0] >= x0 + c0 * gsd) & (pts[:, 0] < x0 + (c0 + K) * gsd) & \
            (pts[:, 1] > y1 - (r0 + K) * gsd) & (pts[:, 1] <= y1 - r0 * gsd)
        ax.scatter(pts[m, 0], pts[m, 1], s=2.5, c=col, linewidths=0)
        ax.set_title(f"{title} @({c0},{r0}) {a.zoom} m（绿=真值中轴 青=模型轴 橙=交付骨架）",
                     fontsize=10)
        ax.set_xlim(x0 + c0 * gsd, x0 + (c0 + K) * gsd)
        ax.set_ylim(y1 - (r0 + K) * gsd, y1 - r0 * gsd)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    tag = f"{a.truth}{'_main' if a.truth_main else ''}"
    fig.suptitle(f"{site}：预测 vs 真值（{tag}）  FP {fp_n:,} / FN {fn_n:,}", fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG, f"pred_vs_truth_{site}_{tag}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  写出 {out}")

    # ---- 横向偏移剖面：沿模型轴每 10 m 取断面，量真值/概率带相对轴的偏移 ----
    path = order_path(axis)
    if path is None or len(path) < 20:
        print("  跳过偏移剖面（轴线过短）")
        return
    step = max(1, int(round(10 / gsd)))
    st = path[::step]
    TR = np.arange(-20, 21)                     # 法向采样 -20..20 m
    rows, d_skel, d_prob, d_gtc = [], [], [], []
    for i in range(1, len(st) - 1):
        r, c = st[i]
        dr, dc = st[i + 1][0] - st[i - 1][0], st[i + 1][1] - st[i - 1][1]
        L = np.hypot(dr, dc)
        if L < 1e-6:
            continue
        nr, nc = -dc / L, dr / L                # 法向
        rr = np.clip(np.round(r + nr * TR).astype(int), 0, H - 1)
        cc = np.clip(np.round(c + nc * TR).astype(int), 0, W - 1)
        g_ = gt[rr, cc]
        s_ = gt_skel[rr, cc]
        p_ = prob[rr, cc]
        if not g_.any():
            continue
        rows.append(i * step * gsd)
        d_skel.append(float(TR[s_].mean()) if s_.any() else np.nan)
        d_gtc.append(float(TR[g_].mean()))      # 真值面中心（断面加权）
        d_prob.append(float((TR * p_).sum() / max(p_.sum(), 1e-9)) if p_.any() else np.nan)

    rows, d_skel = np.array(rows), np.array(d_skel)
    d_gtc, d_prob = np.array(d_gtc), np.array(d_prob)
    off = d_gtc - d_prob                         # 真值中心 - 概率带中心（正=偏移方向）
    med = float(np.nanmedian(off))
    amed = float(np.nanmedian(np.abs(off)))
    print(f"  真值中心 vs 概率带中心：中位偏移 {med:+.2f} m，绝对中位 {amed:.2f} m")
    # 落盘，供正文引用（此前该数字只存在于图注，无法追溯）
    jp = os.path.join(RES, "offset_stats.json")
    stats = json.load(open(jp, encoding="utf-8")) if os.path.exists(jp) else {}
    stats[site] = dict(median_m=round(med, 3), abs_median_m=round(amed, 3),
                       n_sections=int(len(off)), gsd_m=round(float(gsd), 3),
                       npz=os.path.basename(npz), truth=tag)
    json.dump(stats, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 {jp}")

    fig2, ax2 = plt.subplots(2, 1, figsize=(13, 8),
                             gridspec_kw={"height_ratios": [2, 1.6]})
    ax2[0].plot(rows, d_gtc, "-", color="#8a6d3b", lw=1.4, label="真值面断面中心")
    ax2[0].plot(rows, d_prob, "-", color="#7b2fd0", lw=1.4, label="概率带断面中心")
    ax2[0].plot(rows, d_skel, ".", color="#00aa44", ms=3, label="真值骨架交点")
    ax2[0].axhline(0, color="#00b7cc", lw=1.0, ls="--", label="模型轴（=0）")
    ax2[0].fill_between(rows, d_gtc, d_prob, color="#e3001b", alpha=0.12,
                        label=f"偏移（中位 {med:+.2f} m）")
    ax2[0].set_xlabel("沿模型轴距离 (m)")
    ax2[0].set_ylabel("法向偏移 (m)（正=轴法向一侧）")
    ax2[0].set_title(f"{site}：沿程横向偏移（真值中心 vs 概率带中心）")
    ax2[0].legend(fontsize=8, ncol=2)
    ax2[0].grid(alpha=0.25)

    # 3 个代表性断面（偏移最大 / 中位 / 反号）
    order = np.argsort(np.abs(off - med))
    picks = [int(order[-1]), int(order[len(order) // 2]),
             (int(np.argmax(off)) if np.nanmax(off) > 0 else 0)]
    for k, pi in enumerate(picks):
        i = int(round(rows[pi] / gsd / step))
        r, c = st[min(i, len(st) - 1)]
        dr, dc = st[min(i + 1, len(st) - 1)][0] - st[max(i - 1, 0)][0], \
            st[min(i + 1, len(st) - 1)][1] - st[max(i - 1, 0)][1]
        L = np.hypot(dr, dc) or 1.0
        nr, nc = -dc / L, dr / L
        rr = np.clip(np.round(r + nr * TR).astype(int), 0, H - 1)
        cc = np.clip(np.round(c + nc * TR).astype(int), 0, W - 1)
        ax2[1].plot(TR, gt[rr, cc].astype(float) + k * 1.4, "-", color="#8a6d3b",
                    lw=1.4, label="真值面" if k == 0 else None)
        ax2[1].plot(TR, prob[rr, cc] + k * 1.4, "-", color="#7b2fd0", lw=1.2,
                    label="概率 P" if k == 0 else None)
        ax2[1].axvline(0, color="#00b7cc", ls="--", lw=1.0)
        ax2[1].text(TR[-1], k * 1.4 + 0.25, f"@{rows[pi]:.0f} m", ha="right", fontsize=8)
    ax2[1].set_xlabel("法向距离 (m)（0=模型轴，向上叠放 3 个断面）")
    ax2[1].set_title("断面示例：真值面与概率带的位置关系")
    ax2[1].legend(fontsize=8)
    ax2[1].grid(alpha=0.25)
    fig2.tight_layout()
    out2 = os.path.join(FIG, f"offset_profile_{site}_{tag}.png")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"  写出 {out2}")


if __name__ == "__main__":
    main()
