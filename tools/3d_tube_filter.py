# -*- coding: utf-8 -*-
"""3D 廊道（管道）约束滤波：3d_tube_filter.py。

核心逻辑
--------
1. **3D 中线**：跑 fusion_train 的分支 1（起止点 → openness 代价 + h1 方向惩罚的
   状态增广 Dijkstra），得到 1px 路径；把路径像素映射回 UTM，沿路径取 DTM 高程，
   按 0.25 m 弧长重采样，得到有序 3D 折线 `(x, y, z_terrain)`。
2. **管道约束**：对每个激光点求到该 3D 折线的最近欧氏距离 `D3d`（密度重采样后
   最近点即为折线距离，误差 ≤0.125 m），保留 `D3d ≤ --tube`（默认 2.5 m）。
3. **多回波辅助**：对管道内的点，用 0.5 m 体素质心 + KD-Tree(r=1.5 m) 统计邻域内
   `rn < nr`（多次回波，植被特征）的比例，剔除比例 > `--multi`（默认 10%）的点。
   用体素是为了避免 195 点/m² 下 r=1.5 m 球形邻域（≈1400 点）造成的不可行内存。
4. **评价**：SHP 只作裁判。对照：
   · 纯 2D 掩膜（P≥0.5/0.3/0.2，来自 fusion_train 的 CNN 概率图）
   · 9 月纯回波法
   · 本方法（3D 管道 / 3D 管道 + 多回波）

用法
----
    python tools/3d_tube_filter.py
    python tools/3d_tube_filter.py --tube 2.0 --multi 0.10
    python tools/3d_tube_filter.py --sites 杨家坝 --no-echo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from fusion_train import build_f1, load_terrain, shp_endpoints            # noqa: E402
from model_guided_3d_filter import bilinear                               # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")
SITES = ("杨家坝", "西坝村")
CHUNK = 4_000_000
NB8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# ------------------------------------------------ 3D 中线 ----
def order_path(path):
    """把 8 连通的路径像元按行进顺序排成一维序列（从任一端开始贪心游走）。"""
    px = [tuple(p) for p in np.argwhere(path)]
    if len(px) < 2:
        return np.array(px, dtype=np.int32)
    S = set(px)

    def nbrs(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in NB8
                if (p[0] + dy, p[1] + dx) in S]

    ends = [p for p in px if len(nbrs(p)) == 1]
    cur = ends[0] if ends else px[0]
    seq, seen = [cur], {cur}
    while True:
        nx = [q for q in nbrs(cur) if q not in seen]
        if not nx:
            break
        # 优先直线前进（与上一步同向），避免锯齿误拐
        if len(nx) > 1 and len(seq) > 1:
            v0 = (seq[-1][0] - seq[-2][0], seq[-1][1] - seq[-2][1])
            nx.sort(key=lambda q: abs(q[0] - seq[-1][0] - v0[0]) +
                    abs(q[1] - seq[-1][1] - v0[1]))
        cur = nx[0]
        seq.append(cur); seen.add(cur)
    # 若还有未走到的分支，按最近邻追加（正常路径是单链，不会触发）
    for p in px:
        if p not in seen:
            seq.append(p); seen.add(p)
    return np.array(seq, dtype=np.int32)


def path_polyline_3d(site, path, dem, meta, step=0.25, log=print):
    """路径像素 → 有序 UTM 折线（x,y,z_terrain），按弧长 step 重采样。"""
    gsd = float(meta["gsd_m"])
    x0, y1 = float(meta["x0"]), float(meta["y1"])
    seq = order_path(path)
    if len(seq) < 2:
        log("  [FAIL] 路径过短")
        return None
    # 像元中心 → UTM
    x = x0 + (seq[:, 1] + 0.5) * gsd
    y = y1 - (seq[:, 0] + 0.5) * gsd
    z = dem[seq[:, 0], seq[:, 1]]
    xy = np.stack([x, y], axis=1)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    n = max(int(arc[-1] / step) + 1, 2)
    s = np.linspace(0.0, arc[-1], n)
    dx = np.interp(s, arc, x); dy = np.interp(s, arc, y)
    # DEM 在重采样点上取双线性值（用像元坐标换回）
    c = (dx - x0) / gsd - 0.5
    r = (y1 - dy) / gsd - 0.5
    dz = bilinear(dem, r, c)
    P = np.stack([dx, dy, dz], axis=1)
    # 相邻重采样点的 3D 间距（诊断：垂直起伏是否被解析出来）
    d3 = np.linalg.norm(np.diff(P, axis=0), axis=1)
    log(f"  3D 中线：路径 {len(seq)} px、弧长 {arc[-1]:.1f} m → 重采样 {n} 点"
        f"（步长 {step} m，3D 间距中位 {np.median(d3):.3f} m）")
    log(f"    高程沿程 {dz.min():.1f}~{dz.max():.1f} m（落差 {dz.max() - dz.min():.1f} m）；"
        f"起点 ({dx[0]:.1f},{dy[0]:.1f},{dz[0]:.1f}) 终点 "
        f"({dx[-1]:.1f},{dy[-1]:.1f},{dz[-1]:.1f})")
    return P, dict(n_pts=int(len(seq)), arc_m=round(float(arc[-1]), 1),
                   resample=int(n), z_range=[float(dz.min()), float(dz.max())])


# ------------------------------------------------ 管道距离 ----
def tube_keep(site, line, tubes, log):
    """逐点求到 3D 折线的最近距离（折线已按 0.25 m 重采样，最近点即折线距离）。

    一次查询同时产出多个半径的保留掩膜（避免重复 KD-Tree 查询）。
    """
    from scipy.spatial import cKDTree

    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    n = xyz.shape[0]
    tree = cKDTree(line)
    keeps = {t: np.zeros(n, dtype=bool) for t in tubes}
    hist = np.zeros(81, dtype=np.int64)      # D3d 直方图（0~10m，0.125m 一格 + 溢出）
    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        p = np.asarray(xyz[s:e], dtype=np.float64) + o[None, :]
        d = tree.query(p, workers=-1)[0]
        for t, k in keeps.items():
            k[s:e] = d <= t
        h = np.minimum((d / 0.125).astype(np.int32), 80)
        hist += np.bincount(h, minlength=81)
        if (s // CHUNK) % 5 == 0:
            log(f"    点 {e / 1e6:.0f}M / {n / 1e6:.1f}M  "
                f"D3d≤{tubes[0]:g}m {keeps[tubes[0]][:e].sum() / e * 100:.2f}%"
                f"（{time.time() - t0:.0f}s）")
    pct = np.cumsum(hist) / hist.sum()
    d125 = np.arange(81) * 0.125
    q = {f"P{p}": float(d125[min(int(np.searchsorted(pct, p / 100.0)), 80)])
         for p in (50, 90, 99)}
    sr = dict(d3d_pct=q, seconds=round(time.time() - t0, 1))
    for t, k in keeps.items():
        log(f"  D3d≤{t:g}m 保留 {int(k.sum()):,}（{k.mean() * 100:.3f}%）")
        sr[f"keep_frac_{t:g}"] = round(float(k.mean()), 5)
    log("  D3d 分位 " + " / ".join(f"{k_} {v:.2f}m" for k_, v in q.items()))
    return keeps, sr


# ------------------------------------------------ 多回波 ----
def multi_echo_drop(keep, rn, nr, vox=0.5, radius=1.5, multi=0.10, log=print,
                    origin=None, xyz=None):
    """体素化后统计 r=1.5 m 邻域的多次回波（rn<nr）比例，返回需剔除的掩膜。"""
    from scipy.spatial import cKDTree

    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return np.zeros_like(keep)
    p = np.asarray(xyz[idx], dtype=np.float64) + origin[None, :]
    key = np.floor((p - p.min(0)) / vox).astype(np.int64)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    me = (np.rint(rn[idx]) < np.rint(nr[idx])).astype(np.int32)
    nech = np.bincount(inv, weights=me, minlength=len(uniq))
    cent = uniq * vox + vox / 2.0 + p.min(0)

    tree = cKDTree(cent)
    nb = tree.query_ball_point(cent, radius, workers=-1)
    n_tot = np.array([cnt[l].sum() for l in nb], dtype=np.int64)
    n_mul = np.array([nech[l].sum() for l in nb], dtype=np.int64)
    frac_vox = n_mul / np.maximum(n_tot, 1)
    frac = frac_vox[inv]                       # 广播到点

    drop_idx = idx[frac > multi]
    drop = np.zeros_like(keep)
    drop[drop_idx] = True
    log(f"  多回波：体素 {len(uniq):,}（{vox} m），邻域中位 {int(np.median(n_tot))} 点；"
        f"多次回波比例 中位 {np.median(frac) * 100:.1f}%、P90 "
        f"{np.percentile(frac, 90) * 100:.1f}%；>"
        f"{multi * 100:g}% 剔除 {len(drop_idx):,}"
        f"（管道内的 {len(drop_idx) / max(len(idx), 1) * 100:.1f}%）")
    return drop, dict(voxels=int(len(uniq)), nb_median=int(np.median(n_tot)),
                     frac_median=round(float(np.median(frac)), 4),
                     frac_p90=round(float(np.percentile(frac, 90)), 4),
                     dropped=int(len(drop_idx)))


# ------------------------------------------------ 评价 ----
def metrics(keep, label):
    tp = int((keep & label).sum())
    fp = int((keep & ~label).sum())
    fn = int((~keep & label).sum())
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, keep=int(keep.sum()),
                keep_frac=round(float(keep.mean()), 5), truth=int(label.sum()),
                precision=float(pr), recall=float(rc),
                f1=float(2 * pr * rc / max(pr + rc, 1e-9)))


def main():
    p = argparse.ArgumentParser(description="3D 廊道（管道）约束滤波")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--ep", type=str, default=None, help='端点 "r,c;r,c"；缺省用 SHP 骨架最远两点')
    p.add_argument("--tube", type=float, default=2.5, help="3D 管道主半径 m")
    p.add_argument("--tube-sweep", nargs="*", type=float, default=[5.0, 10.0],
                   help="对照用的附加管道半径（一次 KD 查询顺带产出，不额外耗时）")
    p.add_argument("--step", type=float, default=0.25, help="中线重采样步长 m")
    p.add_argument("--vox", type=float, default=0.5, help="多回波统计体素边长 m")
    p.add_argument("--radius", type=float, default=1.5, help="多回波邻域半径 m")
    p.add_argument("--multi", type=float, default=0.10, help="多次回波比例上限")
    p.add_argument("--no-echo", action="store_true", help="跳过多回波辅助")
    p.add_argument("--gate", type=float, default=3.0)
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "3d_tube_filter.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True); lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"3D 廊道约束滤波 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"管道 D3d≤{a.tube:g} m" +
        (f" → 多回波比例≤{a.multi * 100:g}%（体素 {a.vox}m，邻域 {a.radius}m）"
         if not a.no_echo else "（跳过多回波）"))
    log("SHP 只在最后评分")

    out = []
    for s in a.sites:
        t0 = time.time()
        log(f"\n[{s}]")
        dem, hs, valid = load_terrain(s)
        meta = json.load(open(os.path.join(DATA, f"meta_{s}.json"), encoding="utf-8"))

        if a.ep:
            ep = [np.array([float(v) for v in q.split(",")]) for q in a.ep.split(";")]
            src = "cli"
        else:
            ep, info = shp_endpoints(s, valid, log=log)
            src = info["source"]

        r1 = build_f1(s, valid, ep, gate=a.gate, dir_lambda=a.dir_lambda,
                      turn_lambda=a.turn_lambda, turn_min=a.turn_min, log=log)
        if r1 is None:
            continue
        line, linfo = path_polyline_3d(s, r1["path"], dem, meta, step=a.step, log=log)

        tubes = sorted({float(a.tube), *[float(t) for t in a.tube_sweep]})
        keeps, tinfo = tube_keep(s, line, tubes, log)

        o = np.load(os.path.join(REAL, f"{s}_origin.npy"))
        label = np.asarray(np.load(os.path.join(REAL, f"{s}_label.npy"),
                                   mmap_mode="r"), dtype=bool)
        keep_t = keeps[float(a.tube)]
        rows = []
        for t in tubes:
            is_main = abs(t - a.tube) < 1e-9
            rows.append(dict(name=f"3D 管道 (D3d≤{t:g}m)" + ("" if is_main else " 扫描"),
                             kind="tube" if is_main else "tube_sweep", radius=t,
                             **metrics(keeps[t], label)))

        keep = keep_t
        einfo = None
        if not a.no_echo:
            ft = np.load(os.path.join(REAL, f"{s}_lasefeat.npy"), mmap_mode="r")
            xyz = np.load(os.path.join(REAL, f"{s}_xyz_local.npy"), mmap_mode="r")
            drop, einfo = multi_echo_drop(keep_t, np.asarray(ft[:, 0]), np.asarray(ft[:, 1]),
                                          vox=a.vox, radius=a.radius, multi=a.multi,
                                          log=log, origin=o, xyz=xyz)
            keep = keep_t & ~drop
            rows.append(dict(name=f"管道 + 多回波≤{a.multi * 100:g}%", kind="tube_echo",
                             radius=a.tube, **metrics(keep, label)))

        # ---- 对照：纯 2D 掩膜 / 9 月回波法 ----
        pc = os.path.join(RES, f"model_guided_prob_{s}.npz")
        if os.path.exists(pc):
            z = np.load(pc)
            prob, vmask = z["prob"].astype(np.float64), z["valid"] == 1
            xyz = np.load(os.path.join(REAL, f"{s}_xyz_local.npy"), mmap_mode="r")
            n = xyz.shape[0]
            pv = np.zeros(n, dtype=np.float32)
            H, W = prob.shape
            gsd, x0, y1 = meta["gsd_m"], meta["x0"], meta["y1"]
            for s0 in range(0, n, CHUNK):
                e0 = min(s0 + CHUNK, n)
                xl = np.asarray(xyz[s0:e0], dtype=np.float64)
                c = (xl[:, 0] + o[0] - x0) / gsd - 0.5
                r = (y1 - xl[:, 1] - o[1]) / gsd - 0.5
                ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
                ii = np.flatnonzero(ins)
                v = bilinear(prob, r[ii], c[ii])
                vb = bilinear(vmask.astype(np.float64), r[ii], c[ii])
                pv[s0 + ii] = np.where(vb > 0.5, v, 0.0)
            for t in (0.5, 0.3, 0.2):
                rows.append(dict(name=f"纯2D (P≥{t:g})", kind="2d", thr=t,
                                 **metrics(pv >= t, label)))
        bp = os.path.join(REAL, f"{s}_filtered_回波法_idx.npy")
        if os.path.exists(bp):
            bi = np.asarray(np.load(bp, mmap_mode="r"))
            bk = np.zeros(label.shape[0], dtype=bool); bk[bi] = True
            rows.append(dict(name="9月纯回波法", kind="echo", **metrics(bk, label)))

        for r_ in rows:
            log(f"  {r_['name']:<22s} 保留 {r_['keep']:>10,d}（{r_['keep_frac'] * 100:6.2f}%）"
                f"  P={r_['precision']:.4f} R={r_['recall']:.4f} F1={r_['f1']:.4f}")

        # ---- 图 ----
        os.makedirs(FIG, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4), dpi=145)
        # Hillshade 用 UTM extent 显示，中线按 UTM 坐标叠加（两者坐标系必须一致）
        H_, W_ = hs.shape
        gsd_, x0_, y1_ = float(meta["gsd_m"]), float(meta["x0"]), float(meta["y1"])
        ext = [x0_, x0_ + W_ * gsd_, y1_ - H_ * gsd_, y1_]
        ax[0].imshow(np.where(valid, hs, np.nan), cmap="gray", vmin=0, vmax=1,
                     extent=ext)
        # 中线（青实线）+ 主半径的 XY 投影边界（青虚线），便于对照真值走廊宽度
        dxs = np.gradient(line[:, 0]); dys = np.gradient(line[:, 1])
        nrm = np.hypot(dxs, dys); nrm[nrm < 1e-9] = 1.0
        nx, ny = -dys / nrm, dxs / nrm
        ax[0].plot(line[:, 0], line[:, 1], "-", lw=1.8, c="#00e5ff", zorder=7)
        for sg in (1.0, -1.0):
            ax[0].plot(line[:, 0] + sg * nx * a.tube, line[:, 1] + sg * ny * a.tube,
                       "--", lw=0.6, c="#66d9ff", alpha=0.8, zorder=4)
        truth_ras = np.load(os.path.join(RES, f"mask_1m_{s}.npy")).astype(bool)
        Xg = x0_ + (np.arange(W_) + 0.5) * gsd_
        Yg = y1_ - (np.arange(H_) + 0.5) * gsd_
        ax[0].contour(Xg, Yg, truth_ras, levels=[0.5], colors="#00ff66", linewidths=0.9,
                      zorder=6)
        ax[0].set_title(f"① 3D 中线（{linfo['n_pts']}px → {linfo['resample']} 点）\n"
                        f"弧长 {linfo['arc_m']:.0f} m，落差 "
                        f"{linfo['z_range'][1] - linfo['z_range'][0]:.1f} m"
                        f"（青=中线±{a.tube:g}m，绿=SHP）", fontsize=9.5)
        ax[0].set_xticks([]); ax[0].set_yticks([])

        def _lbl(r_):
            k = r_["kind"]
            if k in ("tube", "tube_sweep"):
                return f"管道\n≤{r_['radius']:g}m"
            if k == "tube_echo":
                return f"管道+回波\n≤{a.multi * 100:g}%"
            if k == "2d":
                return f"纯2D\nP≥{r_['thr']:g}"
            return "9月回波法"

        names = [_lbl(r_) for r_ in rows]
        f1s = [r_["f1"] for r_ in rows]
        cmap = dict(tube="#1f77b4", tube_sweep="#9ecae1", tube_echo="#17becf",
                    **{"2d": "#ff7f0e"}, echo="#9aa0a6")
        bars = ax[1].bar(range(len(rows)), f1s,
                         color=[cmap.get(r_["kind"], "#333") for r_ in rows], width=0.68)
        for b, r_ in zip(bars, rows):
            ax[1].text(b.get_x() + b.get_width() / 2, r_["f1"] + max(f1s) * 0.02,
                       f"{r_['f1']:.3f}\nP{r_['precision']:.2f}/R{r_['recall']:.2f}",
                       ha="center", fontsize=6.5)
        ax[1].set_xticks(range(len(rows))); ax[1].set_xticklabels(names, fontsize=7)
        ax[1].set_ylabel("F1"); ax[1].set_ylim(0, max(f1s) * 1.34 + 0.01)
        ax[1].grid(axis="y", alpha=0.25)
        ax[1].set_title("F1 对照（蓝=管道，青=管道+多回波，橙=纯2D，灰=回波）", fontsize=9.5)
        fig.suptitle(f"{s} · 3D 廊道约束滤波", fontsize=11)
        fig.tight_layout()
        fp = os.path.join(FIG, f"3dtube_{s}.png")
        fig.savefig(fp); plt.close(fig)

        out.append(dict(site=s, ep_source=src, tube=a.tube, multi=a.multi,
                        line=linfo, tube_stats=tinfo, echo_stats=einfo,
                        rows=rows, figure=os.path.relpath(fp, ROOT),
                        seconds=round(time.time() - t0, 1)))

    log("\n" + "=" * 78)
    log(f"{'站点':>8s} {'方案':>24s} {'保留点':>12s} {'保留率':>8s} {'TP':>11s} "
        f"{'FP':>12s} {'FN':>10s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    log("-" * 108)
    for r_ in out:
        for d in r_["rows"]:
            log(f"{r_['site']:>8s} {d['name']:>24s} {d['keep']:>12,d} "
                f"{d['keep_frac'] * 100:>7.2f}% {d['tp']:>11,d} {d['fp']:>12,d} "
                f"{d['fn']:>10,d} {d['precision']:>7.4f} {d['recall']:>7.4f} {d['f1']:>7.4f}")

    jp = os.path.join(RES, f"3d_tube_filter_t{a.tube:g}_m{a.multi:g}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="3D 管道约束（D3d≤tube）+ 多回波辅助；对照纯2D与回波法",
                       tube=a.tube, multi=a.multi, vox=a.vox, radius=a.radius,
                       sites=out), f, ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
