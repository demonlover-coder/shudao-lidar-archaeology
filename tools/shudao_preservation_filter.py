# -*- coding: utf-8 -*-
"""2D 引导 + 3D 几何的蜀道石板点联合滤波（shudao_preservation_filter.py）。

流程
----
1. **2D 概率圈地**：复用 `model_guided_3d_filter.prob_map()`（起止点 → 分支1 → AE 瓶颈
   PCA4 + Hillshade + CSSC → CNN），逐点双线性取 `P`，只保留 `P ≥ thr`（默认 0.2，放宽
   以保召回）。
2. **3D 局部法向**：对候选点建 KD-Tree（半径 `--radius`，默认 0.5 m），对每点取邻域做
   PCA，法向取最小特征值方向；保留法向与 +Z 夹角 `< --angle`（默认 30°）的点。
3. **3D 局部平坦度**：邻域内 Z 标准差 `> --std`（默认 0.1 m）的点剔除。
   2、3 同时满足才保留。邻域点数 `< --min-nb`（默认 8）无法估计几何量，按不通过计。
4. **输出**：最终保留点云 LAS（局部坐标 + 原点 = UTM 48N），并计算与 SHP 面内真值点
   （`data_real/{site}_label.npy`，含树冠，同口径）的 P/R/F1，对照 9 月纯回波法。

口径提醒
--------
真值 `label` 是"SHP 多边形投影内的所有点"，含走廊上方 5~13 m 的树冠点。本方法刻意剔除
树冠，因此召回天然受限（上一轮实测：真值点中 H_agl≤1m 的只占 13.2%/27.4%）。脚本会额外
输出"真值点中贴地部分"的分层召回，用来区分"漏掉路面"与"漏掉树冠"。

用法
----
    python tools/shudao_preservation_filter.py
    python tools/shudao_preservation_filter.py --sites 杨家坝 --thr 0.2
    python tools/shudao_preservation_filter.py --angle 25 --std 0.08
    python tools/shudao_preservation_filter.py --no-las        # 不写 LAS（默认写）
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
from model_guided_3d_filter import bilinear, prob_map                      # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")
SITES = ("杨家坝", "西坝村")
CHUNK = 4_000_000
GEOM_STEP = 100_000


# ------------------------------------------------- 第 1 步：2D 概率圈地 ----
def gate_2d(site, prob, valid, thr, log):
    """返回 (候选点局部坐标 (M,3), 候选点在 原序列 中的索引, 概率, 统计)。"""
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
    gsd, x0, y1 = float(meta["gsd_m"]), float(meta["x0"]), float(meta["y1"])
    H, W = prob.shape
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    n = xyz.shape[0]

    sel_idx, sel_p = [], []
    n_in = 0
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        xl = np.asarray(xyz[s:e], dtype=np.float64)
        x = xl[:, 0] + float(o[0]); y = xl[:, 1] + float(o[1])
        c = (x - x0) / gsd - 0.5
        r = (y1 - y) / gsd - 0.5
        ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
        pv = np.zeros(len(x))
        if ins.any():
            ii = np.flatnonzero(ins)
            pv[ii] = bilinear(prob, r[ii], c[ii])
            vb = bilinear(valid.astype(np.float64), r[ii], c[ii])
            pv[ii] = np.where(vb > 0.5, pv[ii], 0.0)
        n_in += int(ins.sum())
        k = pv >= thr
        if k.any():
            sel_idx.append(np.flatnonzero(k) + s)
            sel_p.append(pv[k])
    idx = np.concatenate(sel_idx) if sel_idx else np.array([], dtype=np.int64)
    pvv = np.concatenate(sel_p) if sel_p else np.array([])
    pts = np.asarray(xyz[idx], dtype=np.float64) if len(idx) else np.zeros((0, 3))
    st = dict(n=int(n), in_dom=n_in, cand=int(len(idx)),
              cand_frac=round(float(len(idx) / n), 5))
    log(f"  第1步 2D 圈地（P≥{thr:g}）：域内 {n_in / 1e6:.2f}M / {n / 1e6:.2f}M，"
        f"候选 {len(idx):,}（{st['cand_frac'] * 100:.2f}%），"
        f"P 分位 P50/P90/P99 = " +
        " / ".join(f"{v:.3f}" for v in np.percentile(pvv, [50, 90, 99])) if len(pvv) else "")
    return pts, idx, pvv, st


# ------------------------------------------------- 第 2/3 步：几何 ----
def surface_filter(pts, a, log):
    """KD-Tree + PCA：返回 (keep, 诊断) —— 法向夹角、Z 标准差、邻域点数。"""
    from scipy.spatial import cKDTree

    n = len(pts)
    if n == 0:
        z = np.zeros(0, bool)
        return z, dict(cnt=z, ang=z, sd=z, curv=z)

    t0 = time.time()
    tree = cKDTree(pts)
    log(f"    KD-Tree 构建 {n:,} 点（{time.time() - t0:.1f}s），"
        f"查询半径 {a.radius} m、k≤{a.k}")

    cnt = np.zeros(n, dtype=np.int32)
    ang = np.full(n, np.nan, dtype=np.float32)
    sd = np.full(n, np.nan, dtype=np.float32)
    curv = np.full(n, np.nan, dtype=np.float32)

    hi = float(a.angle)
    for s in range(0, n, GEOM_STEP):
        e = min(s + GEOM_STEP, n)
        d, idx = tree.query(pts[s:e], k=a.k, distance_upper_bound=a.radius,
                            workers=-1)
        ok = np.isfinite(d)                       # (m,k)
        m = np.clip(idx, 0, n - 1)
        w = ok.astype(np.float64)[..., None]      # (m,k,1)
        P = pts[m]                                # (m,k,3)
        cnt[s:e] = ok.sum(1)
        c = np.maximum(cnt[s:e], 1).astype(np.float64)
        mu = (P * w).sum(1) / c[:, None]
        D = (P - mu[:, None, :]) * w
        C = np.einsum("mki,mkj->mij", D, D) / c[:, None, None]
        ev, evec = np.linalg.eigh(C)              # ev 升序；evec[:, :, 0] = 最小特征值方向
        nrm = evec[:, :, 0]
        ang[s:e] = np.degrees(np.arccos(np.clip(np.abs(nrm[:, 2]), 0.0, 1.0)))
        sd[s:e] = np.sqrt(np.maximum(C[:, 2, 2], 0.0))
        tr = np.maximum(ev.sum(1), 1e-12)
        curv[s:e] = ev[:, 0] / tr

    enough = cnt >= a.min_nb
    ok_n = enough & (ang < hi)
    ok_f = enough & (sd <= a.std)
    keep = ok_n & ok_f
    diag = dict(cnt=cnt, ang=ang, sd=sd, curv=curv, enough=enough,
                ok_n=ok_n, ok_f=ok_f)
    log(f"    邻域：中位 {int(np.median(cnt))} 点；法向夹角中位 {np.nanmedian(ang):.1f}°"
        f"（<{hi:g}° 占 {(ang < hi).mean() * 100:.1f}%）；"
        f"Z 标准差中位 {np.nanmedian(sd):.3f} m（≤{a.std:g} 占 {(sd <= a.std).mean() * 100:.1f}%）")
    log(f"    邻域不足 {int((~enough).sum()):,} 点；仅法向通过 {int(ok_n.sum()):,}；"
        f"仅平坦通过 {int(ok_f.sum()):,}；两者同时 {int(keep.sum()):,}"
        f"（候选的 {keep.mean() * 100:.2f}%）")
    return keep, diag


# ------------------------------------------------- 评价 ----
def metrics(keep_full, label, name_sel=None):
    tp = int((keep_full & label).sum())
    fp = int((keep_full & ~label).sum())
    fn = int((~keep_full & label).sum())
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, keep=int(keep_full.sum()), truth=int(label.sum()),
                precision=float(pr), recall=float(rc),
                f1=float(2 * pr * rc / max(pr + rc, 1e-9)))


def write_las(path, xyz_utm, rn, nr, inten, score):
    import laspy
    hdr = laspy.LasHeader(point_format=0, version="1.2")
    hdr.offsets = [float(xyz_utm[:, 0].min()), float(xyz_utm[:, 1].min()),
                   float(xyz_utm[:, 2].min())]
    hdr.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(hdr)
    las.x = xyz_utm[:, 0].astype(np.float64)
    las.y = xyz_utm[:, 1].astype(np.float64)
    las.z = xyz_utm[:, 2].astype(np.float64)
    las.return_number = np.clip(rn, 1, 7).astype(np.uint8)
    las.number_of_returns = np.clip(nr, 1, 7).astype(np.uint8)
    las.intensity = np.clip(inten, 0, 65535).astype(np.uint16)
    las.classification = np.ones(len(xyz_utm), np.uint8)
    las.user_data = np.clip(score, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    las.write(os.path.abspath(path))
    return os.path.abspath(path)


def main():
    p = argparse.ArgumentParser(description="2D 引导 + 3D 几何的蜀道石板点联合滤波")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--ep", type=str, default=None, help='端点 "r,c;r,c"；缺省用 SHP 骨架最远两点')
    p.add_argument("--thr", type=float, default=0.2, help="2D 概率圈地阈值")
    p.add_argument("--radius", type=float, default=0.5, help="KD-Tree 邻域半径 m")
    p.add_argument("--k", type=int, default=64, help="每点最多取的邻居数（上限保护）")
    p.add_argument("--angle", type=float, default=30.0, help="法向与 +Z 夹角上限（度）")
    p.add_argument("--std", type=float, default=0.1, help="邻域 Z 标准差上限 m")
    p.add_argument("--min-nb", type=int, default=8, help="最少邻域点数")
    p.add_argument("--gate", type=float, default=3.0)
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    p.add_argument("--ae-epochs", type=int, default=20)
    p.add_argument("--ae-patches", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-las", action="store_true", help="不写保留点云 LAS")
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "shudao_preservation_filter.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True); lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"蜀道石板点联合滤波 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"P≥{a.thr:g} → KD-Tree(r={a.radius}m, k≤{a.k}) → 法向<{a.angle:g}° 且 "
        f"邻域Zσ≤{a.std:g}m")
    log("SHP 只在最后评分")

    out = []
    for s in a.sites:
        t0 = time.time()
        log(f"\n[{s}]")
        prob, dem, valid, info = prob_map(s, a.ep, a, log)
        if prob is None:
            continue

        pts, idx, pv, st1 = gate_2d(s, prob, valid, a.thr, log)
        keep_c, dg = surface_filter(pts, a, log)

        n_all = st1["n"]
        keep = np.zeros(n_all, dtype=bool)
        keep[idx[keep_c]] = True

        label = np.asarray(np.load(os.path.join(REAL, f"{s}_label.npy"),
                                   mmap_mode="r"), dtype=bool)
        m = metrics(keep, label)

        # 分层：真值点里"候选（2D 命中）"与"贴地"两类的召回
        hit = np.zeros(n_all, dtype=bool); hit[idx] = True
        ts = int(label.sum())
        r_2d = float((hit & label).sum() / max(ts, 1))
        agl = np.full(n_all, np.nan)
        xyz = np.load(os.path.join(REAL, f"{s}_xyz_local.npy"), mmap_mode="r")
        o = np.load(os.path.join(REAL, f"{s}_origin.npy"))
        meta = json.load(open(os.path.join(DATA, f"meta_{s}.json"), encoding="utf-8"))
        gsd, x0, y1 = meta["gsd_m"], meta["x0"], meta["y1"]
        H, W = prob.shape
        for s0 in range(0, n_all, CHUNK):
            e0 = min(s0 + CHUNK, n_all)
            xl = np.asarray(xyz[s0:e0], dtype=np.float64)
            c = (xl[:, 0] + o[0] - x0) / gsd - 0.5
            r = (y1 - xl[:, 1] - o[1]) / gsd - 0.5
            ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
            ii = np.flatnonzero(ins)
            agl[s0:e0][ii] = xl[ii, 2] + o[2] - bilinear(dem, r[ii], c[ii])
        low = label & (agl <= 1.0)
        m_low = metrics(keep, low)

        log(f"  最终保留 {int(keep.sum()):,} 点（原始 {n_all:,} 的 "
            f"{keep.sum() / n_all * 100:.3f}%）")
        log(f"  对 SHP 面内真值（含树冠）：P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f}（TP {m['tp']:,}/FP {m['fp']:,}/FN {m['fn']:,}）")
        log(f"  分层：2D 圈地召回 {r_2d:.4f}（{int((hit & label).sum()):,}/{ts:,}）；"
            f"对'贴地真值'（AGL≤1m，{int(low.sum()):,} 点）"
            f"P={m_low['precision']:.4f} R={m_low['recall']:.4f} F1={m_low['f1']:.4f}")

        base_p = os.path.join(REAL, f"{s}_filtered_回波法_idx.npy")
        m_base = None
        if os.path.exists(base_p):
            bi = np.load(base_p, mmap_mode="r")
            bk = np.zeros(n_all, dtype=bool); bk[np.asarray(bi)] = True
            m_base = metrics(bk, label)
            log(f"  对照 9月回波法：P={m_base['precision']:.4f} R={m_base['recall']:.4f} "
                f"F1={m_base['f1']:.4f}")

        las_path = None
        if not a.no_las:
            ft = np.load(os.path.join(REAL, f"{s}_lasefeat.npy"), mmap_mode="r")
            fidx = np.flatnonzero(keep)
            xyz_utm = np.asarray(xyz[fidx], dtype=np.float64) + o[None, :]
            sc = np.clip(pv[keep_c] * 255.0, 0, 255) if len(pv) else np.zeros(len(fidx))
            las_path = os.path.join(REAL, f"{s}_shudao_preserve.las")
            write_las(las_path, xyz_utm,
                      np.rint(np.asarray(ft[fidx, 0])).astype(np.int32),
                      np.rint(np.asarray(ft[fidx, 1])).astype(np.int32),
                      np.asarray(ft[fidx, 2]).astype(np.float64), sc)
            log(f"  导出 {os.path.relpath(las_path, ROOT)}（{len(fidx):,} 点）")

        # ---------------- 图 ----------------
        os.makedirs(FIG, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(1, 3, figsize=(16.8, 5.4), dpi=145)
        ax[0].imshow(np.where(valid, prob, np.nan), cmap="magma")
        ax[0].set_title(f"① CNN 走廊概率（P≥{a.thr:g} 圈地）", fontsize=10)

        sub = np.random.default_rng(a.seed).choice(len(pts), size=min(len(pts), 400_000),
                                                   replace=False) if len(pts) else np.array([], int)
        cs, rs = pts[sub, 1], pts[sub, 0]
        ax[1].imshow(np.where(valid, prob, np.nan), cmap="gray")
        ax[1].scatter(cs[~keep_c[sub]], rs[~keep_c[sub]], s=0.35, c="#d62728",
                      label="3D 剔除", alpha=0.55, linewidths=0)
        ax[1].scatter(cs[keep_c[sub]], rs[keep_c[sub]], s=0.35, c="#00e5ff",
                      label="保留", alpha=0.75, linewidths=0)
        ax[1].legend(fontsize=8, markerscale=12, loc="lower left")
        ax[1].set_title(f"② 候选点 3D 判定（候选 {len(pts):,}）", fontsize=10)

        # 真值点 vs 保留点的 AGL 分布（解释召回分层）
        ax[2].hist(agl[label & np.isfinite(agl)], bins=60, range=(-1, 20), color="#9aa0a6",
                   alpha=0.75, label=f"SHP 面内点（{ts:,}）")
        ax[2].hist(agl[keep & np.isfinite(agl)], bins=60, range=(-1, 20), color="#1f77b4",
                   alpha=0.75, label=f"最终保留（{int(keep.sum()):,}）")
        ax[2].set_xlabel("离地高度 H_agl (m)"); ax[2].set_ylabel("点数")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)
        ax[2].set_title("③ 保留点的离地高度分布", fontsize=10)

        for a_ in ax[:2]:
            a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f"{s} · 2D 引导 + 3D 几何联合滤波   "
                     f"保留 {keep.sum() / n_all * 100:.2f}%  "
                     f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}", fontsize=11)
        fig.tight_layout()
        fp = os.path.join(FIG, f"shudao_preserve_{s}.png")
        fig.savefig(fp); plt.close(fig)

        out.append(dict(site=s, thr=a.thr, radius=a.radius, k=a.k, angle=a.angle,
                        std=a.std, min_nb=a.min_nb, endpoints=info,
                        gate2d=st1,
                        geom=dict(nb_median=float(np.median(dg["cnt"])),
                                  ang_median=float(np.nanmedian(dg["ang"])),
                                  sd_median=float(np.nanmedian(dg["sd"])),
                                  nb_short=int((~dg["enough"]).sum()),
                                  ok_normal=int(dg["ok_n"].sum()),
                                  ok_flat=int(dg["ok_f"].sum()),
                                  passed=int(keep_c.sum())),
                        result=m, echo_baseline=m_base,
                        recall_2d=r_2d, truth_low=m_low, las=las_path,
                        figure=os.path.relpath(fp, ROOT),
                        seconds=round(time.time() - t0, 1)))

    log("\n" + "=" * 78)
    log(f"{'站点':>8s} {'方法':>12s} {'保留点':>11s} {'TP':>11s} {'FP':>12s} "
        f"{'FN':>10s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    log("-" * 92)
    for r in out:
        for nm, mm in (("9月回波法", r["echo_baseline"]), ("联合滤波", r["result"])):
            if not mm:
                continue
            log(f"{r['site']:>8s} {nm:>12s} {mm['keep']:>11,d} {mm['tp']:>11,d} "
                f"{mm['fp']:>12,d} {mm['fn']:>10,d} {mm['precision']:>7.4f} "
                f"{mm['recall']:>7.4f} {mm['f1']:>7.4f}")

    jp = os.path.join(RES, f"shudao_preservation_filter_t{a.thr:g}_r{a.radius:g}"
                          f"_ang{a.angle:g}_sd{a.std:g}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="2D 概率圈地 + 3D 法向/平坦度联合滤波", sites=out),
                  f, ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
