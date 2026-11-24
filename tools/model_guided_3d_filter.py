# -*- coding: utf-8 -*-
"""用已训练的 2D 融合模型指导 3D 点云滤波（model_guided_3d_filter.py）。

流程
----
1. **概率图**：复用 `fusion_train.py` 的全套前向——起止点 → 分支1（openness 代价 +
   h1 方向惩罚的状态增广 Dijkstra）→ F1(2ch)；加载 `results/ae_{site}.pt` 无监督自编码器
   → 瓶颈 PCA 4 维 + Hillshade + CSSC → F2(6ch)；拼接 8 通道过 `results/fusion_net_{site}.pt`
   （2.7k 参数）→ 全图 `prob_map`。
2. **逐点映射**：`data_real/{site}_xyz_local.npy` 是相对 `{site}_origin.npy` 的局部坐标，
   加回原点得 UTM。按 `data/meta_{site}.json` 的 x0/y1/gsd 映射到像元
   （`col=(x−x0)/gsd−0.5`，`row=(y1−y)/gsd−0.5`），双线性插值取 `P` 与 DTM 高程，
   `H_agl = z − DTM`。域外点 `P=0`（必然剔除）。
3. **滤波判据**：`keep = (P ≥ thr) & (H_agl ≤ agl)`，默认 thr=0.5、agl=1.0 m。
4. **评价**：真值 `data_real/{site}_label.npy`（SHP 面内点）。对照组为 9 月纯回波法
   `data_real/{site}_filtered_回波法_idx.npy`（`keep=(rn==1)|(rn==nr)`），同一批点、
   同一个 label 数组、同一个口径（点级 P/R/F1）。

口径说明
--------
* 点级指标：TP=保留且在真值内，FP=保留但在真值外，FN=剔除但在真值内。
  与 9 月回波法基线完全同口径，可直接对比。
* 本方法是"走廊概率 + 离地高度"双条件，不依赖回波信息，也不读 SHP（SHP 只在最后评分）。
* `--write-las` 才导出保留点云（默认关闭，西坝村全量会写出 ~1.7GB）。

用法
----
    python tools/model_guided_3d_filter.py
    python tools/model_guided_3d_filter.py --thr 0.5 --agl 1.0
    python tools/model_guided_3d_filter.py --sites 杨家坝 --ep "139,283;495,133"
    python tools/model_guided_3d_filter.py --agl 1.5 --write-las
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
from fusion_train import (build_f1, build_f2, build_net, infer,            # noqa: E402
                          load_or_train_ae, load_terrain, shp_endpoints)

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")
SITES = ("杨家坝", "西坝村")
CHUNK = 4_000_000


# ------------------------------------------------------------ 概率图 ----
def prob_map(site, ep_arg, a, log):
    """起止点 → F1 → AE 瓶颈 → F2 → CNN → 全图概率。返回 (prob, dem, valid, ep)."""
    dem, hs, valid = load_terrain(site)
    log(f"  DTM valid {int(valid.sum()):,}px，高程 {dem[valid].min():.1f}~{dem[valid].max():.1f} m")

    if ep_arg:
        p, q = [np.array([float(x) for x in s.split(",")]) for s in ep_arg.split(";")]
        ep, src = (p, q), "cli"
        log(f"  端点（命令行）({p[0]:.0f},{p[1]:.0f}) ↔ ({q[0]:.0f},{q[1]:.0f})")
    else:
        ep, info = shp_endpoints(site, valid, log=log)
        src = info["source"]

    f1 = build_f1(site, valid, ep, gate=a.gate, dir_lambda=a.dir_lambda,
                  turn_lambda=a.turn_lambda, turn_min=a.turn_min, log=log)
    if f1 is None:
        return None, None, None, None

    enc, mse = load_or_train_ae(site, hs, valid, a.ae_epochs, a.ae_patches,
                                a.seed, "cpu", False, log)
    f2 = build_f2(site, hs, valid, enc, "cpu", log)

    X = np.concatenate([f1["f1"], f2], axis=0)
    X = np.where(valid[None], X, 0.0)

    import torch
    ck = os.path.join(RES, f"fusion_net_{site}.pt")
    net = build_net(X.shape[0])
    net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
    log(f"  载入融合网络 {os.path.relpath(ck, ROOT)}"
        f"（{sum(p.numel() for p in net.parameters()) / 1e3:.1f}k 参数）")
    prob = infer(net, X, valid, "cpu", log=log)
    return prob, dem, valid, dict(ep=[list(map(float, ep[0])), list(map(float, ep[1]))],
                                  source=src, ae_mse=mse,
                                  mask_px=int((prob > a.thr).sum()))


# ------------------------------------------------------------ 采样 ----
def bilinear(img, r, c):
    """双线性插值（越界按边缘夹取）。r/c 为浮点像元坐标（中心约定）。"""
    H, W = img.shape
    r = np.asarray(r, dtype=np.float64); c = np.asarray(c, dtype=np.float64)
    ri, ci = np.floor(r), np.floor(c)
    dr, dc = r - ri, c - ci
    r0 = np.clip(ri, 0, H - 1).astype(np.int32); r1 = np.clip(ri + 1, 0, H - 1).astype(np.int32)
    c0 = np.clip(ci, 0, W - 1).astype(np.int32); c1 = np.clip(ci + 1, 0, W - 1).astype(np.int32)
    return ((img[r0, c0] * (1 - dr) * (1 - dc) + img[r0, c1] * (1 - dr) * dc +
             img[r1, c0] * dr * (1 - dc) + img[r1, c1] * dr * dc))


def point_keep(site, prob, dem, valid, a, log):
    """逐点算 P 与 H_agl，返回 (keep 掩膜, 统计)。分块读取避免爆内存。"""
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
    gsd, x0, y1 = float(meta["gsd_m"]), float(meta["x0"]), float(meta["y1"])
    H, W = prob.shape
    xyz_p = os.path.join(REAL, f"{site}_xyz_local.npy")
    xyz = np.load(xyz_p, mmap_mode="r")
    n = xyz.shape[0]
    keep = np.zeros(n, dtype=bool)
    st = dict(n=n, agl_ok=0, p_ok=0, in_dom=0,
              agl_pct=[], p_pct=[])
    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        xl = np.asarray(xyz[s:e], dtype=np.float64)
        x = xl[:, 0] + float(o[0]); y = xl[:, 1] + float(o[1]); z = xl[:, 2] + float(o[2])
        c = (x - x0) / gsd - 0.5
        r = (y1 - y) / gsd - 0.5
        inside = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
        pv = np.zeros(len(x)); agl = np.full(len(x), np.inf)
        if inside.any():
            ii = np.flatnonzero(inside)
            pv[ii] = bilinear(prob, r[ii], c[ii])
            # 域外/空洞处概率置 0（prob 在 valid 外是网络对全零输入的响应，不可信）
            vb = bilinear(valid.astype(np.float64), r[ii], c[ii])
            pv[ii] = np.where(vb > 0.5, pv[ii], 0.0)
            agl[ii] = z[ii] - bilinear(dem, r[ii], c[ii])
        k = (pv >= a.thr) & (agl <= a.agl)
        keep[s:e] = k
        st["in_dom"] += int(inside.sum())
        st["p_ok"] += int((pv >= a.thr).sum())
        st["agl_ok"] += int((agl <= a.agl).sum())
        if s == 0:
            st["agl_pct"] = [float(np.percentile(agl, q)) for q in (1, 5, 50, 95)]
            st["p_pct"] = [float(np.percentile(pv, q)) for q in (50, 90, 99)]
        if (s // CHUNK) % 5 == 0:
            log(f"    点 {e / 1e6:.0f}M / {n / 1e6:.1f}M  ({time.time() - t0:.0f}s)")
    st["keep_px"] = int(keep.sum())
    st["keep_frac"] = round(float(keep.sum() / n), 5)
    log(f"  逐点：域内 {st['in_dom'] / 1e6:.2f}M（{st['in_dom'] / n * 100:.1f}%）；"
        f"P≥{a.thr:g} {st['p_ok'] / 1e6:.2f}M；H_agl≤{a.agl:g}m "
        f"{st['agl_ok'] / 1e6:.2f}M；同时满足 {st['keep_px'] / 1e6:.2f}M"
        f"（{st['keep_frac'] * 100:.2f}%）")
    log(f"  H_agl 分位 P1/P5/P50/P95 = " +
        " / ".join(f"{v:.2f}" for v in st["agl_pct"]) + " m；"
        f"P 分位 P50/P90/P99 = " + " / ".join(f"{v:.3f}" for v in st["p_pct"]))
    return keep, st


# ------------------------------------------------------------ 评价 ----
def metrics(keep, label, sel=None):
    """点级 P/R/F1（与 9 月回波法同口径）。sub 抽样时 sel 为布尔掩膜。"""
    if sel is not None:
        keep, label = keep[sel], label[sel]
    tp = int((keep & label).sum())
    fp = int((keep & ~label).sum())
    fn = int((~keep & label).sum())
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, keep_px=int(keep.sum()),
                truth_px=int(label.sum()),
                precision=float(pr), recall=float(rc),
                f1=float(2 * pr * rc / max(pr + rc, 1e-9)),
                iou=float(tp / max(tp + fp + fn, 1)))


def echo_baseline(site, n):
    """9 月纯回波法：从缓存的保留索引重建 keep 掩膜。"""
    p = os.path.join(REAL, f"{site}_filtered_回波法_idx.npy")
    if not os.path.exists(p):
        return None
    idx = np.load(p, mmap_mode="r")
    keep = np.zeros(n, dtype=bool)
    keep[np.asarray(idx)] = True
    return keep


def main():
    p = argparse.ArgumentParser(description="2D 融合模型指导的 3D 点云滤波（P≥thr 且 H_agl≤agl）")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--ep", type=str, default=None, help='端点 "r,c;r,c"；缺省用 SHP 骨架最远两点')
    p.add_argument("--thr", type=float, default=0.5, help="走廊概率阈值")
    p.add_argument("--agl", type=float, default=1.0, help="离地高度上限（m）")
    p.add_argument("--gate", type=float, default=3.0, help="分支1 距离场带宽 px")
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    p.add_argument("--ae-epochs", type=int, default=20)
    p.add_argument("--ae-patches", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write-las", action="store_true", help="导出保留点的 LAS（大文件）")
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "model_guided_3d_filter.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True); lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"2D 模型指导的 3D 滤波 · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"判据 keep = (P ≥ {a.thr:g}) & (H_agl ≤ {a.agl:g} m)；对照组=9 月纯回波法 "
        f"(rn==1)|(rn==nr)")
    log("SHP 只在最后评分，不参与概率图生成与滤波")

    out = []
    for s in a.sites:
        t0 = time.time()
        log(f"\n[{s}]")
        prob, dem, valid, info = prob_map(s, a.ep, a, log)
        if prob is None:
            continue
        np.savez_compressed(os.path.join(RES, f"model_guided_prob_{s}.npz"),
                            prob=prob.astype(np.float32), valid=valid.astype(np.uint8),
                            ep0=np.array(info["ep"][0]), ep1=np.array(info["ep"][1]))
        keep, pst = point_keep(s, prob, dem, valid, a, log)

        label = np.asarray(np.load(os.path.join(REAL, f"{s}_label.npy"),
                                   mmap_mode="r"), dtype=bool)
        m_new = metrics(keep, label)
        base = echo_baseline(s, pst["n"])
        m_base = metrics(base, label) if base is not None else None

        log(f"  新方法：P={m_new['precision']:.4f} R={m_new['recall']:.4f} "
            f"F1={m_new['f1']:.4f}（TP {m_new['tp']:,}/FP {m_new['fp']:,}/"
            f"FN {m_new['fn']:,}）")
        if m_base:
            log(f"  回波法：P={m_base['precision']:.4f} R={m_base['recall']:.4f} "
                f"F1={m_base['f1']:.4f}（TP {m_base['tp']:,}/FP {m_base['fp']:,}/"
                f"FN {m_base['fn']:,}）")
            log(f"  增益：F1 {m_base['f1']:.4f}→{m_new['f1']:.4f}"
                f"（{(m_new['f1'] - m_base['f1']) / max(m_base['f1'], 1e-9) * 100:+.1f}%），"
                f"P {m_base['precision']:.4f}→{m_new['precision']:.4f}"
                f"（×{m_new['precision'] / max(m_base['precision'], 1e-9):.1f}），"
                f"R {m_base['recall']:.4f}→{m_new['recall']:.4f}")

        if a.write_las:
            from pointcloud_hyper_filter import write_las
            xyz = np.load(os.path.join(REAL, f"{s}_xyz_local.npy"), mmap_mode="r")
            ft = np.load(os.path.join(REAL, f"{s}_lasefeat.npy"), mmap_mode="r")
            idx = np.flatnonzero(keep)
            out_p = os.path.join(REAL, f"{s}_model_guided.las")
            write_las(out_p, np.asarray(xyz[idx]), np.rint(np.asarray(ft[idx, 2])).astype(np.int16),
                      np.rint(np.asarray(ft[idx, 0])).astype(np.uint8),
                      np.rint(np.asarray(ft[idx, 1])).astype(np.uint8))
            log(f"  导出 {os.path.relpath(out_p, ROOT)}（{len(idx):,} 点）")

        # ---------------- 图 ----------------
        os.makedirs(FIG, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(1, 2, figsize=(11.4, 5.2), dpi=145)
        ax[0].imshow(np.where(valid, prob, np.nan), cmap="magma")
        ax[0].contour((prob > a.thr) & valid, levels=[0.5], colors="#00e5ff", linewidths=1.0)
        ax[0].set_title(f"CNN 走廊概率（青线=P>{a.thr:g}）", fontsize=10)
        ax[0].set_xticks([]); ax[0].set_yticks([])

        names = ["精确率", "召回率", "F1"]
        nv = [m_new["precision"], m_new["recall"], m_new["f1"]]
        bv = [m_base["precision"], m_base["recall"], m_base["f1"]] if m_base else [0, 0, 0]
        xi = np.arange(3)
        ax[1].bar(xi - 0.2, bv, 0.38, label="9月纯回波法", color="#9aa0a6")
        ax[1].bar(xi + 0.2, nv, 0.38, label=f"P≥{a.thr:g} & AGL≤{a.agl:g}m", color="#1f77b4")
        for i, v in enumerate(bv):
            ax[1].text(i - 0.2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5)
        for i, v in enumerate(nv):
            ax[1].text(i + 0.2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5)
        ax[1].set_xticks(xi); ax[1].set_xticklabels(names)
        ax[1].set_ylim(0, max(max(nv), max(bv)) * 1.25 + 0.02)
        ax[1].legend(fontsize=9); ax[1].set_title("点级指标（同一 label、同口径）", fontsize=10)
        ax[1].grid(axis="y", alpha=0.25)
        fig.suptitle(f"{s} · 2D 模型指导的 3D 滤波  "
                     f"保留 {m_new['keep_px'] / 1e6:.2f}M / {pst['n'] / 1e6:.1f}M 点", fontsize=11)
        fig.tight_layout()
        fp = os.path.join(FIG, f"model_guided_3d_{s}.png")
        fig.savefig(fp); plt.close(fig)

        out.append(dict(site=s, thr=a.thr, agl=a.agl, endpoints=info,
                        points=pst, new=m_new, echo_baseline=m_base,
                        seconds=round(time.time() - t0, 1),
                        figure=os.path.relpath(fp, ROOT)))

    log("\n" + "=" * 78)
    log(f"{'站点':>8s} {'方法':>10s} {'保留点':>10s} {'TP':>10s} {'FP':>12s} {'FN':>9s} "
        f"{'P':>7s} {'R':>7s} {'F1':>7s}")
    log("-" * 92)
    for r in out:
        for nm, m in (("回波法", r["echo_baseline"]), ("模型+AGL", r["new"])):
            if not m:
                continue
            log(f"{r['site']:>8s} {nm:>10s} {m['keep_px']:>10,d} {m['tp']:>10,d} "
                f"{m['fp']:>12,d} {m['fn']:>9,d} {m['precision']:>7.4f} "
                f"{m['recall']:>7.4f} {m['f1']:>7.4f}")

    jp = os.path.join(RES, f"model_guided_3d_filter_t{a.thr:g}_agl{a.agl:g}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="2D 融合模型指导的 3D 滤波：keep=(P≥thr)&(H_agl≤agl)；"
                            "对照 9 月纯回波法",
                       thr=a.thr, agl=a.agl, sites=out), f, ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
