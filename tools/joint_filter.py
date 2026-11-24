# -*- coding: utf-8 -*-
"""回波法初筛 + 2D 概率掩膜的联合滤波（joint_filter.py）。

流程
----
1. **回波法初筛**：`keep = (return_number == 1) | (return_number == number_of_returns)`
   （即首末回波，项目既有基线）。
2. **2D 概率投影**：`(x,y)` 按 `data/meta_{site}.json` 的 x0/y1/gsd 映射到概率图像元，
   双线性取值；域外或 valid 外概率置 0。
3. **空间掩膜**：在初筛结果上再叠加 `P ≥ thr`，thr 扫描 {0.5, 0.3, 0.2}。
4. **评价**：真值 `data_real/{site}_label.npy`（SHP 面内点）。三组对照：
   · 纯回波法（只用步骤 1）
   · 纯 2D 法（不筛回波，只用步骤 3）
   · 联合（步骤 1 + 步骤 3）

概率图来源
----------
默认读缓存 `results/model_guided_prob_{site}.npz`（由 model_guided_3d_filter.py 生成，
即 fusion_train 的 CNN 对 F1(2ch)+F2(6ch) 的全图推理）；加 `--refresh-prob` 重算。

用法
----
    python tools/joint_filter.py
    python tools/joint_filter.py --refresh-prob
    python tools/joint_filter.py --sites 杨家坝 --thrs 0.5 0.3 0.2 0.1
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


def load_prob(site, a, log):
    p = os.path.join(RES, f"model_guided_prob_{site}.npz")
    if os.path.exists(p) and not a.refresh_prob:
        z = np.load(p)
        log(f"  概率图：读缓存 {os.path.relpath(p, ROOT)}（max {z['prob'].max():.3f}）")
        return z["prob"].astype(np.float64), z["valid"] == 1
    prob, dem, valid, info = prob_map(site, a.ep, a, log)
    return prob, valid


def project_prob(site, prob, valid, log):
    """逐点双线性取概率，同时返回回波字段与真值（分块读，只保留必要数组）。"""
    o = np.load(os.path.join(REAL, f"{site}_origin.npy"))
    meta = json.load(open(os.path.join(DATA, f"meta_{site}.json"), encoding="utf-8"))
    gsd, x0, y1 = float(meta["gsd_m"]), float(meta["x0"]), float(meta["y1"])
    H, W = prob.shape
    xyz = np.load(os.path.join(REAL, f"{site}_xyz_local.npy"), mmap_mode="r")
    ft = np.load(os.path.join(REAL, f"{site}_lasefeat.npy"), mmap_mode="r")
    lb = np.load(os.path.join(REAL, f"{site}_label.npy"), mmap_mode="r")
    n = xyz.shape[0]

    pv = np.zeros(n, dtype=np.float32)
    echo = np.zeros(n, dtype=bool)
    n_in = 0
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        xl = np.asarray(xyz[s:e], dtype=np.float64)
        f = np.asarray(ft[s:e])
        c = (xl[:, 0] + o[0] - x0) / gsd - 0.5
        r = (y1 - xl[:, 1] - o[1]) / gsd - 0.5
        ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
        if ins.any():
            ii = np.flatnonzero(ins)                      # 块内局部索引
            v = bilinear(prob, r[ii], c[ii])
            vb = bilinear(valid.astype(np.float64), r[ii], c[ii])
            pv[s + ii] = np.where(vb > 0.5, v, 0.0)        # 必须加块偏移 s
        n_in += int(ins.sum())
        rn, nr = np.rint(f[:, 0]).astype(np.int32), np.rint(f[:, 1]).astype(np.int32)
        echo[s:e] = (rn == 1) | (rn == nr)

    label = np.asarray(lb, dtype=bool)
    log(f"  投影：{n:,} 点，域内 {n_in / n * 100:.1f}%；回波初筛保留 "
        f"{int(echo.sum()):,}（{echo.mean() * 100:.2f}%）；"
        f"P 分位 P50/P90/P99 = " +
        " / ".join(f"{v:.3f}" for v in np.percentile(pv, [50, 90, 99])))
    return pv, echo, label, n


def main():
    p = argparse.ArgumentParser(description="回波法初筛 + 2D 概率掩膜的联合滤波")
    p.add_argument("--sites", nargs="*", default=list(SITES))
    p.add_argument("--thrs", nargs="*", type=float, default=[0.5, 0.3, 0.2])
    p.add_argument("--ep", type=str, default=None, help='端点 "r,c;r,c"；缺省用 SHP 骨架最远两点')
    p.add_argument("--refresh-prob", action="store_true")
    p.add_argument("--gate", type=float, default=3.0)
    p.add_argument("--dir-lambda", type=float, default=1.5)
    p.add_argument("--turn-lambda", type=float, default=0.2)
    p.add_argument("--turn-min", type=int, default=1)
    p.add_argument("--ae-epochs", type=int, default=20)
    p.add_argument("--ae-patches", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.makedirs(RES, exist_ok=True)
    lf = open(os.path.join(RES, "joint_filter.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True); lf.write(m + "\n"); lf.flush()

    log("\n" + "=" * 78)
    log(f"联合滤波 joint_filter · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"步骤：回波初筛 (rn==1)|(rn==nr) → P≥thr，thr ∈ {a.thrs}")
    log("SHP 只在最后评分")

    out = []
    for s in a.sites:
        t0 = time.time()
        log(f"\n[{s}]")
        prob, valid = load_prob(s, a, log)
        pv, echo, label, n = project_prob(s, prob, valid, log)

        rows = []
        m_echo = metrics(echo, label)
        rows.append(dict(name="纯回波法", kind="echo", thr=None, **m_echo))
        for t in a.thrs:
            pm = pv >= t
            rows.append(dict(name=f"纯2D (P≥{t:g})", kind="2d", thr=t,
                             **metrics(pm, label)))
            rows.append(dict(name=f"联合 (回波+P≥{t:g})", kind="joint", thr=t,
                             **metrics(echo & pm, label)))
        for r in rows:
            log(f"  {r['name']:<18s} 保留 {r['keep']:>10,d}（{r['keep_frac'] * 100:6.2f}%）"
                f"  P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")

        best = max((r for r in rows if r["kind"] != "echo"), key=lambda r: r["f1"])
        e = next(r for r in rows if r["kind"] == "echo")
        log(f"  最优：{best['name']} F1={best['f1']:.4f}"
            f"（回波法 {e['f1']:.4f}，相对 "
            f"{(best['f1'] - e['f1']) / max(e['f1'], 1e-9) * 100:+.1f}%）")

        os.makedirs(FIG, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=145)
        rmask = (prob >= a.thrs[0]) & valid
        truth_ras = np.load(os.path.join(RES, f"mask_1m_{s}.npy")).astype(bool)
        ax[0].imshow(np.where(valid, prob, np.nan), cmap="magma")
        ax[0].contour(rmask, levels=[0.5], colors="#00e5ff", linewidths=1.0)
        ax[0].contour(truth_ras, levels=[0.5], colors="#00ff66", linewidths=1.0)
        ax[0].set_title(f"概率图（青=P≥{a.thrs[0]:g}，绿=SHP）", fontsize=10)
        ax[0].set_xticks([]); ax[0].set_yticks([])

        names = [r["name"].replace(" (", "\n(") for r in rows]
        f1s = [r["f1"] for r in rows]
        cols = ["#9aa0a6" if r["kind"] == "echo" else
                ("#ff7f0e" if r["kind"] == "2d" else "#1f77b4") for r in rows]
        bars = ax[1].bar(range(len(rows)), f1s, color=cols, width=0.66)
        for b, r in zip(bars, rows):
            ax[1].text(b.get_x() + b.get_width() / 2, r["f1"] + max(f1s) * 0.02,
                       f"{r['f1']:.3f}\nP{r['precision']:.2f}/R{r['recall']:.2f}",
                       ha="center", fontsize=6.8)
        ax[1].set_xticks(range(len(rows))); ax[1].set_xticklabels(names, fontsize=7.5)
        ax[1].set_ylabel("F1"); ax[1].set_ylim(0, max(f1s) * 1.32 + 0.01)
        ax[1].grid(axis="y", alpha=0.25)
        ax[1].set_title("三组对照的 F1（灰=回波，橙=纯2D，蓝=联合）", fontsize=10)
        fig.suptitle(f"{s} · 回波初筛 + 2D 概率掩膜联合滤波", fontsize=11)
        fig.tight_layout()
        fp = os.path.join(FIG, f"joint_filter_{s}.png")
        fig.savefig(fp); plt.close(fig)

        out.append(dict(site=s, n=n, rows=rows,
                        best=best["name"],
                        figure=os.path.relpath(fp, ROOT),
                        seconds=round(time.time() - t0, 1)))

    log("\n" + "=" * 78)
    log(f"{'站点':>8s} {'方案':>20s} {'保留点':>12s} {'保留率':>8s} {'TP':>10s} "
        f"{'FP':>12s} {'FN':>10s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    log("-" * 104)
    for r in out:
        for d in r["rows"]:
            log(f"{r['site']:>8s} {d['name']:>20s} {d['keep']:>12,d} "
                f"{d['keep_frac'] * 100:>7.2f}% {d['tp']:>10,d} {d['fp']:>12,d} "
                f"{d['fn']:>10,d} {d['precision']:>7.4f} {d['recall']:>7.4f} {d['f1']:>7.4f}")

    jp = os.path.join(RES, "joint_filter.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(desc="回波初筛 (rn==1)|(rn==nr) + 2D 概率掩膜 P≥thr；对照纯回波/纯2D",
                       thrs=a.thrs, sites=out), f, ensure_ascii=False, indent=2)
    log(f"\n保存 {os.path.relpath(jp, ROOT)}")
    lf.close()


if __name__ == "__main__":
    main()
