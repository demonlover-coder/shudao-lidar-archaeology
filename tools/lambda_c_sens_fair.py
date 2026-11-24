# -*- coding: utf-8 -*-
"""两站 λc 敏感性（最终公平点级 IoU）：λc=1..10。

与现有 `lambda_c_sens_yang.py`（路径级、两站口径不一致）不同，本脚本统一采用
`tools/eval_freeaxis.py` 的**公平点级**口径：

    P/R/IoU = point_fair（AGL≤0.3 m 的地面真值点；候选 P≥0.5；
              到轴线 3D 距离≤dist3 → 交付）

轴线为模型自产概率图的骨架（--axis skel），SHP 只在最后评分，不参与寻路/训练。

结果链：
    · 宏观寻路（openness + h1 方向先验 + 方向余弦惩罚 λc）
    → 路径膨胀 gate px 伪标签
    → 1.5k 微型 CNN 现场重训（96px patch、BCE+pos_weight、Adam、早停、seed=0）
    → 概率图（npz 的 prob）
    → 公平点级 IoU + 10 m 窗纵坡 P95

复用策略：
    · 杨家坝：`cold_start_杨家坝_lcfair{lc}.npz` 已存在则复用；λc=6 复用
      `cold_start_杨家坝_lc6.npz`（同一链路、同参数）；其余现场重训。
    · 西坝村：`cold_start_西坝村_lcs{lc}.npz` / `cold_start_西坝村_lc{lc}.npz`
      含 prob 即复用；缺则按 gate=5 补训。

用法：
    python tools/lambda_c_sens_fair.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from argparse import Namespace

import numpy as np
from scipy.ndimage import label as cc_label

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import cold_start_transfer as C                                  # noqa: E402
import path_extraction as PE                                     # noqa: E402
import p2_dem as PD                                              # noqa: E402
from eval_freeaxis import SS, axis_field, fill_nodata            # noqa: E402
from eval_line_iou import poly_mask_hr                           # noqa: E402
from eval_new_shp import LAS, RES, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402
from fig_pred_vs_truth import order_path                         # noqa: E402
from fusion_train import infer, train_net                        # noqa: E402

FIG = os.path.join(ROOT, "figures")
LCS = list(range(1, 11))
CH = 3_000_000

CFG = {
    "杨家坝": dict(gate=3.0, dist3=3.5, agl=2.0,
                   truth="old", truth_main=False,
                   ep_json="cold_start_杨家坝_ep.json"),
    "西坝村": dict(gate=5.0, dist3=3.5, agl=2.0,
                   truth="new", truth_main=True,
                   ep_json="cold_start_西坝村_lc6.json"),
}
# 与 cold_start_transfer 主模型完全一致的训练超参
TRAIN = dict(epochs=40, batch=32, lr=2e-3, patch=96, seed=0, device="cpu", in_ch=4)


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ 特征
def build_features(site, ep_json):
    """一次性构建与 λc 无关的地形特征与端点（杨家坝现场重训用）。"""
    cfg = CFG[site]
    dem_f, valid = C.load_dem(site, "mean", log)
    dem_min, _ = C.load_dem(site, "min", log)
    hs = C.hillshade_map(dem_f, valid)
    log("  特征 1/3：Hillshade")
    op, n_row, n_col, conf, minfo = C.macro_fields(dem_f, valid)
    log(f"  特征 2/3：openness/h1（h1 门控 P60={minfo['h1_gate_deg']}°，"
        f"{minfo['conf_px']:,}px）")
    cssc = C.cssc_map(dem_f, valid, n_row, n_col, log)
    log("  特征 3/3：CSSC")
    epj = json.load(open(os.path.join(RES, ep_json), encoding="utf-8"))
    ep = [np.asarray(PE.snap_to_domain(np.asarray(p, np.float64), valid), np.float64)
          for p in (epj["endpoints"]["ep0"], epj["endpoints"]["ep1"])]
    log(f"  端点固定 ({ep[0][0]:.0f},{ep[0][1]:.0f}) ↔ ({ep[1][0]:.0f},{ep[1][1]:.0f})")
    return dict(dem_f=dem_f, dem_min=dem_min, valid=valid, hs=hs, op=op,
                n_row=n_row, n_col=n_col, conf=conf, cssc=cssc, ep=ep, minfo=minfo)


# ------------------------------------------------------------------ 训练
def train_one(site, lc, F):
    """完整链路：λc 寻路 → gate 伪标签 → 现场训练 → 概率图 → npz。"""
    cfg = CFG[site]
    a = Namespace(z_lambda=C.LAMBDA_Z, lam_c=float(lc), slope_lambda=0.0,
                  dir_lambda=1.5, turn_lambda=0.2, turn_min=1,
                  max_grade=0.0, grade_lag=5,
                  gate=float(cfg["gate"]), **TRAIN)
    t0 = time.time()
    path, dist, cost, _ = C.macro_path(F["op"], F["valid"], F["ep"], F["n_row"],
                                       F["n_col"], F["conf"], a, log=log, dem=F["dem_f"])
    pseudo = (dist <= float(cfg["gate"])) & F["valid"]
    log(f"  λc={lc}: 伪标签 {int(pseudo.sum()):,}px（路径 {int(path.sum())}px，"
        f"gate={cfg['gate']:g}px）")
    X, _ = C.build_inputs(F["hs"], F["cssc"], F["valid"], dist, cost,
                          a.in_ch, a.gate, log)
    net, hist = train_net(X, pseudo.astype(np.float32), F["valid"], a.epochs,
                          a.batch, a.lr, a.patch, a.seed, a.device, log)
    prob = infer(net, X, F["valid"], a.device, log=log)
    name = f"cold_start_{site}_lcfair{lc}.npz"
    np.savez_compressed(os.path.join(RES, name),
                        prob=prob.astype(np.float32),
                        path=path.astype(np.uint8),
                        pseudo=pseudo.astype(np.float32),
                        cssc=F["cssc"].astype(np.float32),
                        openness=F["op"].astype(np.float32))
    log(f"  λc={lc}: 保存 {name}（{len(hist)} epoch，{time.time() - t0:.0f}s）")
    return os.path.join(RES, name), name, dict(epochs=len(hist), seed=a.seed,
                                               seconds=round(time.time() - t0, 1))


def find_npz(site, lc):
    """按优先级找含 prob 的现成 npz。"""
    names = [f"cold_start_{site}_lcfair{lc}.npz"]
    if site == "杨家坝" and lc in (4, 5, 6, 7):
        names.append(f"cold_start_{site}_lc{lc}.npz")
    if site == "西坝村":
        names += [f"cold_start_{site}_lcs{lc}.npz", f"cold_start_{site}_lc{lc}.npz"]
    for n in names:
        p = os.path.join(RES, n)
        if os.path.exists(p) and "prob" in np.load(p).files:
            return p, n
    return None, None


def ensure_npzs(site):
    """返回 {lc: dict(npz, name, retrained)}；缺失/无 prob 的现场补训。"""
    cfg = CFG[site]
    out, F = {}, None
    log(f"\n[{site}] 收集 λc=1..10 的概率图")
    for lc in LCS:
        p, name = find_npz(site, lc)
        if p is not None:
            # lcfair* = 本脚本现场重训产物；其余为既有实验 npz（复用）
            out[lc] = dict(npz=p, name=name, retrained=("lcfair" in name), train=None)
            log(f"  λc={lc:>2d}: 复用 {name}")
            continue
        if F is None:
            log(f"  [{site}] 缺少可用 npz → 现场构建特征并补训（gate={cfg['gate']:g}）")
            F = build_features(site, cfg["ep_json"])
        p, name, tinfo = train_one(site, lc, F)
        out[lc] = dict(npz=p, name=name, retrained=True, train=tinfo)
    return out


# ------------------------------------------------------------------ 评分
def build_truth(site, valid):
    """评分真值（与 eval_freeaxis --truth/--truth-main 完全一致）。"""
    cfg = CFG[site]
    x0, y1, _wm, _hm, gsd, d = PD.dem_window(site)
    H, W = valid.shape
    if cfg["truth"] == "old":
        gt = np.load(os.path.join(RES, f"mask_1m_{site}.npy")).astype(bool) & valid
        src = f"旧 SHP 栅格 mask_1m_{site}.npy"
    else:
        _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        mh = poly_mask_hr(polys, x0, y1, gsd, H, W, SS)
        gt = (mh.reshape(H, SS, W, SS).mean(axis=(1, 3)) >= 0.5) & valid
        src = f"新 SHP {SHP_NEW[site]['poly']}"
    if cfg["truth_main"]:
        lb, ncc = cc_label(gt)
        if ncc > 1:
            sz = np.bincount(lb.ravel())
            sz[0] = 0
            gt = lb == sz.argmax()
    from skimage.morphology import skeletonize
    gt_skel = skeletonize(gt)
    log(f"  真值[{src}{' + 主板' if cfg['truth_main'] else ''}]：{int(gt.sum()):,}px，"
        f"骨架 {int(gt_skel.sum()):,}px")
    return gt, gt_skel


def grade_p95(dem, path):
    """10 m 窗纵坡 P95（%，|z(i+10)−z(i)|/10 m），与 diag_path_grade 同算法。"""
    pl = order_path(np.asarray(path, bool))
    if pl is None or len(pl) < 12:
        return float("nan")
    p = np.asarray(pl, np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    si = np.arange(0.0, s[-1], 1.0)
    rr = np.clip(np.round(np.interp(si, s, p[:, 0])).astype(int), 0, dem.shape[0] - 1)
    cc = np.clip(np.round(np.interp(si, s, p[:, 1])).astype(int), 0, dem.shape[1] - 1)
    zz = dem[rr, cc]
    g = np.abs(zz[10:] - zz[:-10]) / 10.0
    return float(np.percentile(g, 95) * 100)


def score_site(site, npzs):
    """单遍流式读 LAS，同时算 10 个 λc 的 point_fair（与 eval_freeaxis 同式）。"""
    cfg = CFG[site]
    x0, y1, _wm, _hm, gsd, d = PD.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    dem = fill_nodata(d["dem_min"], valid)
    gt, _gt_skel = build_truth(site, valid)

    log(f"  [{site}] 轴线/距离场（skel，P≥0.5）")
    fields, prob_any = {}, np.zeros((H, W), np.float32)
    grades = {}
    for lc in LCS:
        p = npzs[lc]["npz"]
        z = np.load(p)
        prob = np.where(valid, z["prob"].astype(np.float64), 0.0)
        d2, zax = axis_field(site, prob, valid, H, W, gsd, x0, y1, None,
                             dem, "skel", p)
        fields[lc] = (prob, d2, zax)
        prob_any = np.maximum(prob_any, prob.astype(np.float32))
        grades[lc] = round(grade_p95(dem, z["path"]), 4)

    tp_f = {lc: 0 for lc in LCS}
    fp_f = {lc: 0 for lc in LCS}
    n_cand = {lc: 0 for lc in LCS}
    n_tr_g = 0
    n_pts = 0
    t0 = time.time()
    import laspy
    with laspy.open(LAS[site]) as f:
        for pts in f.chunk_iterator(CH):
            X = np.asarray(pts.x, np.float64)
            Y = np.asarray(pts.y, np.float64)
            Z = np.asarray(pts.z, np.float64)
            n_pts += X.size
            c = np.floor((X - x0) / gsd).astype(np.int64)
            r = np.floor((y1 - Y) / gsd).astype(np.int64)
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            if not ok.any():
                continue
            X, Y, Z, c, r = X[ok], Y[ok], Z[ok], c[ok], r[ok]
            agl = Z - dem[r, c]
            tr = gt[r, c]
            n_tr_g += int((tr & (agl <= 0.3)).sum())     # FN 分母（地面口径）
            sub = np.flatnonzero(prob_any[r, c] >= 0.5)  # 任一 λc 的候选并集
            if sub.size == 0:
                continue
            rr, cc2, zz, ag, trs = r[sub], c[sub], Z[sub], agl[sub], tr[sub]
            for lc, (prob, d2, zax) in fields.items():
                cand = prob[rr, cc2] >= 0.5
                if not cand.any():
                    continue
                n_cand[lc] += int(cand.sum())
                d3 = np.hypot(d2[rr, cc2], zz - zax[rr, cc2])
                keep = cand & (d3 <= cfg["dist3"]) & (ag <= cfg["agl"])
                kg = keep & (ag <= 0.3)
                tp_f[lc] += int((kg & trs).sum())
                fp_f[lc] += int((kg & ~trs).sum())
            log(f"    点云 {n_pts / 1e6:.1f}M（{time.time() - t0:.0f}s）")
    log(f"  [{site}] 地面真值点 {n_tr_g:,}；点云 {n_pts:,}；"
        f"评分 {time.time() - t0:.0f}s")

    rows = {}
    for lc in LCS:
        tp, fp = tp_f[lc], fp_f[lc]
        p = tp / max(tp + fp, 1)
        rc = tp / max(tp + (n_tr_g - tp), 1)
        rows[lc] = dict(P=round(p, 4), R=round(rc, 4),
                        IoU=round(tp / max(fp + n_tr_g, 1), 4),
                        grade_p95=grades[lc], n_cand=n_cand[lc])
    return rows


# ------------------------------------------------------------------ 汇总输出
def plateau(rows, tol=0.01):
    """IoU ≥ max−tol 的最大连续 λc 区间。"""
    v = {lc: r["IoU"] for lc, r in rows.items() if r.get("IoU") is not None}
    if not v:
        return None
    mx = max(v.values())
    good = [lc for lc in sorted(v) if v[lc] >= mx - tol]
    runs, cur = [], [good[0]]
    for lc in good[1:]:
        if lc == cur[-1] + 1:
            cur.append(lc)
        else:
            runs.append(cur)
            cur = [lc]
    runs.append(cur)
    best = max(runs, key=len)
    return best[0], best[-1], min(v[lc] for lc in best), mx


def largest_drop(rows):
    v = {lc: r["IoU"] for lc, r in rows.items() if r.get("IoU") is not None}
    pairs = [(v[b] - v[a], a, b) for a, b in zip(sorted(v)[:-1], sorted(v)[1:])]
    return max(pairs, key=lambda t: abs(t[0])) if pairs else None


def make_figure(results, path_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    sites = ["杨家坝", "西坝村"]
    colors = {"杨家坝": "#1f4fd8", "西坝村": "#e07b00"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, site in zip(axes, sites):
        rows = {int(k): v for k, v in results[site].items()}
        xs = [lc for lc in LCS if rows[lc]["IoU"] is not None]
        iou = [rows[lc]["IoU"] for lc in xs]
        gr = [rows[lc]["grade_p95"] for lc in xs]
        pl = plateau(rows)
        ax.plot(xs, iou, "o-", color=colors[site], lw=2, ms=6, label="公平点级 IoU")
        for x, y in zip(xs, iou):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color=colors[site])
        if pl and pl[0] < pl[1]:
            ax.axvspan(pl[0], pl[1], color="#9ad0f5", alpha=0.22,
                       label=f"平台区 {pl[0]}–{pl[1]}")
        ax.axvline(6, color="#e3001b", ls="--", lw=1, label="锁定值 λc=6")
        ax.set_xlabel("λc（方向余弦惩罚权重）")
        ax.set_ylabel("公平点级 IoU", color=colors[site])
        ax.tick_params(axis="y", labelcolor=colors[site])
        ax.set_xticks(LCS)
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(xs, gr, "s--", color="#2ca02c", lw=1.4, ms=5, label="路径纵坡 P95")
        ax2.axhline(20, color="#2ca02c", ls=":", lw=1, alpha=0.7)
        ax2.set_ylabel("10 m 窗纵坡 P95 (%)", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right", framealpha=0.95)
        ax.margins(y=0.16)
        ax2.margins(y=0.16)
        if pl:
            ax.set_title(f"{site}：平台区 λc={pl[0]}–{pl[1]}"
                         f"（IoU {pl[2]:.3f}–{pl[3]:.3f}）", fontsize=11)
        else:
            ax.set_title(site, fontsize=11)
    fig.tight_layout()
    fig.savefig(path_out, dpi=160)
    plt.close(fig)
    log(f"写出 {path_out}")


def patch_legacy_json():
    """在西坝村路径级数据旁标注废弃，不改动原始数值。"""
    p = os.path.join(RES, "lambda_c_sensitivity_杨家坝.json")
    txt = open(p, encoding="utf-8").read()
    if '"deprecated"' in txt:
        return
    needle = '  "西坝村": {\n   "gate_px": 5.0,'
    repl = ('  "西坝村": {\n'
            '   "deprecated": true,\n'
            '   "reason": "path 数组未栅格化，数值无效；以 lambda_c_sensitivity_fair.json 为准",\n'
            '   "gate_px": 5.0,')
    if needle not in txt:
        raise RuntimeError("未找到西坝村路径级数据锚点，放弃修改旧 JSON")
    open(p, "w", encoding="utf-8").write(txt.replace(needle, repl, 1))
    log(f"标注旧 JSON：{p}（西坝村路径级数据 deprecated=true）")


def main():
    t_all = time.time()
    os.makedirs(FIG, exist_ok=True)

    npzs = {site: ensure_npzs(site) for site in ("杨家坝", "西坝村")}
    scores = {site: score_site(site, npzs[site]) for site in ("杨家坝", "西坝村")}

    out = {}
    for site in ("杨家坝", "西坝村"):
        out[site] = {}
        for lc in LCS:
            s = scores[site][lc]
            out[site][str(lc)] = {
                "P": s["P"], "R": s["R"], "IoU": s["IoU"],
                "grade_p95": s["grade_p95"],
                "npz": npzs[site][lc]["name"],
                "retrained": bool(npzs[site][lc]["retrained"]),
            }

    jp = os.path.join(RES, "lambda_c_sensitivity_fair.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"写出 {jp}")

    cp = os.path.join(RES, "lambda_c_sensitivity_fair.csv")
    with open(cp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "lam_c", "P", "R", "IoU", "grade_p95_pct", "npz", "retrained"])
        for site in ("杨家坝", "西坝村"):
            for lc in LCS:
                r = out[site][str(lc)]
                w.writerow([site, lc, r["P"], r["R"], r["IoU"], r["grade_p95"],
                            r["npz"], r["retrained"]])
    log(f"写出 {cp}")

    make_figure(out, os.path.join(FIG, "lambda_c_sensitivity_fair.png"))
    patch_legacy_json()

    for site in ("杨家坝", "西坝村"):
        rows = scores[site]
        pl = plateau(rows)
        dp = largest_drop(rows)
        log(f"\n[{site}] λc 1..10 公平点级 IoU")
        for lc in LCS:
            r = rows[lc]
            log(f"  λc={lc:>2d}  P={r['P']:.4f}  R={r['R']:.4f}  IoU={r['IoU']:.4f}  "
                f"P95={r['grade_p95']:5.1f}%  {'重训' if npzs[site][lc]['retrained'] else '复用'}")
        if pl:
            log(f"  平台区 λc={pl[0]}–{pl[1]}（IoU {pl[2]:.4f}–{pl[3]:.4f}）")
        if dp:
            log(f"  最大相变 λc={dp[1]}→{dp[2]}（ΔIoU {dp[0]:+.4f}）")
    log(f"\n总耗时 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
