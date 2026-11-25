# -*- coding: utf-8 -*-
"""用更新后的 SHP 重新评分（只读，不改动 data/ 与既有 results 结论）。

新几何：
    杨家坝  `杨家坝_蜀道_中轴线.shp`  PolyLineZ，3 段，总长 438.6 m
    杨家坝  `杨家坝_蜀道_本体范围.shp` PolygonZ，3 面，合计 1,315.6 m²
    西坝村  `XBC_orig_clean.shp`      Polygon，7 面，与旧 `西坝村-f1.shp` 字节一致

真值口径（全部栅格化到各站 1 m 网格，与 meta 的 x0/y1/gsd 一致）：
    line_rX   到中轴线距离 ≤ X m（X = 1.0/1.5/2.0/2.5/3.0/5.0）
    poly      面要素 4× 超采样 + 覆盖≥0.5（与 build_dataset 同口径）
    old       `data/{site}_1m.npz` 里的旧标签（校验用，应复现已发布数字）

被评对象：
    p05       2D 概率 P≥0.5 的全部点（冷启动 4ch，`cold_start_{site}_ep.npz`）
    dplus     冷启动 D+ 交付点（`cold_start_{site}_ep_keep_idx.npy`）
    mg05      域内 8ch 概率 P≥0.5（`model_guided_prob_{site}.npz`，有则评）

输出：
    results/eval_new_shp.json    各真值口径 × 被评对象的 P/R/F1/IoU
    results/eval_new_shp.log     运行日志（追加）
    results/dist_centerline_{site}.npy   到中轴线距离场（1 m，米）

用法：
    python tools/eval_new_shp.py --site 杨家坝
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
_DATA_ROOT = os.environ.get("SHUDAO_DATA_ROOT", os.path.join(ROOT, "data"))
REAL = os.path.join(_DATA_ROOT, "data_real")
SHP_BASE = os.path.join(_DATA_ROOT, "survey-shp")

SHP_NEW = {
    "杨家坝": {"line": r"杨家坝\杨家坝_蜀道_中轴线.shp",
               "poly": r"杨家坝\杨家坝_蜀道_本体范围.shp"},
    "西坝村": {"poly": r"西坝村\XBC_orig_clean.shp"},
}
LAS = {
    "杨家坝": os.path.join(SHP_BASE, "杨家坝", "yangjiaba.las"),
    "西坝村": os.path.join(SHP_BASE, "西坝村", "西坝村栈道-lidar.las"),
}
LINE_R = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]


def log(msg: str, path: str | None = None) -> None:
    print(msg)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def read_shp_any(path: str):
    """读 SHP：支持 线(3/13/23) 与 面(5/15/25)，返回 (type, [{'parts','pts'}])。"""
    raw = open(path, "rb").read()
    ftype = struct.unpack_from("<i", raw, 32)[0]
    shapes, pos = [], 100
    while pos + 8 <= len(raw):
        _n, clen = struct.unpack_from(">ii", raw, pos)
        pos += 8
        end = pos + clen * 2
        st = struct.unpack_from("<i", raw, pos)[0]
        if st in (3, 5, 13, 15, 23, 25):
            nparts, npoints = struct.unpack_from("<ii", raw, pos + 36)
            parts = struct.unpack_from("<%di" % nparts, raw, pos + 44)
            pts = np.frombuffer(raw, dtype="<f8", count=npoints * 2,
                                offset=pos + 44 + 4 * nparts).reshape(-1, 2)
            shapes.append({"parts": parts, "pts": pts})
        pos = end
    return ftype, shapes


def parts_of(sh):
    out = []
    for j, p0 in enumerate(sh["parts"]):
        p1 = sh["parts"][j + 1] if j + 1 < len(sh["parts"]) else len(sh["pts"])
        out.append(sh["pts"][p0:p1])
    return out


def poly_mask(shapes, x0, y1, gsd, H, W, ss=4):
    """面要素 → 1 m 掩膜（ss 倍超采样，覆盖≥0.5；与 build_dataset 同口径）。"""
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    img = Image.new("1", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(img)
    for sh in shapes:
        for xy in parts_of(sh):
            cols = (xy[:, 0] - x0) / gsd * ss
            rows = (y1 - xy[:, 1]) / gsd * ss
            d.polygon([(float(c), float(r)) for c, r in zip(cols, rows)], fill=1)
    hi = np.asarray(img, dtype=np.uint8).reshape(H, ss, W, ss)
    return (hi.mean(axis=(1, 3)) >= 0.5)


def line_dist_map(shapes, x0, y1, gsd, H, W):
    """每个 1 m 像元中心到折线的最近距离（米）。"""
    xs = x0 + (np.arange(W) + 0.5) * gsd
    ys = y1 - (np.arange(H) + 0.5) * gsd
    gx, gy = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    segs = []
    for sh in shapes:
        for xy in parts_of(sh):
            for k in range(len(xy) - 1):
                segs.append((xy[k], xy[k + 1]))
    d = np.full((H, W), np.inf, dtype=np.float32)
    for p0, p1 in segs:
        ax, ay = float(p0[0]), float(p0[1])
        bx, by = float(p1[0]), float(p1[1])
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        if L2 <= 1e-12:
            t = np.zeros_like(gx)
        else:
            t = np.clip(((gx - ax) * vx + (gy - ay) * vy) / L2, 0.0, 1.0)
        dx = gx - (ax + t * vx)
        dy = gy - (ay + t * vy)
        np.minimum(d, np.sqrt(dx * dx + dy * dy), out=d)
    return d


def prf(tp, fp, fn):
    P_ = tp / max(tp + fp, 1)
    R_ = tp / max(tp + fn, 1)
    F1 = 2 * P_ * R_ / max(P_ + R_, 1e-9)
    IoU = tp / max(tp + fp + fn, 1)
    return dict(precision=round(float(P_), 4), recall=round(float(R_), 4),
                f1=round(float(F1), 4), iou=round(float(IoU), 4))


def pixel_eval(pred, truth):
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    out = prf(tp, fp, fn)
    out.update({"truth_px": int(truth.sum()), "pred_px": int(pred.sum()),
                "tp": tp, "fp": fp, "fn": fn})
    return out


def bilinear(P, r, c):
    """双线性采样（索引约定：像元中心在整数索引处，与 cold_start 的 project_points 一致）。"""
    H, W = P.shape
    r = np.clip(r, 0, H - 1)
    c = np.clip(c, 0, W - 1)
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    r1 = np.minimum(r0 + 1, H - 1)
    c1 = np.minimum(c0 + 1, W - 1)
    fr = (r - r0).astype(np.float32)
    fc = (c - c0).astype(np.float32)
    return ((1 - fr) * (1 - fc) * P[r0, c0] + (1 - fr) * fc * P[r0, c1]
            + fr * (1 - fc) * P[r1, c0] + fr * fc * P[r1, c1])


def seg_dist(X, Y, segs):
    """点集到折线的最短距离（逐段向量化）。"""
    best = np.full(X.size, np.inf, dtype=np.float32)
    for p0, p1 in segs:
        ax, ay = float(p0[0]), float(p0[1])
        vx, vy = float(p1[0]) - ax, float(p1[1]) - ay
        L2 = vx * vx + vy * vy
        if L2 <= 1e-12:
            d = np.hypot(X - ax, Y - ay)
        else:
            t = np.clip(((X - ax) * vx + (Y - ay) * vy) / L2, 0.0, 1.0)
            d = np.hypot(X - (ax + t * vx), Y - (ay + t * vy))
        np.minimum(best, d.astype(np.float32), out=best)
    return best


def point_eval(site, las_path, line_shapes, poly_shapes, preds, chunk=3_000_000):
    """逐点精确真值：中轴线 = 点到折线距离；面 = 点在多边形内；另带旧标签对照。

    真值口径全部**逐点计算**（不走栅格查表），阈值 = 线距离 ≤ r 与点在面内；
    `old` 行用 `data_real/{site}_label.npy` 复现已发布数字，作为管线校验。
    """
    import laspy
    from matplotlib.path import Path

    z = np.load(os.path.join(ROOT, "data", f"{site}_1m.npz"))
    meta = json.load(open(os.path.join(ROOT, "data", f"meta_{site}.json"), encoding="utf-8"))
    x0, y1, gsd = float(meta["x0"]), float(meta["y1"]), float(meta["gsd_m"])
    H, W = z["dem_mean"].shape
    prob = preds["_prob"]
    valid = preds["_valid"]
    mg = preds.get("_mg")
    keep_idx = preds.get("_keep_idx")
    old = np.load(os.path.join(REAL, f"{site}_label.npy"), mmap_mode="r")

    segs = [ (xy[k], xy[k + 1]) for sh in (line_shapes or []) for xy in parts_of(sh)
             for k in range(len(xy) - 1) ]
    paths = [Path(xy) for sh in (poly_shapes or []) for xy in parts_of(sh)]
    dist_r = None
    if segs:
        dist_r = np.load(os.path.join(RES, f"dist_centerline_{site}.npy"))
    names = ["old"] + [f"line_{r:g}m" for r in LINE_R] + (["poly"] if paths else [])

    sets = [p for p in preds if not p.startswith("_")]
    cm = {n: {p: [0, 0, 0] for p in sets} for n in names}
    npos = {n: 0 for n in names}
    npred = {p: 0 for p in sets}
    keep_mask = None
    if keep_idx is not None:
        with laspy.open(las_path) as f:
            N = int(f.header.point_count)
        keep_mask = np.zeros(N, dtype=bool)
        keep_mask[keep_idx] = True
        npred["dplus"] = 0
        for n in names:
            cm[n]["dplus"] = [0, 0, 0]

    d_all, n_total = [], 0
    with laspy.open(las_path) as f:
        for pts in f.chunk_iterator(chunk):
            X = np.asarray(pts.x, dtype=np.float64)
            Y = np.asarray(pts.y, dtype=np.float64)
            c = (X - x0) / gsd - 0.5
            r = (y1 - Y) / gsd - 0.5
            ins = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
            pb = bilinear(prob, r, c)
            vb = bilinear(valid.astype(np.float32), r, c)
            inside = ins & (vb > 0.5)
            pset = {}
            if "p05" in preds:
                pset["p05"] = inside & (pb >= 0.5)
            if "mg05" in preds and mg is not None:
                pset["mg05"] = inside & (bilinear(mg, r, c) >= 0.5)
            if keep_mask is not None:
                pset["dplus"] = keep_mask[n_total:n_total + X.size]

            # ---- 逐点真值
            truths = {"old": np.asarray(old[n_total:n_total + X.size])}
            if segs:
                cr = np.clip(np.rint(r), 0, H - 1).astype(np.int64)
                cc = np.clip(np.rint(c), 0, W - 1).astype(np.int64)
                cand = ins & (dist_r[cr, cc] <= max(LINE_R) + 1.2)
                d = np.full(X.size, 999.0, dtype=np.float32)
                ii = np.flatnonzero(cand)
                if ii.size:
                    d[ii] = seg_dist(X[ii], Y[ii], segs)
                d_all.append(d)
                for rr in LINE_R:
                    truths[f"line_{rr:g}m"] = d <= rr
            for j, pa in enumerate(paths):
                if j == 0:
                    inpoly = np.zeros(X.size, dtype=bool)
                bb = pa.get_extents()
                sel = (X >= bb.x0 - 2) & (X <= bb.x1 + 2) & (Y >= bb.y0 - 2) & (Y <= bb.y1 + 2)
                kk = np.flatnonzero(sel)
                if kk.size:
                    inpoly[kk] |= pa.contains_points(np.column_stack([X[kk], Y[kk]]))
            if paths:
                truths["poly"] = inpoly & ins

            for n, tv in truths.items():
                npos[n] += int(tv.sum())
                for pn, pm in pset.items():
                    cm[n][pn][0] += int((pm & tv).sum())
                    cm[n][pn][1] += int((pm & ~tv).sum())
                    cm[n][pn][2] += int((~pm & tv).sum())
            for pn, pm in pset.items():
                npred[pn] += int(pm.sum())
            n_total += X.size
    return cm, npos, npred, n_total, (np.concatenate(d_all) if d_all else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="杨家坝")
    ap.add_argument("--no-points", action="store_true", help="只做栅格口径，不读 LAS")
    ap.add_argument("--npz", default="", help="概率图 npz（默认 cold_start_{site}_ep.npz）")
    ap.add_argument("--out", default="eval_new_shp", help="输出 JSON 前缀")
    a = ap.parse_args()
    site = a.site
    LOG = os.path.join(RES, "eval_new_shp.log")
    log(f"\n{'='*78}\n[eval_new_shp] {site}  {time.strftime('%Y-%m-%d %H:%M:%S')}", LOG)

    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1
    old_label = (d["label"] == 1) if "label" in d.files else np.zeros((H, W), bool)

    truths, info = {"old": old_label}, {}
    line_shapes, poly_shapes = None, None
    if "line" in SHP_NEW[site]:
        ftype, line_shapes = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["line"]))
        dist = line_dist_map(line_shapes, x0, y1, gsd, H, W)
        np.save(os.path.join(RES, f"dist_centerline_{site}.npy"), dist)
        L = sum(float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
                for sh in line_shapes for xy in parts_of(sh))
        info["line"] = {"file": SHP_NEW[site]["line"], "shp_type": ftype,
                        "features": len(line_shapes), "length_m": round(L, 1)}
        for r in LINE_R:
            truths[f"line_{r:g}m"] = (dist <= r) & valid
    if "poly" in SHP_NEW[site]:
        ftype, poly_shapes = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
        pm = poly_mask(poly_shapes, x0, y1, gsd, H, W)
        truths["poly"] = pm & valid
        A = 0.0
        for sh in poly_shapes:
            for q in parts_of(sh):
                A += 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1))
                               - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
        info["poly"] = {"file": SHP_NEW[site]["poly"], "shp_type": ftype,
                        "features": len(poly_shapes), "area_m2": round(float(A), 1)}

    log(f"  网格 {H}×{W} @ {gsd} m，valid {int(valid.sum()):,} px", LOG)
    for n, t in truths.items():
        log(f"  真值 {n:<12} {int(t.sum()):>7,} px  ({t.sum()*gsd*gsd:>9.0f} m²)"
            f"  与旧标签重合 {int((t & old_label).sum()):>6,}", LOG)

    # ---- 栅格口径：被评对象（有什么评什么）
    pix_pred, prob = {}, None
    zc_path = os.path.join(RES, a.npz or f"cold_start_{site}_ep.npz")
    if os.path.exists(zc_path):
        zc = np.load(zc_path)
        prob = (np.where(zc["valid"] == 1, zc["prob"], 0.0)
                if "valid" in zc.files else zc["prob"].astype(np.float64))
        pix_pred["p05"] = (prob >= 0.5) & valid
        if "pseudo" in zc.files:
            pix_pred["pseudo"] = zc["pseudo"].astype(bool) & valid
    else:
        log(f"  （无 {os.path.basename(zc_path)}：跳过 2D 概率类被评对象）", LOG)
    mg_path = os.path.join(RES, f"model_guided_prob_{site}.npz")
    mg = None
    if os.path.exists(mg_path):
        zm = np.load(mg_path)
        key = "prob" if "prob" in zm.files else list(zm.files)[0]
        mg = zm[key].astype(np.float64)
        if mg.shape == (H, W):
            pix_pred["mg05"] = (mg >= 0.5) & valid

    result = {"site": site, "grid": [H, W], "shp": info, "pixel": {}, "point": {}}
    if pix_pred:
        log("\n  [栅格口径] 预测 vs 真值", LOG)
        for pn, pm in pix_pred.items():
            result["pixel"][pn] = {}
            for n, tr in truths.items():
                result["pixel"][pn][n] = pixel_eval(pm, tr)
            row = "  ".join(f"{n}:IoU={result['pixel'][pn][n]['iou']:.4f}"
                            f"/F1={result['pixel'][pn][n]['f1']:.4f}" for n in truths)
            log(f"    {pn:<7} pred {int(pm.sum()):>6,} px | {row}", LOG)

    keep_path = os.path.join(RES, f"cold_start_{site}_ep_keep_idx.npy")
    if not a.no_points and prob is not None and os.path.exists(keep_path):
        segs = [(xy[k], xy[k + 1]) for sh in (line_shapes or []) for xy in parts_of(sh)
                for k in range(len(xy) - 1)]
        t0 = time.time()
        cm, npos, npred, n_total, dpts = point_eval(
            site, LAS[site], line_shapes, poly_shapes,
            {"p05": True, "mg05": True, "_prob": prob, "_valid": valid, "_mg": mg,
             "_keep_idx": np.load(keep_path)})
        log(f"\n  [点级口径] 点云 {n_total:,} 点（{time.time()-t0:.0f}s）", LOG)
        for n in cm:
            log(f"    真值 {n:<12} {npos[n]:>9,} 点", LOG)
        for pn in ("p05", "mg05", "dplus"):
            if pn not in npred:
                continue
            result["point"][pn] = {}
            for n in cm:
                tp, fp, fn = cm[n][pn]
                if tp + fp + fn == 0:
                    continue
                r = prf(tp, fp, fn)
                r.update({"keep": npred[pn], "tp": tp, "fp": fp, "fn": fn})
                result["point"][pn][n] = r
            row = "  ".join(f"{n}:{result['point'][pn][n]['iou']:.4f}"
                            for n in result["point"][pn])
            log(f"    {pn:<7} 保留 {npred[pn]:>9,} 点 | IoU {row}", LOG)

        # 中轴线距离诊断（D+ 交付点离中轴线有多远；直接量交付 LAS，逐点精确）
        if line_shapes and os.path.exists(os.path.join(RES, f"cold_start_{site}_ep_corridor.las")):
            import laspy
            dl = []
            with laspy.open(os.path.join(RES, f"cold_start_{site}_ep_corridor.las")) as f:
                for pts in f.chunk_iterator(1_500_000):
                    dl.append(seg_dist(np.asarray(pts.x, dtype=np.float64),
                                       np.asarray(pts.y, dtype=np.float64), segs))
            dl = np.concatenate(dl)
            result["dplus_to_centerline_m"] = {
                "n": int(dl.size),
                "p50": round(float(np.percentile(dl, 50)), 2),
                "p90": round(float(np.percentile(dl, 90)), 2),
                "frac_le2m": round(float((dl <= 2).mean()), 4),
                "frac_le5m": round(float((dl <= 5).mean()), 4)}
            log(f"    D+ 到中轴线距离（{dl.size:,} 点）：中位 "
                f"{result['dplus_to_centerline_m']['p50']} m，P90 "
                f"{result['dplus_to_centerline_m']['p90']} m，≤2 m "
                f"{result['dplus_to_centerline_m']['frac_le2m']:.2%}，≤5 m "
                f"{result['dplus_to_centerline_m']['frac_le5m']:.2%}", LOG)

    out = os.path.join(RES, f"{a.out}_{site}.json")
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"\n  写出 {os.path.relpath(out, ROOT)}", LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
