# -*- coding: utf-8 -*-
"""物理尺度（亚像元宽度）评估：骨架对骨架的线状 F1 与中心线 Buffer IoU。

蜀道真实宽度约 0.7 m，在 1 m 网格下不足 1 个像元，因此用**线状地物**口径评估：
    真值骨架：SHP 面 → 0.25 m 栅格 → skeletonize → 降采样到 1 m
    预测骨架：收窄交付点云 → 0.25 m 占用格 → skeletonize → 降采样到 1 m

预测骨架三步后处理（真值完全不动）：
    ① 闭运算：骨架提取前对掩膜做 binary_closing，补 1 px 断裂
    ② 剪枝：连通域 < `--prune-min` px 的碎段直接删除；端点向内 ≤`--spur` px
       就撞到分叉的短支（毛刺）整段删除
    ③ 端点延伸：以端点处切线方向延长 `--extend` px（不参考真值外包框，避免信息泄漏）

指标：
    A. 线状 F1（容差 1 px）：P = 预测骨架命中率；R = 真值骨架被覆盖比例
    B. 中心线 Buffer IoU（互膨胀 1 px）

用法：
    python tools/eval_physical_width.py --site 杨家坝 --fig
    python tools/eval_physical_width.py --site 杨家坝 --pred p05
    python tools/eval_physical_width.py --site 杨家坝 --no-close --extend 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_dem as P  # noqa: E402
from eval_line_iou import poly_mask_hr  # noqa: E402
from eval_new_shp import RES, ROOT, SHP_BASE, SHP_NEW, read_shp_any  # noqa: E402

FIG = os.path.join(ROOT, "figures")
SS = 4                       # 骨架提取分辨率倍率（0.25 m）
N8 = np.ones((3, 3), np.uint8)
N8[1, 1] = 0


def up_skeleton(mask01, H, W, ss):
    """0.25 m 布尔面 → 骨架 → 降到 1 m（任一像元命中即命中）。"""
    from skimage.morphology import skeletonize
    sk = skeletonize(np.ascontiguousarray(mask01.astype(np.uint8)))
    return sk[:H * ss, :W * ss].reshape(H, ss, W, ss).any(axis=(1, 3))


def nbr_count(sk):
    from scipy.ndimage import convolve
    return convolve(sk.astype(np.uint8), N8, mode="constant") * sk


def prune_skeleton(sk, min_px=15, spur=10, rounds=4):
    """① 连通域过滤（< min_px 删除）② 毛刺裁剪（端点向内 ≤spur 撞分叉则删）。"""
    from collections import deque
    from scipy.ndimage import label as cc_label
    sk = sk.copy()
    lab, n = cc_label(sk)
    if n:
        cnt = np.bincount(lab.ravel())
        small = np.isin(lab, [k for k in range(1, n + 1) if cnt[k] < min_px])
        sk &= ~small
    H, W = sk.shape
    for _ in range(rounds):
        nb = nbr_count(sk)
        eps = np.argwhere(sk & (nb == 1))
        if eps.size == 0:
            break
        remove = np.zeros_like(sk)
        for r0, c0 in eps:
            r0, c0 = int(r0), int(c0)
            if not sk[r0, c0]:
                continue
            path = [(r0, c0)]
            seen = {(r0, c0)}
            q = deque([(r0, c0)])
            hit = False
            while q and len(path) <= spur and not hit:
                pr, pc = q.popleft()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = pr + dr, pc + dc
                        if not (0 <= rr < H and 0 <= cc < W) or not sk[rr, cc]:
                            continue
                        if (rr, cc) in seen:
                            continue
                        if nb[rr, cc] >= 3:            # 撞到分叉 → 该支为毛刺
                            hit = True
                            break
                        seen.add((rr, cc))
                        q.append((rr, cc))
                        path.append((rr, cc))
                    if hit:
                        break
            if hit and len(path) <= spur:
                for rr, cc in path:
                    remove[rr, cc] = True
        if not remove.any():
            break
        sk &= ~remove
    return sk


def extend_ends(sk, steps=8, probe=8):
    """只延伸**最大连通域的两个端点**。

    用图测地距离找两端（BFS 两次），而不是重跑 skeletonize —— 1 m 降采样后的
    骨架再做细化会把斜向链节切成孤立点（实测 24 段会变成 97 段）。
    返回 (骨架, 新增像素数)。
    """
    from collections import deque
    from scipy.ndimage import label as cc_label
    if steps <= 0:
        return sk, 0
    H, W = sk.shape
    lab, n = cc_label(sk)
    if n == 0:
        return sk, 0
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    comp = lab == int(cnt.argmax())

    def bfs(start):
        dist = -np.ones((H, W), np.int32)
        par = {}
        q = deque([start])
        dist[start] = 0
        while q:
            r, c = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and comp[rr, cc] and dist[rr, cc] < 0:
                        dist[rr, cc] = dist[r, c] + 1
                        par[(rr, cc)] = (r, c)
                        q.append((rr, cc))
        far = np.unravel_index(int(dist.argmax()), dist.shape)
        return (int(far[0]), int(far[1])), par

    a0 = tuple(np.argwhere(comp)[0])
    end_a, _ = bfs(a0)
    end_b, par = bfs(end_a)
    out = sk.copy()
    added = 0
    for end in (end_a, end_b):
        path = [end]
        cur = end
        while cur in par and len(path) <= probe + 1:
            cur = par[cur]
            path.append(cur)
        if len(path) < 3:
            continue
        dr = path[0][0] - path[-1][0]
        dc = path[0][1] - path[-1][1]
        nrm = float(np.hypot(dr, dc))
        if nrm < 1e-6:
            continue
        for t in range(1, steps + 1):
            rr = int(round(path[0][0] + dr / nrm * t))
            cc = int(round(path[0][1] + dc / nrm * t))
            if 0 <= rr < H and 0 <= cc < W and not out[rr, cc]:
                out[rr, cc] = True
                added += 1
    return out, added


def pred_occ_from_points(path, x0, y1, gsd, H, W, ss):
    """点云 → 0.25 m 占用格（不做闭运算，闭运算在评估阶梯里单独控制）。"""
    import laspy
    occ = np.zeros((H * ss, W * ss), dtype=bool)
    n = 0
    with laspy.open(path) as f:
        for pts in f.chunk_iterator(3_000_000):
            X = np.asarray(pts.x, dtype=np.float64)
            Y = np.asarray(pts.y, dtype=np.float64)
            c = np.floor((X - x0) / gsd * ss).astype(np.int64)
            r = np.floor((y1 - Y) / gsd * ss).astype(np.int64)
            ok = (c >= 0) & (c < W * ss) & (r >= 0) & (r < H * ss)
            occ[r[ok], c[ok]] = True
            n += X.size
    return occ, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="杨家坝")
    ap.add_argument("--pred", default="narrow", choices=["narrow", "p05"])
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--ss", type=int, default=SS)
    ap.add_argument("--phys-width", type=float, default=0.7)
    ap.add_argument("--close", dest="close", action="store_true", default=True,
                    help="骨架提取前对掩膜做 binary_closing（默认开）")
    ap.add_argument("--no-close", dest="close", action="store_false")
    ap.add_argument("--close-iter", type=int, default=1, help="0.25 m 上闭运算迭代次数")
    ap.add_argument("--prune-min", type=int, default=15, help="删除 <N px 的碎段（0=关）")
    ap.add_argument("--spur", type=int, default=10, help="毛刺长度上限（0=关）")
    ap.add_argument("--extend", type=int, default=8, help="端点延伸 px（0=关）")
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()
    site, ss = a.site, max(int(a.ss), 1)
    from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, label

    x0, y1, wm, hm, gsd, d = P.dem_window(site)
    H, W = d["dem_mean"].shape
    valid = d["valid"] == 1

    # ---------- 真值骨架（不参与任何后处理） ----------
    _, polys = read_shp_any(os.path.join(SHP_BASE, SHP_NEW[site]["poly"]))
    gt = up_skeleton(poly_mask_hr(polys, x0, y1, gsd, H, W, ss), H, W, ss) & valid

    # ---------- 预测掩膜 ----------
    if a.pred == "narrow":
        las = os.path.join(RES, f"narrow_{site}_corridor.las")
        if not os.path.exists(las):
            print(f"[错误] 缺少 {las}，先跑 tools/line_metrics_narrow.py")
            return 2
        occ, npt = pred_occ_from_points(las, x0, y1, gsd, H, W, ss)
        src = f"{os.path.basename(las)}（{npt:,} 点）"
    else:
        z = np.load(os.path.join(RES, f"cold_start_{site}_ep.npz"))
        m1 = (z["prob"] >= 0.5) & ((z["valid"] == 1) if "valid" in z.files else (z["prob"] > 0))
        occ = np.repeat(np.repeat(m1, ss, 0), ss, 1)
        npt = None
        src = "cold_start P≥0.5 栅格"

    Dg = distance_transform_edt(~gt, sampling=gsd)      # 到真值骨架

    def score(sk, tag):
        sk = sk & valid
        labg, ng = label(gt)
        labp, npc = label(sk)
        Dp = distance_transform_edt(~sk, sampling=gsd)
        hit_p = Dg[sk] <= a.tol
        hit_g = Dp[gt] <= a.tol
        P_ = float(hit_p.mean()) if sk.any() else 0.0
        R_ = float(hit_g.mean())
        F_ = 2 * P_ * R_ / max(P_ + R_, 1e-9)
        bp = binary_dilation(sk, iterations=int(a.tol)) if a.tol else sk
        bg = binary_dilation(gt, iterations=int(a.tol)) if a.tol else gt
        iou = float((bp & bg).sum() / max((bp | bg).sum(), 1))
        print(f"  {tag:<14} 骨架 {int(sk.sum()):>4} px / {npc:>3} 段 | "
              f"P {P_:.4f}  R {R_:.4f}  F1 {F_:.4f} | Buffer IoU {iou:.4f}")
        return {"tag": tag, "skeleton_px": int(sk.sum()), "components": int(npc),
                "precision": round(P_, 4), "recall": round(R_, 4), "f1": round(F_, 4),
                "buffer_iou": round(iou, 4)}

    print(f"[{site}] 网格 {H}×{W} @ {gsd} m（骨架在 {gsd / ss:g} m 上提取）；"
          f"物理宽度 {a.phys_width} m = {a.phys_width / gsd:.2f} px")
    print(f"  预测源：{src}")
    print(f"  真值骨架：{int(gt.sum())} px / {label(gt)[1]} 段")

    # ---------- 后处理阶梯 ----------
    print(f"\n[后处理阶梯]（容差 {a.tol:g} px；GT 不动）")
    rows = []
    sk0 = up_skeleton(occ, H, W, ss)
    rows.append(score(sk0, "① 原始骨架"))
    cur = sk0
    if a.close:
        occ_c = binary_closing(occ, np.ones((3, 3), dtype=bool), iterations=a.close_iter)
        cur = up_skeleton(occ_c, H, W, ss)
        rows.append(score(cur, f"② +闭运算×{a.close_iter}"))
    # 碎段尺寸分布（判断"多出来的段"到底是毛刺还是线段）
    lab_c, n_c = label(cur & valid)
    if n_c:
        sz = sorted([int((lab_c == k).sum()) for k in range(1, n_c + 1)], reverse=True)
        print(f"     碎段尺寸：{sz[:12]}{' …' if len(sz) > 12 else ''}；"
              f"<5 px {sum(1 for v in sz if v < 5)} 段，<15 px {sum(1 for v in sz if v < 15)} 段")
    if a.prune_min > 0 or a.spur > 0:
        cur = prune_skeleton(cur, min_px=a.prune_min, spur=a.spur)
        rows.append(score(cur, "③ +剪枝"))
    if a.extend > 0:
        cur2, added = extend_ends(cur, steps=a.extend)
        print(f"     端点延伸：新增 {added} px，其中落在有效区 "
              f"{int((cur2 & ~cur & valid).sum())} px")
        cur = cur2
        rows.append(score(cur, "④ +端点延伸"))

    best = rows[-1]
    print(f"\n[结论] 最终线状 F1 = {best['f1']:.4f}，Buffer IoU = {best['buffer_iou']:.4f}"
          f" → {'突破 0.70' if best['f1'] >= 0.70 else '未达 0.70'}")
    print(f"  精度 {rows[0]['precision']:.4f} → {best['precision']:.4f}；"
          f"召回 {rows[0]['recall']:.4f} → {best['recall']:.4f}；"
          f"碎段 {rows[0]['components']} → {best['components']}")

    Dp = distance_transform_edt(~(cur & valid), sampling=gsd)
    print(f"\n[距离] 预测→真值 中位 {np.median(Dg[cur]):.2f} m / P90 "
          f"{np.percentile(Dg[cur], 90):.2f} m；真值→预测 中位 {np.median(Dp[gt]):.2f} m / "
          f"P90 {np.percentile(Dp[gt], 90):.2f} m")

    # ---------- 容差敏感性（对后处理后的骨架） ----------
    print("\n[容差敏感性]")
    sweep = {}
    Dp2 = distance_transform_edt(~(cur & valid), sampling=gsd)
    for t in (1.0, 1.5, 2.0, 3.0):
        pp = float((Dg[cur] <= t).mean())
        rr = float((Dp2[gt] <= t).mean())
        ff = 2 * pp * rr / max(pp + rr, 1e-9)
        bpp = binary_dilation(cur & valid, iterations=int(np.ceil(t)))
        bgg = binary_dilation(gt, iterations=int(np.ceil(t)))
        ii = float((bpp & bgg).sum() / max((bpp | bgg).sum(), 1))
        sweep[f"{t:g}px"] = {"precision": round(pp, 4), "recall": round(rr, 4),
                             "f1": round(ff, 4), "buffer_iou": round(ii, 4)}
        print(f"  {t:>3g} px：P {pp:.4f}  R {rr:.4f}  F1 {ff:.4f}  Buffer IoU {ii:.4f}")

    res = {"site": site, "pred": a.pred, "pred_source": src, "grid": [H, W], "gsd_m": gsd,
           "phys_width_m": a.phys_width, "phys_width_px": round(a.phys_width / gsd, 3),
           "skel_res_m": gsd / ss, "tol_px": a.tol,
           "params": {"close": a.close, "prune_min": a.prune_min, "spur": a.spur,
                      "extend": a.extend},
           "gt_skeleton_px": int(gt.sum()), "gt_components": int(label(gt)[1]),
           "ladder": rows, "final": best, "sweep": sweep,
           "pass_0p70": bool(best["f1"] >= 0.70)}
    out = os.path.join(RES, f"eval_physical_width_{site}_{a.pred}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  写出 results/{os.path.basename(out)}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        ys, xs = np.where(gt | cur)
        y0, y1_ = max(int(ys.min()) - 15, 0), min(int(ys.max()) + 15, H)
        c0, c1 = max(int(xs.min()) - 15, 0), min(int(xs.max()) + 15, W)
        sl = (slice(y0, y1_), slice(c0, c1))
        rgb = np.ones((y1_ - y0, c1 - c0, 3), dtype=np.float32)
        rgb[binary_dilation(gt, iterations=1)[sl]] = [0.80, 0.92, 0.80]
        rgb[binary_dilation(cur & valid, iterations=1)[sl]] = [0.96, 0.84, 0.84]
        rgb[gt[sl]] = [0.0, 0.40, 0.0]
        rgb[(cur & valid)[sl]] = [0.85, 0.10, 0.10]
        fig, ax = plt.subplots(figsize=(13, 13))
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{site}（{a.pred}）：骨架对骨架，容差 {a.tol:g} px\n"
                     f"线状 F1 = {best['f1']:.4f}   Buffer IoU = {best['buffer_iou']:.4f}")
        ax.axis("off")
        fig.tight_layout()
        pth = os.path.join(FIG, f"eval_physical_width_{site}_{a.pred}.png")
        fig.savefig(pth, dpi=140)
        plt.close(fig)
        print(f"  写出 figures/{os.path.basename(pth)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
