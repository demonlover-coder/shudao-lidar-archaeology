# -*- coding: utf-8 -*-
"""用激光脉冲物理特性捞回被回波法误杀的硬质路面（石板）。

背景
----
现行滤波 = 回波法 `keep = (rn == 1) | (rn == nr)`（并集 `cls == 2` 实测零增益，
因为所有地面点本来就是首/末回波）。它把 `rn = 2, nr = 3` 的**中间回波**全部丢掉。

物理假设
--------
同一脉冲打在石板上：光斑内只有极薄的一层硬质面，多次回波高差 Δz 很小；
打在树冠/灌木上：首末回波跨越冠层，Δz 大。因此

    Δz = max_z(pulse) − min_z(pulse) < h0   →   该脉冲的中间回波属于硬质面，捞回。

与用户原始实现的差异（必须说明）
--------------------------------
1. **Δz 在完整脉冲上算，不在被剔除点上算。**
   被剔除点恒为 `rn = 2, nr = 3` 的**唯一**中间回波，若只在被剔除点上分组，
   每组只有 1 个点 → `max_z == min_z` → `Δz ≡ 0 < h0` → 无条件捞回全部中间回波，
   规则 2 退化为空操作。本脚本对**整个脉冲**（rn = 1..nr）求 min/max。
2. **必须用原始全回波 LAS。**
   `yangjiaba_lastreturn.las` / `西坝村栈道-lidar_回波法.las` 已被回波法抽成
   单点（每个 gps_time 唯一、rn == nr），Δz 恒为 0，无法判别。
3. **全局分组。**
   实测同脉冲的点在 LAS 文件里**不相邻**（连续率 0%），分块局部成组会把一个
   脉冲拆到不同块。本脚本用三遍扫描：
     Pass 1  收集被剔除点的 gps_time 键（全局唯一集合，约 165 万，仅 13 MB）
     Pass 2  扫全部点，只对命中这些键的点累加脉冲 min_z / max_z
     Pass 3  重扫，算 keep、写 LAS、统计
   内存只随"被剔除点数"增长，与总点数无关。

输出
----
* `data_real/{site}_hard_surface.las`   新 LAS（仅保留点，原字段全部保留）
* `results/las_hard_surface.json`       点数 / 密度 / Δz 分位 / 阈值敏感性
* `figures/las_hard_surface.png`        (a) Δz 直方图  (b) 阈值扫描  (c) 密度提升

用法
----
    python tools/las_hard_surface.py                  # 两站，h0 = 0.5
    python tools/las_hard_surface.py --site 杨家坝 --h0 0.3
    python tools/las_hard_surface.py --no-write       # 只统计不写 LAS
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
PCL_ROOT = os.environ.get("SHUDAO_DATA_ROOT", os.path.join(ROOT, "data"))
REAL = os.path.join(PCL_ROOT, "data_real")

# 原始全回波 LAS（**不是** lastreturn / 回波法版本）
SRC_LAS = {
    "杨家坝": os.path.join(PCL_ROOT, "survey-shp", "杨家坝", "yangjiaba.las"),
    "西坝村": os.path.join(PCL_ROOT, "survey-shp", "西坝村", "西坝村栈道-lidar.las"),
}
LABEL = {s: os.path.join(REAL, f"{s}_label.npy") for s in SRC_LAS}
MASK = {s: os.path.join(RES, f"mask_1m_{s}.npy") for s in SRC_LAS}
OUT_LAS = {s: os.path.join(REAL, f"{s}_hard_surface.las") for s in SRC_LAS}

H0_GRID = (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0)
CHUNK = 2_000_000
KEY_SCALE = 1_000_000.0        # gps_time 保留 6 位小数


# ---------------------------------------------------------------- 基础 ----
def tkey(gt):
    """gps_time → 整数键（保留 6 位小数，与 np.round(gt, 6) 等价但更快）。"""
    return np.rint(np.asarray(gt, dtype=np.float64) * KEY_SCALE).astype(np.int64)


def rule1(rn, nr, cls):
    """规则 1：首/末回波 ∪ 地面类。实测并集 cls==2 零增益（地面点本就全在首/末回波里）。"""
    return (rn == 1) | (rn == nr) | (cls == 2)


def iter_las(path, chunk, limit=None):
    import laspy
    n = 0
    with laspy.open(path) as fh:
        for pts in fh.chunk_iterator(chunk):
            yield pts
            n += len(pts)
            if limit and n >= limit:
                return


# ------------------------------------------------------- Pass 1 / 2 ----
def pass1_dropped_keys(path, chunk, limit=None):
    """收集规则 1 剔除点的 gps_time 键（全局唯一）。"""
    parts, n = [], 0
    for pts in iter_las(path, chunk, limit):
        rn = np.asarray(pts.return_number)
        nr = np.asarray(pts.number_of_returns)
        cls = np.asarray(pts.classification)
        drop = ~rule1(rn, nr, cls)
        if drop.any():
            parts.append(tkey(np.asarray(pts.gps_time)[drop]))
        n += len(rn)
        print(f"    [P1] {n:,} 点，累计剔除键 {sum(len(p) for p in parts):,}", flush=True)
    if not parts:
        return np.zeros(0, dtype=np.int64), 0
    return np.unique(np.concatenate(parts)), n


def pass2_pulse_z(path, keys, chunk, limit=None):
    """扫全部点，对命中 keys 的点累加脉冲 min_z / max_z / 回波数。"""
    m = keys.size
    minz = np.full(m, np.inf, dtype=np.float64)
    maxz = np.full(m, -np.inf, dtype=np.float64)
    cnt = np.zeros(m, dtype=np.int64)
    for pts in iter_las(path, chunk, limit):
        k = tkey(pts.gps_time)
        pos = np.searchsorted(keys, k)
        np.clip(pos, 0, m - 1, out=pos)
        hit = keys[pos] == k
        if not hit.any():
            continue
        z = np.asarray(pts.z, dtype=np.float64)[hit]
        p = pos[hit]
        np.minimum.at(minz, p, z)
        np.maximum.at(maxz, p, z)
        np.add.at(cnt, p, 1)
    miss = ~np.isfinite(minz)
    if miss.any():
        print(f"    [P2] 警告：{int(miss.sum())} 个键未在扫描中命中（提供 limit 时会如此）",
              flush=True)
        minz[miss] = 0.0
        maxz[miss] = 0.0
    return minz, maxz, cnt


# ------------------------------------------------------------ Pass 3 ----
def pass3_apply(path, out_path, keys, dz_key, cnt_key, h0, chunk, lab_path,
                write_las=True, limit=None):
    """重扫：算 keep、写 LAS、累积统计量。

    判定口径（**关键**）：只有脉冲组内回波数 `cnt >= 2` 才允许用 Δz 判硬质面。
    单点组的 Δz 恒为 0（max_z == min_z），若据此外推会把"无法判定"误当成
    "薄层硬质面"无条件捞回——实测 6 位小数取整会让约 1/3 的多回波脉冲被拆成
    单点组（同脉冲回波的时间要么位级相同、要么相差 ≥1e-6 s），因此必须设这道闸。
    """
    import laspy
    lab = np.load(lab_path, mmap_mode="r")
    n_total = n_old = n_new = 0
    stats = {}
    for tag in ("走廊", "背景"):
        stats[tag] = {"old": 0, "new": 0}
    dz_parts, lab_parts, cnt_parts = [], [], []
    off = 0
    writer = None
    if write_las:
        with laspy.open(path) as rfh:
            header = copy.deepcopy(rfh.header)
        header.point_count = 0
        writer = laspy.open(out_path, mode="w", header=header)

    try:
        for pts in iter_las(path, chunk, limit):
            n = len(pts)
            rn = np.asarray(pts.return_number)
            nr = np.asarray(pts.number_of_returns)
            cls = np.asarray(pts.classification)
            k1 = rule1(rn, nr, cls)
            keep = k1.copy()

            didx = np.flatnonzero(~k1)
            if didx.size:
                kk = tkey(np.asarray(pts.gps_time)[didx])
                pos = np.searchsorted(keys, kk)
                np.clip(pos, 0, keys.size - 1, out=pos)
                ok = keys[pos] == kk
                dz = np.where(ok, dz_key[pos], np.inf)          # 未命中 → 视为不捞
                nc = np.where(ok, cnt_key[pos], 0)
                rescue = ok & (nc >= 2) & (dz < h0)             # 必须 ≥2 回波才可判定
                keep[didx[rescue]] = True
                # 诊断：脉冲 Δz + 组内回波数 + 逐点走廊标签
                dz_parts.append(dz.astype(np.float32))
                cnt_parts.append(nc.astype(np.int16))
                lab_parts.append(np.asarray(lab[off:off + n])[didx].astype(bool))

            lab_n = np.asarray(lab[off:off + n]).astype(bool)
            n_old += int(k1.sum())
            n_new += int(keep.sum())
            n_total += n
            stats["走廊"]["old"] += int(k1[lab_n].sum())
            stats["走廊"]["new"] += int(keep[lab_n].sum())
            stats["背景"]["old"] += int(k1[~lab_n].sum())
            stats["背景"]["new"] += int(keep[~lab_n].sum())

            if writer is not None:
                writer.write_points(pts[keep])
            off += n
            print(f"    [P3] {off:,} 点  保留 {n_new:,} (+{n_new - n_old:,})", flush=True)

        if writer is not None:
            writer.header.point_count = int(n_new)
            writer.close()
            writer = None
    finally:
        if writer is not None:
            writer.close()

    dz = np.concatenate(dz_parts) if dz_parts else np.zeros(0, np.float32)
    lb = np.concatenate(lab_parts) if lab_parts else np.zeros(0, bool)
    nc = np.concatenate(cnt_parts) if cnt_parts else np.zeros(0, np.int16)
    return (dict(n_total=n_total, n_old=n_old, n_new=n_new, stats=stats),
            dz, lb, nc)


# ------------------------------------------------------------ 统计图 ----
def summarize(site, dz, lb, nc, meta, h0, area_corr, fig_path):
    """产出统计 + 三联图。

    两种口径必须同时报（否则会误读）：
      * `literal`：用户字面规格——组内只有 1 个点也算脉冲，Δz = 0 → 无条件捞回；
      * `judgeable`：物理口径——只有组内 ≥2 回波才判定 Δz，单点组视为"无法判定"不捞。
    前者会把"取整把脉冲拆散"的技术噪声误当成"薄层硬质面"，必须显式区分。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fin = np.isfinite(dz)
    dz, lb, nc = dz[fin], lb[fin], nc[fin]
    judge = nc >= 2

    out = {"site": site, "h0": h0, **meta,
           "area_corridor_m2": float(area_corr),
           "dropped": {"走廊": int(lb.sum()), "背景": int((~lb).sum())},
           "unjudgeable": {
               "走廊": int((lb & ~judge).sum()), "背景": int((~lb & ~judge).sum()),
               "走廊占比": float((lb & ~judge).sum() / max(int(lb.sum()), 1)),
               "背景占比": float((~lb & ~judge).sum() / max(int((~lb).sum()), 1)),
           },
           "hz_grid": []}

    def cnt_of(region, t, conv):
        m = judge if conv == "judgeable" else np.ones(dz.size, bool)
        return int(((dz < t) & m & region).sum())

    for t in H0_GRID:
        row = {"h0": t}
        for conv in ("judgeable", "literal"):
            rc = cnt_of(lb, t, conv)
            rb = cnt_of(~lb, t, conv)
            row[f"{conv}_corridor"] = rc
            row[f"{conv}_background"] = rb
            row[f"{conv}_corridor_rate"] = rc / max(int(lb.sum()), 1)
            row[f"{conv}_background_rate"] = rb / max(int((~lb).sum()), 1)
        out["hz_grid"].append(row)

    def pct(a, q):
        return float(np.percentile(a, q)) if a.size else float("nan")

    # Δz 分位：judgeable 才是有效样本
    for tag, reg in (("走廊", lb), ("背景", ~lb)):
        arr = dz[reg & judge]
        out[f"dz_{tag}"] = {
            "n_judgeable": int(arr.size),
            "min": float(arr.min()) if arr.size else float("nan"),
            "p10": pct(arr, 10), "p25": pct(arr, 25), "p50": pct(arr, 50),
            "p75": pct(arr, 75), "p90": pct(arr, 90),
        }

    st = meta["stats"]
    for tag in ("走廊", "背景"):
        a = area_corr if tag == "走廊" else meta["area_background_m2"]
        old, new = st[tag]["old"], st[tag]["new"]
        out[f"density_{tag}"] = {
            "old_pts": old, "new_pts": new, "added": new - old,
            "old_pts_per_m2": old / a, "new_pts_per_m2": new / a,
            "gain": (new - old) / a, "gain_pct": (new / max(old, 1) - 1.0),
        }
    out["added_total"] = meta["n_new"] - meta["n_old"]
    out["added_total_pct"] = meta["n_new"] / max(meta["n_old"], 1) - 1.0
    out["added_corridor_share"] = (out["density_走廊"]["added"] /
                                   max(out["added_total"], 1))

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    xmax = 4.0
    bins = np.linspace(0, xmax, 81)
    for tag, reg, c in (("走廊", lb, "#c0392b"), ("背景", ~lb, "#2c3e50")):
        arr = dz[reg & judge]
        if arr.size:
            a = arr[arr <= xmax]
            ax.hist(a, bins=bins, density=True, histtype="step", lw=1.8, color=c,
                    label=f"{tag}·可判定（n={arr.size:,}，>{xmax:g} m 占"
                          f"{(arr > xmax).mean()*100:.1f}%）")
    ax.axvline(h0, color="#e67e22", ls="--", lw=2, label=f"h0 = {h0:g} m")
    uj = out["unjudgeable"]
    zero = int(((dz == 0) & judge)[lb].sum()), int(((dz == 0) & judge)[~lb].sum())
    ax.text(0.97, 0.97,
            f"已排除单点组（Δz 不可判定）：\n"
            f"  走廊 {uj['走廊']:,}（{uj['走廊占比']*100:.1f}%）/ "
            f"背景 {uj['背景']:,}（{uj['背景占比']*100:.1f}%）\n"
            f"Δz≡0 但 ≥2 回波：走廊 {zero[0]:,} / 背景 {zero[1]:,}",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(fc="white", ec="#bdc3c7", alpha=.9, boxstyle="round,pad=.3"))
    ax.set_xlabel("Δz = 脉冲 max_z − min_z（m）")
    ax.set_ylabel("概率密度")
    ax.set_title("(a) 被剔除点所属脉冲的 Δz（仅 ≥2 回波）")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=.3)

    ax = axes[1]
    g = out["hz_grid"]
    hs = [d["h0"] for d in g]
    ax.plot(hs, [d["judgeable_corridor_rate"] * 100 for d in g], "o-",
            color="#c0392b", label="走廊·可判定口径")
    ax.plot(hs, [d["judgeable_background_rate"] * 100 for d in g], "s-",
            color="#2c3e50", label="背景·可判定口径")
    ax.plot(hs, [d["literal_corridor_rate"] * 100 for d in g], "^--", lw=1,
            color="#c0392b", alpha=.45, label="走廊·字面规格（含单点组）")
    ax.plot(hs, [d["literal_background_rate"] * 100 for d in g], "v--", lw=1,
            color="#2c3e50", alpha=.45, label="背景·字面规格（含单点组）")
    ax.axvline(h0, color="#e67e22", ls="--", lw=2)
    ax.set_xlabel("h0（m）")
    ax.set_ylabel("占该区域被剔除点的比例（%）")
    ax.set_title("(b) 阈值敏感性（两种口径）")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)

    ax = axes[2]
    tags = ["走廊", "背景"]
    dens = [st[t]["old"] / (area_corr if t == "走廊" else meta["area_background_m2"])
            for t in tags]
    new = [st[t]["new"] / (area_corr if t == "走廊" else meta["area_background_m2"])
           for t in tags]
    x = np.arange(len(tags))
    ax.bar(x - .2, dens, .4, label="规则 1（回波法）", color="#7f8c8d")
    ax.bar(x + .2, new, .4, label=f"规则 1 + 高差（h0={h0:g}）", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("点密度（点 / m²）")
    ax.set_title("(c) 有效点密度")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    for i, (o, nw) in enumerate(zip(dens, new)):
        ax.text(i, max(o, nw) * 1.02, f"+{(nw / max(o, 1e-9) - 1) * 100:.2f}%",
                ha="center", fontsize=9)

    fig.suptitle(f"{site} · 回波物理特性捞回硬质面（h0 = {h0:g} m）", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)
    return out


# -------------------------------------------------------------- main ----
def run_site(site, h0, chunk, write_las, limit, las_path=None, lab_path=None):
    src = las_path or SRC_LAS[site]
    lab_path = lab_path or LABEL[site]
    print(f"\n{'='*74}\n=== {site}\n  源 LAS : {src}\n  label : {lab_path}", flush=True)
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    keys, n_seen = pass1_dropped_keys(src, chunk, limit)
    print(f"  [P1] 扫描 {n_seen:,} 点；被规则 1 剔除的 gps_time 键 {keys.size:,} 个",
          flush=True)

    if keys.size == 0:
        print("  无被剔除点，规则 2 无作用；直接退出。")
        return {"site": site, "note": "无被剔除点"}

    minz, maxz, cnt = pass2_pulse_z(src, keys, chunk, limit)
    dz_key = np.maximum(maxz - minz, 0.0)
    print(f"  [P2] 脉冲回波数分布 "
          f"{dict(zip(*np.unique(cnt, return_counts=True)))}；"
          f"Δz 中位 {np.median(dz_key):.3f} m", flush=True)

    out_las = OUT_LAS[site]
    meta, dz, lb, nc = pass3_apply(src, out_las, keys, dz_key, cnt, h0, chunk,
                                   lab_path, write_las=write_las, limit=limit)
    mask = np.load(MASK[site]).astype(bool)
    area_corr = float(mask.sum())                     # GSD = 1 m → 1 px = 1 m²
    area_bg = float(mask.size - mask.sum())
    meta["area_background_m2"] = area_bg
    meta["out_las"] = out_las if write_las else None
    meta["src_las"] = src
    if write_las:
        print(f"  已写 {out_las}（{os.path.getsize(out_las)/2**30:.2f} GB）", flush=True)

    fig_path = os.path.join(FIG, f"las_hard_surface_{site}.png")
    rec = summarize(site, dz, lb, nc, meta, h0, area_corr, fig_path)
    np.savez_compressed(os.path.join(RES, f"las_hard_surface_dz_{site}.npz"),
                        dz=dz, lb=lb, nc=nc)      # 供日后免重扫重绘
    rec["pulse_multi_echo_ratio"] = float((cnt > 1).mean())
    print(f"  总点数 {meta['n_old']:,} → {meta['n_new']:,} "
          f"(+{meta['n_new']-meta['n_old']:,}, {rec['added_total_pct']*100:+.2f}%)", flush=True)
    print(f"  走廊密度 {rec['density_走廊']['old_pts_per_m2']:.4f} → "
          f"{rec['density_走廊']['new_pts_per_m2']:.4f} 点/m² "
          f"({rec['density_走廊']['gain_pct']*100:+.2f}%)", flush=True)
    print(f"  图 → {fig_path}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description="回波物理特性捞回硬质面（石板）")
    ap.add_argument("--site", choices=list(SRC_LAS), action="append",
                    help="可重复；默认两站")
    ap.add_argument("--h0", type=float, default=0.5, help="首末高差阈值（m），默认 0.5")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--no-write", action="store_true", help="只统计，不写 LAS")
    ap.add_argument("--limit", type=int, default=None, help="调试：每站只处理前 N 点")
    ap.add_argument("--las", default=None, help="覆盖源 LAS 路径（仅 --site 单站时有效）")
    ap.add_argument("--label", default=None, help="覆盖逐点 label .npy 路径")
    a = ap.parse_args()

    os.makedirs(FIG, exist_ok=True)
    sites = a.site or list(SRC_LAS)
    if (a.las or a.label) and len(sites) != 1:
        ap.error("--las / --label 只能与单个 --site 搭配")
    out = [run_site(s, a.h0, a.chunk, not a.no_write, a.limit, a.las, a.label)
           for s in sites]

    p = os.path.join(RES, "las_hard_surface.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n汇总 → {p}")


if __name__ == "__main__":
    main()
