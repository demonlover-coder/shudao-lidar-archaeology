# -*- coding: utf-8 -*-
"""审稿防御实验包（四个定量实验，共用同一套冷启动输入/伪标签/交付口径）。

    --exp shortcut  去掉距离场通道（3ch：Hillshade + CSSC + openness），重训 CNN/U-Net
    --exp cross     零样本跨站：杨模型推西、西模型推杨，不做任何微调
    --exp shuffle   地形置换：DEM 高程随机打乱后重跑 λc=6 寻路，看 IoU 是否塌
    --exp seeds     随机种子 42/2023/2024 重复训练+交付，报公平 IoU 均值±标准差

评分一律走 `eval_freeaxis.py`（骨架轴线 + D+ 交付），SHP 只在最后评分。

用法：
    python tools/review_experiments.py --exp shortcut
    python tools/review_experiments.py --exp cross
    python tools/review_experiments.py --exp shuffle
    python tools/review_experiments.py --exp seeds
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import cold_start_transfer as C                                       # noqa: E402
import path_extraction as P                                           # noqa: E402
import p2_dem as PD                                                   # noqa: E402
import train_seg_compare as T                                         # noqa: E402
from eval_freeaxis import fill_nodata                                 # noqa: E402

RES = os.path.join(ROOT, "results")
SITES = ("杨家坝", "西坝村")
TRUTH_ARGS = {"杨家坝": ["--truth", "old", "--dist3", "3.5"],
              "西坝村": ["--truth", "new", "--truth-main", "--dist3", "3.5"]}
DEV = "cuda"


def log(msg):
    print(msg, flush=True)


def train_args(seed=0):
    return SimpleNamespace(device=DEV, seed=seed, patch=96, batch=32,
                           patch_per_epoch=512, lr=2e-3, epochs=300,
                           patience=20, min_epochs=30)


def eval_npz(site, npz_name, out, axis="skel"):
    """调 eval_freeaxis，返回该次运行的 json。"""
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "eval_freeaxis.py"),
           "--site", site, "--axis", axis, "--npz", npz_name, "--out", out]
    cmd += TRUTH_ARGS[site]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    f = os.path.join(RES, f"{out}_{site}_{axis}.json")
    if p.returncode != 0 or not os.path.exists(f):
        log(p.stdout[-800:] + "\n" + p.stderr[-800:])
        raise RuntimeError(f"eval 失败：{site} {npz_name}")
    e = json.load(open(f, encoding="utf-8"))
    pf = e["point_fair"]
    return dict(IoU=pf["IoU"], P=pf["P"], R=pf["R"],
                skelF1=e["skel_f1"]["2"]["F1"],
                bufIoU=e["buffer_iou_5m_band"]["1"], json=f)


def fit_predict(site, X, Y, valid, net_name, tag, seed=0, logf=log, prev=None):
    npz = os.path.join(RES, f"{tag}.npz")
    if os.path.exists(npz) and "prob" in np.load(npz).files:
        npar = sum(p.numel() for p in T.build_model(net_name, X.shape[0]).parameters())
        logf(f"    复用 {os.path.basename(npz)}（不重训，{npar / 1e3:.1f}k 参数，"
             f"{X.shape[0]}ch）")
        pv = prev or {}
        return dict(params=int(npar), epochs=pv.get("epochs"),
                    seconds=pv.get("seconds"))
    net = T.build_model(net_name, X.shape[0])
    npar = sum(p.numel() for p in net.parameters())
    a = train_args(seed)
    net = net.to(a.device)
    logf(f"    训练 {net_name}（{npar / 1e3:.1f}k 参数，{X.shape[0]}ch，seed {seed}）")
    hist, sec = T.train_one(net, X, Y, valid, a, logf)
    T.recalibrate_bn(net, X, valid, a, logf)
    prob = T.infer(net, X, valid, a.device, log=logf)
    np.savez_compressed(os.path.join(RES, f"{tag}.npz"), prob=prob.astype(np.float32))
    return dict(params=int(npar), epochs=len(hist), seconds=round(sec, 1))


# ================================================================ 1 捷径
def exp_shortcut():
    fprev = os.path.join(RES, "exp_shortcut.json")
    prev = json.load(open(fprev, encoding="utf-8")) if os.path.exists(fprev) else {}
    out = {}
    for site in SITES:
        log(f"\n[{site}] 去掉距离场通道（3ch）")
        d = T.load_case(site, log)
        X3 = d["X"][[0, 1, 3]]
        for net in ("cnn", "unet"):
            tag = f"nodeist_{site}_{net}"
            m = fit_predict(site, X3, d["Y"], d["valid"], net, tag,
                            prev=prev.get(f"{site}_{net}"))
            e = eval_npz(site, f"{tag}.npz", f"nodeist_{net}")
            log(f"    {net:<5s} IoU {e['IoU']:.4f}  P {e['P']:.4f}  R {e['R']:.4f}  "
                f"骨架F1 {e['skelF1']:.4f}")
            out[f"{site}_{net}"] = {**m, **e}
    json.dump(out, open(os.path.join(RES, "exp_shortcut.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("写出 results/exp_shortcut.json")


# ================================================================ 2 跨站
def exp_cross(net_name="cnn"):
    cases = {s: T.load_case(s, log) for s in SITES}
    out = {}
    for src, dst in (("杨家坝", "西坝村"), ("西坝村", "杨家坝")):
        for nch, sel in (("4ch", [0, 1, 2, 3]), ("3ch", [0, 1, 3])):
            log(f"\n[{src} → {dst}] 零样本（不微调，{nch}，{net_name}）")
            Xs, Xd = cases[src]["X"][sel], cases[dst]["X"][sel]
            tag = f"cross_{src}to{dst}_{nch}_{net_name}"
            npz = os.path.join(RES, f"{tag}.npz")
            if os.path.exists(npz) and "prob" in np.load(npz).files:
                log(f"    复用 {os.path.basename(npz)}（不重训）")
            else:
                net = T.build_model(net_name, Xs.shape[0]).to(DEV)
                a = train_args(0)
                T.train_one(net, Xs, cases[src]["Y"], cases[src]["valid"], a, log)
                T.recalibrate_bn(net, Xs, cases[src]["valid"], a, log)
                prob = T.infer(net, Xd, cases[dst]["valid"], a.device, log=log)
                np.savez_compressed(npz, prob=prob.astype(np.float32))
            e = eval_npz(dst, f"{tag}.npz", f"cross_{src}to{dst}_{nch}_{net_name}")
            log(f"    {dst} 上 IoU {e['IoU']:.4f}  P {e['P']:.4f}  R {e['R']:.4f}")
            out[f"{src}->{dst}_{nch}"] = e
    json.dump(out, open(os.path.join(RES, f"exp_cross_{net_name}.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"写出 results/exp_cross_{net_name}.json")


# ================================================================ 3 地形置换
def path_endpoints(mask):
    pts = np.argwhere(P.find_endpoints(mask))
    if len(pts) < 2:
        pts = np.argwhere(mask)
    a, b = P.merge_and_pick_pair(pts)
    return (a, b)


def exp_shuffle():
    out = {}
    for site in SITES:
        log(f"\n[{site}] 地形置换")
        x0, y1, wm, hm, gsd, d = PD.dem_window(site)
        valid = d["valid"] == 1
        dem = fill_nodata(d["dem_mean"], valid)
        z = np.load(os.path.join(RES, f"cold_start_{site}_lc6.npz"))
        real = z["path"].astype(bool)
        prob = z["prob"].astype(np.float32)
        ep = path_endpoints(real)
        log(f"  端点 {ep[0]} ↔ {ep[1]}")

        # 真实地形上的 λc=6 路径（同口径基线）
        a = SimpleNamespace(dir_lambda=0.0, turn_lambda=0.0, turn_min=2,
                            slope_lambda=0.0, z_lambda=2.0, lam_c=6.0,
                            max_grade=0.0, grade_lag=5)
        op, nr, nc, conf, _ = C.macro_fields(dem, valid)
        path_real, _, _, _ = C.macro_path(op, valid, ep, nr, nc, conf, a,
                                          log=log, dem=dem)
        np.savez_compressed(os.path.join(RES, f"pathreal_{site}.npz"),
                            prob=prob, path=path_real.astype(np.uint8))
        e0 = eval_npz(site, f"pathreal_{site}.npz", f"pathreal", axis="path")
        log(f"  真地形 λc=6：IoU {e0['IoU']:.4f}")

        rng = np.random.default_rng(42)
        sh = dem.copy()
        sh[valid] = rng.permutation(sh[valid])
        op2, nr2, nc2, conf2, _ = C.macro_fields(sh, valid)
        path_sh, _, _, _ = C.macro_path(op2, valid, ep, nr2, nc2, conf2, a,
                                        log=log, dem=sh)
        np.savez_compressed(os.path.join(RES, f"shuffle_{site}.npz"),
                            prob=prob, path=path_sh.astype(np.uint8))
        e1 = eval_npz(site, f"shuffle_{site}.npz", f"shuffle", axis="path")
        log(f"  置换地形 λc=6：IoU {e1['IoU']:.4f}（掉落 "
            f"{(e0['IoU'] - e1['IoU']) * 100:.1f} 个点）")
        out[site] = dict(real=e0, shuffled=e1)
    json.dump(out, open(os.path.join(RES, "exp_shuffle.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("写出 results/exp_shuffle.json")


# ================================================================ 4 种子
def exp_seeds(net_name="cnn"):
    fprev = os.path.join(RES, f"exp_seeds_{net_name}.json")
    prev = json.load(open(fprev, encoding="utf-8")) if os.path.exists(fprev) else {}
    out = {}
    for site in SITES:
        log(f"\n[{site}] 种子稳定性（{net_name}）")
        d = T.load_case(site, log)
        pmap = {r["seed"]: r for r in prev.get(site, {}).get("runs", [])}
        rows = []
        for seed in (42, 2023, 2024):
            tag = f"seed{seed}_{site}_{net_name}"
            fit_predict(site, d["X"], d["Y"], d["valid"], net_name, tag, seed=seed,
                        prev=pmap.get(seed))
            e = eval_npz(site, f"{tag}.npz", f"seed{seed}_{net_name}")
            rows.append(dict(seed=seed, **e))
            log(f"    seed {seed}: IoU {e['IoU']:.4f}  P {e['P']:.4f}  R {e['R']:.4f}")
        iou = np.array([r["IoU"] for r in rows])
        out[site] = dict(runs=rows, mean=round(float(iou.mean()), 4),
                         std=round(float(iou.std(ddof=1)), 4),
                         min=round(float(iou.min()), 4), max=round(float(iou.max()), 4))
        log(f"    → 公平 IoU = {iou.mean():.3f} ± {iou.std(ddof=1):.3f}")
    json.dump(out, open(os.path.join(RES, f"exp_seeds_{net_name}.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"写出 results/exp_seeds_{net_name}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["shortcut", "cross", "shuffle", "seeds"])
    ap.add_argument("--net", default="cnn", choices=["cnn", "unet"])
    a = ap.parse_args()
    log("=" * 76)
    log(f"审稿防御实验：{a.exp}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    {"shortcut": exp_shortcut, "cross": lambda: exp_cross(a.net),
     "shuffle": exp_shuffle, "seeds": lambda: exp_seeds(a.net)}[a.exp]()


if __name__ == "__main__":
    main()
