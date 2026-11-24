# -*- coding: utf-8 -*-
"""v6 补洞增益的分层验证：**有测点覆盖区** vs **无测点覆盖区**。

动机
----
P2 地形因子在**无地面点覆盖区**是噪声（P2 纯坐标 hashgrid 在那里外推 ±100 m，
见 D12 #11），而 v6 的补洞增益据称全部来自地形通道。于是必须回答：

    v6 的增益集中在有测点覆盖区（真增益，地形因子在那里可信），
    还是集中在无测点覆盖区（可能是"用噪声填噪声"的伪增益，评估协议没抓到）？

协议（与 eval_d10 严格同口径，保证数字可比）
  - **同一个**合成丢块 mask：seed=777、EVAL_FRAC=(0.02,0.04)、EVAL_BS=(16,48)
    （直接复用 `train_mt_v2.eval_drop_mask`，绝不重新生成）；
  - **像元级**分层：把被评估的 drop 像元按是否落在地面点覆盖区拆成两个子集
    （一个 16×48 块可能横跨两区，故必须按像元而非按块分层）；
  - 子集内分别算 RMSE（各模型 + **最近邻基线**），并报**样本数**；
  - 三方对照：v4（无地形基线）/ v6（地形+热启动）/ v6ab（地形，随机初始化）。

判读
----
  * 增益集中在覆盖区 → 地形因子可信，**真增益**；
  * 增益集中在无覆盖区 → 需警惕"用噪声填噪声"；
  * 无覆盖层样本 <200 px → 该层结论标 **不可靠**。

注意（评估盲区的残余风险）
  U-Net 感受野 170 px：即使目标像元在覆盖区，邻近无覆盖区的**噪声地形因子**仍会
  通过感受野渗入预测。故像元级分层是必要但不充分的；若覆盖区改善幅度与整体一致，
  则确认增益为真，感受野渗入为次要风险。

用法
----
    python tools/eval_v6_stratified.py
    python tools/eval_v6_stratified.py --stage pre --base 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from train_mt_v2 import MultiTaskUNetV2, eval_drop_mask, full_infer   # noqa: E402
from p2_dem import dem_window                                         # noqa: E402
from p2_export_all import project_covered                             # noqa: E402

DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")

DIRECTIONS = (("杨家坝", "西坝村"), ("西坝村", "杨家坝"))
# (前缀, 显示名)。v4=无地形基线；v6=地形+热启动；v6ab=地形+随机初始化（消融）
MODELS = (("v4_", "v4"), ("v6_", "v6"), ("v6ab_", "v6ab"))
UNC_RELIABLE_MIN = 200      # 无覆盖层样本数下限，低于此标"不可靠"


def covered_mask(site: str) -> np.ndarray:
    """地面点 3×3 膨胀覆盖掩码（与 p2_export_all 同口径，即 P2 可信域）。"""
    x0, y1, _wm, _hm, gsd, d = dem_window(site)
    H, W = d["dem_mean"].shape
    return project_covered(site, x0, y1, gsd, H, W)


def build_infer(prefix: str, stage: str, base: int, dev, train: str, val: str):
    """载入检查点并返回 `infer(drop) -> out`（自动区分是否含地形通道）。"""
    ckpt = os.path.join(CKPT, f"{prefix}{stage}_{train}_{val}_base{base}.pth")
    if not os.path.exists(ckpt):
        return None, f"检查点不存在: {os.path.basename(ckpt)}"
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    in_ch = int(sd["enc.0.body.0.weight"].shape[1])   # 10=无坐标, 12=含坐标, 15/17=+5 地形
    terrain = in_ch > 12
    coord = (in_ch - 5 == 12) if terrain else (in_ch == 12)
    has_ms = any(k.startswith("seg_heads.") for k in sd)
    model = MultiTaskUNetV2(in_ch=in_ch, base=base, norm="none",
                            multi_scale=has_ms).to(dev)
    model.load_state_dict(sd)
    model.eval()
    if terrain:
        from train_mt_v3 import full_infer_v3, load_terrain_norm
        terr = load_terrain_norm(val)

        def infer(drop):
            return full_infer_v3(model, val, dev, terr, coord=coord, drop=drop)
    else:
        def infer(drop):
            return full_infer(model, val, dev, coord=coord, drop=drop)
    return infer, f"in_ch={in_ch} coord={coord} 地形={terrain}"


def rmse(pred: np.ndarray, gt: np.ndarray, m: np.ndarray):
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    e = (pred[m] - gt[m]).astype(np.float64)
    return float(np.sqrt((e ** 2).mean())), n


def run_direction(train: str, val: str, stage: str, base: int, dev, log):
    log(f"\n{'=' * 72}")
    log(f"方向: {train} → {val}   stage={stage} base={base}")

    drop = eval_drop_mask(val, seed=777)         # 与 eval_d10 同一个 mask
    cov = covered_mask(val)

    # 分层（像元级）：全体 drop / 覆盖区 drop / 无覆盖区 drop
    layers = {
        "全体": drop,
        "覆盖区": drop & cov,
        "无覆盖区": drop & (~cov),
    }
    for name, m in layers.items():
        n = int(m.sum())
        share = f"{n / max(int(drop.sum()), 1):.1%}"
        log(f"  drop 像元[{name}]: {n:,}（占全部 drop {share}）")
    n_unc = int(layers["无覆盖区"].sum())
    unc_reliable = n_unc >= UNC_RELIABLE_MIN

    # 参考地板：最近邻填充（与模型无关）
    d0 = np.load(os.path.join(DATA, f"{val}_1m.npz"))
    valid = d0["valid"] == 1
    dem_m = d0["dem_mean"].astype(np.float32)
    nn_idx = distance_transform_edt(~(valid & (~drop)), return_distances=False,
                                    return_indices=True)
    nn_fill = dem_m[tuple(nn_idx)]

    rows = {}
    for prefix, name in MODELS:
        infer, info = build_infer(prefix, stage, base, dev, train, val)
        if infer is None:
            log(f"  [SKIP] {name}: {info}")
            continue
        r = infer(drop)
        pred = r["dem_pred_m"]
        for lname, m in layers.items():
            v, n = rmse(pred, dem_m, m)
            rows.setdefault(lname, {})[name] = dict(rmse=v, n=n)
            # 最近邻基线只在"全体/覆盖/无覆盖"各自子集上算一次
            if "nn" not in rows[lname]:
                nv, _ = rmse(nn_fill.astype(np.float32), dem_m, m)
                rows[lname]["nn"] = dict(rmse=nv, n=n)
        # 用"全体 drop"与 eval_d10 交叉核对
        log(f"  模型 {name} [{info}]  全体 drop RMSE={rows['全体'][name]['rmse']:.4f} m")

    # ---- 打印分层表 ----
    names = [n for _, n in MODELS if n in rows.get("全体", {})]
    log(f"\n  分层补洞 RMSE（m）——模型 vs 真值，同 mask、像元级分层：")
    head = f"  {'层':<8s}{'像元':>10s}" + "".join(f"{n:>10s}" for n in names) \
        + f"{'最近邻':>10s}{'v6−v4':>10s}{'相对':>9s}"
    log(head)
    log("  " + "-" * (len(head) - 2))
    for lname in ("全体", "覆盖区", "无覆盖区"):
        if lname not in rows:
            continue
        n = rows[lname]["nn"]["n"]
        cells = "".join(f"{rows[lname].get(n, {}).get('rmse', float('nan')):>10.4f}"
                        for n in names)
        nnv = rows[lname]["nn"]["rmse"]
        dv6 = rows[lname].get("v6", {}).get("rmse", float("nan"))
        dv4 = rows[lname].get("v4", {}).get("rmse", float("nan"))
        delta = dv6 - dv4
        rel = delta / dv4 if dv4 and np.isfinite(dv4) else float("nan")
        log(f"  {lname:<8s}{n:>10,d}{cells}{nnv:>10.4f}{delta:>+10.4f}{rel:>+9.1%}")

    # ---- 判读（用**相对**增益 + MSE 质量分解，绝对 Δ 因两层基线不同会误导）----
    def _rel(lname):
        dv6 = rows.get(lname, {}).get("v6", {}).get("rmse", float("nan"))
        dv4 = rows.get(lname, {}).get("v4", {}).get("rmse", float("nan"))
        return (dv6 - dv4) / dv4 if dv4 and np.isfinite(dv4) else float("nan")

    log("\n  判读：")
    n_cov = int(layers["覆盖区"].sum())
    n_all = int(drop.sum())
    # 各层对**总 MSE 降幅**的贡献（MSE 可加，RMSE 不可；这才是"增益来自哪"的度量）
    def _mse(lname, name):
        v = rows.get(lname, {}).get(name, {}).get("rmse", float("nan"))
        return v ** 2 if np.isfinite(v) else float("nan")
    drop_v4 = _mse("全体", "v4") - _mse("全体", "v6")
    share_cov = ((n_cov * (_mse("覆盖区", "v4") - _mse("覆盖区", "v6")))
                 / (n_all * drop_v4)) if drop_v4 > 0 else float("nan")
    log(f"    v6−v4 相对增益：覆盖区 {_rel('覆盖区'):+.1%}（{n_cov:,} px，"
        f"占 drop {n_cov / max(n_all, 1):.1%}）；无覆盖区 {_rel('无覆盖区'):+.1%}"
        f"（{n_unc:,} px，占 drop {n_unc / max(n_all, 1):.1%}）")
    log(f"    **总 MSE 降幅中来自覆盖区的份额 = {share_cov:.1%}**"
        f"（覆盖区占 drop {n_cov / max(n_all, 1):.1%}，份额 ≈ 占比则增益均匀分布，非集中于噪声区）")
    if not unc_reliable:
        log(f"    ⚠️ 无覆盖层仅 {n_unc} px（<{UNC_RELIABLE_MIN}）→ **不可靠**，该层结论仅供参考。")
    rel_cov, rel_unc = _rel("覆盖区"), _rel("无覆盖区")
    if np.isfinite(rel_cov) and rel_cov <= 0 and (not unc_reliable or rel_cov <= rel_unc):
        log("    ⇒ 增益在**有测点覆盖区**（地形因子可信处）成立且不弱于无覆盖区 → **真增益**。")
    elif unc_reliable and rel_unc < rel_cov:
        log("    ⇒ 增益更多集中在**无测点覆盖区**→ 需警惕'用噪声填噪声'，须复核。")
    log("    残余风险：RF=170px，覆盖区像元仍可能受邻近无覆盖区噪声地形因子的感受野渗入。")

    return dict(direction=f"{train}→{val}", stage=stage, base=base,
                n_drop=int(drop.sum()), n_cov=int(n_cov), n_unc=n_unc,
                unc_reliable=bool(unc_reliable), layers=rows)


def plot_stratified(results, out_path):
    """分层补洞 RMSE 柱状图（每个方向一子图：覆盖区 / 无覆盖区 × v4/v6/v6ab + 最近邻）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    names = [n for _, n in MODELS]
    layers = ["覆盖区", "无覆盖区"]
    x = np.arange(len(layers))
    w = 0.22
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5.5),
                             squeeze=False)
    for ax, res in zip(axes[0], results):
        for i, n in enumerate(names):
            vals = [res["layers"].get(L, {}).get(n, {}).get("rmse", float("nan"))
                    for L in layers]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=n)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=1)
        nnv = [res["layers"].get(L, {}).get("nn", {}).get("rmse", float("nan"))
               for L in layers]
        ax.plot(x, nnv, "k--o", lw=1.2, ms=5, label="最近邻基线")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{L}\n({res['layers'].get(L, {}).get('nn', {}).get('n', 0):,} px)"
                            for L in layers])
        ax.set_ylabel("合成丢块补洞 RMSE (m)")
        ax.set_title(f"{res['direction']}  stage={res['stage']}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("v6 补洞增益分层验证：有测点覆盖区 vs 无测点覆盖区")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  写出 {os.path.relpath(out_path, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pre", choices=["pre", "ft"],
                    help="pre=按补洞 RMSE 选出的权重（补洞指标口径，默认）；ft=微调权重")
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--val-site", default=None, choices=["杨家坝", "西坝村"],
                    help="只跑以该站为**评估站**的方向；缺省两个方向都跑")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = os.path.join(RES, "eval_v6_stratified.txt")
    lf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n")
        lf.flush()

    log(f"v6 补洞增益分层验证：有测点覆盖区 vs 无测点覆盖区")
    log(f"stage={a.stage} base={a.base} 设备={dev} 评估站={a.val_site or '全部'}")
    log(f"协议：同 eval_d10 的合成丢块 mask（seed=777, frac=2–4%, bs=16–48），"
        f"像元级分层；模型 {[n for _, n in MODELS]} + 最近邻基线")

    dirs = [d for d in DIRECTIONS if a.val_site is None or d[1] == a.val_site]
    out = [run_direction(tr, va, a.stage, a.base, dev, log) for tr, va in dirs]

    jp = os.path.join(RES, "eval_v6_stratified.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\n保存: {jp}")

    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    suffix = f"_{a.val_site}" if a.val_site else ""
    plot_stratified(out, os.path.join(fig_dir, f"eval_v6_stratified{suffix}.png"))
    lf.close()


if __name__ == "__main__":
    main()
