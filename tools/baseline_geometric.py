"""分割侧几何基线（下界）：坡度阈值 + 形态学，跨站双向协议。

协议（与 D10 四件套一致）：
  * 阈值在**训练站**上选（最大化 IoU），在**验证站**上报告 → 与后续模型完全同协议
  * 只在有效像元（DEM 非空洞）∩ 内部区域（距图像边界 >= RF/2 = 85 px）内计算
  * 主指标 IoU，附 precision / recall / F1 / 预测面积占比
  * 分层：坡度 P20/40/60/80 四层

用法：python tools/baseline_geometric.py
产出：results/baseline_geometric.md / .json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
BAND = 85          # RF/2 边界排除带
SITES = ("杨家坝", "西坝村")


def load(site: str) -> dict:
    z = np.load(os.path.join(DATA, f"{site}_1m.npz"))
    return {k: z[k] for k in z.files}


def interior(shape: tuple[int, int], band: int = BAND) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[band:shape[0] - band, band:shape[1] - band] = True
    return m


def morph(mask: np.ndarray, open_r: int) -> np.ndarray:
    """先开后闭（PIL MinFilter/MaxFilter，尺寸 2r+1）。"""
    if open_r <= 0:
        return mask
    k = 2 * open_r + 1
    im = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    im = im.filter(ImageFilter.MinFilter(k))     # 腐蚀 → 开
    im = im.filter(ImageFilter.MaxFilter(k))     # 膨胀
    im = im.filter(ImageFilter.MaxFilter(k))     # 闭
    im = im.filter(ImageFilter.MinFilter(k))
    return np.asarray(im) > 127


def predict(d: dict, slope_max: float, open_r: int, valid: np.ndarray) -> np.ndarray:
    """几何基线：坡度 < 阈值 → 候选；再形态学清理。"""
    pred = (d["slope_deg"] < slope_max) & valid
    return morph(pred, open_r)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"IoU": round(iou, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "F1": round(f1, 4), "pred_px": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    data = {s: load(s) for s in SITES}
    valid = {s: (data[s]["valid"] == 1) for s in SITES}
    inner = {s: interior(data[s]["label"].shape) for s in SITES}
    gt = {s: (data[s]["label"] == 1) for s in SITES}
    # 评估域：有效 ∩ 内部
    domain = {s: valid[s] & inner[s] for s in SITES}

    lines: list[str] = []
    out: dict = {"band_px": BAND, "directions": []}

    def echo(s: str = "") -> None:
        lines.append(s)
        print(s)

    echo("# 分割侧几何基线（坡度阈值 + 形态学，跨站双向）\n")
    echo("协议：阈值在训练站选（最大化 IoU），在验证站报告；评估域 = 有效像元 ∩ "
          f"距图像边界 ≥ {BAND} px 的内部区域。\n")
    echo("| 站点 | 评估域像元 | 域内走廊 | 域内正样本率 | 走廊全在有效区? |")
    echo("|---|---|---|---|---|")
    for s in SITES:
        n = int(domain[s].sum())
        p = int((gt[s] & domain[s]).sum())
        echo(f"| {s} | {n} | {p} | {p / max(n,1):.2%} | "
             f"{'是' if int((gt[s] & ~valid[s]).sum()) == 0 else '否'} |")

    echo("\n## 一、训练站上的阈值搜索（最大化 IoU）\n")
    echo("| 方向 | 训练站 | 最优坡度阈值 (°) | 形态学半径 | 训练 IoU |")
    echo("|---|---|---|---|---|")
    best = {}
    for tr, te in ((SITES[0], SITES[1]), (SITES[1], SITES[0])):
        rows = []
        for t in (5, 10, 15, 20, 25, 30, 40):
            for r in (0, 1, 2, 3):
                pred = predict(data[tr], t, r, valid[tr]) & domain[tr]
                m = metrics(pred, gt[tr] & domain[tr])
                rows.append((m["IoU"], t, r, m))
        rows.sort(reverse=True)
        iou, t, r, m = rows[0]
        best[(tr, te)] = (t, r)
        echo(f"| {tr}→{te} | {tr} | {t} | {r} | {iou} |")

    echo("\n## 二、跨站迁移结果（几何基线的下界）\n")
    echo("| 方向 | 阈值 | IoU | Precision | Recall | F1 | 预测像元 | 预测面积占比 |")
    echo("|---|---|---|---|---|---|---|---|")
    for tr, te in ((SITES[0], SITES[1]), (SITES[1], SITES[0])):
        t, r = best[(tr, te)]
        pred = predict(data[te], t, r, valid[te]) & domain[te]
        m = metrics(pred, gt[te] & domain[te])
        area = m["pred_px"] / max(int(domain[te].sum()), 1)
        out["directions"].append({"train": tr, "test": te, "slope_max": t, "open_r": r,
                                  "test": m, "pred_area_frac": round(area, 4)})
        echo(f"| {tr}→{te} | {t}° | **{m['IoU']}** | {m['precision']} | {m['recall']} | "
             f"{m['F1']} | {m['pred_px']} | {area:.1%} |")

    echo("\n## 三、按坡度分层（验证站，使用训练站选定阈值）\n")
    echo("| 方向 | 坡度层 | 层内像元 | 层内走廊 | 该层 IoU |")
    echo("|---|---|---|---|---|")
    for tr, te in ((SITES[0], SITES[1]), (SITES[1], SITES[0])):
        t, r = best[(tr, te)]
        pred = predict(data[te], t, r, valid[te]) & domain[te]
        g, p, dm = gt[te] & domain[te], pred, domain[te]
        sl = data[te]["slope_deg"][dm]
        qs = np.percentile(sl, [20, 40, 60, 80])
        edges = [-np.inf, *qs, np.inf]
        for i in range(5):
            layer = dm & (data[te]["slope_deg"] >= edges[i]) & (data[te]["slope_deg"] < edges[i + 1])
            m = metrics(p & layer, g & layer)
            echo(f"| {tr}→{te} | P{20*(i+1) if i < 4 else 100} "
                 f"[{edges[i] if np.isfinite(edges[i]) else 0:.1f},"
                 f"{edges[i+1] if np.isfinite(edges[i+1]) else sl.max():.1f}) | "
                 f"{int(layer.sum())} | {int((g & layer).sum())} | {m['IoU']} |")

    echo("\n## 四、判读\n")
    ious = [d["test"]["IoU"] for d in out["directions"]]
    echo(f"- 几何基线的跨站 IoU = **{ious[0]} / {ious[1]}**（杨→西 / 西→杨）。"
         "这就是模型的**下界**：模型必须显著高于它才有意义。")
    echo("- 若某方向 IoU 极低（< 0.05），说明「坡度平坦」这类几何特征对走廊不具排他性，"
         "与项目既有结论 #15/#18/#19（窄带/走廊/全局路径三条几何路线全部负结果）一致 —— "
         "此时模型的提升要谨慎解读：小样本下模型也可能只是记住了训练站的局部纹理。")
    echo("- 论文写法：几何基线作为对照下限列出，并与点云侧的几何路线负结果互相印证。")

    open(os.path.join(RESULTS, "baseline_geometric.md"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(RESULTS, "baseline_geometric.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n写出: {os.path.join(RESULTS, 'baseline_geometric.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
