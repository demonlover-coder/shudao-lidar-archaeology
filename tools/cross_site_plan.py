"""跨站划分方案测算（栅格侧，GSD = 1 m）。

回答"跨站划分具体是什么样"：像元数、走廊正样本分布、tile 数、边界带影响、
两种方向（杨训西验 / 西训杨验）的类别不平衡。

不需要 GDAL：用 PIL 把 SHP 面要素栅格化到 1 m 网格（原点/像元取自各站外部 DEM）。

产出：
  results/cross_site_plan.md / .json
  results/mask_1m_{site}.npy      1 m 走廊掩膜（0/1），供后续 Step 5 复用

用法：python tools/cross_site_plan.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_data import BASE, read_shp  # noqa: E402
from tile_ops import stitch, tile_starts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
AUDIT = os.path.join(RESULTS, "audit.json")
GSD = 1.0
TILE, OVERLAP = 704, 128

SITE_SHP = {
    "杨家坝": r"杨家坝\杨家坝-feature1.shp",
    "西坝村": r"西坝村\西坝村-f1.shp",
}


def site_grid(facts: dict, site: str) -> dict:
    """由该站外部 DEM 的 tiepoint / pixel_scale 定义 1 m 网格。"""
    ras = facts["sites"][site]["rasters"]
    dem = next(r for r in ras if r["file"].endswith("-dem.tif"))
    scale = dem["pixel_scale_m"][0]
    x0 = dem["tiepoint"][3]
    y1 = dem["tiepoint"][4]          # tiepoint 在左上角
    w_m, h_m = dem["width"] * scale, dem["height"] * scale
    return {"site": site, "src_gsd_m": scale, "dem_size_px": [dem["width"], dem["height"]],
            "extent_m": [round(w_m, 1), round(h_m, 1)],
            "x0": x0, "y1": y1,
            "w": int(np.ceil(w_m / GSD)), "h": int(np.ceil(h_m / GSD))}


def rasterize(site: str, grid: dict, ss: int = 4) -> np.ndarray:
    """SHP 面要素 → 1 m 掩膜（0/1）。

    用 ss 倍超采样再按面积平均 + 0.5 阈值，消掉 PIL 多边形填充的边界像素正偏
    （直接填充会比面要素面积大 ~15%，即周长/2 量级的偏差）。
    """
    path = os.path.join(BASE, SITE_SHP[site])
    _ftype, shapes = read_shp(path)
    W, H = grid["w"] * ss, grid["h"] * ss
    img = Image.new("1", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for sh in shapes:
        for j, p0 in enumerate(sh["parts"]):
            p1 = sh["parts"][j + 1] if j + 1 < len(sh["parts"]) else len(sh["pts"])
            xy = sh["pts"][p0:p1]
            cols = (xy[:, 0] - grid["x0"]) / GSD * ss
            rows = (grid["y1"] - xy[:, 1]) / GSD * ss
            draw.polygon([(float(c), float(r)) for c, r in zip(cols, rows)], fill=1)
    hi = np.asarray(img, dtype=np.uint8).reshape(grid["h"], ss, grid["w"], ss)
    cover = hi.mean(axis=(1, 3))
    return (cover >= 0.5).astype(np.uint8)


def tile_stats(mask: np.ndarray) -> dict:
    """按契约 tile/overlap 统计：tile 数、含正样本的 tile 数、tile 内正样本分位数。"""
    h, w = mask.shape
    halo = OVERLAP // 2
    pad = [(halo, halo), (halo, halo)]
    mp = np.pad(mask, pad, mode="constant", constant_values=0)
    hp, wp = mp.shape
    ys = tile_starts(hp, TILE, TILE - OVERLAP)
    xs = tile_starts(wp, TILE, TILE - OVERLAP)
    pos_in_tile = []
    for y0 in ys:
        for x0 in xs:
            sub = mp[y0:y0 + TILE, x0:x0 + TILE]
            pos_in_tile.append(int(sub.sum()))
    n_tiles = len(pos_in_tile)
    with_pos = sum(1 for p in pos_in_tile if p > 0)
    return {"n_tiles": n_tiles, "tiles_with_positive": with_pos,
            "tiles_all_negative": n_tiles - with_pos,
            "positive_per_tile": sorted(pos_in_tile, reverse=True)}


def border_excluded(mask: np.ndarray, band: int) -> np.ndarray:
    """去掉距图像边界 band px 的带（评估协议要求）。"""
    m = mask.copy()
    if band > 0:
        m[:band, :] = 0
        m[-band:, :] = 0
        m[:, :band] = 0
        m[:, -band:] = 0
    return m


def main() -> int:
    facts = json.load(open(AUDIT, encoding="utf-8"))
    lines: list[str] = []
    out: dict = {"gsd_m": GSD, "tile": TILE, "overlap": OVERLAP, "sites": {}, "schemes": []}

    def echo(s: str = "") -> None:
        lines.append(s)
        print(s)

    echo("# 跨站划分方案测算（栅格侧 GSD=1 m，自动生成）\n")
    echo(f"tile={TILE} / overlap={OVERLAP} / 边界排除带 = RF/2 = 85 px\n")

    masks = {}
    echo("## 一、两站 1 m 网格与走廊掩膜\n")
    echo("| 站点 | 源 DEM GSD | 1 m 网格 (px) | 覆盖 (m) | 面积 (km²) | 走廊像元 | 正样本率 | 负:正 |")
    echo("|---|---|---|---|---|---|---|---|")
    for site in SITE_SHP:
        g = site_grid(facts, site)
        mask = rasterize(site, g)
        masks[site] = mask
        np.save(os.path.join(RESULTS, f"mask_1m_{site}.npy"), mask)
        pos = int(mask.sum())
        tot = mask.size
        g.update(pos_px=pos, tot_px=tot, pos_rate=round(pos / tot, 5),
                 neg_to_pos=round((tot - pos) / max(pos, 1), 1))
        out["sites"][site] = g
        echo(f"| {site} | {g['src_gsd_m']} m | {g['w']}×{g['h']} | "
             f"{g['extent_m'][0]:.0f}×{g['extent_m'][1]:.0f} | "
             f"{tot / 1e6:.3f} | {pos} | {pos/tot:.3%} | {g['neg_to_pos']}:1 |")

    echo("\n## 二、tile 覆盖与训练可行性（tile 内是否含正样本）\n")
    echo("| 站点 | tile 数 | 含正样本 tile | 全负 tile | 单 tile 正样本中位/最大 |")
    echo("|---|---|---|---|---|")
    for site, mask in masks.items():
        ts = tile_stats(mask)
        med = int(np.median(ts["positive_per_tile"]))
        mx = max(ts["positive_per_tile"])
        out["sites"][site].update(ts)
        echo(f"| {site} | {ts['n_tiles']} | {ts['tiles_with_positive']} | "
             f"{ts['tiles_all_negative']} | {med} / {mx} |")

    echo("\n## 三、边界排除带的影响（评估协议：距图像边界 ≥ RF/2）\n")
    echo("| 站点 | 原始正样本 | 排除边界带后 | 损失 | 排除后正样本率 |")
    echo("|---|---|---|---|---|")
    for site, mask in masks.items():
        band = 85
        m2 = border_excluded(mask, band)
        pos0, pos1 = int(mask.sum()), int(m2.sum())
        tot1 = (mask.shape[0] - 2 * band) * (mask.shape[1] - 2 * band)
        out["sites"][site]["pos_after_border_exclude"] = pos1
        echo(f"| {site} | {pos0} | {pos1} | {pos0 - pos1} ({(pos0-pos1)/max(pos0,1):.1%}) | "
             f"{pos1/max(tot1,1):.3%} |")

    echo("\n## 四、两种跨站方案\n")
    echo("| 方案 | 训练站（正样本/总像元/占比） | 验证站 | 训练 tile | 验证 tile | 训练负:正 | 说明 |")
    echo("|---|---|---|---|---|---|---|")
    for tr, va in (("杨家坝", "西坝村"), ("西坝村", "杨家坝")):
        s = out["sites"][tr]
        v = out["sites"][va]
        out["schemes"].append({"train": tr, "test": va,
                               "train_pos_px": s["pos_px"], "train_tot_px": s["tot_px"],
                               "test_pos_px": v["pos_px"], "test_tot_px": v["tot_px"],
                               "train_tiles": s["n_tiles"], "test_tiles": v["n_tiles"]})
        echo(f"| S{1 if tr == '杨家坝' else 2} | {tr}（{s['pos_px']} / {s['tot_px']} / "
             f"{s['pos_rate']:.2%}） | {va} | {s['n_tiles']} | {v['n_tiles']} | "
             f"{s['neg_to_pos']}:1 | 中心距 35.27 km，无感受野泄漏 |")

    echo("\n## 五、判读\n")
    s1, s2 = out["schemes"][0], out["schemes"][1]
    echo(f"- **S1（杨训西验）**：训练 {s1['train_tot_px']} 像元 / {s1['train_pos_px']} 正样本"
         f"（{s1['train_tiles']} 个 tile 覆盖整站）；验证 {s1['test_tot_px']} 像元 / "
         f"{s1['test_pos_px']} 正样本。训练侧正样本偏少 → 需 patch 采样 + 类别加权。")
    echo(f"- **S2（西训杨验）**：训练像元多 3 倍，但验证站只有 "
         f"{s2['test_tot_px']} 像元 / {s2['test_pos_px']} 正样本 → 验证集小，指标方差会大。")
    rates = "、".join(f"{out['sites'][s]['pos_rate']:.2%}" for s in SITE_SHP)
    ratios = "~".join(f"{out['sites'][s]['neg_to_pos']:.0f}" for s in SITE_SHP)
    echo(f"- 两站正样本率 **{rates}**（负:正 ≈ {ratios}:1）→ 极端不平衡二分类，"
         "指标必须报 IoU/F1/Dice，报准确率没有意义。")
    echo("- **数据量是硬约束**：训练侧只有 0.33 km²（杨）/ 1.01 km²（西）。这与项目既有结论"
         "（#21「深度学习无法突破数据层信息边界」）一致 → 栅格侧应定位为**辅助验证**，"
         "而不是主结果；模型规模必须小（参数量 ~1~3 M），靠增广与 patch 采样撑。")
    echo("- 建议：**两个方向都跑**（与点云侧 `run_cross_site_v2` 的双向协议一致），"
         "报两方向指标，避免单向结论。")
    echo("- `results/mask_1m_{site}.npy` 已保存（4× 超采样栅格化，面积偏差已消），"
         "可直接作为 Step 5 标签；DEM 四通道/DOM/坡度/曲率按同一网格生成即可。")

    open(os.path.join(RESULTS, "cross_site_plan.md"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(RESULTS, "cross_site_plan.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(f"\n写出: {os.path.join(RESULTS, 'cross_site_plan.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
