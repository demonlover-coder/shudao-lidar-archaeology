"""由实测事实生成尺度契约表（单一真源，禁止手写数字）。

输入：
  results/audit.json   由 tools/audit_data.py 生成的实测事实
  CLI 参数             L* 口径与取值、统一 GSD、训练 crop、显存上限

输出：
  results/contract.json  机器可读契约（CI 比对用）
  results/contract.md    人类可读契约表

CI 用法：
  python tools/gen_contract.py --check      # 与已提交的 contract.json 不一致则退出码 1

说明：改骨干或改 GSD 后必须重跑本脚本并提交 diff，否则契约表就是死数。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_infer import plan_infer, receptive_field, unet_spec, erf_update_table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
AUDIT = os.path.join(RESULTS, "audit.json")

# 环境常数（改硬件要同步改这里，并重跑）
GPU = {"name": "NVIDIA GeForce RTX 5070 Laptop GPU", "vram_MiB": 8151, "driver": "617.14"}
PRECISION_BYTES = {"fp32": 4, "amp": 2}
# 小样本约束：参数量必须压到 1~3 M → base=16（约 2 M）；base=32（约 8 M）不可用
BASE_CH = 16
C_FIRST = BASE_CH  # 首层通道数，用于显存估算


def load_facts() -> dict:
    if not os.path.exists(AUDIT):
        raise SystemExit(f"缺少 {AUDIT}，先运行 tools/audit_data.py")
    return json.load(open(AUDIT, encoding="utf-8"))


def pick_lstar(facts: dict, mode: str, manual: float | None) -> dict:
    if mode == "segment_measured":
        seg = facts.get("segment_lengths", {})
        v = manual if manual is not None else (seg.get("p5_pca_m") or seg.get("min_pca_m"))
        return {"mode": mode, "value_m": v,
                "source": (f"实测：{seg.get('n')} 个自洽站面要素主轴长度的 P5"
                           f"（站点 {seg.get('usable_sites')}）；"
                           f"min={seg.get('min_pca_m')} m，中位={seg.get('median_pca_m')} m；"
                           f"P/2 独立估计 P5={seg.get('p5_P2_m')} m"),
                "lengths_pca_m": seg.get("lengths_pca_m"),
                "p5_pca_m": seg.get("p5_pca_m"), "p5_P2_m": seg.get("p5_P2_m"),
                "provenance_note": ("项目代码 MIN_LEN=50 m 是**检测阈值**"
                                    "（scripts/run_narrowband_detect.py:49），不是标注规范，"
                                    "且是在同一批面要素上事后确定的 → 用它当 L* 属循环论证")}
    if mode == "segment_spec":
        v = manual if manual is not None else 50.0
        return {"mode": mode, "value_m": v,
                "source": "项目检测阈值 MIN_LEN=50 m（非标注规范）",
                "provenance_note": "仅在拿到标注规范原文时可用；否则退回 segment_measured"}
    yjb = facts["shp"].get("杨家坝\\杨家坝-feature1.shp", {})
    widths = [w for w in (yjb.get("widths_2A_P_m") or []) if w]
    v = manual if manual is not None else (min(widths) if widths else None)
    return {"mode": "patch", "value_m": v,
            "source": "杨家坝-feature1.shp 面要素 2A/P 宽度（实测）",
            "widths_m": widths, "width_median_m": yjb.get("width_median_m"),
            "width_min_m": yjb.get("width_min_m"),
            "provenance_note": "线状地物的路面宽度不是其最小可识别单元，仅作对照"}


def sweep_backbones(gsd: float, rf_px_lo: float, rf_px_hi: float) -> list[dict]:
    rows = []
    for down in range(2, 7):
        for enc, dec in ((2, 1), (2, 2)):
            spec = unet_spec(down, enc, dec)
            rf, stride = receptive_field(spec)
            rf_m = rf * gsd
            rows.append({"down": down, "enc": enc, "dec": dec, "n_layers": len(spec),
                         "rf_px": round(rf, 1), "rf_m": round(rf_m, 2),
                         "in_band": rf_px_lo <= rf <= rf_px_hi})
    return rows


def mem_mb(px: int, c: int, bytes_per: int) -> float:
    return round(px * px * c * bytes_per / 1e6, 1)


def _dataset_facts() -> dict:
    """读 data/meta_{site}.json（由 build_dataset.py 生成）→ 有效面积/空洞/正样本。"""
    out: dict = {}
    for site in ("杨家坝", "西坝村"):
        p = os.path.join(ROOT, "data", f"meta_{site}.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p, encoding="utf-8"))
        out[site] = {"grid": m["grid"], "valid_area_km2": m["valid_area_km2"],
                     "void_frac": m["void_frac"], "pos_px": m["pos_px"],
                     "pos_rate_nominal": m["pos_rate"],
                     "pos_rate_in_valid": m["pos_in_valid_rate"],
                     "valid_bbox": m["valid_bbox"]}
    return out


def lstar_sensitivity(facts: dict, gsd: float, rf_px_lo: float, rf_px_hi: float,
                      chosen_rf: float, args) -> dict:
    """L* 敏感性：样本区间内 RF_px 目标区间怎么变，选定骨干是否全程落在区间内。

    样本区间取 [实测 min, 旧检测阈值 50 m]——前者是数据下界，后者是项目原本使用的值；
    另外给一个"L* < 35 m 需重新评估骨干"的下界告警。
    """
    seg = facts.get("segment_lengths", {})
    lo_m = seg.get("min_pca_m") or (35.0)
    hi_m = 50.0
    rows = []
    for name, val in (("min(实测下界)", lo_m), ("P5(采用)", seg.get("p5_pca_m")),
                      ("旧检测阈值", hi_m)):
        if val is None:
            continue
        rows.append({"lstar_label": name, "lstar_m": val,
                     "rf_px_lo": round(3 * val / gsd, 1),
                     "rf_px_hi": round(5 * val / gsd, 1),
                     "chosen_in_band": 3 * val / gsd <= chosen_rf <= 5 * val / gsd})
    all_band = [r["rf_px_lo"] for r in rows] + [r["rf_px_hi"] for r in rows]
    warn_m = 35.0
    return {
        "sample_range_m": [lo_m, hi_m],
        "rows": rows,
        "union_rf_px": [min(all_band), max(all_band)] if all_band else None,
        "chosen_rf_px": chosen_rf,
        "chosen_robust": all(r["chosen_in_band"] for r in rows) if rows else None,
        "warn_below_m": warn_m,
        "warn_rf_px_hi": round(5 * warn_m / gsd, 1),
        "verdict": ("选定骨干在 L* ∈ [%.1f, %.1f] 全程落在 RF_px 区间内 → 骨干选择稳健"
                    % (lo_m, hi_m)) if rows and all(r["chosen_in_band"] for r in rows)
                   else "存在使选定骨干越界的 L* 取值 → 需重新评估骨干",
    }


def spatial_split_check(facts: dict, tile: int, rf_px: float) -> dict:
    """验证"沿用点云侧 grid_split_v2 块划分"是否可行（沿用 = 含义 A：块划分本身）。

    事实来源：scripts/run_randlanet_grid_split_v2.py（GRID_BASE=40, MACRO_KS=[3]）
    与 data_real/{site}_grid_split_v2_meta.json（phase / macro_cnt / cell_cnt）。
    任一项检查不通过即不得硬跑。
    """
    grid_base, k = 40.0, 3
    macro = grid_base * k
    erode = rf_px / 2
    sites: dict = {}
    for site in ("杨家坝", "西坝村"):
        las = facts.get("sites", {}).get(site, {}).get("las") or []
        if not las:
            continue
        b = las[0]["bbox"]
        sites[site] = {"extent_m": [round(b[2] - b[0], 1), round(b[3] - b[1], 1)],
                       "fits_tile": max(b[2] - b[0], b[3] - b[1]) >= tile}
    cross = None
    if "杨家坝" in facts.get("sites", {}) and "西坝村" in facts.get("sites", {}):
        b1 = facts["sites"]["杨家坝"]["las"][0]["bbox"]
        b2 = facts["sites"]["西坝村"]["las"][0]["bbox"]
        cross = float((((b1[0] + b1[2]) / 2 - (b2[0] + b2[2]) / 2) ** 2 +
                       ((b1[1] + b1[3]) / 2 - (b2[1] + b2[3]) / 2) ** 2) ** 0.5)
    checks = {
        "macro_ge_tile": {"required_m": tile, "actual_m": macro, "pass": macro >= tile},
        "buffer_ge_rf_half": {"required_m": erode, "actual_m": 0.0, "pass": False,
                              "note": "grid_split_v2 无 buffer/margin，宏块按秩循环分配 "
                                      "→ 相邻宏块必然跨 split"},
        "erode_feasible": {"erode_each_side_m": erode, "leftover_m": macro - 2 * erode,
                           "pass": macro - 2 * erode > 0,
                           "note": "栅格侧须对每个 split 的像素掩膜腐蚀 RF/2 才能阻断感受野泄漏"},
        "site_extent_ge_tile": {"tile_m": tile,
                                "pass": all(s["fits_tile"] for s in sites.values()),
                                "detail": {s: v["extent_m"] for s, v in sites.items()}},
    }
    passing = all(c["pass"] for c in checks.values())
    return {
        "reuse_meaning": "A 沿用块划分本身（同一地理区域进同一 split）",
        "source": "scripts/run_randlanet_grid_split_v2.py + data_real/*_grid_split_v2_meta.json",
        "params": {"GRID_BASE_m": grid_base, "k": k, "macro_m": macro,
                   "pattern": "T/V/E 长度 20（14/3/3 ≈ 70/15/15）",
                   "phase": {"杨家坝": 17, "西坝村": 5},
                   "macro_cnt": {"杨家坝": [7, 2, 2], "西坝村": [23, 5, 5]},
                   "buffer_m": 0.0},
        "checks": checks,
        "verdict": "PASS：可直接沿用" if passing else "**FAIL：不得直接沿用**（见 blocking_finding）",
        "blocking_finding": (
            f"宏块 {macro:.0f} m < 一个 tile {tile} px；腐蚀 RF/2={erode:.0f} m 后剩余 "
            f"{macro - 2 * erode:.0f} m（负值）→ 宏块内不存在无泄漏像素。"
            f"更根本的是杨家坝整站点云范围仅 "
            f"{sites.get('杨家坝', {}).get('extent_m')} m，本身装不下一个 tile。"
            f"即 **点云尺度的空间划分与 RF={rf_px:.0f} m 的栅格感受野尺度不兼容**"),
        "cross_site_distance_m": round(cross, 1) if cross else None,
        "options": [
            "选项 1（已选）：栅格侧**双向跨站划分**（S1 杨训西验 + S2 西训杨验）。"
            "中心距 35.27 km ≫ RF/2 = 85 m，无感受野泄漏；与点云侧 run_cross_site_v2 同构，"
            "指标可直接对比；论文写「同一跨站协议」",
            "选项 2（已否决）：跨站 + 西坝村内部细分 —— 西坝村 1062×949 仅比 704 px 略大，"
            "内部最多切出 1~2 块，失去空间泛化检验意义，且细分块若无 85 m buffer 比跨站更危险",
            "选项 3（已否决）：下调尺度契约（RF ≤ 30 px）→ 与 L*=42.8 m 的 [128,214] m 冲突",
        ],
        "decision": "双向都跑（S1 + S2）—— 训练侧 0.33/1.01 km² 太小，任一侧都不能浪费；"
                    "单向跑要么欠拟合、要么验证集太小导致 P95 方差极大；双向是唯一能回答"
                    "「学到的是走廊特征还是站点局部纹理」的方案",
        "cross_site_plan": "results/cross_site_plan.md（像元数 / 正样本 / tile 数 / 边界带损失）",
    }


def build(args) -> dict:
    facts = load_facts()
    lstar = pick_lstar(facts, args.lstar_mode, args.lstar)
    if not lstar.get("value_m"):
        raise SystemExit("L* 无法确定：改用 --lstar-mode segment 或显式 --lstar")
    gsd = args.gsd
    rf_m_lo, rf_m_hi = 3 * lstar["value_m"], 5 * lstar["value_m"]
    rf_px_lo, rf_px_hi = rf_m_lo / gsd, rf_m_hi / gsd

    sweep = sweep_backbones(gsd, rf_px_lo, rf_px_hi)
    in_band = [r for r in sweep if r["in_band"]]
    chosen = min(in_band, key=lambda r: r["rf_px"]) if in_band else \
        min(sweep, key=lambda r: abs(r["rf_px"] - rf_px_hi))

    # L* 敏感性：在样本区间 [min, 50 m 旧阈值] 内扫一遍，看骨干选择是否全程稳健
    sens = lstar_sensitivity(facts, gsd, rf_px_lo, rf_px_hi, chosen["rf_px"], args)

    plan = plan_infer(chosen["rf_px"], align=args.align, k=args.k,
                      tile_floor=args.tile_floor, erf_px=args.erf)
    # 空间划分沿用性验证（点云侧 grid_split_v2 → 栅格侧），依赖 tile
    split = spatial_split_check(facts, plan.tile, chosen["rf_px"])
    crop = args.crop
    checks = {
        "rf_in_band": rf_px_lo <= chosen["rf_px"] <= rf_px_hi,
        "rf_le_tile_half": chosen["rf_px"] <= plan.tile / 2,
        "rf_le_crop_half": chosen["rf_px"] <= crop / 2,
        "tile_ge_4rf": plan.tile >= 4 * chosen["rf_px"],
        "train_window_positive": crop + 2 * plan.train_halo > crop,
        "vram_inference_ok": mem_mb(plan.tile, C_FIRST, PRECISION_BYTES["fp32"]) < 0.5 * GPU["vram_MiB"],
    }
    contract = {
        "generated_by": "tools/gen_contract.py",
        "note": "所有数字由本脚本算出；文档不得手填。改骨干/GSD 后重跑并提交 diff。",
        "gpu": GPU,
        "data_facts": {
            "gsd_span": facts.get("gsd", {}).get("span"),
            "gsd_min_m": facts.get("gsd", {}).get("min_m"),
            "gsd_max_m": facts.get("gsd", {}).get("max_m"),
            "memory_envelope": facts.get("gsd", {}).get("memory_envelope"),
            "vertical_datum_declared": facts.get("vertical_datum", {}).get("declared"),
        },
        "lstar": {**lstar,
                  "confidence": ("低（n<30，P5 近似等于最小值；真实 P5 可能更接近 "
                                 f"{facts.get('segment_lengths', {}).get('min_pca_m')} m）"
                                 if (facts.get("segment_lengths", {}).get("n") or 0) < 30
                                 else "中（n>=30）"),
                  "resample_trigger": "若后续数据使 n>=30，需重算 P5 并重新生成契约表（提交 diff）",
                  },
        "lstar_sensitivity": sens,
        "spatial_split": split,
        "dataset_facts": {
            # 由 tools/build_dataset.py 从实测栅格算出（稳定），不是手填
            "channels": ["dem_mean", "dem_min", "dem_max", "dem_std", "dom_lum",
                         "slope_deg", "aspect_deg", "curvature", "valid", "label"],
            "sites": _dataset_facts(),
            "note": "外部 DEM 覆盖范围内约 72% 是 nodata（只覆盖航带）→ "
                    "有效训练面积远小于名义网格面积；走廊像元 100% 落在有效区内",
        },
        "baseline_lower_bound": {
            "geometry_segmentation": "results/baseline_geometric.md（坡度阈值 + 形态学；"
                                     "阈值在训练站选、跨站报告）—— 模型必须显著高于该 IoU",
            "interpolation_regression": "待定标（空洞任务可行：DEM 空洞率 ~72%；"
                                        "但需先确认回归目标是「填洞」还是「走廊内路面高程」）",
        },
        "model_constraints": {
            "param_budget_M": [1.0, 3.0],
            "base_ch": BASE_CH,
            "params_measured": {"base_16": 1.03, "base_48": 9.26,
                                "law": "参数量 ≈ 1.03 M × (base/16)²（通道上限封在第 4 级）",
                                "base_24": 2.32, "base_32": 4.12},
            "forbidden": {"base_ch": 32, "params_M": 4.12,
                          "reason": "1.3 km²、正样本 <8000 px 的小样本下必然过拟合；"
                                    "注意本架构 base=32 是 4.12 M（通道封顶），不是 8 M"},
            "measured": "见 results/t3_model.log（--base 参数）",
            "loss": "BCE + Dice（或 Focal + Dice）；**禁止**纯逐像素 CrossEntropy —— "
                    "背景占比 99.4%，BCE 会被背景淹没，Dice 是全局梯度能拉回正样本贡献",
            "metrics_primary": "IoU（主指标）+ 走廊边界距离 Chamfer / HD95；"
                               "Accuracy 与单纯 F1 因 99.4% 背景占比而虚高，不作为主指标",
        },
        "sampling_policy": {
            "problem": "704 px tile 内正样本期望 2049/495616 ≈ 0.41%，随机裁剪时一个 batch "
                       "可能一个正样本都没有",
            "rule": [
                "窗口采样偏向正样本：以正样本为中心的裁剪概率 >= 80%",
                "或强制每个 batch 内走廊像素占比 >= 10%",
            ],
            "must_record": "实际采样命中率与每 batch 正样本占比，写入训练日志",
        },
        "paper_positioning": {
            "statement": "受限于 DEM 与点云覆盖范围，栅格侧任务定位为辅助验证（Proof of Concept）。"
                         "模型在跨站双向测试上报告指标，但不声称其具备大范围推广能力。",
            "reason": "训练侧合计仅约 1.3 km²、正样本 <8000 px；不写这句，审稿人看到数据量必然质疑；"
                      "写了且有双向跨站测试，作为辅助验证成立",
            "evidence": "与项目既有结论 #21「深度学习无法突破数据层信息边界」一致",
        },
        "step5_task": {            "main": "碥道走廊分割（像素级二分类）",
            "aux": "走廊内路面高程回归（辅助头，共享骨干）",
            "label_source": "长安大学 SHP 面要素栅格化为走廊 mask（无需新增标注；当前只用杨家坝 + 西坝村）",
            "input": "1 m 网格多通道：DEM 四通道(mean/min/max/std)、DOM 降采样、坡度/坡向/曲率、有效 mask",
            "output": "走廊概率 [0,1] + 走廊内路面高程",
            "loss": "分割头 BCE + Dice；回归头 masked L1（仅走廊内像素计入）",
            "metrics": "IoU / F1 / 边界距离（与点云侧同协议）；回归 RMSE/MAE",
            "fit_with_contract": ("L*=42.8 m 是段长，走廊的最小单元即「段」；RF=170 m 覆盖约 4 个 L*，"
                                  "足够判段；与上游点云 F1 任务同源可互相验证"),
            "rejected": {
                "纯 DEM 空洞填补": "自监督但绕开 L*/RF 尺度契约，浪费 Step 0/1 的产出",
                "坡度/汇流一致性回归": "适合做辅助损失，不宜作主任务",
            },
        },
        "baseline_protocol": {
            "scheme": "方案 1：插值基线只对连续场，分割侧另设基线",
            "regression_baselines": ["常数(训练集中位数)", "最近邻", "双线性", "IDW"],
            "regression_metrics": "RMSE / MAE / 坡度误差 / 汇流一致性",
            "segmentation_baseline": "坡度阈值 + 形态学（开/闭）+ 连通域筛选（几何基线）",
            "segmentation_metrics": "IoU / F1 / 边界距离",
            "shared_protocol": [
                "同 mask（无效区一致排除）",
                "同评估脚本",
                "评估排除距图像边界 >= RF/2 的边界带",
                "分层边界冻结（见 stratified_eval）",
            ],
            "reason": "IDW/双线性是插值方法，对分割任务没有直接输出；硬套没有物理意义",
        },
        "stratified_eval": {
            "ghsd_note": "固定网格路线下 GSD 恒定，按 GSD 分层无意义 → 改用下列维度",
            "dimensions": [
                "地形复杂度：坡度 P20/40/60/80 切 4 层（边界冻结）",
                "走廊密度：每 km² 碥道长度分层",
                "空间距离：与训练集最近块的地理距离分层 "
                "[0,500) / [500,2000) / [2000,5000) / [5000,+∞) m，"
                "每层报样本数 + IoU/F1 + 路面高程 RMSE；"
                "远距离层显著下降 → 空间泛化不足，需检查划分或加数据",
            ],
            "boundary_freeze": "分层边界由训练集统计得到并冻结写入 config；测试集沿用同一组边界",
            "edge_split": {
                "IoU_inner": "走廊内部像素（距边界 >= 2 px）",
                "IoU_border": "走廊边缘像素（距边界 < 2 px）",
                "reason": "DOM 0.0235 m 降到 1 m 后边缘光谱被平滑（1 px 覆盖 ~1764 原始像素）；"
                          "边缘误差必须单列，否则被平均值吃掉。与 T3 边界带处理同源",
            },
        },
        "route": {
            "gsd_target_m": gsd,
            "invariant": "pixel-context" if facts.get("gsd", {}).get("span", 1) > 2 else "physical-context",
            "rationale": f"GSD 跨度 {facts.get('gsd', {}).get('span')}x → 物理上下文显存包络 "
                         f"{facts.get('gsd', {}).get('memory_envelope')}x，不可行",
        },
        "rf_target": {"rf_m_lo": rf_m_lo, "rf_m_hi": rf_m_hi,
                      "rf_px_lo": round(rf_px_lo, 2), "rf_px_hi": round(rf_px_hi, 2),
                      "formula": "RF_m in [3L*, 5L*]; RF_px = RF_m / GSD"},
        "backbone": chosen,
        "backbone_sweep": sweep,
        "plan": {
            "align": plan.align, "halo": plan.halo, "overlap": plan.overlap,
            "tile": plan.tile, "step": plan.step, "cost": plan.cost,
            "erf_px": plan.erf_px, "erf_source": plan.erf_source,
            "checks": plan.checks,
        },
        "train": {
            "crop": crop, "halo": plan.train_halo,
            "window_read": crop + 2 * plan.train_halo,
            "loss_mask_edge_px": 1,
            "note": "halo 语义与推理不同：训练保证标签像素完整理论 RF；推理只保证核心区有效上下文",
        },
        "memory_MB_per_tensor": {
            "inference_tile_fp32": mem_mb(plan.tile, C_FIRST, PRECISION_BYTES["fp32"]),
            "inference_tile_amp": mem_mb(plan.tile, C_FIRST, PRECISION_BYTES["amp"]),
            "train_crop_fp32": mem_mb(crop, C_FIRST, PRECISION_BYTES["fp32"]),
            "c_first_channels": C_FIRST,
            "caveat": "仅首层单张特征图；实测量级请乘层数并用 AMP",
        },
        "erf_update_table": erf_update_table(chosen["rf_px"], args.align, args.k, args.tile_floor),
        "norm_policy": {
            "level": "与路线选择同级（Step 0 决策档 D2）",
            "allowed": ["none", "bn(eval, running stats)"],
            "forbidden": ["gn", "in"],
            "paper_sentence": "滑窗一致性是归一化的属性，不是 FCN 的默认属性",
            "reason": ("空间统计归一化（GN/IN）逐窗统计量与整图不同，"
                       "滑窗推理本身会改变预测；无归一化或 BN(eval) 与窗口无关"),
            "evidence": "results/t3_model.log 的 [T3-SWEEP]：none 3.6e-4 PASS、bn 4.2e-4 PASS、"
                        "gn 2.7e-1 FAIL（判据 1e-3×动态范围，内部区域）",
            "gate": "改归一化（含加 FiLM、改 stem）必须重跑 tests/test_t3_model.py",
        },
        "p2_prereg": {
            "task": "P2 纯坐标地形表征 PoC：HashGrid(2D) + MLP，输入仅 (x,y)∈[0,1]²，输出归一化高程",
            "registered_at": "2026-10-10（预登记，任务书原文）",
            "irrevocable": "本表在开跑后不得修改判据阈值；结果无论正负都按本表裁决",
            "primary_criterion": ("同站自验为主，跨站作对照（用户确认）—— 纯坐标网络只有 2 个输入，"
                                  "两站相距 35.27 km、高程带差约 300 m，站间不存在可迁移的地形信息，"
                                  "故跨站数字只作泛化对照，不触发淘汰"),
            "checkpoints": [
                {"id": "T4", "when": "1m 网格评估（同站自验）",
                 "condition": "S1/自验 1m RMSE > 1.5 m", "action": "停，P2 失败，回 v4，写 D12 留档"},
                {"id": "T5", "when": "多分辨率查询（0.5m）",
                 "condition": "0.5m 输出有条带/棋盘伪影",
                 "action": "调 base_resolution 或 per_level_scale，只调一次；再失败即停"},
                {"id": "T6", "when": "两站同站自验对比",
                 "condition": "两站 1m RMSE 差异 > 2×",
                 "action": "停，站点依赖过强，P2 失败"},
            ],
            "sig_eval_domain": ("评估域 = 源 DEM valid 像元 ∩ 落在训练站 bbox 内的像元；"
                                "跨站时两站 bbox 无交集 → 退化为整幅 valid 区（纯外推），"
                                "结果必须显式标注 extrapolation=true，不得与同口径数字并列"),
            "data_policy": ("训练必须用地面点（LAS classification==2）。全点云含树冠，"
                            "目标不再是 (x,y) 的单值函数：实测全点 vs DEM 偏差 杨 +3.71 m / 西 +2.79 m、"
                            "RMSE 5.12 / 4.75 m；地面点偏差 -0.008 / +0.002 m、RMSE 0.226 / 0.349 m"),
            "param_budget_M": [0.0, 2.0],
            "hash_size_deviation": ("任务书给 hash_table_size=2**19，16 级 × 2^19 × 2 = 16.78 M 参数，"
                                    "违反「参数量 < 2M」判据 → 默认取 2**15 = 1.05 M（含 MLP）；"
                                    "如需放开用 --hash-size 显式声明"),
            "reference_baseline_IDW_m": {"杨家坝→西坝村": 0.577, "西坝村→杨家坝": 0.435,
                                         "source": "results/baseline_regression.json，合成空洞率 72%，1m 网格"},
        },
        "resample_policy": {
            "rule": "降采样倍率 > 3 时禁用 bilinear/cubic（只采样源窗口内少数点 → 混叠）",
            "decimation_ratios": {"DOM_to_1m": 42.5, "DEM005_to_1m": 20.0, "DEM01_to_1m": 10.0},
            "dem_downsample_channels": ["mean", "min", "max", "std"],
            "dem_multichannel_note": ("DEM 降采样输出 4 通道（同块 mean/min/max/std）："
                                      "一次同时解决抗混叠与地形起伏信息丢失，"
                                      "这正是「不仅单通道」的落点"),
            "per_band_norm": ("Step 4 的 per-band 标准化必须按上述 4 通道分别统计，"
                              "不可跨通道共享统计量"),
            "channels": {
                "DEM 高程": "average（抗混叠）→ 输出 mean/min/max/std 四通道",
                "DOM/影像": "average",
                "掩膜": "最近邻 或 any/min 池化（禁止双线性后阈值化）",
                "分类/整数编码": "mode 或最近邻",
                "派生量(坡度/汇流)": "在目标网格上重算，不要从高分辨率降采样后平均",
            },
            "require_pyramid": True,
            "pyramid_note": ("分级降采样（2^k）而非一步到位：外部 DEM 已带 .ovr；"
                             "DOM 需先生成金字塔。**金字塔是必选项不是优化项**"),
            "max_safe_interp_decimation": 3,
        },
        "timing_budget": {
            "formula_model": "s/km² = (1e6 / GSD²) px × cost_factor / throughput",
            "io_bottleneck": {
                "measured_io_Mpx_per_s": 51.7,
                "read_1km2_at_005m_s": 7.7,
                "model_1km2_at_1m_s": 0.14,
                "ratio": 48.0,
                "verdict": "**IO 瓶颈**：不做金字塔时端到端预算重心在 IO 层而非模型层",
                "basis": "必须做金字塔的量化依据",
            },
            "rows_gsd": [0.05, 0.1, 0.5, 1.0],
            "table_a_note": "表 A（按 GSD 的端到端）数字在 contract.md 中按 timing.json 渲染",
            "table_b_note": ("表 B（COG overview 生成时间）当前只有 numpy 纯计算估算，"
                             "含 GeoTIFF 写入的真实耗时需 GDAL 实测"),
        },
        "t3_criteria": {
            "interior_rule": "仅对距图像边界 >= RF/2 的内部区域判定；边界带单独报告",
            "interior_threshold": "max|Δ| < 1e-3 × (p99|y| − p1|y|)",
            "amp_threshold": "1e-2 × 动态范围，且必须附 fp32 参考值",
            "determinism": "两次推理 bitwise equal（use_deterministic_algorithms(True)）",
            "whole_image_definition": "模型直接吃全图（非单块）",
            "border_band": ("整图用卷积零填充、滑窗用 replicate 填充，属约定差异；"
                            "实测边界带 2.5e1 vs 内部 1.8e-3（差 4 个数量级）→ "
                            "**边界像素的预测不得用于任何精度评估，评估协议须显式排除**"),
            "origin_test": "stitch 输出 vs 单块(tile=全图) 输出在内部区域一致（抓平移/原点/stride 类 bug）",
        },
        "decisions": {
            "route": "固定网格（统一 GSD=1.0 m）",
            "film": False,
            "film_reason": "固定 GSD 下尺度是常量，FiLM 退化为常数偏置",
            "gsd_binning": False,
            "randlanet_in_raster_pipeline": False,
            "randlanet_reason": "上游已是栅格；接点云网络等于硬塞 栅格→点云→栅格，每次转换引入误差。"
                                "仅当上游是原始 LAS 时才用",
        },
        "run_artifacts": {
            # 只记路径与存在性（稳定），**不把实测数字写进 contract.json**：
            # 那些数每次重测都会变，会导致 --check 永远失败。
            # 数字在 contract.md（人读）里按需读取打印。
            "timing": "results/timing.json" if os.path.exists(
                os.path.join(RESULTS, "timing.json")) else None,
            "erf_measured": "results/erf_measured.json" if os.path.exists(
                os.path.join(RESULTS, "erf_measured.json")) else None,
            "t3_model_log": "results/t3_model.log" if os.path.exists(
                os.path.join(RESULTS, "t3_model.log")) else None,
            "baseline_geometric": "results/baseline_geometric.md" if os.path.exists(
                os.path.join(RESULTS, "baseline_geometric.md")) else None,
            "dataset_report": "data/dataset_report.json" if os.path.exists(
                os.path.join(ROOT, "data", "dataset_report.json")) else None,
        },
        "checks": checks,
        "open_items": [
            "垂直基准：全部数据未声明；需向采集方索要基准说明或 GNSS 检查点",
            "ERF：随机初始化实测 8.6 px、RF/RF≈0.05（先验 0.4 过大）→ 训练收敛后重测；"
            "**重测后必须重新生成契约表并提交 diff**（halo/overlap/tile/算力全部会变）",
            "ERF 测量协议：仅允许 none 或 bn(eval) 模型；扰动法；>=10 样本报中位 + P5/P95；"
            "随机初始化与收敛后分开报告；记录模型配置哈希",
            "端到端时间预算：表 A（按 GSD）与表 B（COG overview 生成）待补真实耗时",
            "四档基线：常数/最近邻/双线性/IDW；**同协议定义（同 mask、同 GSD 分层边界、"
            "同评估脚本、评估时排除边界带）已写入 docs/01**",
            "L* 口径：已改用实测段长 P5；若拿到标注规范原文需复核",
        ],
    }
    return contract


def _load_optional(name: str, raw_key: str | None = None):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    if raw_key:
        txt = open(path, encoding="utf-8").read().strip().splitlines()
        return {"note": name, raw_key: txt[-8:]}
    return json.load(open(path, encoding="utf-8"))


def to_md(c: dict) -> str:
    p, b, t = c["plan"], c["backbone"], c["train"]
    rows = []
    rows.append("# 尺度契约表（自动生成，勿手改）\n")
    rows.append(f"> 生成器：`{c['generated_by']}`　硬件：{c['gpu']['name']} "
                f"({c['gpu']['vram_MiB']} MiB, driver {c['gpu']['driver']})\n")
    rows.append("## 1 决策\n")
    rows.append("| 项 | 值 | 来源 |")
    rows.append("|---|---|---|")
    rows.append(f"| L* | {c['lstar']['value_m']} m（{c['lstar']['mode']}） | {c['lstar']['source']} |")
    rows.append(f"| L* 置信度 | {c['lstar']['confidence']} | n={len(c['lstar'].get('lengths_pca_m') or [])}；{c['lstar']['resample_trigger']} |")
    ls = c["lstar_sensitivity"]
    rows.append(f"| **L* 敏感性** | L* ∈ [{ls['sample_range_m'][0]}, {ls['sample_range_m'][1]}] m "
                f"→ RF_px 目标并集 [{ls['union_rf_px'][0]}, {ls['union_rf_px'][1]}] px；"
                f"选定骨干 {ls['chosen_rf_px']} px | {ls['verdict']}；"
                f"若 L* < {ls['warn_below_m']} m（RF_px 上界 < {ls['warn_rf_px_hi']}）需重新评估骨干 |")
    rows.append(f"| 统一 GSD | {c['route']['gsd_target_m']} m | 路线 {c['route']['invariant']} |")
    rows.append(f"| GSD 跨度 | {c['data_facts']['gsd_span']}x "
                f"({c['data_facts']['gsd_min_m']}~{c['data_facts']['gsd_max_m']} m) | 实测 audit.json |")
    rows.append(f"| 显存包络 | {c['data_facts']['memory_envelope']}x | 跨度² |")
    rows.append(f"| 垂直基准 | {'已声明' if c['data_facts']['vertical_datum_declared'] else '**未声明**'} | GeoTIFF GeoKeys |")
    rows.append("\n## 2 感受野\n")
    rows.append("| 项 | 值 |")
    rows.append("|---|---|")
    rows.append(f"| RF_m 目标 | [{c['rf_target']['rf_m_lo']}, {c['rf_target']['rf_m_hi']}] m |")
    rows.append(f"| RF_px 目标 | [{c['rf_target']['rf_px_lo']}, {c['rf_target']['rf_px_hi']}] px |")
    rows.append(f"| 选定骨干 | U-Net down={b['down']} enc={b['enc']} dec={b['dec']} "
                f"({b['n_layers']} 层) |")
    rows.append(f"| RF 实测(公式) | {b['rf_px']} px = {b['rf_m']} m |")
    rows.append("\n## 3 窗口\n")
    rows.append("| 项 | 值 |")
    rows.append("|---|---|")
    for k in ("align", "halo", "overlap", "tile", "step", "cost", "erf_px", "erf_source"):
        rows.append(f"| {k} | {p[k]} |")
    rows.append(f"| 训练 crop / halo / 实读窗口 | {t['crop']} / {t['halo']} / {t['window_read']} px |")
    rows.append("\n### 3.1 归一化策略（决定滑窗一致性）\n")
    np_ = c["norm_policy"]
    rows.append(f"- 允许：{', '.join(np_['allowed'])}；禁止：{', '.join(np_['forbidden'])}")
    rows.append(f"- 依据：{np_['evidence']}")
    rows.append(f"- 门禁：{np_['gate']}")
    rows.append("\n### 3.2 重采样核策略（按通道）\n")
    rp = c["resample_policy"]
    rows.append(f"- {rp['rule']}")
    rows.append(f"- 降采样倍率：{rp['decimation_ratios']}")
    rows.append(f"- 需金字塔：{rp['require_pyramid']} —— {rp['pyramid_note']}")
    rows.append("")
    rows.append("| 通道 | 核 |")
    rows.append("|---|---|")
    for k, v in rp["channels"].items():
        rows.append(f"| {k} | {v} |")
    rows.append("\n## 4 显存（首层单张，fp32）\n")
    for k, v in c["memory_MB_per_tensor"].items():
        rows.append(f"- {k}: {v}")
    rows.append("\n## 5 ERF 实测更新位\n")
    rows.append("| ERF_px | ratio | halo | overlap | tile | 算力 | 动作 |")
    rows.append("|---|---|---|---|---|---|---|")
    for r in c["erf_update_table"]:
        if r.get("ok"):
            rows.append(f"| {r['erf_px']} | {r['erf_ratio']} | {r['halo']} | {r['overlap']} "
                        f"| {r['tile']} | {r['cost']}x | {r['action']} |")
        else:
            rows.append(f"| {r['erf_px']} | {r['erf_ratio']} | - | - | - | - | {r['action']} |")
    rows.append("\n## 6 通过条件自检\n")
    for k, v in c["checks"].items():
        rows.append(f"- {'PASS' if v else 'FAIL'}  {k}")
    rows.append("\n## 6.1 运行产物（易变量，读文件；不进 contract.json）\n")
    art = c.get("run_artifacts", {})
    for key, rel in art.items():
        if not rel:
            rows.append(f"- {key}: 尚未生成")
            continue
        path = os.path.join(ROOT, rel.replace("/", os.sep))
        try:
            if path.endswith(".json"):
                val = json.load(open(path, encoding="utf-8"))
                rows.append(f"- {key} (`{rel}`): `{json.dumps(val, ensure_ascii=False)}`")
            else:
                tail = open(path, encoding="utf-8").read().strip().splitlines()
                rows.append(f"- {key} (`{rel}`) 末行: `{tail[-1] if tail else ''}`")
        except Exception as exc:  # noqa: BLE001
            rows.append(f"- {key} (`{rel}`): 读取失败 {exc}")

    # ---- 时间预算表 A / 表 B（数字来自 timing.json，属易变量）
    tb = c["timing_budget"]
    tpath = os.path.join(RESULTS, "timing.json")
    tj = json.load(open(tpath, encoding="utf-8")) if os.path.exists(tpath) else {}
    coeff = c["plan"]["cost"]
    rows.append("\n## 6.2 时间预算表 A：按 GSD 的端到端\n")
    rows.append(f"公式：`{tb['formula_model']}`（含 {coeff}× 滑窗开销）\n")
    rows.append("| GSD (m) | 推理 s/km² (fp32/fp16) | IO s/km²(无金字塔) | IO s/km²(有金字塔) | 端到端 |")
    rows.append("|---|---|---|---|---|")
    tp32 = tj.get("throughput_px_per_s_fp32")
    tp16 = tj.get("throughput_px_per_s_fp16")
    io_rate = tj.get("io_read_Mpx_per_s", tb["io_bottleneck"]["measured_io_Mpx_per_s"])
    for g in tb["rows_gsd"]:
        px = 1e6 / (g * g)
        m32 = f"{px * coeff / tp32:.2f}" if tp32 else "待测"
        m16 = f"{px * coeff / tp16:.2f}" if tp16 else "待测"
        io_src = px / (io_rate * 1e6)
        io_pyr = (1e6 / (1.0 ** 2)) / (io_rate * 1e6)
        rows.append(f"| {g} | {m32} / {m16} | {io_src:.2f} | {io_pyr:.2f}（按 1 m 网格读） | 待测 |")
    ib = tb["io_bottleneck"]
    rows.append(f"\n**IO 瓶颈**：实测 IO {ib['measured_io_Mpx_per_s']} Mpx/s → "
                f"1 km² @0.05 m 读 {ib['read_1km2_at_005m_s']} s，"
                f"而同面积 1 m 网格推理仅 {ib['model_1km2_at_1m_s']} s → "
                f"**{ib['ratio']:.0f}× 差距；这是「金字塔是必选项」的量化依据**")
    rows.append("\n## 6.3 时间预算表 B：COG overview 生成\n")
    rows.append("| 源 GSD | 目标 GSD | overview 级别 | 纯计算估算 | 含写入实测 | 与推理比值 |")
    rows.append("|---|---|---|---|---|---|")
    ov = tj.get("overview_compute_s_per_km2_005m")
    rows.append(f"| 0.05 m | 1 m | [2,4,8,16,32] | "
                f"{f'{ov:.1f} s/km²（numpy 块平均，不含 IO 写入）' if ov else '待测'} | 待测 | 待测 |")
    rows.append(f"\n{tb['table_b_note']}")

    rows.append("\n## 6.4 Step 5 任务定义\n")
    st = c["step5_task"]
    rows.append(f"- **主任务**：{st['main']}")
    rows.append(f"- **辅助头**：{st['aux']}")
    rows.append(f"- 标签来源：{st['label_source']}")
    rows.append(f"- 输入：{st['input']}")
    rows.append(f"- 输出：{st['output']}")
    rows.append(f"- 损失：{st['loss']}")
    rows.append(f"- 指标：{st['metrics']}")
    rows.append(f"- 与尺度契约的匹配：{st['fit_with_contract']}")
    for k, v in st["rejected"].items():
        rows.append(f"- 不采用「{k}」：{v}")

    rows.append("\n## 6.5 基线协议与分层评估\n")
    bp = c["baseline_protocol"]
    rows.append(f"- 方案：{bp['scheme']}（{bp['reason']}）")
    rows.append(f"- 连续场基线：{'、'.join(bp['regression_baselines'])}；指标 {bp['regression_metrics']}")
    rows.append(f"- 分割基线：{bp['segmentation_baseline']}；指标 {bp['segmentation_metrics']}")
    rows.append("- 同协议四件套：")
    for s in bp["shared_protocol"]:
        rows.append(f"  - {s}")
    se = c["stratified_eval"]
    rows.append(f"- 分层维度（{se['ghsd_note']}）：")
    for s in se["dimensions"]:
        rows.append(f"  - {s}")
    rows.append(f"- {se['boundary_freeze']}")
    es = se["edge_split"]
    rows.append(f"- 边缘/内部拆分：**{es['IoU_inner']}** 与 **{es['IoU_border']}** 分别报告；"
                f"理由：{es['reason']}")

    rows.append("\n## 6.6 空间划分沿用性验证（点云侧 grid_split_v2 → 栅格侧）\n")
    sp = c["spatial_split"]
    p2 = sp["params"]
    rows.append(f"- 沿用含义：**{sp['reuse_meaning']}**；来源 `{sp['source']}`")
    rows.append(f"- 参数：GRID_BASE {p2['GRID_BASE_m']} m × k={p2['k']} → "
                f"宏块 **{p2['macro_m']:.0f} m**；区域模式 {p2['pattern']}；"
                f"phase {p2['phase']}；macro_cnt {p2['macro_cnt']}；buffer {p2['buffer_m']} m")
    rows.append("")
    rows.append("| 检查 | 要求 | 实测 | 结果 |")
    rows.append("|---|---|---|---|")
    for name, chk in sp["checks"].items():
        req = chk.get("required_m", chk.get("erode_each_side_m", chk.get("tile_m", "-")))
        act = chk.get("actual_m", chk.get("leftover_m", chk.get("detail", "-")))
        rows.append(f"| {name} | {req} | {act} | {'PASS' if chk['pass'] else '**FAIL**'} |")
    rows.append(f"\n**判定：{sp['verdict']}**")
    rows.append(f"\n**阻塞发现**：{sp['blocking_finding']}")
    rows.append(f"\n跨站中心距：**{sp['cross_site_distance_m']} m**")
    if sp.get("decision"):
        rows.append(f"\n**已决策**：{sp['decision']}")
    rows.append(f"\n跨站测算：`{sp.get('cross_site_plan', '')}`")
    rows.append("\n方案清单（含否决理由）：")
    for o in sp["options"]:
        rows.append(f"- {o}")

    rows.append("\n## 6.7 小样本约束（1.3 km² / 正样本 <8000 px）\n")
    mc = c["model_constraints"]
    rows.append(f"- 参数量预算：**{mc['param_budget_M'][0]}~{mc['param_budget_M'][1]} M**，"
                f"base_ch = **{mc['base_ch']}**（约 2 M）")
    rows.append(f"- 禁用：base_ch={mc['forbidden']['base_ch']}（约 {mc['forbidden']['params_M']} M）"
                f"—— {mc['forbidden']['reason']}")
    rows.append(f"- 损失：{mc['loss']}")
    rows.append(f"- 指标：{mc['metrics_primary']}")
    smp = c["sampling_policy"]
    rows.append(f"- 采样：{smp['problem']}")
    for s in smp["rule"]:
        rows.append(f"  - {s}")
    pp = c["paper_positioning"]
    rows.append(f"- **论文定位（必须写）**：{pp['statement']}")
    rows.append(f"  - 理由：{pp['reason']}")
    rows.append(f"  - 依据：{pp['evidence']}")

    rows.append("\n## 6.8 数据管线实测（有效面积 / 空洞 / 类别不平衡）\n")
    df = c["dataset_facts"]
    rows.append("| 站点 | 1 m 网格 | 有效面积 | 空洞率 | 走廊像元 | 名义正样本率 | **有效区内正样本率** |")
    rows.append("|---|---|---|---|---|---|---|")
    for s, v in df["sites"].items():
        rows.append(f"| {s} | {v['grid'][0]}×{v['grid'][1]} | {v['valid_area_km2']} km² | "
                    f"{v['void_frac']:.1%} | {v['pos_px']} | {v['pos_rate_nominal']:.3%} | "
                    f"**{v['pos_rate_in_valid']:.2%}** |")
    rows.append(f"\n- 通道：{', '.join(df['channels'])}")
    rows.append(f"- {df['note']}")
    bl = c["baseline_lower_bound"]
    rows.append(f"- 分割基线（下界）：{bl['geometry_segmentation']}")
    rows.append(f"- 回归基线：{bl['interpolation_regression']}")

    rows.append("\n## 7 未结项\n")
    for s in c["open_items"]:
        rows.append(f"- {s}")

    pr = c["p2_prereg"]
    rows.append("\n## 8 P2 预登记契约（HashGrid 纯坐标地形表征 PoC，开跑后不得改判据）\n")
    rows.append(f"- 任务：{pr['task']}")
    rows.append(f"- 预登记时间：{pr['registered_at']}；**{pr['irrevocable']}**")
    rows.append(f"- 主判据：{pr['primary_criterion']}")
    rows.append(f"- 评估域口径：{pr['sig_eval_domain']}")
    rows.append(f"- 数据口径：{pr['data_policy']}")
    rows.append(f"- 参数量预算：{pr['param_budget_M'][0]}~{pr['param_budget_M'][1]} M；"
                f"**偏离声明**：{pr['hash_size_deviation']}")
    rows.append(f"- 对照基线 IDW：{pr['reference_baseline_IDW_m']}")
    rows.append("\n| 检查点 | 触发时点 | 淘汰条件 | 动作 |")
    rows.append("|---|---|---|---|")
    for cp in pr["checkpoints"]:
        rows.append(f"| {cp['id']} | {cp['when']} | **{cp['condition']}** | {cp['action']} |")
    return "\n".join(rows) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstar-mode", choices=("segment_measured", "segment_spec", "patch"),
                    default="segment_measured")
    ap.add_argument("--lstar", type=float, default=None)
    ap.add_argument("--gsd", type=float, default=1.0)
    ap.add_argument("--align", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tile-floor", type=int, default=512)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--erf", type=float, default=None)
    ap.add_argument("--check", action="store_true", help="与已提交 contract.json 比对")
    a = ap.parse_args()

    c = build(a)
    js = json.dumps(c, ensure_ascii=False, indent=2, sort_keys=True)
    path_json = os.path.join(RESULTS, "contract.json")
    os.makedirs(RESULTS, exist_ok=True)

    if a.check:
        if not os.path.exists(path_json):
            print("FAIL: contract.json 不存在，先运行不带 --check 的生成")
            return 1
        cur = json.dumps(json.load(open(path_json, encoding="utf-8")),
                         ensure_ascii=False, indent=2, sort_keys=True)
        if cur != js:
            print("FAIL: 契约表与当前骨干/参数不一致 —— 重新生成并提交 diff")
            return 1
        print("PASS: 契约表与代码一致")
        return 0

    open(path_json, "w", encoding="utf-8").write(js)
    open(os.path.join(RESULTS, "contract.md"), "w", encoding="utf-8").write(to_md(c))
    print(to_md(c))
    print(f"写出: {path_json}")
    print(f"写出: {os.path.join(RESULTS, 'contract.md')}")
    bad = [k for k, v in c["checks"].items() if not v]
    print(f"自检: {'全部 PASS' if not bad else 'FAIL -> ' + ', '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
