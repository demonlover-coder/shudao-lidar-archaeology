"""感受野 / halo / tile 规划器。

设计约束（与 Step 0 决策档一致）：
  * 训练 halo  = rf_px // 2                —— 标签像素必须看到完整理论 RF（硬保证）
  * 推理 halo  = ceil(erf_px / 2 / align)*align —— 只保证 tile 核心区的有效上下文（软近似）
  * 两者共用同一套 tile 坐标系与 align 对齐，但语义不同
  * 所有窗口参数由本模块算出，禁止手填进文档（避免文档与骨干脱节）

用法：
    python tools/plan_infer.py --down 5 --enc 2 --dec 1 --align 32 --json
    python tools/plan_infer.py --table          # ERF 实测值 → halo/overlap/算力 更新表
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import ceil


# ---------------------------------------------------------------- 感受野

def receptive_field(spec) -> tuple[float, float]:
    """spec: [(kernel, stride, dilation), ...] 按层顺序。

    返回 (RF_px, 总输出 stride)。上采样层用 (1, 0.5, 1) 表示：
    kernel=1 不增加 RF，stride=0.5 使 jump 减半。
    """
    rf, jump = 1.0, 1.0
    for k, s, d in spec:
        rf += (d * (k - 1)) * jump
        jump *= s
    return rf, jump


def conv_block(k: int = 3, n: int = 2):
    return [(k, 1, 1)] * n


def pool(k: int = 2, s: int = 2):
    return [(k, s, 1)]


def up():
    return [(1, 0.5, 1)]


def unet_spec(down: int = 3, enc: int = 2, dec: int = 1, k: int = 3):
    """标准 U-Net：每级 enc 层 3x3，下采样 maxpool 2x2，解码每级 dec 层 + 最近邻上采样。"""
    spec: list[tuple[int, int, int]] = []
    for _ in range(down):
        spec += conv_block(k, enc)
        spec += pool()
    spec += conv_block(k, enc)          # bottleneck
    for _ in range(down):
        spec += up()
        spec += conv_block(k, dec)
    return spec


# ---------------------------------------------------------------- 规划

@dataclass(frozen=True)
class Plan:
    rf_px: float
    erf_px: float
    erf_source: str
    align: int
    k: int
    tile_floor: int
    halo: int
    overlap: int
    tile: int
    step: int
    cost: float
    checks: dict
    train_halo: int
    note: str


def plan_infer(rf_px: float, align: int = 32, k: int = 4, tile_floor: int = 512,
               erf_px: float | None = None, erf_ratio: float = 0.4) -> Plan:
    if erf_px is not None:
        erf, src = float(erf_px), "measured"
    else:
        erf, src = erf_ratio * rf_px, f"prior(erf_ratio={erf_ratio})"

    halo = max(align, ceil(erf / 2 / align) * align)          # 单侧
    overlap = 2 * halo
    t = max(k * rf_px, tile_floor, overlap + rf_px)
    tile = ceil(t / align) * align
    step = tile - overlap
    cost = (tile / step) ** 2

    checks = {
        "tile_align": tile % align == 0,
        "overlap_align": overlap % align == 0,
        "step_align": step % align == 0,
        "overlap_lt_tile": overlap < tile,
        "core_ge_rf": step >= rf_px,          # 核心区宽度 >= RF
        "rf_le_tile_half": rf_px <= tile / 2,  # 场景尺度上界（= tile 口径）
    }
    failed = [n for n, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"窗口断言失败 {failed}: rf={rf_px} align={align} tile={tile} "

                             f"overlap={overlap} step={step}")

    return Plan(
        rf_px=rf_px, erf_px=erf, erf_source=src, align=align, k=k, tile_floor=tile_floor,
        halo=halo, overlap=overlap, tile=tile, step=step, cost=round(cost, 4),
        checks=checks,
        train_halo=int(rf_px // 2),
        note="core_ge_rf 保证每个 tile 存在宽度>=RF 的可信核心区；融合只依赖核心区",
    )


def erf_update_table(rf_px: float, align: int, k: int, tile_floor: int,
                     candidates=(0.2, 0.4, 0.6, 0.8, 1.0)) -> list[dict]:
    """ERF 实测值 → halo / overlap / 算力系数 的更新表（契约表预留位）。

    含 0.2 档：随机初始化实测 ERF/RF ≈ 0.05，说明 0.4 先验明显过大，
    需要一条"ERF 远小于理论 RF 时下压 halo"的路径。
    """
    rows = []
    for r in candidates:
        erf = r * rf_px
        try:
            p = plan_infer(rf_px, align=align, k=k, tile_floor=tile_floor, erf_px=erf)
            rows.append({"erf_px": round(erf, 1), "erf_ratio": r, "halo": p.halo,
                         "overlap": p.overlap, "tile": p.tile, "step": p.step,
                         "cost": p.cost, "ok": True, "action": _action(r)})
        except AssertionError as exc:
            rows.append({"erf_px": round(erf, 1), "erf_ratio": r, "ok": False,
                         "error": str(exc), "action": "需增大 tile 或降低 k"})
    return rows


def _action(r: float) -> str:
    """ERF/RF 五段判定。"""
    if r < 0.2:
        return "下压 halo 到 ERF 上取整并重算算力；记录「ERF 远小于理论 RF」"
    if r < 0.4:
        return "按 ERF 上取整，算力改善"
    if r <= 0.6:
        return "维持先验，算力不变"
    if r <= 0.8:
        return "扩 halo，复核 tile/显存"
    return "异常：查测量方法或被上采样污染"


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--down", type=int, default=5)
    ap.add_argument("--enc", type=int, default=2)
    ap.add_argument("--dec", type=int, default=1)
    ap.add_argument("--align", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tile-floor", type=int, default=512)
    ap.add_argument("--erf", type=float, default=None, help="ERF 实测值(px)；不给则用 0.4*RF 先验")
    ap.add_argument("--rf", type=float, default=None, help="直接给 RF_px（跳过骨干计算）")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--table", action="store_true", help="输出 ERF 更新表")
    a = ap.parse_args()

    spec = unet_spec(a.down, a.enc, a.dec)
    rf, stride = (a.rf, None) if a.rf else receptive_field(spec)

    if a.table:
        rows = erf_update_table(rf, a.align, a.k, a.tile_floor)
        print(json.dumps({"rf_px": rf, "align": a.align, "rows": rows},
                         ensure_ascii=False, indent=2))
        return

    p = plan_infer(rf, align=a.align, k=a.k, tile_floor=a.tile_floor, erf_px=a.erf)
    out = {"backbone": {"down": a.down, "enc": a.enc, "dec": a.dec,
                        "n_layers": len(spec), "output_stride": stride},
           "plan": asdict(p)}
    print(json.dumps(out, ensure_ascii=False, indent=2) if a.json else
          f"RF={rf:.0f}px  halo={p.halo} overlap={p.overlap} tile={p.tile} "
          f"step={p.step} cost={p.cost}x  训练halo={p.train_halo}")


if __name__ == "__main__":
    main()
