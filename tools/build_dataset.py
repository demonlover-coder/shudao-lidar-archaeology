"""Step 5 数据管线：把实测栅格降到统一的 1 m 网格，生成多通道输入 + 标签。

按契约（resample_policy）：
  DEM 高程 → 同块 4 通道 mean / min / max / std（平均抗混叠 + 保住地形起伏）
  DOM      → 同块 box 平均亮度
  派生量   → 在**目标网格上重算**：坡度 / 坡向 / 曲率（Horn 3x3）
  掩膜     → 有效像元（源 DEM 非 nodata）
  标签     → SHP 面要素 4× 超采样栅格化（results/mask_1m_{site}.npy）

分块读取，不整幅进内存（杨家坝源 DEM 130 Mpx / DOM 549 Mpx）。

产出：data/{site}_1m.npz（含 meta）与 data/meta_{site}.json

用法：
    python tools/build_dataset.py                 # 两站，含 DOM
    python tools/build_dataset.py --no-dom        # 只做 DEM 与派生量
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_data import BASE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
GSD = 1.0

SITE_FILES = {
    "杨家坝": {"dem": r"杨家坝\杨家河坝碥道-dem.tif", "dom": r"杨家坝\杨家河坝碥道-dom.tif",
               "shp": r"杨家坝\杨家坝-feature1.shp"},
    "西坝村": {"dem": r"西坝村\西坝村栈道-dem.tif", "dom": r"西坝村\西坝村栈道-dom.tif",
               "shp": r"西坝村\西坝村-f1.shp"},
}


def dem_grid(path: str) -> dict:
    with Image.open(path) as im:
        tags = dict(im.tag_v2)
        ps = tags[33550]
        tp = tags[33922]
        scale = float(ps[0])
        return {"w_src": im.width, "h_src": im.height, "src_gsd": scale,
                "x0": float(tp[3]), "y1": float(tp[4]),
                "w": int(np.ceil(im.width * scale / GSD)),
                "h": int(np.ceil(im.height * scale / GSD)),
                "block": int(round(GSD / scale)),
                "nodata": float(str(tags.get(42113, "-9999")).strip("\x00") or -9999)}


def dem_channels(path: str, grid: dict, band_rows: int = 2000) -> dict:
    """分块读源 DEM，块内算 mean/min/max/std 四通道（整数倍降采样）。"""
    b = grid["block"]
    h, w = grid["h"], grid["w"]
    out = {k: np.zeros((h, w), dtype=np.float32) for k in ("mean", "min", "max", "std")}
    nod = grid["nodata"]
    with Image.open(path) as im:
        n_bands = int(np.ceil(grid["h_src"] / band_rows))
        for bi in range(n_bands):
            r0 = bi * band_rows
            r1 = min(r0 + band_rows, grid["h_src"])
            arr = np.asarray(im.crop((0, r0, grid["w_src"], r1)), dtype=np.float32)
            valid = arr != nod
            arr = np.where(valid, arr, np.nan)
            # 行方向补齐到 block 的整数倍
            ph = (-arr.shape[0]) % b
            if ph:
                arr = np.pad(arr, ((0, ph), (0, 0)), constant_values=np.nan)
            hh = arr.shape[0] // b
            blk = arr.reshape(hh, b, w, b).transpose(0, 2, 1, 3).reshape(hh, w, b * b)
            or0 = r0 // b
            or1 = min(or0 + hh, h)
            k = or1 - or0
            if k <= 0:
                del arr, blk
                continue
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out["mean"][or0:or1] = np.nanmean(blk[:k], axis=2)
                out["min"][or0:or1] = np.nanmin(blk[:k], axis=2)
                out["max"][or0:or1] = np.nanmax(blk[:k], axis=2)
                out["std"][or0:or1] = np.nanstd(blk[:k], axis=2)
            del arr, blk
    for k in out:
        out[k] = np.nan_to_num(out[k], nan=-9999.0 if k != "std" else 0.0)
        if out[k].shape[0] < h:                     # ceil 网格比块数多 1 行时补齐
            out[k] = np.pad(out[k], ((0, h - out[k].shape[0]), (0, 0)),
                            constant_values=-9999.0 if k != "std" else 0.0)
    out["valid"] = (out["max"] > -9000.0).astype(np.uint8)
    return out


def dom_luminance(path: str, grid: dict, band_out_rows: int = 100) -> np.ndarray:
    """分块把 DOM 降到 1 m 并取亮度（整数边界 box 平均；半像素误差 ≪ 1 m）。"""
    with Image.open(path) as im:
        rw, rh = im.width, im.height
        rb = np.linspace(0, rh, grid["h"] + 1).round().astype(int)
        cb = np.linspace(0, rw, grid["w"] + 1).round().astype(int)
        lut = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        out = np.zeros((grid["h"], grid["w"]), dtype=np.float32)
        for r in range(0, grid["h"], band_out_rows):
            r1 = min(r + band_out_rows, grid["h"])
            band = np.asarray(im.crop((0, int(rb[r]), rw, int(rb[r1]))).convert("RGB"),
                              dtype=np.float32) @ lut
            for rr in range(r, r1):
                rows = band[int(rb[rr]) - int(rb[r]):int(rb[rr + 1]) - int(rb[r])].mean(axis=0)
                cs = np.concatenate([[0.0], np.cumsum(rows, dtype=np.float64)])
                out[rr] = (cs[cb[1:]] - cs[cb[:-1]]) / np.maximum(cb[1:] - cb[:-1], 1)
            del band
    return out.astype(np.float32)


def dom_channels(path: str, grid: dict, band_out_rows: int = 100) -> dict:
    """DOM 彩度双通道：dom_exg（超绿 2G−R−B）与 dom_sat（饱和度 (max−min)/max）。

    源 DOM 是**真彩色 RGBA**；原 `dom_luminance` 用 `convert("RGB")@[0.299,0.587,0.114]`
    把它压成单通道后，走廊区分有效 AUC≈0.51（≈随机，见 results/dom_channel_auc.json）。
    `exg`/`sat` 则是两站方向一致（走廊更绿/更饱和）的强判别通道：有效 AUC
    exg 0.690（西）/0.649（杨）、sat 0.645/0.749。此处把它俩作为模型输入通道。

    降采样映射与 `dom_luminance` **逐位一致**（整数边界 box 平均），只是换了统计量。
    """
    with Image.open(path) as im:
        rw, rh = im.width, im.height
        rb = np.linspace(0, rh, grid["h"] + 1).round().astype(int)
        cb = np.linspace(0, rw, grid["w"] + 1).round().astype(int)
        cw = np.maximum(cb[1:] - cb[:-1], 1)
        out = {"dom_exg": np.zeros((grid["h"], grid["w"]), dtype=np.float32),
               "dom_sat": np.zeros((grid["h"], grid["w"]), dtype=np.float32)}
        for r in range(0, grid["h"], band_out_rows):
            r1 = min(r + band_out_rows, grid["h"])
            band = np.asarray(im.crop((0, int(rb[r]), rw, int(rb[r1]))).convert("RGB"),
                              dtype=np.float32)
            mx = band.max(2)
            fields = {"dom_exg": 2.0 * band[..., 1] - band[..., 0] - band[..., 2],
                      "dom_sat": (mx - band.min(2)) / np.maximum(mx, 1.0)}
            del band, mx
            for rr in range(r, r1):
                s = slice(int(rb[rr]) - int(rb[r]), int(rb[rr + 1]) - int(rb[r]))
                for k, f in fields.items():
                    rows = f[s].mean(axis=0)
                    cs = np.concatenate([[0.0], np.cumsum(rows, dtype=np.float64)])
                    out[k][rr] = (cs[cb[1:]] - cs[cb[:-1]]) / cw
            del fields
    return out


def terrain(dem: np.ndarray, gsd: float = GSD) -> dict:
    """Horn 3x3 坡度/坡向 + Laplacian 曲率，全部在 1 m 目标网格上重算。"""
    p = np.pad(dem, 1, mode="edge")
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * gsd)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * gsd)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)
    aspect = np.degrees(np.arctan2(dzdy, -dzdx)).astype(np.float32) % 360.0
    curv = ((a + b + c + d + f + g + h + i) / 8.0 - p[1:-1, 1:-1]).astype(np.float32)
    return {"slope_deg": slope, "aspect_deg": aspect, "curvature": curv}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=list(SITE_FILES))
    ap.add_argument("--no-dom", action="store_true")
    ap.add_argument("--dom-only", action="store_true",
                    help="只重算 DOM 通道（dom_exg/dom_sat）并写回现有 npz，"
                         "其余通道逐位保留（不重算 DEM，快得多）")
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report: dict = {}

    for site in a.sites:
        t0 = time.perf_counter()
        f = SITE_FILES[site]

        if a.dom_only:
            npz_path = os.path.join(a.out, f"{site}_1m.npz")
            if not os.path.exists(npz_path):
                raise SystemExit(f"{site}: 缺少 {npz_path}，先跑一次完整构建")
            meta_path = os.path.join(a.out, f"meta_{site}.json")
            meta = (json.load(open(meta_path, encoding="utf-8"))
                    if os.path.exists(meta_path) else {"site": site})
            arrays = dict(np.load(npz_path))            # 全量载入（栅格仅数 MB）
            h, w = arrays["valid"].shape
            grid = {"h": h, "w": w, "x0": meta.get("x0"), "y1": meta.get("y1"),
                    "src_gsd": meta.get("src_gsd_m"), "block": meta.get("block"),
                    "nodata": meta.get("nodata_src")}
            arrays.update(dom_channels(os.path.join(BASE, f["dom"]), grid))
            meta["channels"] = sorted(arrays)
            meta["seconds"] = round(time.perf_counter() - t0, 1)
            np.savez_compressed(npz_path, **arrays)
            json.dump(meta, open(meta_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            report[site] = meta
            print(f"[{site}] --dom-only 写回 DOM 彩度通道 → 共 {len(arrays)} 通道 "
                  f"耗时 {meta['seconds']}s")
            continue

        dem_path = os.path.join(BASE, f["dem"])
        grid = dem_grid(dem_path)
        ch = dem_channels(dem_path, grid)
        der = terrain(ch["mean"], GSD)
        label_path = os.path.join(RESULTS, f"mask_1m_{site}.npy")
        label = np.load(label_path).astype(np.uint8) if os.path.exists(label_path) else None
        if label is not None and label.shape != ch["mean"].shape:
            raise SystemExit(f"{site}: 标签 {label.shape} 与网格 {ch['mean'].shape} 不一致")

        arrays = {"dem_mean": ch["mean"], "dem_min": ch["min"], "dem_max": ch["max"],
                  "dem_std": ch["std"], "valid": ch["valid"], **der}
        if not a.no_dom:
            dom_path = os.path.join(BASE, f["dom"])
            # 保留 dom_lum 以保证 v4/v6/v7 可逐位复现；新增彩度双通道供 v8 替换使用
            arrays["dom_lum"] = dom_luminance(dom_path, grid)
            arrays.update(dom_channels(dom_path, grid))
        if label is not None:
            arrays["label"] = label

        # 空洞统计（决定回归侧"空洞填补"任务是否可行）+ 有效范围与标签重叠
        void = int((ch["valid"] == 0).sum())
        rows = np.where(ch["valid"].any(axis=1))[0]
        cols = np.where(ch["valid"].any(axis=0))[0]
        vbbox = ([int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])]
                 if rows.size else None)
        pos_in_valid = (int((label & ch["valid"]).sum()) if label is not None else None)
        meta = {"site": site, "gsd_m": GSD, "grid": [grid["w"], grid["h"]],
                "x0": grid["x0"], "y1": grid["y1"], "src_gsd_m": grid["src_gsd"],
                "block": grid["block"], "nodata_src": grid["nodata"],
                "valid_px": int(ch["valid"].sum()), "void_px": void,
                "void_frac": round(void / ch["valid"].size, 5),
                "valid_bbox": vbbox,
                "valid_area_km2": round(float(ch["valid"].sum()) / 1e6, 4),
                "dem_mean_range": [round(float(ch["mean"][ch["valid"] == 1].min()), 2),
                                   round(float(ch["mean"][ch["valid"] == 1].max()), 2)]
                if ch["valid"].any() else None,
                "pos_px": int(label.sum()) if label is not None else None,
                "pos_rate": round(float(label.mean()), 5) if label is not None else None,
                "pos_in_valid_px": pos_in_valid,
                "pos_in_valid_rate": (round(pos_in_valid / int(ch["valid"].sum()), 5)
                                      if pos_in_valid is not None and ch["valid"].any() else None),
                "channels": sorted(arrays), "seconds": round(time.perf_counter() - t0, 1)}
        np.savez_compressed(os.path.join(a.out, f"{site}_1m.npz"), **arrays)
        json.dump(meta, open(os.path.join(a.out, f"meta_{site}.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=2)
        report[site] = meta
        print(f"[{site}] {meta['grid'][0]}×{meta['grid'][1]} 通道 {len(arrays)} "
              f"有效像元 {meta['valid_px']} 空洞 {void}({meta['void_frac']:.3%}) "
              f"正样本 {meta['pos_px']}({meta['pos_rate']:.3%}) "
              f"耗时 {meta['seconds']}s")

    json.dump(report, open(os.path.join(a.out, "dataset_report.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n写出: {a.out}\\{{site}}_1m.npz + meta_*.json + dataset_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
