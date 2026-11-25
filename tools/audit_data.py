"""实测数据盘点（只读，不修改任何原始数据）。

产出：
  results/audit.json     机器可读的实测事实（供 gen_contract.py 消费，杜绝手填数字）
  results/audit_out.txt  人类可读的完整记录

覆盖：杨家坝 + 西坝村 的 GeoTIFF 元数据、LAS 头部、面要素宽度、GSD 跨度、垂直基准声明情况。

用法：
    python tools/audit_data.py
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 实测 DOM 5.5 亿像素，只读标签不解码全图

_DATA_ROOT = os.environ.get(
    "SHUDAO_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)
BASE = os.path.join(_DATA_ROOT, "survey-shp")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

GEO_KEYS = {
    1024: "GTModelType", 1025: "GTRasterType", 1026: "GeogCitation(as)",
    2048: "GeographicType", 2049: "GeogCitation", 2054: "GeogAngularUnits",
    3072: "ProjectedCSType", 3073: "PCSCitation", 3076: "ProjLinearUnits",
    4096: "VerticalCSType", 4097: "VerticalCitation", 4098: "VerticalDatum",
    4099: "VerticalUnits",
}
COMPRESSION = {1: "none", 5: "LZW", 7: "JPEG", 8: "DEFLATE", 32946: "DEFLATE",
               34712: "JPEG2000", 50000: "ZSTD"}
SAMPLEFORMAT = {1: "uint", 2: "int", 3: "float"}
SHAPE_TYPE = {1: "Point", 3: "PolyLine", 5: "Polygon", 11: "PointZ", 15: "PolygonZ"}


def _shorts(raw) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, bytes):
        return list(struct.unpack("<%dH" % (len(raw) // 2), raw))
    if isinstance(raw, (tuple, list)):
        return [int(v) for v in raw]
    return [int(raw)]


def geo_keys(tags: dict) -> dict:
    vals = _shorts(tags.get(34735))
    if len(vals) < 4:
        return {}
    ascii_raw = tags.get(34737)
    ascii_str = ascii_raw.decode("latin-1") if isinstance(ascii_raw, bytes) else str(ascii_raw or "")
    out: dict = {}
    for i in range(vals[3]):
        k, loc, cnt, off = vals[4 + 4 * i: 8 + 4 * i]
        name = GEO_KEYS.get(k, f"key{k}")
        if loc == 0:
            out[name] = off
        elif loc == 34737:
            out[name] = ascii_str[off:off + cnt].strip("|")
        elif loc == 34736:
            dbl = list(tags.get(34736) or [])
            out[name] = dbl[off:off + cnt]
        else:
            out[name] = f"<tag {loc} off {off}>"
    return out


def tif_info(path: str) -> dict:
    info: dict = {"file": os.path.basename(path),
                  "size_MB": round(os.path.getsize(path) / 1e6, 2)}
    with Image.open(path) as im:
        tags = dict(im.tag_v2)
        info.update(width=im.width, height=im.height, mode=im.mode,
                    bands=len(im.getbands()),
                    dtype=SAMPLEFORMAT.get(tags.get(339), str(tags.get(339))),
                    compression=COMPRESSION.get(tags.get(259), tags.get(259)),
                    tiled=bool(322 in tags))
        ps = tags.get(33550)
        if ps:
            info["pixel_scale_m"] = [round(float(v), 6) for v in ps]
        tp = tags.get(33922)
        if tp:
            info["tiepoint"] = [round(float(v), 4) for v in tp[:6]]
        nd = tags.get(42113)
        if nd is not None:
            info["nodata"] = nd.decode("latin-1") if isinstance(nd, bytes) else nd
        gk = geo_keys(tags)
        if gk:
            info["geo_keys"] = gk
            info["has_vertical_crs"] = bool(gk.get("VerticalCSType"))
        try:
            cw, ch = min(1024, im.width), min(1024, im.height)
            x0, y0 = (im.width - cw) // 2, (im.height - ch) // 2
            arr = np.asarray(im.crop((x0, y0, x0 + cw, y0 + ch)), dtype=np.float64)
            if arr.ndim == 2:
                fin = arr[np.isfinite(arr)]
                if fin.size:
                    info["center_crop_stats"] = {
                        "n": int(fin.size), "min": round(float(fin.min()), 3),
                        "p1": round(float(np.percentile(fin, 1)), 3),
                        "median": round(float(np.median(fin)), 3),
                        "p99": round(float(np.percentile(fin, 99)), 3),
                        "max": round(float(fin.max()), 3)}
        except Exception as exc:  # noqa: BLE001
            info["center_crop_stats"] = f"跳过: {exc}"
    return info


def las_info(path: str) -> dict:
    info: dict = {"file": os.path.basename(path),
                  "size_MB": round(os.path.getsize(path) / 1e6, 2)}
    with open(path, "rb") as f:
        head = f.read(375)
        if head[:4] != b"LASF":
            info["error"] = "不是 LAS"
            return info
        ver = (head[24], head[25])
        hsize = struct.unpack_from("<H", head, 94)[0]
        nvlr = struct.unpack_from("<I", head, 100)[0]
        pf = head[104]
        n = struct.unpack_from("<I", head, 107)[0]
        if ver >= (1, 4):
            f.seek(247)
            n14 = struct.unpack("<Q", f.read(8))[0]
            if n14:
                n = n14
        sx, sy, sz = struct.unpack_from("<3d", head, 131)
        ox, oy, oz = struct.unpack_from("<3d", head, 155)
        maxx, minx, maxy, miny, maxz, minz = struct.unpack_from("<6d", head, 179)
        vlrs = []
        pos = hsize
        for _ in range(min(nvlr, 64)):
            f.seek(pos)
            b = f.read(54)
            if len(b) < 54:
                break
            uid = b[2:18].rstrip(b"\0").decode("latin-1")
            rid, rlen = struct.unpack_from("<HH", b, 18)
            vlrs.append(f"{uid}:{rid}({rlen}B)")
            pos += 54 + rlen
    w, h = maxx - minx, maxy - miny
    area = w * h
    dens = n / area if area > 0 else 0.0
    info.update(las_version=f"{ver[0]}.{ver[1]}", point_format=pf, n_points=int(n),
                scale_m=[round(sx, 6), round(sy, 6), round(sz, 6)],
                extent_m=[round(w, 1), round(h, 1)], area_km2=round(area / 1e6, 4),
                bbox=[round(minx, 2), round(miny, 2), round(maxx, 2), round(maxy, 2)],
                z_range=[round(minz, 2), round(maxz, 2)],
                density_pts_per_m2=round(dens, 2),
                nominal_spacing_m=round(1.0 / np.sqrt(dens), 4) if dens else None,
                vlr_summary=vlrs)
    return info


# ---------------------------------------------------------------- SHP

def read_shp(path: str):
    raw = open(path, "rb").read()
    ftype = struct.unpack_from("<i", raw, 32)[0]
    shapes, pos = [], 100
    while pos + 8 <= len(raw):
        _n, clen = struct.unpack_from(">ii", raw, pos)
        pos += 8
        end = pos + clen * 2
        st = struct.unpack_from("<i", raw, pos)[0]
        if st in (5, 15, 25):
            nparts, npoints = struct.unpack_from("<ii", raw, pos + 36)
            parts = struct.unpack_from("<%di" % nparts, raw, pos + 44)
            pts = np.frombuffer(raw, dtype="<f8", count=npoints * 2,
                                offset=pos + 44 + 4 * nparts).reshape(-1, 2)
            shapes.append({"parts": parts, "pts": pts})
        pos = end
    return ftype, shapes


def ring_metrics(xy: np.ndarray) -> dict:
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perim = float(np.sum(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))))
    c = xy.mean(axis=0)
    _u, _s, vt = np.linalg.svd(xy - c, full_matrices=False)
    proj = (xy - c) @ vt.T
    major = float(proj[:, 0].max() - proj[:, 0].min())
    minor = float(proj[:, 1].max() - proj[:, 1].min())
    return {"area_m2": round(float(area), 2), "perimeter_m": round(perim, 2),
            "width_2A_P_m": round(2 * area / perim, 3) if perim else None,
            # 段长（线状地物 L* 的依据）：主轴跨度 与 周长/2 两种独立估计
            "length_pca_m": round(major, 2),
            "length_P2_m": round(perim / 2, 2),
            "pca_minor_m": round(minor, 2),
            "n_vertices": int(len(xy))}


def shp_infos() -> list[dict]:
    out = []
    for shp in sorted(glob.glob(os.path.join(BASE, "*", "*.shp"))):
        if os.path.getsize(shp) == 0:
            continue
        try:
            ftype, shapes = read_shp(shp)
        except Exception as exc:  # noqa: BLE001
            out.append({"file": shp, "error": str(exc)})
            continue
        rings, allx, ally = [], [], []
        for sh in shapes:
            for j, p0 in enumerate(sh["parts"]):
                p1 = sh["parts"][j + 1] if j + 1 < len(sh["parts"]) else len(sh["pts"])
                sub = sh["pts"][p0:p1]
                rings.append(ring_metrics(sub))
                allx += [sub[:, 0].min(), sub[:, 0].max()]
                ally += [sub[:, 1].min(), sub[:, 1].max()]
        out.append({
            "file": os.path.relpath(shp, BASE), "shape_type": SHAPE_TYPE.get(ftype, ftype),
            "n_features": len(shapes), "n_rings": len(rings),
            "widths_2A_P_m": [r["width_2A_P_m"] for r in rings],
            "width_min_m": min(r["width_2A_P_m"] for r in rings) if rings else None,
            "width_median_m": float(np.median([r["width_2A_P_m"] for r in rings])) if rings else None,
            "lengths_pca_m": [r["length_pca_m"] for r in rings],
            "lengths_P2_m": [r["length_P2_m"] for r in rings],
            "length_pca_min_m": min(r["length_pca_m"] for r in rings) if rings else None,
            "length_pca_median_m": float(np.median([r["length_pca_m"] for r in rings])) if rings else None,
            "length_P2_min_m": min(r["length_P2_m"] for r in rings) if rings else None,
            "area_total_m2": round(sum(r["area_m2"] for r in rings), 2),
            "bbox": [round(min(allx), 1), round(min(ally), 1),
                     round(max(allx), 1), round(max(ally), 1)] if allx else None,
            "rings": rings,
        })
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    facts: dict = {"base": BASE, "sites": {}, "shp": {}, "findings": []}
    lines: list[str] = []

    def echo(s: str = "") -> None:
        lines.append(s)
        print(s)

    echo("=" * 96)
    echo("一、栅格与点云元数据")
    echo("=" * 96)
    for site in sorted(os.listdir(BASE)):
        sdir = os.path.join(BASE, site)
        if not os.path.isdir(sdir):
            continue
        site_facts: dict = {"rasters": [], "las": []}
        for name in sorted(os.listdir(sdir)):
            p = os.path.join(sdir, name)
            low = name.lower()
            try:
                if low.endswith((".tif", ".tiff")):
                    info = tif_info(p)
                    site_facts["rasters"].append(info)
                elif low.endswith((".las", ".laz")):
                    info = las_info(p)
                    site_facts["las"].append(info)
            except Exception as exc:  # noqa: BLE001
                echo(f"[跳过] {p}: {exc}")
        facts["sites"][site] = site_facts
        echo(f"\n--- {site} ---")
        for r in site_facts["rasters"]:
            gk = r.get("geo_keys", {})
            echo(f"  栅格 {r['file']:34s} {r['width']}x{r['height']} "
                 f"bands={r['bands']} {r['dtype']} gsd={r.get('pixel_scale_m')} "
                 f"nodata={r.get('nodata')} 垂直基准={'有' if r.get('has_vertical_crs') else '未声明'}"
                 f" EPSG={gk.get('ProjectedCSType')}")
        for l in site_facts["las"]:
            echo(f"  点云 {l['file']:34s} {l['n_points']/1e6:.2f}M 点 "
                 f"密度={l['density_pts_per_m2']}/m2 间距={l['nominal_spacing_m']}m "
                 f"范围={l['extent_m']}m Z=[{l['z_range'][0]},{l['z_range'][1]}] "
                 f"{l['area_km2']}km2")

    echo()
    echo("=" * 96)
    echo("二、面要素（碥道）宽度")
    echo("=" * 96)
    for s in shp_infos():
        facts["shp"][s["file"]] = s
        if "error" in s:
            echo(f"  {s['file']}: {s['error']}")
            continue
        echo(f"  {s['file']:28s} 要素{s['n_features']:2d} 环{s['n_rings']:2d} "
             f"宽度 min={s['width_min_m']}m 中位={s['width_median_m']}m | "
             f"段长(PCA) min={s['length_pca_min_m']}m 中位={s['length_pca_median_m']}m | "
             f"段长(P/2) min={s['length_P2_min_m']}m | "
             f"总面积={s['area_total_m2']}m2 bbox={s['bbox']}")

    # 段长汇总（线状地物 L* 的实测依据；仅取与本站点云自洽的站点）
    usable = [r"杨家坝\杨家坝-feature1.shp", r"西坝村\西坝村-f1.shp"]
    seg = {"usable_sites": usable, "lengths_pca_m": [], "lengths_P2_m": [],
           "note": "项目代码中的 50 m 是检测阈值(MIN_LEN)，不是标注规范；L* 必须用面要素实测段长"}
    for f in usable:
        s = facts["shp"].get(f)
        if s and "error" not in s:
            seg["lengths_pca_m"] += s["lengths_pca_m"]
            seg["lengths_P2_m"] += s["lengths_P2_m"]
    if seg["lengths_pca_m"]:
        seg["n"] = len(seg["lengths_pca_m"])
        seg["min_pca_m"] = min(seg["lengths_pca_m"])
        seg["median_pca_m"] = round(float(np.median(seg["lengths_pca_m"])), 2)
        # L* 采用 P5（规范缺失时用实测分位数，比单点最小值稳健）
        seg["p5_pca_m"] = round(float(np.percentile(seg["lengths_pca_m"], 5)), 2)
        seg["p5_P2_m"] = round(float(np.percentile(seg["lengths_P2_m"], 5)), 2)
        seg["min_P2_m"] = min(seg["lengths_P2_m"])
        seg["median_P2_m"] = round(float(np.median(seg["lengths_P2_m"])), 2)
        echo()
        echo(f"  可用站点段长实测（n={seg['n']}）：PCA 主轴 P5={seg['p5_pca_m']}m "
             f"min={seg['min_pca_m']}m 中位={seg['median_pca_m']}m；"
             f"P/2 P5={seg['p5_P2_m']}m 中位={seg['median_P2_m']}m")
        echo("  注：项目 MIN_LEN=50 m 是检测阈值（run_narrowband_detect.py:49），非标注规范，"
             "不能当 L*")
    facts["segment_lengths"] = seg

    echo()
    echo("=" * 96)
    echo("三、GSD 跨度与垂直基准")
    echo("=" * 96)
    gsds: list[tuple[str, float]] = []
    for site, sf in facts["sites"].items():
        for r in sf["rasters"]:
            if r.get("pixel_scale_m"):
                gsds.append((f"{site}/{r['file']}", float(r["pixel_scale_m"][0])))
        for l in sf["las"]:
            if l.get("nominal_spacing_m"):
                gsds.append((f"{site}/{l['file']}(点间距)", l["nominal_spacing_m"]))
    if gsds:
        lo = min(g[1] for g in gsds)
        hi = max(g[1] for g in gsds)
        for n, v in sorted(gsds, key=lambda t: t[1]):
            echo(f"  {v:>9.4f} m   {n}")
        echo(f"\n  GSD 跨度 = {hi/lo:.1f}x   ({lo} m ~ {hi} m)  → 显存包络 ≈ {(hi/lo)**2:.0f}x")
        facts["gsd"] = {"min_m": lo, "max_m": hi, "span": round(hi / lo, 2),
                        "memory_envelope": round((hi / lo) ** 2, 1),
                        "list": [{"where": n, "gsd_m": v} for n, v in gsds]}

    vert = []
    for site, sf in facts["sites"].items():
        for r in sf["rasters"]:
            vert.append({"site": site, "file": r["file"],
                         "has_vertical_crs": bool(r.get("has_vertical_crs")),
                         "epsg": (r.get("geo_keys") or {}).get("ProjectedCSType")})
    facts["vertical_datum"] = {"declared": any(v["has_vertical_crs"] for v in vert),
                               "items": vert}
    echo(f"  垂直基准：{'至少一处声明' if facts['vertical_datum']['declared'] else '全部未声明'}")

    # 完整性交叉检查：SHP bbox 是否落在对应站点点云 bbox 内
    for sfile, s in facts["shp"].items():
        if "error" in s or not s.get("bbox"):
            continue
        site = sfile.split(os.sep)[0]
        las = facts["sites"].get(site, {}).get("las", [])
        if not las:
            continue
        bx0, by0, bx1, by1 = s["bbox"]
        lx0, ly0, lx1, ly1 = las[0]["bbox"]
        inside = bx0 >= lx0 - 1 and by0 >= ly0 - 1 and bx1 <= lx1 + 1 and by1 <= ly1 + 1
        tag = "自洽" if inside else "超出点云范围"
        facts["findings"].append({"shp": sfile, "site": site, "verdict": tag,
                                  "shp_bbox": s["bbox"], "las_bbox": las[0]["bbox"]})
        echo(f"  {sfile:28s} {tag}")

    echo()
    echo("=" * 96)
    echo("四、数据完整性问题（供论文核对）")
    echo("=" * 96)
    shps = {k: v for k, v in facts["shp"].items() if "error" not in v}
    names = list(shps)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if shps[a]["bbox"] == shps[b]["bbox"] and shps[a]["n_features"] == shps[b]["n_features"]:
                echo(f"  [重复] {a} 与 {b} 完全相同（bbox 与要素数一致）")
    for n in names:
        s = shps[n]
        bx0, by0, bx1, by1 = s["bbox"]
        span = max(bx1 - bx0, by1 - by0)
        if span > 5000:
            echo(f"  [合并] {n} 跨度 {span/1000:.1f} km，疑似把多个站点的面要素合并在一起")

    with open(os.path.join(OUT_DIR, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "audit_out.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n写出: {os.path.join(OUT_DIR, 'audit.json')}")
    print(f"写出: {os.path.join(OUT_DIR, 'audit_out.txt')}")


if __name__ == "__main__":
    sys.exit(main())
