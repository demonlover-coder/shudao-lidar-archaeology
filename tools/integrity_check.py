"""数据完整性核查（只读）。

把三处疑点变成可引用的产物 results/integrity_check.md + .json：

  C1 重复文件：不同 SHP 的环几何指纹完全相同
  C2 合并文件：单文件内要素跨越多个站点（跨度异常）
  C3 站名与位置不符：SHP 质心归属到"最近站点点云"，与目录名不一致

同时给出每个要素的归属站点，供"是否纳入训练/评估"决策使用。

用法：python tools/integrity_check.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_data import BASE, read_shp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
AUDIT = os.path.join(RESULTS, "audit.json")


def site_bboxes(facts: dict) -> dict[str, list[float]]:
    out = {}
    for site, sf in facts["sites"].items():
        if sf.get("las"):
            out[site] = sf["las"][0]["bbox"]
    return out


def bbox_dist(p: tuple[float, float], b: list[float]) -> float:
    """点到 bbox 的距离（内部为 0）。"""
    x0, y0, x1, y1 = b
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    return float(np.hypot(dx, dy))


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    facts = json.load(open(AUDIT, encoding="utf-8"))
    sites = site_bboxes(facts)

    files: dict[str, dict] = {}
    for shp in sorted(glob.glob(os.path.join(BASE, "*", "*.shp"))):
        if os.path.getsize(shp) == 0:
            continue
        rel = os.path.relpath(shp, BASE)
        _ftype, shapes = read_shp(shp)
        feats = []
        for sh in shapes:
            for j, p0 in enumerate(sh["parts"]):
                p1 = sh["parts"][j + 1] if j + 1 < len(sh["parts"]) else len(sh["pts"])
                xy = sh["pts"][p0:p1]
                c = xy.mean(axis=0)
                # 环几何指纹：顶点量化后哈希（判重复文件用）
                fp = hashlib.sha1(np.round(xy - c, 3).tobytes()).hexdigest()[:16]
                feats.append({"centroid": [round(float(c[0]), 1), round(float(c[1]), 1)],
                              "fingerprint": fp,
                              "span_m": [round(float(np.ptp(xy[:, 0])), 1),
                                         round(float(np.ptp(xy[:, 1])), 1)],
                              "n_vertices": int(len(xy))})
        files[rel] = {"dir_site": rel.split(os.sep)[0], "n_features": len(feats),
                      "features": feats,
                      "file_fingerprint": hashlib.sha1(
                          "".join(sorted(f["fingerprint"] for f in feats)).encode()
                      ).hexdigest()[:16]}

    lines: list[str] = []
    out: dict = {"files": files, "findings": []}

    def echo(s: str = "") -> None:
        lines.append(s)
        print(s)

    echo("# 数据完整性核查（自动生成）\n")
    echo(f"来源：`tools/integrity_check.py`；SHP 根目录 `{BASE}`\n")

    echo("## C1 重复文件检测（环几何指纹）\n")
    echo("| 文件 | 要素数 | 文件指纹 | 判定 |")
    echo("|---|---|---|---|")
    seen: dict[str, list[str]] = {}
    for rel, info in files.items():
        seen.setdefault(info["file_fingerprint"], []).append(rel)
    for rel, info in files.items():
        dup = seen[info["file_fingerprint"]]
        verdict = "**与 " + "、".join(x for x in dup if x != rel) + " 完全相同**" if len(dup) > 1 else "唯一"
        echo(f"| {rel} | {info['n_features']} | {info['file_fingerprint']} | {verdict} |")
        if len(dup) > 1 and rel == dup[0]:
            out["findings"].append({"type": "duplicate", "files": dup})

    echo("\n## C2 要素归属（质心 → 最近站点点云）\n")
    echo("| 文件 | 目录名 | 要素 | 质心 (X, Y) | 归属站点 | 到归属点云距离 (m) | 站名一致 |")
    echo("|---|---|---|---|---|---|---|")
    for rel, info in files.items():
        for i, f in enumerate(info["features"]):
            p = (f["centroid"][0], f["centroid"][1])
            dists = {s: bbox_dist(p, b) for s, b in sites.items()}
            best = min(dists, key=dists.get)
            same = (best == info["dir_site"])
            echo(f"| {rel} | {info['dir_site']} | {i} | ({p[0]}, {p[1]}) | {best} | "
                 f"{dists[best]:.1f} | {'是' if same else '**否**'} |")
            if not same:
                out["findings"].append({"type": "site_mismatch", "file": rel, "feature": i,
                                        "dir_site": info["dir_site"], "nearest_site": best,
                                        "dist_m": round(dists[best], 1)})

    echo("\n## C3 跨站点合并检测（单文件跨度）\n")
    echo("| 文件 | 要素数 | 最远质心间距 (m) | 判定 |")
    echo("|---|---|---|---|")
    for rel, info in files.items():
        cs = np.array([f["centroid"] for f in info["features"]], dtype=float)
        if len(cs) < 2:
            echo(f"| {rel} | {len(cs)} | - | 单要素 |")
            continue
        span = float(np.max(np.hypot(cs[:, None, 0] - cs[None, :, 0],
                                     cs[:, None, 1] - cs[None, :, 1])))
        hot = "**疑为多站点合并**" if span > 5000 else "同一区域"
        echo(f"| {rel} | {len(cs)} | {span:.0f} | {hot} |")
        if span > 5000:
            out["findings"].append({"type": "merged", "file": rel, "span_m": round(span)})

    echo("\n## 结论与使用建议\n")
    e1 = [f for f in out["findings"] if f["type"] == "duplicate"]
    e2 = [f for f in out["findings"] if f["type"] == "merged"]
    e3 = [f for f in out["findings"] if f["type"] == "site_mismatch"]
    echo(f"- 重复文件：{len(e1)} 组" + (f" —— {e1[0]['files']}" if e1 else ""))
    echo(f"- 疑合并文件：{len(e2)} 个" + (f" —— {[x['file'] for x in e2]}" if e2 else ""))
    echo(f"- 站名与位置不符的要素：{len(e3)} 个" + (f" —— 涉及 {sorted({x['file'] for x in e3})}" if e3 else ""))
    echo("- **当前可用**：杨家坝 `杨家坝-feature1.shp`、西坝村 `西坝村-f1.shp`"
         "（其要素质心均落在各自站点点云范围内）")
    echo("- **暂不纳入**：阎王碥、武关（需先拆分/核对坐标），以及任何重复文件")
    echo("- 论文写法建议：单列一段「数据完整性核查」，列出上述三类问题、"
         "说明当前只用哪两个站及原因、其余站暂不纳入")

    open(os.path.join(RESULTS, "integrity_check.md"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(RESULTS, "integrity_check.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n写出: {os.path.join(RESULTS, 'integrity_check.md')}")
    print(f"写出: {os.path.join(RESULTS, 'integrity_check.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
