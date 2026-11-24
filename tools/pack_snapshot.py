# -*- coding: utf-8 -*-
"""数据保全快照：清单（含校验和）+ 关键产物打包。

不移动、不删除任何原始文件；只新增 backup/ 下的两个产物：
    backup/manifest_{date}.csv    全量文件清单（大小 / 时间 / 校验和）
    backup/shudao_backup_{date}.zip
        代码 tools/*.py、docs、results 的 json/log/csv/md/txt/npz/pt/pth/tif/png、
        figures/*.png —— 即"不可再生或代价高"的产物
        （.las / .npy 体积大，保留在原地并在清单里登记，不做重复拷贝）

校验和：≤64 MB 全量 SHA-256；更大的文件用"首尾各 32 MB + 文件大小"的采样指纹，
并在方法列标注 sampled，避免把 28 GB 全读一遍。

用法：
    python tools/pack_snapshot.py
    python tools/pack_snapshot.py --full-hash      # 所有文件全量 SHA-256（慢）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
REAL = os.path.join(os.environ.get("SHUDAO_DATA_ROOT", os.path.join(SCRIPT_DIR, "..", "data")), "data_real")
DIRS = [("results", os.path.join(ROOT, "results")),
        ("figures", os.path.join(ROOT, "figures")),
        ("tools", os.path.join(ROOT, "tools")),
        ("docs", os.path.join(ROOT, "docs")),
        ("data", os.path.join(ROOT, "data")),
        ("data_real", REAL)]
ZIP_EXT = {".py", ".json", ".log", ".csv", ".md", ".txt", ".npz", ".pt", ".pth",
           ".tif", ".png", ".yml", ".yaml", ".ini", ".toml"}
SMALL = 64 * 1024 * 1024


def digest(path, full=False):
    n = os.path.getsize(path)
    h = hashlib.sha256()
    if not full and n > SMALL:
        with open(path, "rb") as f:
            h.update(f.read(32 * 1024 * 1024))
            f.seek(-32 * 1024 * 1024, os.SEEK_END)
            h.update(f.read(32 * 1024 * 1024))
        h.update(str(n).encode())
        return h.hexdigest(), "sampled"
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest(), "full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-hash", action="store_true")
    ap.add_argument("--date", default=time.strftime("%Y%m%d"))
    a = ap.parse_args()

    out = os.path.join(ROOT, "backup")
    os.makedirs(out, exist_ok=True)
    man = os.path.join(out, f"manifest_{a.date}.csv")

    rows, total = [], 0
    t0 = time.time()
    for tag, d in DIRS:
        if not os.path.isdir(d):
            continue
        for cur, _, files in os.walk(d):
            for fn in files:
                p = os.path.join(cur, fn)
                try:
                    n = os.path.getsize(p)
                except OSError:
                    continue
                rel = os.path.relpath(p, ROOT)
                sha, mode = digest(p, a.full_hash)
                rows.append([tag, rel.replace("\\", "/"), n,
                             time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(os.path.getmtime(p))),
                             sha, mode])
                total += n
    with open(man, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "path", "bytes", "mtime", "sha256", "hash_mode"])
        w.writerows(rows)
    print(f"清单：{len(rows):,} 个文件，{total / 2**30:.1f} GiB，{time.time() - t0:.0f}s")
    print(f"  {os.path.relpath(man, ROOT)}")

    # ---------------- 打包关键产物 ----------------
    zp = os.path.join(out, f"shudao_backup_{a.date}.zip")
    n_zip = 0
    t1 = time.time()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(man, os.path.basename(man))
        for tag, d in DIRS[:3]:                     # results / figures / tools + docs
            if not os.path.isdir(d):
                continue
            for cur, _, files in os.walk(d):
                for fn in files:
                    p = os.path.join(cur, fn)
                    if os.path.splitext(fn)[1].lower() not in ZIP_EXT:
                        continue
                    z.write(p, os.path.relpath(p, ROOT))
                    n_zip += 1
        dz = os.path.join(ROOT, "docs")
        if os.path.isdir(dz):
            for cur, _, files in os.walk(dz):
                for fn in files:
                    p = os.path.join(cur, fn)
                    z.write(p, os.path.relpath(p, ROOT))
                    n_zip += 1
    print(f"打包：{n_zip:,} 个文件，{os.path.getsize(zp) / 2**20:.0f} MB，"
          f"{time.time() - t1:.0f}s")
    print(f"  {os.path.relpath(zp, ROOT)}")
    print("原始 results/figures/data/data_real 均未改动（.las/.npy 原地保留）")


if __name__ == "__main__":
    main()
