# -*- coding: utf-8 -*-
"""防泄漏回归测试（SHP 只评分）。

约定：SHP 派生的 `mask_1m_{site}.npy` / `npz.label` 只能用于评分，
不得作为任何流程环节的输入。本测试锁死两个已修复的默认泄漏路径。

不依赖 pytest：
    python tests/test_no_leak_guard.py
"""
from __future__ import annotations

import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
DATA = os.environ.get("SHUDAO_DATA_ROOT", os.path.join(ROOT, "data"))


def test_gsd_requires_cloud_or_prob():
    """不传 --cloud/--prob 必须报错，禁止回退到 SHP 走廊。"""
    if not os.path.exists(os.path.join(DATA, "meta_杨家坝.json")):
        print("  [SKIP] 未提供 data/meta_杨家坝.json（数据不随仓库分发）")
        return
    from p2_adaptive_gsd import build_map

    try:
        build_map("杨家坝")
    except ValueError as e:
        assert "必须提供 --cloud 或 --prob" in str(e), f"报错信息不符：{e}"
        print("  [OK] build_map 无 cloud/prob 时抛 ValueError")
    else:
        raise AssertionError("build_map 未报错：仍可回退到 SHP 走廊")


def test_extract_path_ignores_gt_roi_by_default():
    """roi_radius>0 但未显式 allow_gt_roi 时，搜索域不得被真值限制。"""
    from path_extraction import extract_path

    sig = inspect.signature(extract_path)
    assert "allow_gt_roi" in sig.parameters, "缺少显式开关 allow_gt_roi"
    assert sig.parameters["allow_gt_roi"].default is False, "默认必须为 False"
    print("  [OK] extract_path.allow_gt_roi 默认 False")


if __name__ == "__main__":
    fails = 0
    for fn in (test_gsd_requires_cloud_or_prob,
               test_extract_path_ignores_gt_roi_by_default):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print(f"  [FAIL] {fn.__name__}: {e}")
    print("全部通过" if fails == 0 else f"{fails} 项失败")
    sys.exit(1 if fails else 0)
