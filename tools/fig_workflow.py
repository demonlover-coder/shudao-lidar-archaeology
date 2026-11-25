# -*- coding: utf-8 -*-
"""图 1 工作流（竖版）：输入 → 支路 A → 支路 B → 交付与产品，底部参考与评价。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2E6DB4"
DBLUE = "#1F4E79"
LBLUE = "#BDD7EE"
YELLOW = "#FFC000"
GREEN = "#548235"
RED = "#C00000"
GROUP_FC = "#FDFCE9"
GROUP_EC = "#A6A6A6"

fig, ax = plt.subplots(figsize=(6.3, 9.45), dpi=600)
ax.set_xlim(0, 100)
ax.set_ylim(0, 150)
ax.axis("off")
fig.subplots_adjust(left=0.004, right=0.996, top=0.998, bottom=0.002)


def group(x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.2",
        fc=GROUP_FC, ec=GROUP_EC, lw=1.0, ls=(0, (3, 2)), zorder=1))
    ax.text((x0 + x1) / 2, y1 - 1.4, title, ha="center", va="top",
            fontsize=7.6, fontweight="bold", color="#3A3A3A", zorder=4)


def mod(x, y, w, h, txt, fc=BLUE, tc="white", fs=6.4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
        fc=fc, ec=DBLUE, lw=0.7, zorder=3))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold", zorder=4,
            linespacing=1.25)


def slab(x, y, w, h, txt="", fc=LBLUE, fs=6.0, skew=0.9):
    ax.add_patch(Polygon(
        [(x + skew, y + h), (x + w + skew, y + h), (x + w, y), (x, y)],
        closed=True, fc=fc, ec=DBLUE, lw=0.6, zorder=3))
    if txt:
        ax.text(x + w / 2 + skew / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, color="#1F3B57", zorder=4)


def arr(x0, y0, x1, y1, dashed=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.45",
                                color="#404040", lw=0.9, shrinkA=0, shrinkB=0,
                                ls=(0, (4, 2.5)) if dashed else "-"), zorder=2)


# ================= 输入 =================
group(2, 128, 98, 148, "输入")
slab(5, 133, 14, 7, "点云 (L1)")
slab(23, 133, 14, 7, "dem_min 1 m")
mod(41, 133.5, 11, 6, "CSSC", fs=6.2)
mod(54, 133.5, 11, 6, "open.", fs=6.2)
mod(67, 133.5, 14, 6, "hillshade·法向", fs=5.9)
ax.text(88.5, 136.5, "70 m · 5 m/s\n仿地飞行", ha="center", va="center",
        fontsize=6.0, color="#555555", linespacing=1.5)

arr(50, 128, 50, 124.6)

# ================= 支路 A =================
group(2, 92, 98, 124, "支路 A：宏观寻路")
for i in range(8):
    slab(8 + i * 6.8, 112, 5.2, 4.6, "", fc=LBLUE if i % 2 else "#DEEBF7",
         skew=0.5)
ax.text(66, 114.3, "8 朝向状态图", ha="left", va="center",
        fontsize=6.2, color="#333333")
mod(6, 103.5, 37, 5.6, "代价面  c = open-ness + 0.05", fs=6.4)
mod(45, 103.5, 18, 5.6, "λc·|cos φ|", fc=YELLOW, tc="#3A3A00", fs=6.4)
mod(65, 103.5, 29, 5.6, "最小代价路径", fs=6.4)
mod(6, 95.5, 15, 4.8, "端点（实地）", fc=GREEN, fs=6.0)
slab(24, 95.0, 38, 5.4, "伪标签走廊 + 距离场", fs=6.2)

arr(50, 92, 50, 88.6)
ax.text(50, 90.3, "伪标签 + 距离场（先验→学习）", ha="center", va="center",
        fontsize=6.0, color="#404040", zorder=6,
        bbox=dict(fc="white", ec="none", pad=0.8))

# ================= 支路 B =================
group(2, 58, 98, 88, "支路 B：弱监督分割（≈1.5 k 参数 CNN）")
for i, t in enumerate(["hillshade", "CSSC", "距离场", "openness"]):
    slab(8 + i * 20, 76.5, 15, 5.2, t, fc="#DEEBF7", fs=6.2, skew=0.6)
mod(8, 68.5, 20, 5.6, "Conv3×3 ×2", fs=6.4)
mod(30, 68.5, 12, 5.6, "池化", fs=6.4)
mod(44, 68.5, 15, 5.6, "上采样", fs=6.4)
ax.text(63, 71.3, "96×96 patch\nBCE + 正样本加权 · Adam", ha="left", va="center",
        fontsize=5.9, color="#555555", linespacing=1.5)
mod(8, 60.5, 16, 5.0, "1×1 → P", fs=6.4)
slab(28, 59.9, 20, 5.4, "概率图 P", fc="#9DC3E6", fs=6.2, skew=0.6)
mod(52, 60.5, 18, 5.0, "P ≥ 0.5 → 骨架", fc=GREEN, fs=5.6)

arr(50, 58, 50, 54.6)

# ================= 交付与产品 =================
group(2, 28, 98, 54, "交付与产品")
mod(6, 38, 25, 7, "交付\nd3 3.5 m · AGL 2.0 m", fs=6.2)
mod(35, 38, 26, 7, "保路滤波\n内 0.5 m / 外 0.3 m", fs=6.2)
mod(65, 38, 27, 7, "自适应分辨率 DEM\n0.5 m 子格 + 羽化", fs=6.2)
arr(31, 41.5, 35, 41.5)
arr(61, 41.5, 65, 41.5)
ax.text(50, 32.5, "路面点保留 91–92% · 接缝残差 −9.9% / −6.9%",
        ha="center", va="center", fontsize=6.0, color="#3A6B2E")

# ================= 参考与评价 =================
ax.add_patch(FancyBboxPatch((2, 2), 42, 22,
                            boxstyle="round,pad=0,rounding_size=1.0",
                            fc="white", ec=GROUP_EC, lw=1.0, ls=(0, (3, 2)),
                            zorder=1))
ax.text(23, 22.4, "参考走廊（SHP）", ha="center", va="top",
        fontsize=7.2, fontweight="bold", color="#3A3A3A", zorder=4)
slab(6, 12.5, 13, 5.2, "线状参考", fc="#F2F2F2", fs=5.9)
slab(23, 12.5, 13, 5.2, "评分掩膜", fc="#F2F2F2", fs=5.9)
ax.text(23, 9.2, "历史文献 + 考古专家知识标注", ha="center", va="center",
        fontsize=5.9, color="#333333")
ax.text(23, 5.2, "不进入特征 / 寻路 / 训练 / 交付", ha="center", va="center",
        fontsize=6.0, color=RED, fontweight="bold")

ax.add_patch(FancyBboxPatch((52, 2), 46, 22,
                            boxstyle="round,pad=0,rounding_size=1.0",
                            fc="white", ec=GROUP_EC, lw=1.0, ls=(0, (3, 2)),
                            zorder=1))
ax.text(75, 22.4, "评价（独立口径）", ha="center", va="top",
        fontsize=7.2, fontweight="bold", color="#3A3A3A", zorder=4)
ax.text(75, 17.0, "公平点级 P / R / IoU\n骨架 F1 · Buffer IoU\n线 IoU / COM / COR（r = 2 m）",
        ha="center", va="top", fontsize=6.0, color="#333333", linespacing=1.7)

arr(44, 13, 52, 13, dashed=True)
arr(80, 28, 80, 24, dashed=True)
ax.text(81, 26, "预测轴线", ha="left", va="center", fontsize=5.9, color="#404040")

fig.savefig(r"D:\点云和3D视觉\蜀道问题\DEM自适应\figures\workflow.png",
            dpi=600, facecolor="white")
print("saved")
