"""毕设模型架构图（论文级矢量图）.

遵循 scientific-schematics 出版规范：矢量 PDF 优先 + 300dpi PNG 备份、
双栏宽度(183mm)、sans-serif、最小字号 6pt、色盲安全配色、无图内编号。
布局：数据层 → 估计层(拟合份) → 推理层(测试点) → 融合 → 评估。
实线=推理数据流(测试份)，虚线=估计/训练数据流(拟合/λ份)。

用法:
    ./.venv/Scripts/python.exe scripts/draw_architecture.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for f in ("simhei.ttf", "msyh.ttc"):
    p = Path("C:/Windows/Fonts") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 色盲安全学术配色（face, edge）
C_DATA = ("#f1f5f9", "#64748b")     # 数据层: 灰蓝
C_STAT = ("#dbeafe", "#2563eb")     # 统计先验: 蓝
C_LLM = ("#ffedd5", "#ea580c")      # LLM 意图模型: 橙
C_LAM = ("#ede9fe", "#7c3aed")      # λ(s) 学习: 紫
C_FUSE = ("#dcfce7", "#16a34a")     # 融合: 绿
C_EVAL = ("#ffe4e6", "#e11d48")     # 评估: 玫红

FIG_W, FIG_H = 16.5, 11.5  # in，1:1 输出，正文缩放至双栏 183mm


def box(ax, x, y, w, h, title, lines=(), color=C_DATA, fs_t=9, fs_b=7,
        ls="-", lw=1.2, title_dy=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                fc=color[0], ec=color[1], lw=lw, ls=ls,
                                zorder=3))
    tdy = title_dy if title_dy is not None else h - 3.2
    ax.text(x + w / 2, y + tdy, title, ha="center", va="center",
            fontsize=fs_t, fontweight="bold", color=color[1], zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + tdy - 4.0 - i * 3.3, ln, ha="center",
                va="center", fontsize=fs_b, color="#1e293b", zorder=4)


def arrow(ax, x1, y1, x2, y2, label="", dashed=False, color="#475569",
          lw=1.4, lab_off=(0, 1.6), fs=6.5, ha="center", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw,
                                 ls=(0, (5, 3)) if dashed else "-",
                                 connectionstyle=
                                 f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((x1 + x2) / 2 + lab_off[0], (y1 + y2) / 2 + lab_off[1],
                label, ha=ha, va="center", fontsize=fs, color=color,
                zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.6))


def stage_label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=8.5, fontweight="bold", color=color,
            rotation=90, va="bottom", ha="center")


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 165)
    ax.set_ylim(0, 115)
    ax.axis("off")

    # ============ 阶段底色带 ============
    for (x0, x1, y0, y1, fc) in [
            (2, 163, 95, 113, "#f8fafc"),      # 数据层
            (2, 163, 57, 92, "#fafdff"),       # 估计层
            (2, 163, 27, 55, "#fffdf8"),       # 推理层
            (2, 163, 2, 25, "#fafff9")]:       # 融合+评估
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.3",
                                    fc=fc, ec="#cbd5e1", lw=0.8, ls=":",
                                    zorder=0))

    # ============ 数据层（顶） ============
    y = 96
    box(ax, 4, y, 26, 14, "168 条真实游记路线",
        ["routes_xhs_holdout.npy", "List[List[int]]  POI id 序列",
         "共 1660 次相邻转移 (km)"])
    box(ax, 34, y, 24, 14, "POI 库 10K",
        ["poi_metadata.csv", "经纬度 / 6 类活动类型",
         "评分·人气·季节"])
    box(ax, 62, y, 24, 14, "距离矩阵",
        ["distance_matrix.npy", "float32[10K×10K]",
         "Haversine 距离 (km)"])
    box(ax, 90, y, 24, 14, "空间聚类",
        ["clusters.npy / cluster_id.npy",
         "90 簇 → 10K POI 映射",
         "int[10000]"])
    box(ax, 118, y, 24, 14, "三分索引（固定 seed）",
        ["fit 84 / λ 42 / test 42",
         "按路线长度分层随机",
         "KS 检验三份同分布"])

    box(ax, 146, y, 16, 14, "数据纪律",
        ["拟合份→估计", "λ份→学权重", "测试份→仅评一次"], fs_t=8, fs_b=6.3)

    # ============ 估计层（拟合份 84 条） ============
    # 统计先验
    box(ax, 4, 74, 30, 16, "(1) 区域划分 RegionMap",
        ["90 簇质心 → Ward 层次合并", "→ K = 8 区域",
         "输出 region_of_poi: int[10000]",
         "(K 由计数≥3 覆盖曲线选定)"], color=C_STAT)
    box(ax, 4, 58, 30, 14, "(2) 二阶区域转移矩阵",
        ["836 次转移计数 8×8×8",
         "+ Dirichlet 平滑 α=0.1 (敏感性扫描)",
         "P2(下一区 | 前两区): float[8,8,8]"], color=C_STAT)
    box(ax, 38, 66, 28, 24, "(3) 距离衰减 f(d)",
        ["候选池离散选择 (条件 logit)",
         "拟合份 752 池 MLE 选型：",
         "  w·exp(−d/ρ) + (1−w)·d^(−β)",
         "  w=0.95, ρ=0.3, β=0.05",
         "λ份确认：mix 1.88 < 幂律 2.08",
         "  < 指数 2.09 (log-loss)",
         "→ 否决旧规则 exp(−d/3)"], color=C_STAT)
    box(ax, 70, 66, 26, 24, "(4) 异质距离混合 EM",
        ["转移距离 836 个:",
         "w·LogNormal(中位0.58km)",
         "+ (1−w)·LogGamma(重尾≈25km)",
         "w=0.45（就近/跨区双峰）",
         "→ posterior_near(d)",
         "供 λ(s) 作模式后验特征"], color=C_STAT)

    # LLM
    box(ax, 100, 66, 28, 24, "(5) 意图模型（冻结）",
        ["Qwen3.5-4B 基座",
         "4-bit NF4 量化 (RTX 4090)",
         "+ SFT LoRA adapter (<24MB)",
         "路线生成指令数据训练",
         "本管线不更新参数",
         "（微调=后续敏感性实验）"], color=C_LLM)

    # λ(s) 学习
    box(ax, 131, 58, 31, 32, "(6) λ(s) 状态加权学习",
        ["特征 x(s)∈R⁸（推理时可得）:",
         "  posterior_near(min/med d)",
         "  转移熵 H(P2(state))",
         "  候选区分布 max/std",
         "  log 前缀长度 / log d",
         "λ份 368 点 · 逻辑回归 σ(w·x+b)",
         "L2=0.1 · 嵌套 CV 5 折",
         "结果: 分布塌缩 [0.52,0.57]",
         "≈ 固定 λ=0.55"], color=C_LAM)

    # ============ 推理层（测试点 372） ============
    box(ax, 4, 30, 30, 22, "(7) 预测点构建 s",
        ["对每条测试路线 t=2..L−1 全取",
         "RAG 检索(末站中心25) + 干扰采样",
         "≤4 同类型 + 异类型 → 8 选 1 池",
         "s = (prefix: int[],",
         "     cands: int[8], true: int)",
         "测试份 372 点（功效粗算裁定）"])
    box(ax, 38, 30, 28, 22, "(8) 统计先验得分",
        ["log P_prior(c) =",
         "  log P2(reg_c | reg_prev, reg_cur)",
         "  + log f(d(末站, c))",
         "逐候选 → float[8]",
         "(二阶状态 × 混合衰减)"], color=C_STAT)
    box(ax, 70, 30, 26, 22, "(9) LLM 候选概率",
        ["prompt: 系统+已游路线+",
         "  编号候选列表",
         "1 次前向 (确定性) →",
         "首 token「1」..「8」 logits",
         "softmax 重归一 → P_llm: float[8]"], color=C_LLM)
    box(ax, 100, 30, 28, 22, "(10) 状态权重 λ(s)",
        ["x(s) 与 (6) 同源",
         "λ(s) = σ(w·x(s)+b) ∈ (0,1)",
         "对照: 固定 λ=0.55",
         "（消融链的一环）"], color=C_LAM)

    # ============ 融合层 ============
    box(ax, 38, 4, 58, 20, "(11) log-linear 池化融合",
        ["log P_final(c) = λ(s)·log P_prior(c) + (1−λ(s))·log P_llm(c)",
         "softmax → P_final: float[8] → argmax 出下一站",
         "误差互补: LLM 单独 26.6% < 规则, 融合后反超全部基线"], color=C_FUSE,
        fs_t=10)

    # ============ 评估层 ============
    box(ax, 100, 4, 62, 20, "(12) 一次性评估（测试份 372 点 8 选 1）",
        ["配对精确 McNemar + Wilson 95% CI",
         "random 13.2 | rule_markov 30.4 | rule_near 32.3",
         "llm 26.6 | 先验单独 37.1 (p=1e-4)",
         "固定λ融合 43.3 (p≈0) | 状态λ 43.0",
         "状态λ vs 固定λ: p=1.0（负结果，如实报告）"], color=C_EVAL)

    # ============ 箭头（数据流） ============
    # 数据层 → 估计层（虚线 = 估计流）
    arrow(ax, 17, 96, 17, 90, "84 条拟合路线", dashed=True, color="#2563eb")
    arrow(ax, 46, 96, 46, 90, "经纬度/类型", dashed=True, color="#2563eb")
    arrow(ax, 74, 96, 52, 90, "距离", dashed=True, color="#2563eb")
    arrow(ax, 102, 96, 102, 66, "簇质心", dashed=True, color="#2563eb")
    arrow(ax, 102, 96, 17, 90, "", dashed=True, color="#94a3b8", rad=0.15)
    arrow(ax, 130, 96, 130, 90, "聚类→POI", dashed=True, color="#2563eb")
    arrow(ax, 143, 96, 143, 90, "λ份 42 条", dashed=True, color="#7c3aed")
    # 估计层内部
    arrow(ax, 19, 74, 19, 72, "区域序列", dashed=True, color="#2563eb")
    arrow(ax, 34, 81, 38, 81, "", dashed=True, color="#2563eb")
    arrow(ax, 34, 63, 70, 63, "距离衰减参与先验", dashed=True,
          color="#2563eb", lab_off=(0, -2.2))
    arrow(ax, 70, 74, 131, 74, "posterior_near 特征", dashed=True,
          color="#7c3aed", lab_off=(0, 1.8))
    arrow(ax, 34, 62, 131, 62, "P2 转移熵特征", dashed=True,
          color="#7c3aed", lab_off=(0, -2.4), rad=-0.12)

    # 估计层 → 推理层（实线 = 推理流，参数移交）
    arrow(ax, 19, 58, 19, 52, "P2 (α=0.1)", color="#2563eb")
    arrow(ax, 52, 66, 48, 52, "f(d) 参数", color="#2563eb")
    arrow(ax, 114, 66, 83, 52, "冻结权重", color="#ea580c", lab_off=(0, 1.8))
    arrow(ax, 146, 58, 114, 52, "w, b", color="#7c3aed", lab_off=(0, 1.8))

    # 推理层内部 → 融合
    arrow(ax, 68, 41, 38, 18, "log P_prior: float[8]", color="#2563eb",
          lab_off=(2, 6), rad=0.1)
    arrow(ax, 96, 41, 38, 14, "P_llm: float[8]", color="#ea580c",
          lab_off=(-4, 2), rad=0.08)
    arrow(ax, 128, 41, 96, 22, "λ(s)∈(0,1)", color="#7c3aed",
          lab_off=(-2, 2.5), rad=0.05)
    # 数据 → 推理
    arrow(ax, 74, 96, 74, 52, "test 42 条", color="#475569",
          lab_off=(5.5, 0))
    arrow(ax, 46, 96, 30, 52, "检索池", color="#475569", lab_off=(-6, 10),
          rad=0.12)
    # 融合 → 评估
    arrow(ax, 96, 14, 100, 14, "P_final", color="#16a34a")

    # ============ 图例 ============
    lx, ly = 5.5, 111
    arrow(ax, lx, ly, lx + 7, ly, "", color="#475569")
    ax.text(lx + 8.5, ly, "推理数据流（测试份）", fontsize=7, va="center")
    arrow(ax, lx + 34, ly, lx + 41, ly, "", dashed=True, color="#2563eb")
    ax.text(lx + 42.5, ly, "估计/训练流（拟合份/λ份）", fontsize=7,
            va="center")
    for (fc, ec, lab, dx) in [(C_STAT[0], C_STAT[1], "统计先验", 78),
                              (C_LLM[0], C_LLM[1], "LLM 意图模型", 98),
                              (C_LAM[0], C_LAM[1], "λ(s) 学习", 120),
                              (C_FUSE[0], C_FUSE[1], "融合", 138),
                              (C_EVAL[0], C_EVAL[1], "评估", 148)]:
        ax.add_patch(FancyBboxPatch((lx + dx, ly - 1.2), 3, 2.4,
                                    boxstyle="round,pad=0.15", fc=fc,
                                    ec=ec, lw=1))
        ax.text(lx + dx + 4.5, ly, lab, fontsize=7, va="center")

    out_png = "output/figures/model_architecture.png"
    out_pdf = "output/figures/model_architecture.pdf"
    Path("output/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"保存: {out_pdf} / {out_png}")


if __name__ == "__main__":
    main()
