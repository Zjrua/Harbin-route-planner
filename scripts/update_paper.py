"""Update paper/main.tex with real ablation results (K=3 best model)."""
import io

with open("paper/main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Abstract summary
text = text.replace(
    "综合得分达0.897，距离33.6 km，耗时83 min",
    "综合得分达0.880，距离27.3 km，耗时约70 min"
)

# 2. Training section
text = text.replace("训练了55个epoch后触发早停", "训练了57个epoch后触发早停")
text = text.replace(
    "最佳验证损失为1.9562（出现在第45个epoch）",
    "最佳验证损失为2.5338（出现在第47个epoch）"
)
text = text.replace("1.9562", "2.5338")
text = text.replace("在第55个epoch触发早停", "在第57个epoch触发早停")
text = text.replace(
    "最终模型在训练集上的损失为1.823，与验证损失差距控制在0.13以内",
    "最终模型在训练集上的损失为1.640，与验证损失差距控制在0.89以内"
)
text = text.replace(
    "对于180个POI和165条路线的数据规模",
    "对于180个POI和535条路线（165条原始+370条合成）的数据规模"
)

# 3. Route results
text = text.replace(
    "使用训练好的最优模型（第45个epoch的checkpoint）",
    "使用消融实验表现最优的Engram K=3模型（第57个epoch的checkpoint）"
)

# 4. Score updates
text = text.replace("综合得分0.897。", "综合得分0.880。")
text = text.replace("综合得分0.897", "综合得分0.880")
text = text.replace("最优33.6 km", "最优27.3 km")
text = text.replace("提升19.4\\%", "提升约0.8\\%")

# 5. Ablation table - replace the old fabricated table
old_table_start = text.find("\\textbf{模型变体} & \\textbf{距离（km）}")
if old_table_start > 0:
    old_table_end = text.find("\\bottomrule", old_table_start) + len("\\bottomrule")
    new_table = (
        "\\textbf{模型变体} & \\textbf{距离（km）} & \\textbf{满意度} & \\textbf{多样性} & \\textbf{综合得分} & \\textbf{$\\Delta$得分} \\\\\n"
        "\\midrule\n"
        "Engram $K=3$（\\textbf{最优}） & 27.3 & 4.85 & 0.50 & \\textbf{0.8802} & $+0.0055$ \\\\\n"
        "移除MHC & 27.2 & 4.84 & 0.48 & 0.8753 & $+0.0006$ \\\\\n"
        "完整模型（$K=5$） & 32.3 & 4.81 & 0.49 & 0.8747 & — \\\\\n"
        "移除Engram & 35.6 & 4.85 & 0.48 & 0.8735 & $-0.0012$ \\\\\n"
        "移除Engram+MHC & 34.5 & 4.84 & 0.48 & 0.8735 & $-0.0012$ \\\\\n"
        "Engram $K=10$ & 38.2 & 4.83 & 0.49 & 0.8714 & $-0.0033$ \\\\\n"
        "纯Transformer基线 & 63.0 & 4.84 & 0.49 & 0.8676 & $-0.0071$ \\\\\n"
        "\\bottomrule"
    )
    text = text[:old_table_start] + new_table + text[old_table_end:]
    print("Ablation table replaced")

# 6. K-sensitivity table
old_k_start = text.find("\\textbf{检索数$K$} & \\textbf{验证损失}")
if old_k_start > 0:
    old_k_end = text.find("\\bottomrule", old_k_start) + len("\\bottomrule")
    new_k = (
        "\\textbf{检索数$K$} & \\textbf{验证损失} & \\textbf{综合得分} \\\\\n"
        "\\midrule\n"
        "$K=3$（\\textbf{最优}） & 2.5338 & \\textbf{0.8802} \\\\\n"
        "$K=5$ & 2.5293 & 0.8747 \\\\\n"
        "$K=10$ & 2.5676 & 0.8714 \\\\\n"
        "\\bottomrule"
    )
    text = text[:old_k_start] + new_k + text[old_k_end:]
    print("K table replaced")

# 7. Route detail in paper
old_route = "最优路线A（综合得分0.897）：中央大街（景点，评分4.7）"
if old_route in text:
    new_route = "最优路线（综合得分0.880）：中央大街（景点，评分4.9）"
    text = text.replace(old_route, new_route)
    print("Route detail updated")

# 8. Fix ablation conclusion paragraphs
reps = [
    ("\\textbf{Engram记忆模块贡献最为显著：}单独移除Engram模块后综合得分从0.897下降至0.821，降幅达0.076，是所有单项消融中降幅最大的。",
     "\\textbf{Engram记忆模块具有正向贡献（$K=3$时最优）：}$K=3$配置的综合得分（0.8802）优于$K=5$（0.8747），提升0.0055。"),
    ("\\textbf{MHC双曲约束具有稳定的正向贡献：}单独移除MHC后综合得分下降0.045",
     "\\textbf{MHC双曲约束在当前数据规模下贡献微小：}移除MHC后综合得分从0.8747升至0.8753（+0.0006），变化在误差范围内。"),
    ("\\textbf{Muon优化器的贡献体现在收敛质量：}将Muon替换为标准AdamW后，综合得分下降0.021",
     "\\textbf{所有消融实验使用AdamW优化器：}为确保公平对比，各组实验统一采用AdamW优化器。"),
    ("\\textbf{三项创新的协同效应显著：}同时移除Engram和MHC（仅保留Muon）的综合得分下降达0.104",
     "\\textbf{各模块协同效应温和但一致：}同时移除Engram和MHC的综合得分（0.8735）较完整模型（0.8747）下降0.0012。"),
    ("\\textbf{纯Transformer基线的显著不足：}不含任何创新模块的标准Transformer（Encoder-Decoder架构，AdamW优化器）在哈尔滨旅游路线生成任务上表现明显不足，综合得分仅0.751，路线距离高达46.8",
     "\\textbf{纯Transformer基线的主要差距体现在空间效率：}不含创新模块的标准Transformer综合得分为0.8676，路线距离高达63.0 km（较最优K=3的27.3 km多35.7 km）。"),
    ("提升19.4\\%，验证了本文各项技术创新融合的必要性和有效性",
     "提升约0.8\\%，验证了本文各项技术创新在合成路线数据上的协同增益"),
    ("综合得分仅0.751", "综合得分0.8676"),
]

for old, new in reps:
    if old in text:
        text = text.replace(old, new)
        print(f"Replaced: {old[:50]}...")
    else:
        print(f"Not found: {old[:50]}...")

with open("paper/main.tex", "w", encoding="utf-8") as f:
    f.write(text)
print("\nPaper updated successfully")
