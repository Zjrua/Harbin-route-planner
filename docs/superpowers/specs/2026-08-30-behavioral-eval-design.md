# 三层统计评测体系设计（评测标准 v1.0）

> 日期：2026-08-30 ｜ 状态：设计已经用户批准（三层体系 + 三角互证 + 不引入外部效标）
> 分支：`thesis/markov-transformer-fusion`
> 背景：v5 综合分与就近基线同源（proximity 0.25 + area_density 0.20 ≈ 质量分 45% 偏向就近），权重专家拍定且无敏感性分析，无外部效标——不具裁决资格。本设计以纯统计语言重建评测标准。

## 一、构念声明

整套体系只裁决一个构念：**行为符合度**（behavioral fidelity）——生成路线在统计上与真实游客移动分布的一致性。统计表述：H0 = 生成路线与真实路线来自同一行为分布。"路线质量"（normative）不进入裁决；v5 降级为附录分项描述。

## 二、三层结构

### 层 1｜点层（主裁决，模型无关）
- 逐站 8 选 1：top-1 命中率 + Wilson 95% CI + 配对精确 McNemar
- 补充：top-3 命中率、NDCG@3（同一检验框架）
- 功效：372 点 MDE≈6pt（已粗算）；多重比较用 Holm 校正

### 层 2｜个体层（描述性，模型依赖）
- 路线行为对数似然（nat/转移，按长度标准化）：
  LL = Σₜ [log f(dₜ) + log P₂(regionₜ|两步状态) + log P(typeₜ|typeₜ₋₁)]
- 同指令下方法间配对 Wilcoxon 符号秩检验
- **标注义务**：依赖行为模型设定（拟合份），融合方法共用同一统计源 → 只作逐路线可解释分数与方法内诊断，不进结论句

### 层 3｜群体层（群体裁决，模型无关）
- 生成路线集合 vs 真实测试份集合的经验分布距离：
  - 转移距离分布：KS + Wasserstein（双峰形态是否重现：就近峰位置/占比、跨区重尾）
  - 类型节奏：餐饮间隔、住宿位置、首餐位置分布的 KS
- 路线级 cluster bootstrap（B=1000）给距离度量 95% CI
- 回答"方法是否系统性就近偏差"（rule_near 预期在跨区占比上暴露）

## 三、裁决规则（预注册）
1. 方法优劣主判据 = 层 1 配对 McNemar（Holm 校正）
2. 分布层声明 = 层 3 + bootstrap CI
3. 层 2 只进叙述与诊断，不进结论
4. 三角互证：两层无模型 + 一层模型依赖，裁决不依赖单层

## 四、实现
- `src/behavioral_eval.py`：route_loglik / dist_stats / ks_report（复用 behavior_prior 组件）
- planner 增 `markov` / `hybrid` 选择器（每步 log P2 + log f(d)；hybrid 融合 LLM 首 token 概率，单次前向）
- `scripts/run_eval_generation.py`：60 指令 × 3 seed × 5 选择器（rule_near/rule_score/markov/llm/hybrid），产出层 2+3+附录 v5 分项；层 1 沿用 372 点结果
- Sanity check：真实路线自身在层 2/3 必须最优参照，否则行为模型设定有偏

## 五、学习地图（答辩防守所需文献，按层组织）

见 `docs/literature-review.md` 与本 spec 对应关系：
- 层 1：McNemar 1947；Wilson 1927；Holm 1979；Agresti《Categorical Data Analysis》ch.10；Järvelin & Kekäläinen 2002（NDCG）
- 层 2：Casella & Berger《Statistical Inference》；Dempster/Laird/Rubin 1977（EM）；McLachlan & Peel 2000（混合）；Billingsley 1961（马尔可夫推断）；Wilcoxon 1945；Arlot & Celisse 2010（交叉验证/循环论证）
- 层 3：Kolmogorov 1933/Smirnov 1948；Conover《Practical Nonparametric Statistics》；Efron & Tibshirani 1993（bootstrap）；Field & Welsh 2007（cluster bootstrap）；Hartigan & Hartigan 1985（dip）；Villani 2009（Wasserstein）
- 总纲：Cronbach & Meehl 1955（构念效度）；Theis et al. 2016（生成模型评估的似然 vs 样本分布分野）；Dror et al. 2018（NLP 统计检验实践）；Card et al. 2020（NLP 功效分析）；Blyth 1972（Simpson 悖论）
