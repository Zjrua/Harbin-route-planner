# 文献定位与相关工作综述（前置实验 #4）

> 2026-08-25 整理。目的：开题答辩用——梳理相关方向现有方案的优点与不足，
> 并明确本项目模型在文献坐标系中解决的问题。
> 结论速览见 §五；本项目证据见 `output/eval_fusion.json` 及设计文档 §七。

## 一、Next-POI 推荐：三代方法

### 1.1 一代：马尔可夫/矩阵分解（2010–2015）
代表：[FPMC-LR (IJCAI 2013)](https://www.ijcai.org/Proceedings/13/Papers/384.pdf)——因子化个性化马尔可夫链；
PRME（个性化排名度量嵌入 + 地理影响，常作 SOTA 基线，见[对比论文](https://pdfs.semanticscholar.org/709e/f60629d1bba6660e99d01a705201d28cab7a.pdf)）。

- **优点**：参数少、可解释（转移概率显式）、小数据可估；FPMC 至今仍是[常胜基线](https://www.mdpi.com/2227-739/10/11/1838)。
- **不足**：一阶假设（本项目已实证二阶显著更优，p=0.005）；状态空间大时稀疏；
  个性化依赖 user-id，冷启动即失效；距离/空间结构通常只作后处理特征而非生成式分量。

### 1.2 二代：深度序列模型（2016–2022）
代表：[ARNN (AAAI 2019)](https://ojs.aaai.org/index.php/AAAI/article/view/5337/5193)、
ST-RNN、[长短期偏好学习 (TKDE 2022)](https://www.computer.org/csdl/journal/tk/2022/04/09117156/1kGfwz0QfpS)、
[图增强时空网络 (TKDD)](https://dl.acm.org/doi/fullHtml/10.1145/3513092)。

- **优点**：自动学时空表征，benchmark 命中率高于一代。
- **不足**：黑箱、数据饥渴（需百万级 check-in）；在 LBSN 信号上训练，与
  "真实行程规划"的目标分布有偏；本项目五轮实验的旧 ItineraryTransformer
  过拟合正是这类模型小数据失效的实例。

### 1.3 三代：LLM 方法（2023–2026）
代表：[LLM4POI (arXiv 2404.17591)](https://arxiv.org/abs/2404.17591)（QA 化）、
RALLM-POI（检索增强+地理重排）、Refine-POI（RL 微调）、
[Think2Go](https://arxiv.org/html/2607.28997v1)（生成式+推理）；
综述：[Li et al. 2024](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/li-2024-large-arxiv.pdf)、
[IEEE IS 2025](https://www.computer.org/csdl/magazine/ex/2025/03/11031157/27uvzWrY2ha)、
[TKDE 2025 POI 综述](https://dl.acm.org/doi/10.1109/TKDE.2025.3551292)。

- **优点**：零/少样本泛化、理解自然语言约束、可解释对话。
- **不足**（与本项目证据直接对话）：
  - **幻觉/编造 POI**：本项目 GRPO 实验 16 名仅 9 有效是教科书案例；
  - **空间结构弱**：纯 LLM 在空间就近性上系统性输给规则（本项目：26.6% vs 32.3%）；
  - **评估方法薄弱**：多用 benchmark 刷分，缺少与强规则基线的配对统计检验、
    缺少对采样方差与聚合谬误的控制（本项目辛普森悖论实证）；
  - **概率校准被忽视**：[校准研究](https://arxiv.org/html/2608.07419)表明对齐后 LLM
    系统性过度自信，但 POI 推荐工作几乎不校准概率就使用。

## 二、统计先验 × 神经模型融合

- **Log-linear opinion pool / Product of Experts**：理论上最相关的一支。
  [对数池化理论](https://philarchive.org/archive/DIEPOP)、
  [池化先验的校准定权](https://www.researchgate.net/publication/233785075_Log-Linear_Pool_to_Combine_Prior_Distributions_A_Suggestion_for_a_Calibration-Based_Approach)、
  [Hinton PoE](https://en.wikipedia.org/wiki/Product_of_experts)。
  **优点**：几何加权保持单峰/可归一化，权重有"信任度"语义。
  **不足**：权重几乎总设为全局常数，**状态依赖权重的实证研究缺位**——
  正是本项目 λ(s) 实验填充的空白（结论：状态权重的可学信号弱于预期，本身是有价值的负结果）。
- **LLM 融合方向**：[FuseLLM](https://arxiv.org/html/2401.10491v1)（token 分布层融合多个 LLM）、
  [LLM-informed priors](https://www.emergentmind.com/topics/llm-informed-prior-distributions)、
  [Bayesian 集成推荐](https://arxiv.org/html/2504.10753v1)。
  **共性不足**：都是"神经×神经"或"先验×神经"的通用框架，缺少
  **行为统计先验（可解释转移结构）× LLM** 在真实人类行为数据上的
  严格三分评估。
- **Learning to defer / 路由**：[多专家 L2D (AISTATS 2023)](https://proceedings.mlr.press/v206/verma23a/verma23a.pdf)、
  [路由与级联统一框架](https://arxiv.org/html/2410.10347v1)。硬路由（选一个专家）的
  代价不对称问题已被本项目门控实验量化（近桶 52%×66% 基线正确，错路由代价高）；
  软池化（本项目）是其替代方案。

## 三、人类移动的距离衰减与分布形态

- **重力模型距离函数之争**：[Chen 2015](https://arxiv.org/abs/1503.02915)（幂律派 vs
  熵最大化指数派）；[通勤实证](https://journals.sagepub.com/doi/10.1068/a39369)发现
  **两者都不完备**；[城市内多重重力律 (EPJ Data Science 2023)](https://link.springer.com/article/10.1140/epjds/s13688-023-00438-x)。
  本项目贡献：在"旅游行程内转移"这一粒度上用**候选池离散选择**选型，
  混合形式（0.95·exp + 0.05·幂律）胜出，且否决了工程上常拍的 exp(-d/3)——
  与文献"单一函数形式不完备"结论一致并给出粒度化证据。
- **距离分布形态**：文献多用 log-normal 刻画城市内移动距离
  （[MDPI IJGI](https://www.mdpi.com/2227-9964/14/1/39)）；双峰分布见于
  移民研究（[西非迁移](https://link.springer.com/article/10.1140/epjds/s13688-026-00633-6)）；
  **目的地内（intra-destination）旅游转移距离的正式双峰检验未见报道**
  （[目的地内移动综述](https://www.researchgate.net/publication/325030529_Modeling_intra-destination_travel_behavior_of_tourists_through_spatio-temporal_analysis)、
  [McKercher & Chan 距离衰减](https://www.semanticscholar.org/paper/The-Impact-of-Distance-on-International-Tourist-Movements-Mckercher-Chan/5f52c0b8fcddfe923a53203e39acbf1c69937b49)）。
  本项目 dip test（p<1e-4）+ EM 参数化（lognormal 0.58km ⊕ log-gamma 重尾）是首个该粒度的双峰实证。

## 四、LLM 行程规划系统

代表：[Google 混合行程规划](https://research.google/blog/optimizing-llm-based-trip-planning/)
（LLM 出初稿 + 算法优化约束）、[MIT 2025](https://news.mit.edu/2025/inroads-personalized-ai-trip-planning-0610)、
[RETAIL (2025)](https://arxiv.org/html/2508.15335v1)（LLM + 蚁群优化）。

- **优点**：LLM 负责语义/约束理解，优化器负责可行性——分工工程上有效。
- **不足**：几乎全部以"LLM 为中心、算法为修正器"；**行为先验作为一等公民
  参与逐站概率决策**的架构少见；评估多用端到端满意度，缺少对
  "每一站是否符合真实人类流动统计"的逐点检验（本项目的 372 点逐站评估）。

## 五、本项目模型解决的问题（定位）

| 文献缺口 | 本项目对应设计 | 证据（测试份 372 点 8 选 1） |
|---|---|---|
| LLM 单独做空间决策输给简单规则，"LLM 有无边际价值"悬而未决 | log-linear 池化：行为先验 × LLM 候选概率 | LLM 单独 26.6% < 规则 32.3%，融合 43.3%（p≈0）——**价值反转** |
| 距离衰减函数形式争议（指数 vs 幂律）在旅游行程粒度无裁决 | 候选池条件 logit 选型 | 混合形式胜出（log-loss 1.88 vs 2.08/2.09），否决 exp(-d/3) |
| 目的地内转移距离双峰性无正式检验 | dip test + 异质混合 EM | p<1e-4；lognormal 0.58km ⊕ log-gamma 重尾（w=0.55/0.45） |
| 一阶马尔可夫假设在 POI 推荐中很少被检验 | 二阶 vs 一阶 bootstrap LRT | 二阶胜出（p=0.005），进入最终模型 |
| 状态依赖融合权重的实证缺位 | λ(s) 逻辑回归 stacking + 嵌套 CV | 塌缩至 [0.52,0.57]，不优于固定 λ（p=1.0）——**诚实的负结果** |
| LLM 推荐评估缺配对统计与数据纪律 | 三分 + 全转移点 + 配对 McNemar | 功效粗算裁定 372 点可检 ≥6pt；辛普森悖论与过拟合已实证 |

**一句话定位**：在"LLM × 空间推荐"的热区里，文献普遍回答"LLM 能不能做"，
本项目回答"**统计行为先验与 LLM 应该如何可信地组合**"——用可解释的
分层马尔可夫先验承载空间结构，用池化让 LLM 只贡献它独有的语义互补信号，
并用严格的统计纪律（三分、配对检验、负结果预案）完成评估。
