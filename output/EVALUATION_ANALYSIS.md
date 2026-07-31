# 评估实验综合分析（方向 B + C）

> 生成时间：2026-07-26
> 数据来源：output/real_data_evaluation.json + output/baseline_comparison.json
> 核心结论：**Transformer 在真实数据上远超启发式，但 composite 指标存在缺陷**

---

## 实验 1：方向 B — 真实数据 next-POI 预测（最有价值）

**协议**：168 条真实 XHS 路线（从未参与训练的 holdout），给定前 k 个 POI 预测第 k+1 个。

| 方法 | top-1 | top-5 |
|---|---|---|
| most_popular（评分最高） | 0.00% | 0.00% |
| nearest_neighbor（距离最近） | 1.93% | 8.67% |
| **Transformer** | **47.77%** | **78.73%** |

**结论**：Transformer 比最近邻高 **25 倍**。模型确实学到了真实游客的行为模式，泛化到了从未见过的真实路线上。**这彻底打破了"数据循环论证"的质疑。**

---

## 实验 2：方向 C — 路线生成质量对比（揭示指标缺陷）

**协议**：5 个高评分景点起点，生成 10 站路线，用 composite_score 评估。

| 方法 | avg_dist | diversity | composite |
|---|---|---|---|
| random | 214 km | 0.579 | 0.8509 |
| NN / 2-opt / OR-Tools | **1.28 km** | 0.480 | **0.8685** |
| Transformer | 164 km | **0.587** | 0.8606 |

**表面结论**：启发式 composite 略高于 Transformer（+0.8%）。

### 但实际路线揭示了 composite 指标的严重缺陷

**NN/OR-Tools 生成的路线**（起点：阿城清真寺）：
```
景点 → 景点 → 住宿 → 住宿 → 住宿 → 住宿 → 住宿 → 餐饮 → 购物 → 餐饮
```
**连续 5 个住宿！** 在现实中完全不合理——没有游客会一天住 5 家酒店。NN 只看距离，把起点附近的所有酒店挨个串起来了。距离极短（1.5km）但路线**完全不可用**。

**Transformer 生成的路线**：
```
景点 → 餐饮 → 景点 → 餐饮 → 景点 → 餐饮 → 景点 → 餐饮 → 景点 → 住宿
```
**完美的"游-吃"节奏交替**，最后以住宿收尾。距离长（530km，跨区域）但**完全符合旅游逻辑**，与真实 XHS 路线结构一致。

---

## 关键发现：composite_score 指标的三个缺陷

NN/OR-Tools 在 composite 下"赢"Transformer，恰恰证明指标本身有问题：

### 缺陷 1：distance 权重过高（0.30），奖励退化解
composite 的 distance 分量鼓励"距离短"，但"短距离"在这个任务里等价于"在同一区域反复横跳"。NN 找到起点附近的酒店群挨个串起，得到 1.5km 的极短路线，但这不是旅游。

### 缺陷 2：缺乏"活动节奏"约束
composite 没有惩罚"连续同类活动"（如连续 5 个住宿）。真实旅游路线有明确的节奏（景点→餐饮→景点→住宿），但 composite 对此无感知。

### 缺陷 3：distance 归一化方式掩盖问题
distance 用 `max_dist * route_len` 归一化，让 1.5km 和 164km 在 [0,1] 上的差异被压缩，而 diversity/satisfaction 的差异（更小）反而占比更大。

---

## 两个实验的统一解读

| 维度 | 启发式（NN/OR-Tools） | Transformer | 真相 |
|---|---|---|---|
| next-POI 预测（真实数据） | 1.93% | **47.77%** | Transformer 学到了真实模式 |
| composite（生成质量） | **0.8685** | 0.8606 | 指标缺陷导致启发式"虚高" |
| 实际路线合理性 | 连续5住宿（荒谬） | 游吃交替（合理） | Transformer 真正可用 |
| 纯距离优化 | 1.28km（最优） | 164km | 启发式的唯一真实优势 |

**统一结论**：
1. **Transformer 的核心价值不在距离优化**（OR-Tools 在纯距离上确实更强）
2. **Transformer 的价值在于学到真实旅游行为模式**（活动节奏、POI 选择偏好）
3. **composite_score 需要重新设计**：加入活动节奏约束、降低 distance 权重、或引入"路线真实性"指标

---

## 论文写作建议

基于以上结果，方向 B + C 在论文中的定位：

1. **方向 B 作为核心证据**：证明模型在真实数据上的泛化能力（top-1=47.77%）
2. **方向 C 作为补充讨论**：诚实报告 composite 上 Transformer 略低，但通过路线样例揭示指标缺陷
3. **新增"评估指标讨论"**：指出 composite 的局限性，提出未来改进方向
4. **避免过度宣称**：不声称 Transformer 在所有指标上领先，而是聚焦"行为模式学习"这一真实优势

---

## 数据来源
- `output/real_data_evaluation.json` — 方向 B 完整结果
- `output/baseline_comparison.json` — 方向 C 完整结果
- `paper/baseline_comparison.png` — 指标对比图
- 路线样例可通过 `scripts/run_baselines.py` 复现
