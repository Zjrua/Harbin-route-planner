# v5 打分设计：质量 + 需求匹配（含季节）

> 日期：2026-08-02
> 状态：设计评审通过，待实现

## 背景与动机

当前 v4 打分（`src/scoring.py`）只衡量路线**内部质量**（就近性/区域密度/节奏/满意度/多样性 + 硬约束），与用户指令完全脱钩。最突出的失败模式：**"带父母去哈尔滨玩三天，节奏要慢"** 生成 5 站路线，被 v4 推断为"1 日游"反而高分——打分器奖励了一条**不符合用户需求**的路线。这个缺陷会通过 DPO 偏好 / GRPO 奖励被放大固化，必须修复。

## 决策汇总（已确认）

| 项 | 决策 |
|---|---|
| 打分职责 | 质量 + 需求匹配 |
| 约束抽取 | 规则为主 + LLM 兜底（训练固定规则，评估/Demo 可开 LLM） |
| 惩罚结构 | 硬判负（天数严重不符/核心景点缺失/出发地不符）+ 软扣分（预算/偏好/节奏/季节） |
| 实现方案 | 分层架构（A）：新增 `constraint_parser.py` + `composite_score_v5`，不传指令时退回 v4 |

## 一、约束抽取器（src/constraint_parser.py，新增）

```python
@dataclass
class Constraints:
    days: int | None              # 指令天数（"3日游"→3）
    budget_min: float | None      # 预算下界（"预算约1500元"→1500）
    budget_max: float | None      # 预算上界（"预算充足"→None 无上限）
    start: str | None             # 出发地
    core_pois: list[str]          # 核心景点
    preferences: list[str]        # 偏好（美食/购物/景点/文化/自然/亲子…）
    pace: str | None              # slow / normal / fast（"慢节奏/走不快"→slow）
    season: str | None            # winter / summer / 未提
    confidence: str               # high（规则完整）/ low（需 LLM 兜底）
```

### 规则主路径（确定性、训练奖励用）
- 天数：`(\d+)\s*日` / `(\d+)\s*天` / 汉字天数（一/两/三…天）
- 预算：`预算约?(\d+)元`；`预算充足|无上限` → `budget_max=None`
- 出发地：`从(.+?)出发` / `以(.+?)为起点` / `第一站想去(.+?)`
- 核心景点：`以(.+?)为核心|重点(.+?)|必须去(.+?)`
- 偏好：`喜欢(.+?)` 关键词映射（美食/购物/景点/文化/自然/亲子）
- 节奏：`慢节奏|不要太赶|走不快|适合老人|节奏要慢` → slow；`节奏适中` → normal
- 季节：`冰雪季|冬季|12月|1月|2月` → winter；`夏季|暑假|7月|8月` → summer

### LLM 兜底（评估/Demo 用）
- `confidence=high` 直接用规则结果；否则调用 Qwen 转结构化 JSON
- **训练路径固定 use_llm=False**：GRPO/DPO 需确定性奖励，LLM 随机性污染梯度；解析失败的样本降级为"无约束"或跳过

### POI 名归一化
- 指令约束里的 POI 名与路线 POI 匹配统一走 `contains` 匹配（复用现有逻辑），如"中央大街"→"哈尔滨中央大街"

## 二、v5 分数合成（src/scoring.py 扩展）

### 硬约束新增（判负，score=0，优先级在时间/重复/过短之后）

| 硬约束 | 判定规则 | 判负条件 |
|---|---|---|
| days_mismatch | 指令 days=D，路线推断 n_days | `n_days < D`（如 3 日游出 5 站→1 日） |
| missing_core_poi | 路线 POI 无法 contains 匹配任何核心景点 | 核心景点在 POI 库也匹配不到 → 降级为软（不误杀） |
| start_mismatch | 路线前两站都不含出发地 POI | 前两站容错 |

### 软扣分项（进需求匹配度，不进硬判负）

| 项 | 规则 |
|---|---|
| 预算 | 指令 `预算约B`：路线估算花费（POI avg_cost 求和）超 B ≤20% 扣 0.1，≤50% 扣 0.3，>50% 扣 0.6；"充足/未提"不扣 |
| 偏好 | `喜欢美食`：餐饮类占比 ≥0.25 满分，每低 0.1 扣 0.15；`购物` 同理；多偏好取最差 |
| 节奏 | `慢节奏`：每站数 = 去重站数/n_days，>7 站/天扣 0.3，>9 扣 0.6；`快节奏` 相反；未提不扣 |
| 季节 | 见下 |
| 天数超出 | 推断 n_days > 指令 days：超出 1/2 天扣 0.15/0.3 |

### 季节软扣分

数据事实：`season_winter`/`season_summer` 均 0~1 连续分，均值 0.8 分布偏窄；**587 个冬季特色 POI**（winter−summer>0.3，如亚布力滑雪场/中国雪谷），**0 个夏季特色 POI**（数据缺失）。

| 指令季节 | 规则 | 扣分 |
|---|---|---|
| winter | 路线含 ≥1 个冬季特色 POI（`season_winter−season_summer>0.3`） | 满足不扣；不满足扣 0.2 |
| summer | 路线冬季特色 POI 占比 >50% | 扣 0.2（弱规则，因缺夏季专属数据） |
| 未提/四季 | — | 0 |

### 分数合成

```
score = 0.70 × 质量五维加权（proximity 0.25 / area_density 0.20 / rhythm 0.20 / satisfaction 0.20 / diversity 0.15 归一化）
      + 0.30 × requirement_match

requirement_match = max(0, 1 − (预算扣分 + 偏好扣分 + 节奏扣分 + 季节扣分 + 天数超出扣分))
```

- 需求硬判负 → `feasible=False, reason=days_mismatch/missing_core_poi/start_mismatch`
- **向后兼容**：不传 instruction/constraints → 完全退回 v4 行为；`composite_score_v4` 别名保留
- 返回结构新增：`requirement_match`、各软扣分明细（便于 UI/评估展示）

## 三、调用点升级（v4 → v5）

| 调用点 | 改动 |
|---|---|
| `src/constraint_parser.py`（新） | 规则抽取 + LLM 兜底 |
| `src/scoring.py` | 新增 `composite_score_v5`；v4 别名保留 |
| `eval_dpo_vs_sft.py` | 传 instruction，展示 v5 分 + 需求匹配明细 |
| `prepare_dpo_dataset.py` | 偏好对打分用 v5（训练体现需求匹配偏好） |
| `train_grpo.py` | reward func 用 v5 |
| `serve_qwen_demo.py` | 打分卡片加"需求匹配"行 + 硬判负原因 |

## 四、验证策略

1. **回归**：5 条测试指令不传 instruction，v5 纯质量分应与 v4 一致
2. **新增**：5 条带 instruction 打分，"带父母三天"应因 `days_mismatch` 判负、纯三日游高分
3. **季节**：冬季指令给含滑雪场的路线加分，夏季指令给纯滑雪场路线扣分
4. **训练冒烟**：GRPO 小规模（20 step）确认 v5 奖励能上升且不崩

## 风险

- LLM 兜底在训练路径禁用，评估才开 → 避免奖励噪声
- 需求匹配 0.30 权重可能让"质量极好但需求微偏"的路线被低估——软扣分上限已控制（单维度 ≤0.6），风险可控
- 季节列缺夏季特色数据 → 夏季只做弱扣分，未来有数据再加强
