# Harbin Route Planner

基于 **Qwen3.5-4B 微调**的旅游路线生成系统：输入自然语言旅行需求（天数、季节、预算、人群），输出**逐日 POI 路线** + v5 质量评分 + 交互地图。

> 以哈尔滨为案例城市，覆盖 48,961 个真实 POI（含语义标注）。

## 它解决什么问题

大模型直接生成旅游路线有两大通病，本项目通过**混合架构**解决：

1. **编造 POI 名** — 4B 模型记不住 1 万个真实地名，长序列生成时即兴发挥（"南岗区天主教堂"等假名）。
   ✅ **RAG 候选检索 + 候选编号输出**：模型只从真实候选中选编号，编造名从源头消失。
2. **多日游一次性输出崩溃** — "三天半、第二天下午 5 点走"这类需求，单次生成长路线必然退化。
   ✅ **逐日拆分**：按天数（含小数/半日）拆成单日任务，每天 5-6 站，模型稳定输出。

## 功能特性

- 🗺️ **自然语言 → 路线**：解析天数（含 `3.5 天`）、季节、预算、出发地、核心景点、人群（老人/亲子）
- 📅 **逐日规划**：全天/半天混合（"第 2 天下午 5 点走" → 半天模板），跨日不重复 POI
- 🎯 **候选编号生成**：模型从检索出的 8-10 个真实候选里选编号，杜绝编造名
- 🌿 **人类旅游节奏**：住宿在每天末尾、餐饮穿插景点间、半天不安排住宿（规则层）
- 🧭 **RAG 语义过滤**：48,961 个 POI 全量语义标注，过滤健身房/酒吧/银行等非旅游设施；带父母 → 只检索适合老人的
- 📊 **v5 硬约束打分**：质量五维（0.70）+ 需求匹配（0.30），天数不符/核心景点缺失/出发地不符直接判负
- 🖥️ **Web Demo**：SFT / GRPO / DPO 模型一键切换，逐日行程卡片 + folium 地图

## 架构

```
指令 ──→ constraint_parser（约束抽取）
       │  天数(3.5) / 季节 / 预算 / 出发地 / 核心景点 / 人群
       ▼
  逐日拆分 split_days（含半日）  检索 retrieval（语义过滤 + 类型配额 + 就近/季节/偏好加权）
       │                          │
       ▼                          ▼
   Qwen3.5-4B 候选编号输出 ──→ 编号→真实名 ──→ 节奏整理 organize_day_rhythm
                                                       │
                                                       ▼
                                            v5 逐日打分（composite_score_v5）
```

## 快速开始

### 环境

```bash
# Python 3.10-3.12，建议 uv
# Windows：自动探测（NVIDIA → CUDA 版 torch；Intel Arc/无独显 → CPU 版，内置 XPU）
./scripts/install_env.ps1
# macOS (Apple Silicon)：PyPI 官方 torch 自带 MPS 后端
./scripts/install_env.sh
# 或手动指定：
uv sync                        # Mac / Windows CPU+XPU
uv sync --index pytorch-cu128  # Windows NVIDIA CUDA
```

运行时代码自动选设备（`src/device.py`）：`CUDA → MPS → XPU → CPU`。

### 数据准备

```bash
# 1. 语义标注全量 POI（48,961 个，输出 merged_pois_labeled.csv）
uv run python scripts/label_poi_semantics.py

# 2. 生成 Qwen3.5 候选编号 SFT 数据（3,000 条逐日样本）
uv run python scripts/prepare_qwen35_dataset.py --max-samples 3000
```

### 运行 Web Demo

```bash
uv run python scripts/serve_qwen_demo.py --port 8898
# 打开 http://localhost:8898，默认 Qwen3.5-SFT 模型
```

> 模型权重未随仓库分发（见 `.gitignore`），需先完成训练生成 `output/qwen35_route_lora/`。

### 训练

```bash
# SFT（候选编号格式，Qwen3.5-4B + QLoRA）
uv run python scripts/finetune_qwen35.py --epochs 3

# GRPO 在线强化（在 SFT 之上，可选）
uv run python scripts/train_grpo_qwen35.py --max-steps 150

# DPO 偏好对齐（Qwen3 版本）
uv run python scripts/train_dpo.py --epochs 2
```

## 训练与评估结果

### 模型对比（v5 打分，带父母三天逐日规划）

| 模型 | 结果 |
|---|---|
| **Qwen3.5-SFT**（默认） | ✅ 单日 v5 0.88-0.95，18 站全真实 POI，需求匹配 1.0 |
| Qwen3.5-GRPO | 与 SFT 相当（候选编号任务简单，SFT 已收敛） |
| Qwen3-GRPO | v5 0.65，被 Qwen3.5-SFT 单轮超越 |
| Qwen3-DPO | v5 0.34，弱 |

> **架构演进教训**：4B 模型做 1 万级 POI 规划，正确解是**混合系统**（检索 + 规则 + 轻模型排序）。直接生成或大规模强化学习都触天花板，SFT 已足够。

### 数据规模

| 数据 | 规模 |
|---|---|
| 原始 POI（`merged_pois.csv`） | 48,961 个 |
| 语义标注（`merged_pois_labeled.csv`） | 48,961 个，排除 1,447 个非旅游 POI |
| 处理后 POI（`data/processed/`，10K 上限因矩阵内存） | 10,000 个 |
| 训练路线（`data/processed/routes.npy`） | 5,168 条 |
| Qwen3.5 SFT 数据（候选编号逐日） | 3,000 条 |

## 打分技术细节（v5）

路线质量由 `src/scoring.py` 的 `composite_score_v5` 评估，**质量 + 需求匹配双层结构**：

```
score = 0.70 × 质量五维（加权） + 0.30 × 需求匹配度
```

### 质量五维（路线内部属性）

| 维度 | 权重 | 计算 |
|---|---|---|
| **proximity** 就近性 | 0.25 | 步长中位数 `hop_p50`：≤1km 取 0.9；1-8km 线性衰减；>8km 快速衰减（下限 0.2） |
| **area_density** 区域密度 | 0.20 | `hop_p90 / hop_p50` 比值（跳距离散度），≤15 满分，越大越低 |
| **rhythm** 节奏 | 0.20 | `1 − 连续同类POI比例`（景点→餐饮→景点 交替加分） |
| **satisfaction** 满意度 | 0.20 | 路线内 POI 平台评分均值 / 5 |
| **diversity** 多样性 | 0.15 | 去重类别数 / 总站数 |

### 时间模型

```
总耗时 = Σ 交通时间 + Σ 停留时间
每日预算 = 720 分钟（12 小时）
```

停留时间按活动类型：**景点 45min / 餐饮 60min / 住宿 0 / 购物 40min / 出发点 0**。

### 硬约束（违反 → `score=0` 判负）

| 约束 | 判负条件 |
|---|---|
| `time_over_budget` | 总耗时 > 720 × 天数 |
| `repeat_poi` | 路线内 POI 重复 |
| `too_short` | 去重后 < 3 站 |
| `days_mismatch` | 推断天数 ≤ 指令天数 − 2（**渐进式**：3 日游出 1 日量才判负，差 1 天走软扣分） |
| `missing_core_poi` | 指令核心景点不在路线（且 POI 库存在该景点） |
| `start_mismatch` | 出发地不在路线前两站（前两站容错） |

> **天数推断**：按去重站数 `≤10 → 1日`、`≤16 → 2日`、`>16 → 3日`。

### 需求匹配度（软扣分）

`requirement_match = max(0, 1 − Σ 扣分)`，只对指令里**显式出现**的约束扣分：

| 维度 | 扣分规则 |
|---|---|
| **预算超支** | 估算花费超预算 ≤20% 扣 0.1；≤50% 扣 0.3；>50% 扣 0.6 |
| **偏好** | 指令"喜欢美食"但餐饮占比 <0.25：每低 0.1 扣 0.15，封顶 0.45 |
| **节奏** | 慢节奏但每天 >7 站扣 0.3；>9 站扣 0.6 |
| **季节** | 冬季指令但路线无冬季特色 POI（`season_winter − season_summer > 0.3`）扣 0.2 |
| **天数超出** | 推断天数 > 指令天数：超 1 天扣 0.15，超 2 天扣 0.3 |
| **天数偏短** | 推断天数 = 指令天数 − 1（如 3 日游出 2 日量）扣 0.5 |

> 不传 `instruction` 时 `composite_score_v5` 完全退回 v4 行为（纯质量五维），旧调用不受影响。

### 逐日模式下的打分

多日游按天拆分后，**每天按 1 日游独立打分**（天数约束只作用于总行程），逐日分 + 需求匹配汇总成整体分。半天（如"第 2 天下午 5 点走"）按 3-4 站轻量路线评估。

## 项目结构

```
src/
├── constraint_parser.py   # 指令约束抽取（天数/季节/预算/人群/半日）
├── retrieval.py           # RAG 候选检索（语义过滤 + 类型配额）
├── itinerary_planner.py   # 逐日拆分 + 候选编号生成 + 节奏整理
├── scoring.py             # v5 打分（质量 0.70 + 需求匹配 0.30）
├── device.py              # 多平台设备选择（CUDA/MPS/XPU/CPU）
└── visualize.py           # folium 交互地图

scripts/
├── serve_qwen_demo.py         # Web Demo（5 模型切换）
├── finetune_qwen35.py         # Qwen3.5 SFT（候选编号格式）
├── train_grpo_qwen35.py       # Qwen3.5 GRPO
├── train_dpo.py               # DPO 偏好对齐
├── train_grpo.py              # Qwen3 GRPO
├── finetune_qwen.py           # Qwen3 SFT
├── prepare_qwen35_dataset.py  # 候选编号逐日数据
├── label_poi_semantics.py     # POI 语义标注（全量 48,961）
├── augment_elderly_dataset.py # 带父母/老人指令数据增强
└── eval_dpo_vs_sft.py         # SFT vs DPO/GRPO 评估

archive/                       # 历史竞赛代码（已归档，见下文）
```

## 历史归档

`archive/` 保留了早期统计建模竞赛的完整代码与论文（Transformer 编解码器、Engram/MHC/Muon 等），以及被实验证实无效的设计：

- `archive/legacy_transformer/` — 旧 ItineraryTransformer 模型与训练管线
- `archive/legacy_scripts/` — 旧数据管线与竞赛评估脚本
- `archive/paper/` — 竞赛论文（LaTeX）
- `archive/docs/` — 旧开发文档与诊断报告

> **为何归档**：Muon 优化器（对比实验 AdamW 胜出 21%）、MHC 双曲流形（消融为净负贡献）、Engram 记忆（差分在噪声级）均被证实无实质收益。当前 Qwen 主线的"检索 + 规则 + 轻模型排序"架构已取代旧方案。
