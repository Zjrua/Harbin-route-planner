# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2026年全国大学生统计建模大赛参赛作品。基于Transformer架构构建哈尔滨文旅线路优化模型，核心创新点为融合DeepSeek论文中的三项技术：Engram内容寻址记忆、MHC双曲流形约束（庞加莱球模型）、Muon矩阵正交化优化器。

**当前状态：POI 规模 10,000 训练完成，消融实验已完成，论文图表已更新。** 已完成POI规模扩展（180→10,000（可配置））、共享数据加载重构（节省25GB+显存）、AMP混合精度训练、7组消融实验、论文图表生成。

## Commands

```bash
# 环境（使用 uv 管理）
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 数据准备
uv run python scripts/prepare_data.py  # POI 筛选 + 合成路线生成（默认用全部合格 POI）
uv run python scripts/process_xhs_data.py    # 提取XHS餐饮住宿热度
uv run python scripts/prepare_real_data.py   # POI增强 + 路线增强（180-POI旧管线）

# 训练
uv run python -m src.train --config configs/default.yaml
uv run python -m src.train --config configs/default.yaml --resume checkpoints/best_model.pt

# 推理生成路线
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season winter --max_stops 10 --n_routes 3
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season winter --max_stops 14 --days 2

# 消融实验
uv run python scripts/run_ablation.py

# 评估（详见下方"评估实验"章节的三个方向）
uv run python -m scripts.evaluate_on_real --checkpoint checkpoints/best_model.pt   # 方向B：真实数据next-POI
uv run python -m scripts.run_baselines --checkpoint checkpoints/best_model.pt      # 方向C：方法对比

# 测试
uv run pytest tests/ -v

# TensorBoard
uv run tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# Lint
ruff check src/ tests/
```

```bash
# 论文编译（paper/目录下，需 xelatex）
cd paper && xelatex main.tex && xelatex main.tex
```

## Architecture

### 数据流

```
原始数据（data/raw/）：
  - merged_pois.csv             # 48,961个POI（景点28K/餐饮10.7K/住宿7.4K/购物2.3K/交通538）
  - 哈尔滨POI_核心节点.csv      # 135个核心POI
  - 哈尔滨旅游路线数据.csv      # 403条小红书路线
  - 距离矩阵_公里.csv           # 135x135真实路网距离
  - 耗时矩阵_分钟.csv           # 135x135真实路网耗时
  - search_*.jsonl              # 11,959条小红书笔记（餐饮/住宿热度）

  ↓ scripts/prepare_data.py（POI 数据管线）

处理后的数据（data/processed/）：
  ├── poi_metadata.csv          # 10,000个POI（含XHS热度、活动类型）
  ├── poi_features.npy          # [10000, 128] 特征矩阵
  ├── adjacency.npy             # [10000, 10000] 邻接矩阵
  ├── distance_matrix.npy       # [10000, 10000] Haversine距离矩阵（km）
  ├── distance_std.npy          # [10000, 10000] 距离标准差
  ├── poi_activity_types.npy    # [10000] 活动类型标签
  ├── cluster_id.npy            # [10000] 景点步行聚类ID
  ├── clusters.npy              # 步行景点聚类
  └── routes.npy                # 5,168条路线（168 XHS + 5000合成）

  ↓ src/data/dataset.py
ItineraryDataset → get_shared_data() (shared tensors on GPU)
  └─ __getitem__ 只返回 (route, score, route_activity) 三个per-sample数据
```

### POI 数据准备 (`scripts/prepare_data.py`)

1. **POI筛选**：从merged_pois.csv(49K)经`clean_poi_data(max_pois=10000)`筛选
   - 基于quality_score（评分+评论数+类别）选择 top 10,000
   - 类别分布：餐饮4091, 住宿2000, 景点1923, 购物1500, 交通486

2. **矩阵计算**：Haversine球面距离 + 速度因子估算时间，邻接矩阵<30km连通

3. **合成路线生成**（~5000条）：
   - 类别感知近邻随机游走（80%景点/15%购物/5%餐饮）
   - 距离^-2 × 评分加权选择下一个POI
   - 餐饮/住宿增强插入（每2-3景插入餐饮，末尾加住宿）

4. **步行聚类**：Union-Find on scenic POIs within 1km → 90 clusters

### 共享数据加载（GPU显存优化）

面向大规模 POI 的关键重构。原方案DataLoader每样本返回完整[n_pois, n_pois]矩阵，大规模 POI 时每样本800MB × batch=32 → 25.6GB不可行。

优化后：
- `get_shared_data(device)` 一次性加载共享矩阵到GPU（~1.4GB）
- `__getitem__` 只返回per-sample数据（route, score, route_activity_types）
- 编码器运行一次，encoder_output在所有batch间共享（省~12.8GB/batch）
- 距离矩阵用2D索引（`distances[src, dst]`），避免expand到batch维度
- 训练峰值显存：~3.27 GB（RTX 4090 24GB充裕）

### 模型架构 (src/models/)

`ItineraryTransformer` (transformer.py) 是顶层模型。精简配置：d_model=128, n_heads=8, 3层encoder/decoder, d_ff=384, dropout=0.2。

**参数量：4,569,976（POI 词表输出层占用~1.28M）**

| 模块 | 参数 | 占比 | 功能 |
|------|------|------|------|
| EngramDecoder | ~960K | 21% | 自回归解码 + Cross-Attn + Engram |
| GraphAwareEncoder | ~300K | 7% | 邻接矩阵 + 活动类型偏置注入SA |
| Output | ~1,280K | 28% | 128→10000投影（最大单模块） |
| POIEmbedding | ~1,350K | 30% | POI 嵌入表 + 活动类型 + 位置编码 |
| EngramMemory | 67K | 1.5% | 余弦相似度top-k检索 |
| MHC (Poincare) | 12K | 0.3% | 双曲空间嵌入正则化 |
| 其他 | ~600K | 13% | FFN, LayerNorm等 |

1. **POIEmbedding** (embeddings.py) — POI ID + 类别 + 活动类型(6种) + 评分嵌入 + 正弦位置编码
2. **PoincareEmbedding** (mhc.py) — 庞加莱球模型双曲嵌入，提供几何正则化
3. **GraphAwareEncoder** (encoder.py) — 邻接矩阵 + 活动类型相似性偏置注入Self-Attention
4. **EngramDecoder** (decoder.py) — Masked Decoder + Cross-Attention + Engram Attention + 活动类型条件嵌入
5. **EngramMemory** (engram.py) — 余弦相似度top-k检索，可学习门控融合，季节权重
6. **RouteLoss** (losses.py) — CE(1.0) + 距离惩罚(0.01) + MHC正则项(0.05)

### 活动类型条件生成架构

**活动类型（6种）：** 景点(0)、餐饮(1)、住宿(2)、交通(3)、购物(4)、出发点(5)

**转换约束矩阵：**
```
# 景点  餐饮  住宿  交通  购物  出发点
[  0.0,  2.0, -inf,  0.0,  1.0,  0.0],  # 景点后：鼓励餐饮/购物，禁止住宿
[  2.0, -inf,  0.0,  0.0,  0.0, -inf],  # 餐饮后：鼓励景点
[  2.0,  0.0, -inf,  0.0,  0.0,  0.0],  # 住宿后：优先去景点
[  2.0,  0.0, -inf, -inf,  0.0,  0.0],  # 交通后：优先去景点
[  2.0,  0.0, -inf,  0.0, -inf,  0.0],  # 购物后：优先回景点
[  2.0,  0.0, -inf,  0.0,  0.0, -inf],  # 出发点后：优先去景点
```

**推理时约束：**
- 一日游：住宿仅在最后3步出现，选中即终止
- 多日游：住宿允许在day boundary（每天末尾），选后继续
- 连续3个同类型后鼓励切换（景点→餐饮+5，再选景点-3）
- 同团景点不重复访问（cluster masking）
- 已访问POI不重复

### POI步行聚类

基于连通图的聚类（≤1km步行距离），90个团（当前规模下距离更稀疏）。路线生成时访问某团后屏蔽全团，避免走回头路。

### 优化器 (src/optim/)

**MuonOptimizer** — Newton-Schulze迭代(5步)近似正交化梯度。attention_params(lr_attn=1e-4)、ffn_params(lr_ffn=3e-4)、other_params(均值)。仅对≥2D参数做正交化。momentum=0.95, Nesterov=True, weight_decay=1e-4。

### 训练策略 (src/train.py)

- **大规模 POI 优化：** 编码器预计算一次，输出在所有batch共享；距离矩阵2D索引避免batch展开
- **AMP混合精度训练**（`torch.amp.autocast("cuda")` + GradScaler）
- Teacher Forcing + Scheduled Sampling（ratio从0.5线性衰减）
- Early Stopping（patience=15, 监控val_loss）
- TensorBoard记录loss和指标
- 最佳模型保存至 `checkpoints/best_model.pt`
- batch_size=256, epochs=200, 数据划分 0.8/0.1/0.1, num_workers=4, pin_memory=True

### 推理与路线优化 (src/inference.py)

**路线生成：**
- 活动类型约束Beam Search + 集群感知masking
- 支持指定起点（景点/酒店/餐饮）、季节、天数
- 多日游：`--days N`，酒店出现在day boundary

**路线优化：**
- 最近邻重排中间景点（保护起点和末尾住宿）
- 多日游跳过2-opt（保护day boundary酒店位置）

**输出：**
- JSON路线详情（含活动类型序列和天数分隔）
- 交互式地图（序号标记、方向箭头、名称标签、路线对比）

### 可视化 (src/visualize.py)

- `plot_route_on_map()` — 单条路线地图，带序号标记和方向箭头
- `plot_route_comparison()` — 多条路线对比图，带图例和标题
- `plot_route_on_map_with_roads()` — 沿实际道路绘制（需高德API key）

### XHS数据集成 (scripts/process_xhs_data.py)

从11,959条小红书笔记提取POI热度权重：
- 餐饮/住宿POI的XHS提及次数和加权热度
- 热度用于路线增强时选择更受欢迎的餐饮/住宿（综合分 = -距离 + 热度）
- 输出：`poi_xhs_popularity.npy` 和更新 `poi_metadata.csv`

### 强化学习微调计划

已完成监督预训练（CE loss），下一步RL微调提升路线质量：
- **算法：** Self-Critical Seq2Seq 或 DPO
- **奖励函数：** composite_score（距离+时间+满意度+多样性）
- **动机：** 绕过合成数据质量限制，直接优化最终目标

## Key Conventions

- 所有配置通过YAML加载，argparse仅传config路径和device/resume
- 模块必须用 `python -m src.train` 运行（相对导入）
- 距离矩阵使用Haversine公式（非欧氏距离）
- 季节分为冬季(11-2月冰雪季)和夏季(6-8月)
- MHC曲率为负值（默认-1.0），内部 `c = abs(curvature)` 使用
- 坐标系：哈尔滨中心点 (45.80, 126.53)，max_pois=10000
- 推理使用约束Beam Search（beam_size=5）
- 活动类型约束：禁止连续餐饮/住宿/交通等不合理序列
- Linter: ruff, line-length=100, target Python 3.10+
- 测试框架: pytest, testpaths=["tests"]
- 包管理: uv（唯一推荐）

## 论文排版（paper/main.tex）

- 编译器：xelatex（ctexart 中文支持）
- 行距：正文全局 `\setstretch{1.66}`；附录/致谢用 `\setstretch{1.0}` + `\fontsize{12pt}{24pt}` 实现24磅行距
- 表格内容用 `tighttable` 环境（`\setstretch{1.0}`）
- 西文字体：Times New Roman（`\setmainfont` + `\setsansfont`）
- 中文字体：宋体正文、黑体标题、楷书二级标题、方正小标宋论文题目
- 封面页已删除，摘要从第1页开始，摘要标题居中
- 参考文献引用格式：上标 `$^{\cite{...}}$`（如 [1] 显示为上标）
- 表格与插图清单：保留清单页但不加入目录（无 `\addcontentsline`）
- 清单中编号和标题须与实际 `\caption` 一致
- 参考文献标题通过 `\renewcommand{\refname}` 控制格式，避免与 `thebibliography` 重复
- 临时文件通过 `paper/.gitignore` 忽略（.aux/.log/.toc等）

## 评估实验（方向 A/B/C，已完成）

三个独立评估实验验证了模型的核心价值与边界，结果已写入论文：

### 方向 B：真实数据泛化性评估（最强证据）⭐

将168条小红书真实路线从训练集剥离为 holdout（`data/processed/routes_xhs_holdout.npy`），在从未见过的真实路线上测试 next-POI 预测：

| 方法 | top-1 | top-5 |
|---|---|---|
| most_popular | 0.00% | 0.00% |
| nearest_neighbor | 1.93% | 8.67% |
| **Transformer** | **47.77%** | **78.73%** |

模型比最近邻高 24.7 倍，证明学到了真实游客行为（非合成规则复现）。**打破了"合成数据循环论证"质疑。**
- 脚本：`scripts/evaluate_on_real.py`
- 数据：`output/real_data_evaluation.json`

### 方向 C：与启发式/OR 方法对比 + 指标改进

对比 5 个方法（random/NN/2-opt/OR-Tools/Transformer），揭示原始 composite 指标缺陷并改进：

| 方法 | rhythm | composite v1 | composite v2 |
|---|---|---|---|
| NN/OR-Tools | 0.467 | 0.8685 | 0.7619（暴跌） |
| **Transformer** | **1.000** | 0.8606 | **0.8727**（翻盘） |

NN/OR-Tools 生成"连续5住宿"的荒谬路线，Transformer 生成"游-吃交替"的合理节奏。v1 偏袒距离优化，v2 加入 rhythm 惩罚后排名翻转。
- 脚本：`scripts/run_baselines.py`、`src/baselines/`、`src/evaluate.py` 的 `composite_score_v2`
- 数据：`output/baseline_comparison.json`

### 方向 A：优化器对比（Muon 失败，诚实降级）

严格对比证明 Muon 在本任务规模下劣于 AdamW：

| 优化器 | best val_loss | train/val gap |
|---|---|---|
| **AdamW** | **4.8849** | 2.03 |
| Muon | 5.9291 | 3.70（过拟合严重）|

Muon 差 21.4%。论文已把 Muon 从"三大核心创新"降级为"探索性尝试"。核心创新改为：图感知注意力 + Engram + MHC。
- 脚本：`scripts/run_optimizer_comparison.py`
- 数据：`output/optimizer_comparison.json`

## 当前训练结果（POI 规模 10,000）

- **数据规模：** 10,000个POI（餐饮4091/住宿2000/景点1923/购物1500/交通486），5,168条路线（168 XHS + 5000合成）
- **训练划分：** train=4134 / val=517 / test=517（0.8/0.1/0.1）
- **最佳模型：** epoch 106，val_loss=4.9041（早停于epoch 121）
- **模型参数量：** 4,569,976
- **训练配置：** batch_size=256, AMP混合精度, patience=15, AdamW
- **训练耗时：** 121 epochs，每epoch ~40秒
- **GPU显存峰值：** ~3.27 GB（RTX 4090 24GB，大量余量）
- **loss曲线：** train 9.06→2.79, val 8.76→4.90
- **90个步行景点团（≤1km）**

### Loss详细记录

| 阶段 | epoch | train_loss | val_loss |
|------|-------|-----------|----------|
| 初始 | 1 | 9.0631 | 8.7592 |
| 快速下降 | 10 | 7.5619 | 7.6832 |
| 稳步下降 | 20 | 6.4710 | 6.7723 |
| 中期 | 40 | 4.9528 | 5.6041 |
| 放缓 | 60 | 3.9077 | 5.2011 |
| 接近收敛 | 80 | 3.5609 | 5.0148 |
| 最佳 | 106 | 3.0382 | 4.9041 |
| 早停 | 121 | 2.7903 | 4.9474 |

> 完整loss数据保存于 `output/training_loss.csv`（121条记录）

### 消融实验结果（已完成）

7组实验配置（180 POI规模，AdamW优化器）：
1. K=3 Engram（预期最优）
2. 完整模型 K=5
3. K=10
4. 移除Engram（-Engram）
5. 移除MHC（-MHC）
6. 移除Engram+MHC（-Engram-MHC）
7. 纯Transformer基线

### 输出文件

| 文件 | 内容 |
|------|------|
| `output/training_loss.csv` | 121 epoch完整loss记录 |
| `output/ablation_results.json` | 7组消融完整数据 |
| `output/ablation_results.csv` | 消融结果CSV |
| `output/routes_result.json` | 最新生成路线详情 |
| `output/best_route_map.html` | 最优路线交互地图 |
| `checkpoints/best_model.pt` | epoch 106最优模型权重（52MB） |
| `paper/main.tex` | 论文LaTeX源码 |
| `paper/main.pdf` | 论文PDF |
| `paper/training_curve.png` | 论文用训练曲线图 |
| `paper/ablation_comparison.png` | 论文用消融对比图 |
