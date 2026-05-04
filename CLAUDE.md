# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2026年全国大学生统计建模大赛参赛作品。基于Transformer架构构建哈尔滨文旅线路优化模型，核心创新点为融合DeepSeek论文中的三项技术：Engram内容寻址记忆、MHC双曲流形约束（庞加莱球模型）、Muon矩阵正交化优化器。

**当前状态：模型训练完成，推理功能可用。** 已完成真实数据处理、模型训练、路线生成和优化。

## Commands

```bash
# 环境（使用 uv 管理）
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 数据准备（真实数据处理）
uv run python scripts/prepare_real_data.py

# 训练
uv run python -m src.train --config configs/default.yaml
uv run python -m src.train --config configs/default.yaml --resume checkpoints/best_model.pt

# 推理生成路线
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season winter --max_stops 10 --n_routes 3

# 评估
uv run python -m src.evaluate --checkpoint checkpoints/best_model.pt

# 测试
uv run pytest tests/ -v

# TensorBoard
uv run tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# Lint
ruff check src/ tests/
```

## Architecture

### 数据流

```
原始数据（data/raw/）：
  - 哈尔滨POI_核心节点.csv      # 135个核心POI
  - 哈尔滨旅游路线数据.csv      # 403条小红书路线
  - 距离矩阵_公里.csv           # 135x135真实路网距离
  - 耗时矩阵_分钟.csv           # 135x135真实路网耗时
  - merged_pois.csv             # 合并后的POI数据（用于补充餐饮/购物）

  ↓ scripts/prepare_real_data.py

处理后的数据（data/processed/）：
  ├── poi_metadata.csv          # 180个POI元信息（含活动类型）
  ├── poi_features.npy          # [180, 128] 特征矩阵
  ├── adjacency.npy             # [180, 180] 邻接矩阵
  ├── distance_matrix.npy       # [180, 180] 距离矩阵（km）
  ├── distance_std.npy          # [180, 180] 距离标准差
  ├── time_matrix.npy           # [180, 180] 时间矩阵（min）
  ├── time_std.npy              # [180, 180] 时间标准差
  ├── poi_activity_types.npy    # [180] 活动类型标签
  └── routes.npy                # 165条真实路线

  ↓ src/data/dataset.py
HarbinRouteDataset → DataLoader (train/val/test)
```

### 模型架构 (src/models/)

`RouteTransformer` (transformer.py) 是顶层模型，支持活动类型条件生成和约束解码。默认 d_model=128, n_heads=8, 4层encoder/decoder, d_ff=512, max_route_len=20。

1. **POIEmbedding** (embeddings.py) — POI ID + 类别 + 活动类型 + 评分嵌入 + 正弦位置编码
2. **PoincareEmbedding** (mhc.py) — 庞加莱球模型双曲嵌入，expmap/logmap映射，测地距离公式
3. **GraphAwareEncoder** (encoder.py) — 邻接矩阵 + 活动类型相似性偏置注入Self-Attention
4. **EngramDecoder** (decoder.py) — Masked Decoder + Cross-Attention + Engram Attention + 活动类型条件嵌入
5. **EngramMemory** (engram.py) — 余弦相似度top-k检索，可学习门控融合，季节权重
6. **RouteLoss** (losses.py) — CE(1.0) + 距离惩罚(0.01) + MHC正则项(0.05)

### 活动类型条件生成架构

核心创新：通过活动类型约束解码，生成符合真实旅游节奏的路线。

**活动类型（6种）：**
- 景点(0)、餐饮(1)、住宿(2)、交通(3)、购物(4)、出发点(5)

**转换约束矩阵：**
- 景点后：都可以
- 餐饮后：不能连续餐饮或出发点
- 住宿后：不能连续住宿
- 交通后：不能连续交通
- 购物后：不能连续购物
- 出发点后：不能连续出发点

**推理时：** 根据当前POI的活动类型，约束下一个POI的选择，避免不合理的连续活动。

### 优化器 (src/optim/)

**MuonOptimizer** — 继承 `torch.optim.Optimizer`，Newton-Schulze迭代(5步)近似正交化梯度。三组参数：attention_params(lr_attn=1e-4)、ffn_params(lr_ffn=3e-4)、other_params(均值)。仅对≥2D参数做正交化。momentum=0.95, Nesterov=True, weight_decay=1e-4。

### 训练策略 (src/train.py)

- Teacher Forcing + Scheduled Sampling（ratio从0.5线性衰减）
- Early Stopping（patience=10, 监控val_loss）
- TensorBoard记录loss和指标
- 最佳模型保存至 `checkpoints/best_model.pt`
- batch_size=32, epochs=200, 数据划分 0.8/0.1/0.1
- 支持活动类型条件生成（7元素batch格式）

### 推理与路线优化 (src/inference.py)

**路线生成：**
- 活动类型约束Beam Search
- 支持指定起点、季节、最大游览点数
- 生成多条候选路线

**路线优化：**
- 最近邻算法：按距离最近原则重排路线顺序，避免走回头路
- 2-opt优化：消除路线中的交叉，进一步优化路线长度
- 保持起点不变，优化后续景点顺序

**输出：**
- JSON路线详情（含活动类型序列）
- 交互式地图（序号标记、方向箭头、名称标签）
- 路线对比图

### 可视化 (src/visualize.py)

- `plot_route_on_map()` — 单条路线地图，带序号标记和方向箭头
- `plot_route_comparison()` — 多条路线对比图，带图例
- `plot_route_on_map_with_roads()` — 沿实际道路绘制（需要高德地图API key）
- `plot_training_curves()` — 训练曲线
- `plot_ablation_results()` — 消融实验结果

### 配置体系

- `configs/default.yaml` — 全部超参（data/model/engram/mhc/loss/optimizer/training/metrics/experiment）
- `configs/ablation.yaml` — 消融实验配置

### 评估指标 (src/evaluate.py)

四维评价：距离(0.30) + 时间(0.25) + 满意度(0.25) + 多样性(0.20)，各指标先归一化到[0,1]再加权求和。

## Key Conventions

- 所有配置通过YAML加载，argparse仅传config路径和device/resume
- 模块必须用 `python -m src.train` 运行（相对导入）
- 距离矩阵使用Haversine公式（非欧氏距离）
- 季节分为冬季(11-2月冰雪季)和夏季(6-8月)，影响Engram检索权重
- MHC曲率为负值（默认-1.0），内部 `c = abs(curvature)` 使用
- 坐标系：哈尔滨中心点 (45.80, 126.53)，max_pois=180
- 推理使用约束Beam Search（beam_size可配置）
- 路线优化：最近邻 + 2-opt，避免走回头路
- 活动类型约束：禁止连续餐饮/住宿/交通等不合理序列
- Linter: ruff, line-length=100, target Python 3.10+
- 测试框架: pytest, testpaths=["tests"]
- 包管理: uv（推荐），不要用 pip/python 直接运行

## 当前训练结果

- **数据规模：** 180个POI，165条真实路线
- **训练轮次：** 26 epochs（早停）
- **最佳val_loss：** 2.2803
- **模型参数量：** 1,946,524
- **最优路线示例：** 综合得分0.822，距离29.7km，耗时147min
