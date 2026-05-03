# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2026年全国大学生统计建模大赛参赛作品。基于Transformer架构构建哈尔滨文旅线路优化模型，核心创新点为融合DeepSeek论文中的三项技术：Engram内容寻址记忆、MHC双曲流形约束（庞加莱球模型）、Muon矩阵正交化优化器。

**当前状态：骨架代码阶段。** 所有模型模块的核心方法均使用 `raise NotImplementedError` 占位，需逐步填充实现。测试文件中的用例也是空壳（pass/TODO）。

## Commands

```bash
# 环境
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 训练（必须用 -m 运行，因为模块间用相对导入）
python -m src.train --config configs/default.yaml
python -m src.train --config configs/default.yaml --resume checkpoints/best_model.pt --device cuda

# 评估
python -m src.evaluate --checkpoint checkpoints/best_model.pt

# 测试
pytest tests/ -v
pytest tests/test_engram.py -v  # 单个测试文件

# 数据爬取（需要 .env 中的 AMAP_API_KEY）
bash scripts/crawl_data.sh

# 消融实验（遍历 ablation.yaml 中所有实验配置）
bash scripts/run_ablation.sh

# Lint
ruff check src/ tests/
```

## Architecture

### 数据流

```
高德API爬取 → src/data/poi_crawler.py → data/raw/pois_harbin.csv
百度POI xlsx → data/raw/百度poi(旅游相关) - 全集.xlsx
  ↓ src/data/preprocess.py (特征工程，Haversine距离矩阵)
data/processed/
  ├── poi_features.npy      # [n_pois, feature_dim]
  ├── adjacency.npy          # [n_pois, n_pois] 路网邻接
  ├── distance_matrix.npy    # [n_pois, n_pois] Haversine距离
  ├── time_matrix.npy        # [n_pois, n_pois] 通行时间
  ├── routes.npy             # 历史路线数据
  └── poi_metadata.csv       # POI元信息
  ↓ src/data/dataset.py
HarbinRouteDataset → DataLoader (train/val/test)
```

### 模型架构 (src/models/)

`RouteTransformer` (transformer.py) 是顶层模型，`__init__` 已实现，forward/generate 待填充：

1. **POIEmbedding** (embeddings.py) — POI ID + 类别 + 评分嵌入 + 正弦位置编码
2. **PoincareEmbedding** (mhc.py) — 庞加莱球模型双曲嵌入，expmap/logmap映射，测地距离公式，曲率取绝对值使用（默认 c=1.0 from curvature=-1.0）
3. **GraphAwareEncoder** (encoder.py) — 邻接矩阵注入Self-Attention作为偏置
4. **EngramDecoder** (decoder.py) — Masked Decoder + Cross-Attention + 可选Engram Attention
5. **EngramMemory** (engram.py) — register_buffer存储memory_keys/values/scores/mask，余弦相似度top-k检索，可学习门控融合，季节权重参数（冬/夏）
6. **RouteLoss** (losses.py) — CE + 距离惩罚 + MHC正则项的加权组合

### 优化器 (src/optim/)

**MuonOptimizer** — 继承 `torch.optim.Optimizer`，Newton-Schulze迭代(5步)近似正交化梯度。三组参数：attention_params(lr_attn=3e-4)、ffn_params(lr_ffn=1e-3)、other_params(均值)。仅对≥2D参数做正交化。

### 配置体系

- `configs/default.yaml` — 全部超参（data/model/engram/mhc/loss/optimizer/training/metrics/experiment）
- `configs/ablation.yaml` — 消融实验配置，8组实验覆盖：无Engram、无MHC、AdamW替代Muon、纯Transformer baseline、不同top_k(3/10)、不同曲率(-0.5/-2.0)

### 评估指标 (src/evaluate.py)

四维评价：距离(0.30) + 时间(0.25) + 满意度(0.25) + 多样性(0.20)，各指标先归一化到[0,1]再加权求和。

### 可视化 (src/visualize.py)

folium交互式地图（路线绘制）+ matplotlib静态图表（训练曲线、消融对比柱状图）。

## Key Conventions

- 所有配置通过YAML加载，argparse仅传config路径和device/resume
- 模块必须用 `python -m src.train` 运行（相对导入）
- 距离矩阵使用Haversine公式（非欧氏距离）
- 季节分为冬季(11-2月冰雪季)和夏季(6-8月)，影响Engram检索权重
- MHC曲率为负值（默认-1.0），内部 `c = abs(curvature)` 使用
- 坐标系：哈尔滨中心点 (45.80, 126.53)，max_pois=150
- Linter: ruff, line-length=100, target Python 3.10+
- 测试框架: pytest, testpaths=["tests"]
