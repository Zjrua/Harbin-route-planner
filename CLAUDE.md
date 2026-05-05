# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2026年全国大学生统计建模大赛参赛作品。基于Transformer架构构建哈尔滨文旅线路优化模型，核心创新点为融合DeepSeek论文中的三项技术：Engram内容寻址记忆、MHC双曲流形约束（庞加莱球模型）、Muon矩阵正交化优化器。

**当前状态：模型训练完成，推理功能可用。** 已完成真实数据处理、模型训练、路线生成优化、多日游支持。

## Commands

```bash
# 环境（使用 uv 管理）
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 数据准备（XHS热度 + POI增强）
uv run python scripts/process_xhs_data.py    # 提取XHS餐饮住宿热度
uv run python scripts/prepare_real_data.py   # POI增强 + 路线增强

# 训练
uv run python -m src.train --config configs/default.yaml
uv run python -m src.train --config configs/default.yaml --resume checkpoints/best_model.pt

# 推理生成路线
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season winter --max_stops 10 --n_routes 3
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season winter --max_stops 14 --days 2

# 评估
uv run python -m src.evaluate --checkpoint checkpoints/best_model.pt

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
  - 哈尔滨POI_核心节点.csv      # 135个核心POI
  - 哈尔滨旅游路线数据.csv      # 403条小红书路线
  - 距离矩阵_公里.csv           # 135x135真实路网距离
  - 耗时矩阵_分钟.csv           # 135x135真实路网耗时
  - merged_pois.csv             # 合并后的POI数据（补充餐饮/购物）
  - search_*.jsonl              # 11,959条小红书笔记（餐饮/住宿热度）

  ↓ scripts/process_xhs_data.py + scripts/prepare_real_data.py

处理后的数据（data/processed/）：
  ├── poi_metadata.csv          # 180个POI（含XHS热度、活动类型）
  ├── poi_features.npy          # [180, 128] 特征矩阵
  ├── adjacency.npy             # [180, 180] 邻接矩阵
  ├── distance_matrix.npy       # [180, 180] 距离矩阵（km）
  ├── distance_std.npy          # [180, 180] 距离标准差
  ├── time_matrix.npy           # [180, 180] 时间矩阵（min）
  ├── time_std.npy              # [180, 180] 时间标准差
  ├── poi_activity_types.npy    # [180] 活动类型标签
  ├── cluster_id.npy            # [180] 景点步行聚类ID
  ├── clusters.npy              # 15个景点团
  └── routes.npy                # 165条增强路线（含餐饮/住宿）

  ↓ src/data/dataset.py
HarbinRouteDataset → DataLoader (train/val/test)
```

**数据增强：** 原始小红书路线全是景点（94.5%）。`augment_routes_with_dining_and_hotel()` 每2景插入餐饮（用XHS热度加权选择）、末尾加住宿、去掉中间住宿。增强后：景点63.6%、餐饮23.5%、住宿10.4%。

**POI分类自动修正：** `prepare_real_data.py` 自动检测名称含"酒店/民宿/餐厅"等关键字的POI并修正分类（约40个）。

### 模型架构 (src/models/)

`RouteTransformer` (transformer.py) 是顶层模型。精简配置：d_model=128, n_heads=8, 3层encoder/decoder, d_ff=384, dropout=0.2。

**参数量：1,417,756（精简27%）**

| 模块 | 参数 | 占比 | 功能 |
|------|------|------|------|
| EngramDecoder | ~960K | 68% | 自回归解码 + Cross-Attn + Engram |
| GraphAwareEncoder | ~300K | 21% | 邻接矩阵 + 活动类型偏置注入SA |
| EngramMemory | 67K | 5% | 余弦相似度top-k检索 |
| POIEmbedding | 25K | 2% | POI + 活动类型 + 位置编码 |
| MHC (Poincare) | 12K | 0.8% | 双曲空间嵌入正则化 |
| Output | 23K | 2% | 128→180投影 |

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

基于连通图的聚类（≤1km步行距离），15个团/49景入团，52景独立。路线生成时访问某团后屏蔽全团，避免走回头路。

### 优化器 (src/optim/)

**MuonOptimizer** — Newton-Schulze迭代(5步)近似正交化梯度。attention_params(lr_attn=1e-4)、ffn_params(lr_ffn=3e-4)、other_params(均值)。仅对≥2D参数做正交化。momentum=0.95, Nesterov=True, weight_decay=1e-4。

### 训练策略 (src/train.py)

- Teacher Forcing + Scheduled Sampling（ratio从0.5线性衰减）
- Early Stopping（patience=10, 监控val_loss）
- TensorBoard记录loss和指标
- 最佳模型保存至 `checkpoints/best_model.pt`
- batch_size=32, epochs=200, 数据划分 0.8/0.1/0.1

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

## Key Conventions

- 所有配置通过YAML加载，argparse仅传config路径和device/resume
- 模块必须用 `python -m src.train` 运行（相对导入）
- 距离矩阵使用Haversine公式（非欧氏距离）
- 季节分为冬季(11-2月冰雪季)和夏季(6-8月)
- MHC曲率为负值（默认-1.0），内部 `c = abs(curvature)` 使用
- 坐标系：哈尔滨中心点 (45.80, 126.53)，max_pois=180
- 推理使用约束Beam Search（beam_size=5）
- 活动类型约束：禁止连续餐饮/住宿/交通等不合理序列
- Linter: ruff, line-length=100, target Python 3.10+
- 测试框架: pytest, testpaths=["tests"]
- 包管理: uv（唯一推荐）

## 论文排版（paper/main.tex）

- 编译器：xelatex（ctexart 中文支持）
- 行距：全局 `\setstretch{1.66}`（正文），表格内容用 `tighttable` 环境（`\setstretch{1.0}`）
- 其他部分（摘要/参考文献/附录/致谢）用 `\setstretch{1.0}` + `\fontsize{...}{24pt}` 显式指定24磅
- 西文字体：Times New Roman（`\setmainfont` + `\setsansfont`）
- 中文字体：宋体正文、黑体标题、楷书二级标题、方正小标宋论文题目
- 封面页已删除，摘要从第1页开始，摘要标题居中
- 图表编号：3图7表（图1模型架构TikZ、图2训练曲线、图3消融对比）
- 参考文献标题通过 `\renewcommand{\refname}` 控制格式，避免与 `thebibliography` 重复
- 临时文件通过 `paper/.gitignore` 忽略（.aux/.log/.toc等）

## 当前训练结果

- **数据规模：** 180个POI（景点60/住宿64/餐饮37/购物15/交通4），535条路线（165原始+370合成）
- **最佳模型：** Engram K=3（消融最优），57 epochs，val_loss=2.5338，综合得分0.8802
- **模型参数量：** 1,417,756（精简配置：3层编解码器, d_ff=384）
- **15个步行景点团（≤1km），49景入团**

### 消融实验结果（535路线/7组，真实数据）

| 实验 | 距离(km) | 满意度 | 多样性 | 综合得分 | Δ |
|------|----------|--------|--------|----------|-----|
| **K=3（最优）** | 27.3 | 4.85 | 0.50 | **0.8802** | +0.0055 |
| 移除MHC | 27.2 | 4.84 | 0.48 | 0.8753 | +0.0006 |
| 完整模型(K=5) | 32.3 | 4.81 | 0.49 | 0.8747 | — |
| 移除Engram | 35.6 | 4.85 | 0.48 | 0.8735 | -0.0012 |
| 移除Engram+MHC | 34.5 | 4.84 | 0.48 | 0.8735 | -0.0012 |
| K=10 | 38.2 | 4.83 | 0.49 | 0.8714 | -0.0033 |
| 纯Transformer基线 | 63.0 | 4.84 | 0.49 | 0.8676 | -0.0071 |

**结论：** K=3最优，完整模型>基线+0.0071，核心差异在路线距离而非满意度/多样性。

### 输出文件

| 文件 | 内容 |
|------|------|
| `output/training_curve.png` | 训练收敛曲线（中文） |
| `output/ablation_comparison.png` | 消融对比柱状图（中文，300dpi） |
| `output/ablation_results.json` | 7组消融完整数据 |
| `output/routes_result.json` | 最新生成路线详情 |
| `output/best_route_map.html` | 最优路线交互地图 |
| `checkpoints/best_model.pt` | K=3最优模型权重 |
| `paper/main.tex` | 论文LaTeX源码 |
| `paper/main.pdf` | 论文PDF（40页） |
| `paper/training_curve.png` | 论文用训练曲线图（复制自output/） |
| `paper/ablation_comparison.png` | 论文用消融对比图（复制自output/） |
