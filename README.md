# 哈尔滨文旅线路优化模型

基于 Transformer 架构的哈尔滨旅游路线智能规划系统，2026 年全国大学生统计建模大赛参赛作品。

## 创新点

1. **Engram 记忆机制** — 外部内容寻址记忆库，从历史优质路线中检索相似路线辅助决策
2. **MHC 双曲流形约束** — 庞加莱球模型嵌入，在双曲空间中保持地理拓扑关系
3. **Muon 优化器** — 参考 KellerJordan/Muon 和 Kimi 论文，Newton-Schulz quintic 迭代矩阵正交化 + Nesterov 动量 + AdamW 1D 参数回退
4. **随机图建模** — POI 间距离和通行时间建模为概率分布 N(mean, std)，近距波动大、远距波动小，训练时采样提升鲁棒性

## 环境安装

```bash
# 使用 uv 管理环境（推荐）
uv sync

# 安装 PyTorch（CUDA 12.8）
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 或使用 pip
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 数据准备

```bash
# 1. 合并高德 + 百度 POI 数据
uv run python scripts/merge_data.py

# 2. 清洗、筛选、特征提取、生成距离/时间概率分布矩阵
uv run python -c "from src.data.preprocess import prepare_all; prepare_all('data/raw/merged_pois.csv')"

# 输出至 data/processed/:
#   poi_metadata.csv      POI 元信息
#   poi_features.npy      特征矩阵 [500, 128]
#   adjacency.npy         邻接矩阵（带距离衰减权重）
#   distance_matrix.npy   距离均值 [500, 500]
#   distance_std.npy      距离标准差（随机图建模）
#   time_matrix.npy       时间均值
#   time_std.npy          时间标准差
#   routes.npy            历史路线数据
```

数据来源：
- 高德地图 POI API（`data/raw/哈尔滨POI数据_完整版.csv`，5,739 条）
- 百度地图 POI（`data/raw/百度poi(旅游相关) - 全集.xlsx`，72,455 条）

合并后按旅游导向配额筛选 500 个 POI：景点 35%、餐饮 25%、住宿 20%、购物 15%、交通 5%。

## 训练

```bash
# 默认配置（AdamW）
uv run python -m src.train --config configs/default.yaml

# 恢复训练
uv run python -m src.train --resume checkpoints/best_model.pt

# TensorBoard 实时监控
uv run tensorboard --logdir logs/
```

训练特性：Teacher Forcing + Scheduled Sampling、早停（patience=10）、tqdm 终端进度条 + TensorBoard 双轨日志。

## 推理

```bash
# 按名称指定起点
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start "冰雪大世界" --season winter

# 限制游览点数和时间预算
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season summer --max_stops 8 --max_hours 6

# 生成多条候选路线
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --season winter --n_routes 3
```

输出：终端路线详情表 + 综合评估得分 + `output/route_map.html` 交互式地图。

## 评估

```bash
uv run python -m src.evaluate --checkpoint checkpoints/best_model.pt
```

四维评价体系：距离(0.30) + 时间(0.25) + 满意度(0.25) + 多样性(0.20)。

## 测试

```bash
uv run pytest tests/ -v
```

## 项目结构

```
├── configs/
│   ├── default.yaml          # 主配置（data/model/loss/optimizer/training）
│   └── ablation.yaml         # 消融实验配置（8 组）
├── data/
│   ├── raw/                  # 原始数据（高德CSV + 百度XLSX）
│   └── processed/            # 处理后数据（npy + csv）
├── scripts/
│   ├── merge_data.py         # 高德百度数据合并去重
│   ├── crawl_data.sh         # 高德API爬取
│   └── run_ablation.sh       # 消融实验批量运行
├── src/
│   ├── data/
│   │   ├── preprocess.py     # 数据清洗 + 特征工程 + 概率分布矩阵
│   │   └── dataset.py        # PyTorch Dataset + 概率采样
│   ├── models/
│   │   ├── transformer.py    # RouteTransformer（顶层模型）
│   │   ├── encoder.py        # Graph-aware Transformer Encoder
│   │   ├── decoder.py        # Engram增强 Masked Decoder
│   │   ├── engram.py         # Engram 记忆模块
│   │   ├── mhc.py            # Poincaré 球双曲嵌入
│   │   ├── embeddings.py     # POI Embedding + 正弦位置编码
│   │   └── losses.py         # CE + 距离惩罚 + MHC 正则
│   ├── optim/
│   │   └── muon.py           # Muon + AdamW 混合优化器
│   ├── train.py              # 训练主脚本
│   ├── evaluate.py           # 四维评估
│   ├── inference.py          # 推理 + 地图可视化
│   └── visualize.py          # folium地图 + matplotlib图表
└── tests/                    # pytest 单元测试（10个）
```
