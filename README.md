# ItineraryTransformer

基于 Transformer 架构的文旅线路优化模型（以哈尔滨为案例城市），2026 年全国大学生统计建模大赛参赛作品。

## 创新点

1. **Engram 记忆机制** — 外部内容寻址记忆库，从历史优质路线中检索相似路线辅助决策
2. **MHC 双曲流形约束** — 庞加莱球模型嵌入，在双曲空间中保持地理拓扑关系
3. **Muon 优化器** — Newton-Schulz quintic 迭代矩阵正交化 + Nesterov 动量 + AdamW 1D 参数回退
4. **随机图建模** — POI 间距离和通行时间建模为概率分布 N(mean, std)，训练时采样提升鲁棒性
5. **活动类型条件生成** — 6种活动类型约束解码，景点→餐饮→住宿自然节奏
6. **POI步行聚类** — ≤1km连通图聚类，90个景点团，同团不重复访问
7. **XHS数据增强** — 11,959条小红书笔记提取餐饮/住宿热度，加权路线增强

## 环境安装

```bash
# 使用 uv 管理环境（推荐）
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 数据准备

```bash
# 1. POI 筛选 + 合成路线生成（主管线，默认用全部合格 POI，--max-pois N 限定数量）
uv run python scripts/prepare_data.py

# 2. 提取小红书 POI 热度
uv run python scripts/process_xhs_data.py

# 3. POI增强 + 路线增强（180-POI 旧管线，可选）
uv run python scripts/prepare_real_data.py
```

## 数据规模

### `data/raw/`（原始数据，共 10 个文件，约 31 MB）

| 文件 | 规模 | 说明 |
|------|------|------|
| `merged_pois.csv` | 48,961 行 × 10 列（6.5 MB） | 合并去重后的哈尔滨 POI（餐饮 28K / 购物 10.7K / 住宿 7.4K / 景点 2.3K / 交通 538） |
| `百度poi(旅游相关) - 全集.xlsx` | 72,455 行 × 15 列（12.3 MB） | 百度地图旅游相关 POI 全集 |
| `哈尔滨POI数据_完整版.csv` | 5,739 行 × 8 列 | 高德 POI 完整版 |
| `哈尔滨POI_核心节点.csv` | 135 行 × 8 列 | 人工筛选的核心 POI 节点 |
| `哈尔滨旅游路线数据.csv` | 403 行 × 6 列 | 小红书旅游路线 |
| `距离矩阵_公里.csv` | 135 × 136 | 135 核心节点真实路网距离（km） |
| `耗时矩阵_分钟.csv` | 135 × 136 | 135 核心节点真实路网耗时（min） |
| `search_contents_2026-05-04.jsonl` | 1,179 条（3.9 MB） | 小红书笔记内容 |
| `search_contents_2026-05-05.jsonl` | 653 条（1.7 MB） | 小红书笔记内容 |
| `search_comments_2026-05-04.jsonl` | 10,127 条（5.1 MB） | 小红书笔记评论 |

> 三份 JSONL 合计 **11,959 条小红书笔记**，用于提取餐饮/住宿热度。

### `data/processed/`（处理后数据，共 15 个文件，约 1.9 GB）

**当前 POI 规模（主管线输出，默认 10,000）：**

| 文件 | 规模 | 说明 |
|------|------|------|
| `poi_metadata.csv` | 10,000 行 × 15 列 | POI 元信息（名称、坐标、类别、评分、XHS 热度、活动类型） |
| `poi_features.npy` | [10000, 128] float64 | POI 特征矩阵 |
| `adjacency.npy` | [10000, 10000] float32 | 邻接矩阵（距离衰减权重，<30 km 连通） |
| `distance_matrix.npy` | [10000, 10000] float32 | Haversine 球面距离矩阵（km） |
| `distance_std.npy` | [10000, 10000] float32 | 距离标准差（概率分布建模） |
| `time_matrix.npy` | [10000, 10000] float32 | 通行时间矩阵（min） |
| `time_std.npy` | [10000, 10000] float32 | 通行时间标准差 |
| `poi_activity_types.npy` | [10000] int64 | 活动类型标签 |
| `cluster_id.npy` | [10000] int32 | 步行景点聚类 ID |
| `clusters.npy` | 90 个团 | ≤1 km 连通的步行景点聚类 |
| `routes.npy` | 5,168 条 | 训练路线（168 XHS + 5,000 合成，长度 2–33，均值 14.0） |

**XHS 数据提取（`process_xhs_data.py` 输出）：**

| 文件 | 规模 | 说明 |
|------|------|------|
| `xhs_extracted_routes.npy` | 153 条 | 从笔记中提取的路线 |
| `xhs_processing_report.json` | — | 11,959 笔记分类报告（路线 493 / 餐饮 794 / 住宿 355 / 其他 10,317） |

**180-POI 旧管线遗留：**

| 文件 | 规模 | 说明 |
|------|------|------|
| `poi_xhs_popularity.npy` | [180] float64 | 180-POI 规模的 XHS 热度（旧管线，未参与当前训练） |

### POI 类别与活动类型分布（当前规模）

| 类别 | 活动类型 | 数量 | 占比 |
|------|----------|------|------|
| 餐饮 | 餐饮(1) | 4,091 | 40.9% |
| 住宿 | 住宿(2) | 2,000 | 20.0% |
| 景点 | 景点(0) | 1,923 | 19.2% |
| 购物 | 购物(4) | 1,500 | 15.0% |
| 交通 | 出发点(5) | 486 | 4.9% |
| **合计** | — | **10,000** | **100%** |

> 配额筛选目标比例为「景点 0.35 / 餐饮 0.25 / 住宿 0.20 / 购物 0.15 / 交通 0.05」，
> 但原始景点仅 2,317 个未填满 35% 配额，余额按综合分补给了餐饮，故餐饮占比最高。

### 训练数据划分

- **总路线：** 5,168 条（train 4,134 / val 517 / test 517，比例 0.8 / 0.1 / 0.1）

## 训练

```bash
# 默认配置（当前规模：3层, d_ff=384, 4.57M参数）
uv run python -m src.train --config configs/default.yaml

# 恢复训练
uv run python -m src.train --config configs/default.yaml --resume checkpoints/best_model.pt

# TensorBoard 实时监控
uv run tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
```

模型参数量：4,569,976。训练特性：AMP 混合精度、共享数据加载（编码器预计算 + 输出共享）、Teacher Forcing + Scheduled Sampling、早停（patience=15）、tqdm 终端进度条 + TensorBoard 双轨日志。

## 评估实验

```bash
# 方向B：真实数据泛化性（168条XHS holdout上的next-POI预测）
uv run python -m scripts.evaluate_on_real --checkpoint checkpoints/best_model.pt
# 结果：Transformer top1=47.77%，比最近邻高24.7倍

# 方向C：与启发式/OR方法对比（5方法，含composite v1/v2）
uv run python -m scripts.run_baselines --checkpoint checkpoints/best_model.pt
# 结果：v2指标下Transformer(0.8727) > NN/OR-Tools(0.7619)，节奏优势显现

# 方向A：优化器对比（AdamW vs Muon）
uv run python -m scripts.run_optimizer_comparison
# 结果：AdamW(4.88) 显著优于 Muon(5.93)，Muon已降级为探索性尝试

# 生成论文图表
uv run python scripts/plot_optimizer_comparison.py
uv run python scripts/plot_baseline_comparison.py
```

## 推理

```bash
# 一日游（从景点出发）
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season winter

# 从酒店出发
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start_id 104 --season winter

# 限时路线
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start "太阳岛" --season summer --max_hours 6

# 多日游
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season winter --max_stops 14 --days 2

# 多条候选对比
uv run python -m src.inference --checkpoint checkpoints/best_model.pt --season winter --n_routes 3
```

输出：终端路线详情表 + 活动类型序列 + 综合评估得分 + `output/best_route_map.html` 交互式地图。

## 评估

```bash
# 综合评估：路线生成质量（composite v1/v2 + 与启发式对比）
uv run python -m scripts.run_baselines --checkpoint checkpoints/best_model.pt
# 真实数据 next-POI 准确率（168条XHS holdout）
uv run python -m scripts.evaluate_on_real --checkpoint checkpoints/best_model.pt
```

四维评价体系：距离(0.30) + 时间(0.25) + 满意度(0.25) + 多样性(0.20)。

## 测试

```bash
uv run pytest tests/ -v
```

## 项目结构

```
├── configs/
│   ├── default.yaml          # 主配置（data/model/engram/mhc/loss/optimizer/training）
│   └── ablation.yaml         # 消融实验配置
├── data/
│   ├── raw/                  # 原始数据（CSV + XLSX + JSONL）
│   └── processed/            # 处理后数据（npy + csv）
├── scripts/
│   ├── prepare_real_data.py  # POI增强 + 路线增强 + 聚类
│   ├── process_xhs_data.py   # XHS笔记清洗 + POI热度提取
│   └── merge_data.py         # 高德百度数据合并去重
├── src/
│   ├── data/
│   │   ├── preprocess.py     # 数据清洗 + 特征工程 + 概率分布矩阵
│   │   └── dataset.py        # PyTorch Dataset + 概率采样
│   ├── models/
│   │   ├── transformer.py    # ItineraryTransformer（约束Beam Search）
│   │   ├── encoder.py        # Graph-aware Encoder + 活动类型偏置
│   │   ├── decoder.py        # Engram Decoder + 活动类型条件 + 转换约束
│   │   ├── engram.py         # Engram 记忆模块
│   │   ├── mhc.py            # Poincaré 球双曲嵌入
│   │   ├── embeddings.py     # POI + 活动类型 + 位置编码
│   │   └── losses.py         # CE + 距离惩罚 + MHC 正则
│   ├── optim/
│   │   └── muon.py           # Muon + AdamW 混合优化器
│   ├── train.py              # 训练主脚本
│   ├── evaluate.py           # 四维评估
│   ├── inference.py          # 推理 + 路线优化 + 多日游
│   └── visualize.py          # folium交互地图 + 沿道路绘制
└── tests/                    # pytest 单元测试
```

## 当前训练结果（POI 规模 10,000）

- **数据规模：** 10,000 个 POI，5,168 条路线（train 4,134 / val 517 / test 517）
- **模型：** 4,569,976 参数（3 层编解码器, d_ff=384）
- **最佳模型：** epoch 106，val_loss=4.9041（早停于 epoch 121）
- **训练配置：** batch_size=256, AMP 混合精度, patience=15, AdamW
- **loss 曲线：** train 9.06→2.79, val 8.76→4.90
- **GPU 显存峰值：** ~3.27 GB（RTX 4090 24GB）
- **90 个步行景点团（≤1km）**
- **消融实验：** 7 组（K=3/5/10、-Engram、-MHC、-Engram-MHC、纯 Transformer 基线，180-POI 规模）
