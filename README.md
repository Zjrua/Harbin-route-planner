# 哈尔滨文旅线路优化模型

基于 Transformer 架构的哈尔滨旅游路线智能规划系统，2026 年全国大学生统计建模大赛参赛作品。

## 创新点

1. **Engram 记忆机制** — 外部内容寻址记忆库，从历史优质路线中检索相似路线辅助决策
2. **MHC 双曲流形约束** — 庞加莱球模型嵌入，在双曲空间中保持地理拓扑关系
3. **Muon 优化器** — Newton-Schulz quintic 迭代矩阵正交化 + Nesterov 动量 + AdamW 1D 参数回退
4. **随机图建模** — POI 间距离和通行时间建模为概率分布 N(mean, std)，训练时采样提升鲁棒性
5. **活动类型条件生成** — 6种活动类型约束解码，景点→餐饮→住宿自然节奏
6. **POI步行聚类** — ≤1km连通图聚类，15个景点团，同团不重复访问
7. **XHS数据增强** — 11,959条小红书笔记提取餐饮/住宿热度，加权路线增强

## 环境安装

```bash
# 使用 uv 管理环境（推荐）
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 数据准备

```bash
# 1. 提取小红书POI热度
uv run python scripts/process_xhs_data.py

# 2. POI增强 + 路线增强（餐饮/住宿插入 + 聚类）
uv run python scripts/prepare_real_data.py

# 输出至 data/processed/:
#   poi_metadata.csv      POI 元信息（含XHS热度）
#   poi_features.npy      特征矩阵 [180, 128]
#   adjacency.npy         邻接矩阵（距离衰减权重）
#   distance_matrix.npy   距离矩阵 [180, 180]
#   distance_std.npy      距离标准差
#   time_matrix.npy       时间矩阵
#   time_std.npy          时间标准差
#   poi_activity_types.npy 活动类型标签
#   cluster_id.npy        景点步行聚类
#   routes.npy            增强路线（含餐饮/住宿）
```

数据来源：
- 135个核心POI + merged_pois补充餐饮/购物 = 180个
- 403条小红书路线（→ 165条去重后增强）
- 11,959条小红书笔记（餐饮/住宿热度）
- POI分类自动修正：约40个误分类（酒店/民宿标为景点）

类别分布：景点 60、住宿 64、餐饮 37、购物 15、交通 4

## 训练

```bash
# 默认配置（精简模型：3层, d_ff=384, 1.4M参数）
uv run python -m src.train --config configs/default.yaml

# 恢复训练
uv run python -m src.train --resume checkpoints/best_model.pt

# TensorBoard 实时监控
uv run tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
```

模型参数量：1,417,756。训练特性：Teacher Forcing + Scheduled Sampling、早停（patience=10）、tqdm 终端进度条 + TensorBoard 双轨日志。

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
│   │   ├── transformer.py    # RouteTransformer（约束Beam Search）
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

## 当前训练结果

- **val_loss：** 1.9562（epoch 55 早停）
- **模型：** 1,417,756 参数（3层编解码器, d_ff=384）
- **一日游最优得分：** 0.897
- **路线示例：** 中央大街→老厨家锅包肉(餐)→圣索菲亚→省博物馆→墨记(餐)→中华巴洛克→住宿
