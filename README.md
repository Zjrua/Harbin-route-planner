# 哈尔滨文旅线路优化模型

基于 Transformer 架构的哈尔滨旅游路线智能规划系统，2026 年全国大学生统计建模大赛参赛作品。

## 创新点

融合 DeepSeek 论文中的三项关键技术：

1. **Engram 记忆机制** — 外部内容寻址记忆库，从历史优质路线中检索相似路线辅助决策
2. **MHC 双曲流形约束** — 庞加莱球模型嵌入，在双曲空间中更好地保持地理拓扑关系
3. **Muon 优化器** — 矩阵正交化梯度更新，分层学习率提升 Transformer 训练效率

## 环境安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

```bash
# 1. 在 .env 中配置高德地图 API Key
echo "AMAP_API_KEY=your_key_here" > .env

# 2. 爬取 POI 数据
bash scripts/crawl_data.sh

# 3. 数据预处理（在 notebooks/ 中完成特征工程后）
# 将处理后的数据保存至 data/processed/
```

## 训练

```bash
# 使用默认配置训练
python -m src.train --config configs/default.yaml

# 恢复训练
python -m src.train --config configs/default.yaml --resume checkpoints/best_model.pt

# 指定 GPU
python -m src.train --device cuda:0
```

## 评估

```bash
# 评估模型性能
python -m src.evaluate --checkpoint checkpoints/best_model.pt
```

## 消融实验

```bash
# 运行全部消融实验
bash scripts/run_ablation.sh
```

## 项目结构

```
harbin-route-planner/
├── configs/          # 超参配置（default.yaml, ablation.yaml）
├── data/             # 数据目录（raw/ 原始, processed/ 处理后）
├── src/              # 源代码
│   ├── data/         # 数据处理（爬取、预处理、Dataset）
│   ├── models/       # 模型（Transformer、Engram、MHC、Embedding、Loss）
│   ├── optim/        # 优化器（Muon）
│   ├── train.py      # 训练主脚本
│   ├── evaluate.py   # 评估脚本
│   └── visualize.py  # 可视化
├── notebooks/        # Jupyter 探索性分析
├── scripts/          # Shell 运行脚本
└── tests/            # 单元测试
```
