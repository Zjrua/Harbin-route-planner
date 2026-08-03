#!/bin/bash
# 数据爬取一键脚本

set -e

echo "=== 哈尔滨文旅数据爬取 ==="

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "错误：未找到 .env 文件，请创建并配置 AMAP_API_KEY"
    exit 1
fi

source .env

# 创建输出目录
mkdir -p data/raw

# 爬取 POI 数据
echo "[1/3] 爬取 POI 数据..."
python -c "
from src.data.poi_crawler import crawl_pois
import pandas as pd
pois = crawl_pois('$AMAP_API_KEY', city='哈尔滨')
pois.to_csv('data/raw/pois_harbin.csv', index=False)
print(f'共获取 {len(pois)} 个 POI')
"

# 构建路网
echo "[2/3] 构建路网数据..."
python -c "
from src.data.road_network import build_adjacency_matrix, build_travel_time_matrix
import pandas as pd, numpy as np
pois = pd.read_csv('data/raw/pois_harbin.csv')
adj = build_adjacency_matrix(pois)
time = build_travel_time_matrix(adj)
np.save('data/raw/adjacency.npy', adj)
np.save('data/raw/time_matrix.npy', time)
print(f'路网矩阵: {adj.shape}')
"

# 爬取评论
echo "[3/3] 爬取游客评论（可选）..."
echo "评论爬取需手动执行，参见 src/data/review_crawler.py"

echo "=== 数据爬取完成 ==="
