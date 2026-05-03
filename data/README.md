# 数据目录说明

## raw/
存放原始爬取数据，不入 git。

预期文件：
- `pois_harbin.csv` — 哈尔滨市 POI 数据（高德地图爬取）
- `reviews_*.csv` — 各平台游客评论数据
- `baidu_poi_tourism.xlsx` — 百度 POI 旅游相关数据

## processed/
存放经预处理后的模型输入数据，不入 git。

预期文件：
- `poi_features.npy` — POI 特征矩阵 [n_pois, feature_dim]
- `adjacency.npy` — 路网邻接矩阵 [n_pois, n_pois]
- `distance_matrix.npy` — 距离矩阵 [n_pois, n_pois]
- `time_matrix.npy` — 时间矩阵 [n_pois, n_pois]
- `routes.npy` — 历史路线数据
- `poi_metadata.csv` — POI 元信息（名称、坐标、类别、评分）
