# 数据目录说明

## raw/
存放原始爬取数据，已入 git（2026-09-04 起全量提交，约 39MB）。

预期文件：
- `pois_harbin.csv` — 哈尔滨市 POI 数据（高德地图爬取）
- `reviews_*.csv` — 各平台游客评论数据
- `baidu_poi_tourism.xlsx` — 百度 POI 旅游相关数据

## processed/
存放经预处理后的模型输入数据（实际已入 git，与 raw/ 策略不同）。

预期文件：
- `poi_features.npy` — POI 特征矩阵 [n_pois, feature_dim]
- `adjacency.npy` — 路网邻接矩阵 [n_pois, n_pois]
- `distance_matrix.npy` — 距离矩阵 [n_pois, n_pois]
- `time_matrix.npy` — 时间矩阵 [n_pois, n_pois]
- `routes.npy` — 历史路线数据
- `poi_metadata.csv` — POI 元信息（名称、坐标、类别、评分）

## 坐标基准（2026-09-04 修正）

全表已统一 **GCJ-02**（高德/火星坐标）：
- 百度源原为 WGS-84（全量景点实证，转换后中位偏差 11m），已做 `wgs84→gcj02` 纯数学转换；
- 景点类 1923 个已逐个回查高德：1498 个采用高德权威坐标 + uid（实体键，可区分同名连锁），
  100 个漂移 >500m 的标 `needs_review=True` 待人工复核（`poi_amap_verified.csv` / `amap_verify_report.json`）。
- 新增列：`amap_uid, coord_source, match_status, drift_m, needs_review, coord_fixed_on`。
- 复现脚本：`scripts/verify_poi_amap.py`（3 QPS，断点续传）→ `scripts/apply_amap_verification.py`。

## ⚠️ 矩阵滞后警告

`distance_matrix.npy` / `time_matrix.npy` 仍基于**修正前**坐标（Haversine×1.3 + 30/60km/h 估算 +
部分天地图驾车结果）。坐标修正后两者不一致，重建（k 近邻 × 真实路网距离）完成前，
不要将矩阵视为修正后坐标的精确反映。

## 哈尔滨+（进行中）

`neighbor_pois_pilot.csv` — 周边城市风景名胜类(高德 110000)试点采集，
GCJ-02 + uid：伊春 197 / 大庆 221 / 牡丹江 199（2026-09-04，`scripts/collect_neighbor_pois.py`）。
目标：支撑哈尔滨+近邻城市的 3~5 天及以上行程线路。
