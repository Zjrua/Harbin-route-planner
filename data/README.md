# 数据目录说明

## raw/ — 原始数据（已入 git，2026-09-04，约 39MB）

当前管线数据（处理链: 原始 → merged → labeled → processed/）:

| 文件 | 内容 | 谱系 |
|---|---|---|
| `哈尔滨POI数据_完整版.csv`(在 legacy/) | 高德导出 5,740 POI | → merged_pois.csv 的高德源 |
| `百度poi(旅游相关) - 全集.xlsx` | 百度旅游 POI(12MB) | → merged_pois.csv 的百度源 |
| `merged_pois.csv` | 两源合并 48,961 POI | `archive/legacy_scripts/merge_data.py` |
| `merged_pois_labeled.csv` | +语义标签(is_tourism/suitable_*) | `scripts/label_poi_semantics.py`；`src/itinerary_planner.py` 运行时依赖 |
| `search_contents_2026-05-0{4,5}.jsonl` | 小红书笔记 1,832 条 | → routes.npy / 168 条 holdout / 人气分 |
| `search_comments_2026-05-04.jsonl` | 小红书评论 10,127 条 | 笔记+评论=11,959，对上 xhs_processing_report 口径 |

## raw/legacy/ — 早期原型(2025 上半年)数据

`哈尔滨POI_核心节点.csv`(136 节点)、`哈尔滨旅游路线数据.csv`(404 路线, 早期爬取)、
`距离矩阵_公里.csv`/`耗时矩阵_分钟.csv`(136×136, 原型期矩阵)、`哈尔滨POI数据_完整版.csv`。
仅 `archive/` 遗留脚本引用（路径已同步改为 legacy/），当前管线不再读取。

## processed/ — 模型输入数据（已入 git）

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
