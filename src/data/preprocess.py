"""数据清洗与特征工程.

支持随机图建模：POI 间边权（距离/耗时）视为概率分布 N(mean, std)，
训练时采样增加鲁棒性，推理时用均值或蒙特卡洛估计置信区间。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# 哈尔滨大区域范围（含亚布力、雪乡等远郊景点）
HB_LAT_RANGE = (44.2, 46.6)
HB_LNG_RANGE = (125.8, 130.2)

# 随机图建模：不同距离段的基础变异系数 (std/mean)
# 近距离交通波动大（堵车），远距离相对稳定
DISTANCE_CV = {
    "near": 0.20,    # < 10km
    "mid": 0.15,     # 10-50km
    "far": 0.10,     # > 50km
}

# 季节权重模板：按类别 + 是否冰雪项目
SEASON_WEIGHTS = {
    "景点": {"winter": 0.9, "summer": 0.7},
    "餐饮": {"winter": 0.8, "summer": 0.9},
    "住宿": {"winter": 0.8, "summer": 0.8},
    "交通": {"winter": 0.6, "summer": 0.7},
    "购物": {"winter": 0.7, "summer": 0.8},
}

# 冰雪关键词（冬季权重提升）
WINTER_KEYWORDS = ["冰雪", "滑雪", "雪", "冰", "雪乡", "雪谷", "亚布力"]


def haversine(lat1: np.ndarray, lon1: np.ndarray,
              lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Haversine 公式计算两点间的球面距离（km）."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _normalize_column_name(col: str) -> str:
    """统一列名映射."""
    mapping = {
        "名称": "name", "经度": "lng", "纬度": "lat",
        "类别": "category", "评分": "rating", "热度": "popularity",
        "人均消费": "avg_cost", "地址": "address", "POI类型": "poi_type",
    }
    return mapping.get(col.strip(), col.strip())


def _parse_value(val) -> float:
    """安全解析数值，处理 '[]' 等缺失占位符."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if val in ("[]", "", "null", "NaN", "None"):
            return np.nan
        try:
            return float(val)
        except ValueError:
            return np.nan
    return np.nan


def _is_winter_poi(name: str, poi_type: str = "") -> bool:
    """判断 POI 是否为冰雪/冬季特色项目."""
    text = str(name) + str(poi_type)
    return any(kw in text for kw in WINTER_KEYWORDS)


def load_raw_poi_data(path: str) -> pd.DataFrame:
    """加载原始 POI 数据，统一列名和数据类型."""
    if path.endswith(".xlsx") or path.endswith(".xls"):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, encoding="utf-8-sig")

    raw.columns = [_normalize_column_name(c) for c in raw.columns]

    # 解析数值列
    for col in ["lat", "lng", "rating", "popularity", "avg_cost"]:
        if col in raw.columns:
            raw[col] = raw[col].apply(_parse_value)

    return raw


def clean_poi_data(raw_pois: pd.DataFrame, max_pois: int = 150) -> pd.DataFrame:
    """清洗 POI 原始数据：去重、缺失值处理、按质量筛选.

    保留哈尔滨大区域范围（含远郊），按评分+类别均衡筛选到 max_pois.
    """
    df = raw_pois.copy()

    # 必需字段检查
    for col in ["name", "lat", "lng"]:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")

    # 去除坐标缺失的行
    df = df.dropna(subset=["lat", "lng"])

    # 去重（按名称，保留评分最高的）
    if "rating" in df.columns:
        df = df.sort_values("rating", ascending=False).drop_duplicates(subset=["name"])
    else:
        df = df.drop_duplicates(subset=["name"])

    # 坐标范围过滤（大区域，保留郊区景点）
    df = df[
        (df["lat"] >= HB_LAT_RANGE[0]) & (df["lat"] <= HB_LAT_RANGE[1]) &
        (df["lng"] >= HB_LNG_RANGE[0]) & (df["lng"] <= HB_LNG_RANGE[1])
    ]

    # 统一类别字段
    if "category" not in df.columns and "poi_type" in df.columns:
        df["category"] = df["poi_type"].apply(lambda s: str(s).split(";")[0] if pd.notna(s) else "其他")

    # 类别归并
    cat_map = {
        "餐饮服务": "餐饮", "住宿服务": "住宿", "风景名胜": "景点",
        "交通设施服务": "交通", "购物服务": "购物", "体育休闲服务": "景点",
    }
    df["category"] = df["category"].apply(lambda c: cat_map.get(str(c), str(c)))
    # 归并小类别到"其他"
    top_cats = {"景点", "餐饮", "住宿", "交通", "购物"}
    df["category"] = df["category"].apply(lambda c: c if c in top_cats else "其他")

    # 评分处理：保留原始评分的区分度，仅对真正缺失的做填充
    if "rating" in df.columns:
        # 评分 0 视为缺失
        df.loc[df["rating"] == 0, "rating"] = np.nan
        # 用同类别的中位数填充缺失评分
        cat_medians = df.groupby("category")["rating"].transform("median")
        df["rating"] = df["rating"].fillna(cat_medians)
        df["rating"] = df["rating"].fillna(4.0)
    else:
        df["rating"] = 4.0

    # 热度：优先用评论数（百度数据），缺失时用评分近似
    if "comment_num" in df.columns:
        df["popularity"] = df["comment_num"].fillna(0).astype(float)
    elif "popularity" not in df.columns:
        df["popularity"] = df["rating"] * 20
    else:
        df["popularity"] = df["popularity"].fillna(df["rating"] * 20)

    # 季节权重：基于类别 + 冰雪关键词
    poi_type_col = df.get("poi_type", pd.Series([""] * len(df)))
    df["season_winter"] = df.apply(
        lambda row: 1.0 if _is_winter_poi(row["name"], str(poi_type_col.iloc[row.name] if row.name < len(poi_type_col) else ""))
        else SEASON_WEIGHTS.get(row["category"], SEASON_WEIGHTS["景点"])["winter"],
        axis=1,
    )
    df["season_summer"] = df.apply(
        lambda row: 0.3 if _is_winter_poi(row["name"], str(poi_type_col.iloc[row.name] if row.name < len(poi_type_col) else ""))
        else SEASON_WEIGHTS.get(row["category"], SEASON_WEIGHTS["景点"])["summer"],
        axis=1,
    )

    # 按质量筛选到 max_pois：旅游导向的类别配额
    if len(df) > max_pois:
        # 过滤评分低于 3.5 的低质量 POI
        df_high = df[df["rating"] >= 3.5]

        if len(df_high) <= max_pois:
            df = df_high
        else:
            # 旅游导向配额：景点为主，餐饮/住宿/购物配套
            target_ratios = {
                "景点": 0.35, "餐饮": 0.25, "住宿": 0.20,
                "购物": 0.15, "交通": 0.05,
            }
            cat_counts = df_high["category"].value_counts()

            # 按目标比例分配，不超过该类别实际数量
            quota_per_cat = {}
            remaining = max_pois
            for cat, ratio in target_ratios.items():
                n_cat = min(int(cat_counts.get(cat, 0)), int(max_pois * ratio))
                quota_per_cat[cat] = n_cat
                remaining -= n_cat

            # 处理未在 target_ratios 中的类别
            for cat in cat_counts.index:
                if cat not in quota_per_cat:
                    quota_per_cat[cat] = min(int(cat_counts[cat]), max(20, remaining // len(cat_counts)))
                    remaining -= quota_per_cat[cat]

            # 如果总配额不够（某些类别数据不足），把余额分给够数的类别
            if remaining > 0:
                for cat in sorted(quota_per_cat, key=lambda c: cat_counts.get(c, 0), reverse=True):
                    can_add = int(cat_counts.get(cat, 0)) - quota_per_cat[cat]
                    add = min(can_add, remaining)
                    if add > 0:
                        quota_per_cat[cat] += add
                        remaining -= add
                    if remaining <= 0:
                        break

            # 各类别按综合分取 top
            df_high = df_high.copy()
            rating_rank = df_high["rating"].rank(pct=True)
            if "popularity" in df_high.columns and df_high["popularity"].max() > 0:
                pop_rank = df_high["popularity"].rank(pct=True)
            else:
                pop_rank = rating_rank
            df_high["sort_score"] = rating_rank * 0.6 + pop_rank * 0.4

            selected = []
            for cat, n_cat in quota_per_cat.items():
                cat_df = df_high[df_high["category"] == cat]
                cat_df = cat_df.sort_values("sort_score", ascending=False)
                selected.append(cat_df.head(n_cat))

            df = pd.concat(selected).head(max_pois)

    df = df.reset_index(drop=True)
    return df


def extract_poi_features(pois: pd.DataFrame, d_model: int = 128) -> np.ndarray:
    """提取 POI 特征向量并投影到 d_model 维度.

    特征维度：rating(1) + category_onehot(n_cat) + lat_norm(1) + lng_norm(1)
              + popularity(1) + season_winter(1) + season_summer(1) = raw_dim
    再通过随机投影矩阵映射到 d_model。
    """
    n = len(pois)
    features = []

    # 评分归一化到 [0, 1]
    if "rating" in pois.columns:
        ratings = pois["rating"].values.astype(np.float32)
        ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-8)
    else:
        ratings = np.ones(n, dtype=np.float32)
    features.append(ratings[:, None])

    # 类别 one-hot
    if "category" in pois.columns:
        categories = pois["category"].astype(str).values
        unique_cats = sorted(set(categories))
        cat_map = {c: i for i, c in enumerate(unique_cats)}
        onehot = np.zeros((n, len(unique_cats)), dtype=np.float32)
        for i, c in enumerate(categories):
            onehot[i, cat_map[c]] = 1.0
    else:
        onehot = np.ones((n, 1), dtype=np.float32)
    features.append(onehot)

    # 经纬度归一化
    lat_norm = (pois["lat"].values - pois["lat"].min()) / (pois["lat"].max() - pois["lat"].min() + 1e-8)
    lng_norm = (pois["lng"].values - pois["lng"].min()) / (pois["lng"].max() - pois["lng"].min() + 1e-8)
    features.append(lat_norm[:, None].astype(np.float32))
    features.append(lng_norm[:, None].astype(np.float32))

    # 热度归一化
    if "popularity" in pois.columns:
        pop = pois["popularity"].values.astype(np.float32)
        pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-8)
    else:
        pop = np.ones(n, dtype=np.float32) * 0.5
    features.append(pop[:, None])

    # 季节权重
    features.append(pois["season_winter"].values[:, None].astype(np.float32))
    features.append(pois["season_summer"].values[:, None].astype(np.float32))

    raw = np.concatenate(features, axis=1)  # [n, raw_dim]
    raw = normalize_features(raw)

    # 投影到 d_model 维度
    raw_dim = raw.shape[1]
    if raw_dim < d_model:
        # 随机正交投影补齐
        np.random.seed(42)
        proj = np.random.randn(raw_dim, d_model).astype(np.float32) / np.sqrt(raw_dim)
        return raw @ proj
    elif raw_dim > d_model:
        np.random.seed(42)
        proj = np.random.randn(raw_dim, d_model).astype(np.float32) / np.sqrt(d_model)
        return raw @ proj
    return raw


def normalize_features(features: np.ndarray) -> np.ndarray:
    """StandardScaler 标准化."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std


def build_distance_matrix(pois: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """从坐标构建 Haversine 距离矩阵 + 概率分布标准差矩阵.

    Returns:
        (mean, std): 距离均值矩阵 [n, n] (km), 标准差矩阵 [n, n] (km)
        std 按距离段使用不同变异系数：近距波动大、远距波动小。
    """
    n = len(pois)
    lats = pois["lat"].values.astype(np.float64)
    lngs = pois["lng"].values.astype(np.float64)

    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dist[i] = haversine(lats[i], lngs[i], lats, lngs).astype(np.float32)
    np.fill_diagonal(dist, 0.0)

    # 标准差 = 均值 * 变异系数，按距离段区分
    std = np.zeros_like(dist)
    std[dist <= 10] = dist[dist <= 10] * DISTANCE_CV["near"]
    std[(dist > 10) & (dist <= 50)] = dist[(dist > 10) & (dist <= 50)] * DISTANCE_CV["mid"]
    std[dist > 50] = dist[dist > 50] * DISTANCE_CV["far"]
    np.fill_diagonal(std, 0.0)

    return dist, std


def build_time_matrix(distance_matrix: np.ndarray,
                      distance_std: np.ndarray = None,
                      urban_speed: float = 30.0,
                      suburban_speed: float = 60.0,
                      threshold_km: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """从距离矩阵估算耗时矩阵 + 标准差矩阵.

    城区速度和郊区速度不同，近距离交通波动更大。
    标准差 = 时间均值 * 变异系数（近距0.25, 远距0.12）。

    Returns:
        (mean, std): 时间均值矩阵 [n, n] (分钟), 标准差矩阵 [n, n] (分钟)
    """
    speed = np.where(distance_matrix <= threshold_km, urban_speed, suburban_speed)
    time_mean = (distance_matrix / speed * 60).astype(np.float32)

    # 时间标准差：近距波动大（红绿灯/堵车），远距波动小（高速稳定）
    time_cv = np.where(distance_matrix <= 10, 0.25,
                       np.where(distance_matrix <= 50, 0.18, 0.12))
    time_std = (time_mean * time_cv).astype(np.float32)
    np.fill_diagonal(time_std, 0.0)

    return time_mean, time_std


def build_adjacency(distance_matrix: np.ndarray,
                    max_distance_km: float = 20.0) -> np.ndarray:
    """基于距离阈值构建邻接矩阵."""
    adj = (distance_matrix < max_distance_km).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def generate_synthetic_routes(n_pois: int, dist_matrix: np.ndarray,
                              n_routes: int = 500,
                              min_len: int = 3, max_len: int = 12,
                              seed: int = 42) -> np.ndarray:
    """生成基于距离的启发式路线（比纯随机更贴近真实游览模式）.

    策略：随机选起点，每步倾向于选距离近且评分高的下一个 POI。
    """
    rng = np.random.RandomState(seed)
    routes = []
    n_pois_actual = dist_matrix.shape[0]

    for _ in range(n_routes):
        length = rng.randint(min_len, max_len + 1)
        start = rng.randint(0, n_pois_actual)
        route = [start]
        visited = {start}

        for _ in range(length - 1):
            candidates = [i for i in range(n_pois_actual) if i not in visited]
            if not candidates:
                break
            # 距离倒数作为权重（近的更可能被选中）
            dists = dist_matrix[route[-1], candidates]
            weights = 1.0 / (dists + 1.0)
            weights = weights / weights.sum()
            next_poi = rng.choice(candidates, p=weights)
            route.append(next_poi)
            visited.add(next_poi)

        routes.append(np.array(route))
    return np.array(routes, dtype=object)


def build_adjacency(distance_matrix: np.ndarray,
                    max_distance_km: float = 50.0) -> np.ndarray:
    """基于距离阈值构建邻接矩阵.

    阈值较大（50km）以覆盖郊区景点间的连通关系。
    远距离边的权重按距离衰减。
    """
    dist = distance_matrix
    # 连通性：距离内可达
    connected = (dist > 0) & (dist < max_distance_km)
    # 权重：距离越近权重越高
    weights = np.where(connected, 1.0 - dist / max_distance_km, 0.0).astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    return weights


def prepare_all(raw_csv_path: str, output_dir: str = "data/processed",
                d_model: int = 128, max_pois: int = 150,
                max_distance_km: float = 50.0,
                n_synthetic_routes: int = 500) -> None:
    """一键完成全流程：加载 → 清洗 → 筛选 → 特征 → 概率分布矩阵 → 路线 → 保存.

    输出文件:
    - poi_metadata.csv — POI 元信息
    - poi_features.npy — 特征矩阵 [n_pois, d_model]
    - adjacency.npy — 邻接矩阵（带衰减权重）
    - distance_matrix.npy — 距离均值 [n_pois, n_pois] (km)
    - distance_std.npy — 距离标准差 [n_pois, n_pois]
    - time_matrix.npy — 时间均值 [n_pois, n_pois] (分钟)
    - time_std.npy — 时间标准差 [n_pois, n_pois]
    - routes.npy — 历史路线
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 读取原始数据
    raw = load_raw_poi_data(raw_csv_path)
    print(f"读取 {len(raw)} 条原始 POI 数据")

    # 清洗 + 筛选
    pois = clean_poi_data(raw, max_pois=max_pois)
    n_pois = len(pois)
    print(f"清洗筛选后 {n_pois} 条 POI")
    print(f"  类别分布: {pois['category'].value_counts().to_dict()}")

    # 特征提取
    features = extract_poi_features(pois, d_model)
    print(f"特征矩阵: {features.shape}")

    # 距离矩阵（均值 + 标准差）
    dist_mean, dist_std = build_distance_matrix(pois)
    print(f"距离矩阵: 均值 {dist_mean[dist_mean > 0].mean():.1f}km, "
          f"std范围 [{dist_std[dist_std > 0].min():.2f}, {dist_std[dist_std > 0].max():.2f}]")

    # 耗时矩阵（均值 + 标准差）
    time_mean, time_std = build_time_matrix(dist_mean, dist_std)
    print(f"耗时矩阵: 均值 {time_mean[time_mean > 0].mean():.1f}min")

    # 邻接矩阵
    adj = build_adjacency(dist_mean, max_distance_km)
    print(f"邻接矩阵: 连通边数 {(adj > 0).sum()} / {n_pois * n_pois}")

    # 路线数据
    routes_path = out / "routes.npy"
    if routes_path.exists():
        routes = np.load(routes_path, allow_pickle=True)
        print(f"使用已有路线数据: {len(routes)} 条")
    else:
        routes = generate_synthetic_routes(n_pois, dist_mean, n_synthetic_routes)
        print(f"生成启发式路线: {len(routes)} 条")

    # 保存
    pois.to_csv(out / "poi_metadata.csv", index=False, encoding="utf-8")
    np.save(out / "poi_features.npy", features)
    np.save(out / "adjacency.npy", adj)
    np.save(out / "distance_matrix.npy", dist_mean)
    np.save(out / "distance_std.npy", dist_std)
    np.save(out / "time_matrix.npy", time_mean)
    np.save(out / "time_std.npy", time_std)
    np.save(out / "routes.npy", routes)

    print(f"\n全部数据已保存至 {out}/")
    print(f"  poi_metadata.csv   — {n_pois} POI")
    print(f"  poi_features.npy   — {features.shape}")
    print(f"  adjacency.npy      — {adj.shape}")
    print(f"  distance_matrix.npy— {dist_mean.shape} (均值)")
    print(f"  distance_std.npy   — {dist_std.shape} (标准差)")
    print(f"  time_matrix.npy    — {time_mean.shape} (均值)")
    print(f"  time_std.npy       — {time_std.shape} (标准差)")
    print(f"  routes.npy         — {len(routes)} 条路线")
