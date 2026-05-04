"""数据清洗与特征工程."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def haversine(lat1: np.ndarray, lon1: np.ndarray,
              lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Haversine 公式计算两点间的球面距离（km）."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def clean_poi_data(raw_pois: pd.DataFrame) -> pd.DataFrame:
    """清洗 POI 原始数据：去重、缺失值处理、坐标异常过滤."""
    df = raw_pois.copy()

    # 去重（按 name + lat + lng）
    df = df.drop_duplicates(subset=["name", "lat", "lng"])

    # 填充缺失评分
    if "rating" in df.columns:
        df["rating"] = df["rating"].fillna(df["rating"].median())

    # 填充缺失热度
    if "popularity" in df.columns:
        df["popularity"] = df["popularity"].fillna(df["popularity"].median())

    # 坐标过滤：哈尔滨范围 ~[45.0, 46.5] x [125.5, 127.5]
    df = df[(df["lat"] >= 45.0) & (df["lat"] <= 46.5)]
    df = df[(df["lng"] >= 125.5) & (df["lng"] <= 127.5)]

    # 填充季节权重默认值
    for col in ["season_winter", "season_summer"]:
        if col not in df.columns:
            df[col] = 1.0
        else:
            df[col] = df[col].fillna(1.0)

    # 重置索引作为 poi_id
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


def build_distance_matrix(pois: pd.DataFrame) -> np.ndarray:
    """从坐标构建 Haversine 距离矩阵（km）."""
    n = len(pois)
    lats = pois["lat"].values.astype(np.float64)
    lngs = pois["lng"].values.astype(np.float64)

    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dist[i] = haversine(lats[i], lngs[i], lats, lngs).astype(np.float32)

    # 对角线置零
    np.fill_diagonal(dist, 0.0)
    return dist


def build_time_matrix(distance_matrix: np.ndarray, avg_speed_kmh: float = 30.0) -> np.ndarray:
    """从距离矩阵估算耗时矩阵（分钟）."""
    return (distance_matrix / avg_speed_kmh * 60).astype(np.float32)


def build_adjacency(distance_matrix: np.ndarray,
                    max_distance_km: float = 20.0) -> np.ndarray:
    """基于距离阈值构建邻接矩阵."""
    adj = (distance_matrix < max_distance_km).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def generate_synthetic_routes(n_pois: int, n_routes: int = 500,
                              min_len: int = 3, max_len: int = 15,
                              seed: int = 42) -> np.ndarray:
    """生成随机历史路线（用于测试/启动训练）."""
    rng = np.random.RandomState(seed)
    routes = []
    for _ in range(n_routes):
        length = rng.randint(min_len, max_len + 1)
        route = rng.choice(n_pois, size=length, replace=False)
        routes.append(route)
    return np.array(routes, dtype=object)


def prepare_all(raw_csv_path: str, output_dir: str = "data/processed",
                d_model: int = 128, max_distance_km: float = 20.0,
                n_synthetic_routes: int = 500,
                existing_distance_matrix: Optional[str] = None,
                existing_time_matrix: Optional[str] = None) -> None:
    """一键完成全流程：清洗 → 特征 → 距离矩阵 → 邻接 → 路线 → 保存.

    Args:
        raw_csv_path: 原始 POI CSV/Excel 文件路径
        output_dir: 输出目录
        d_model: 特征投影维度
        max_distance_km: 邻接矩阵距离阈值
        n_synthetic_routes: 生成的随机路线数量（没有历史路线时使用）
        existing_distance_matrix: 已有距离矩阵文件路径（npy格式），若提供则不自动计算
        existing_time_matrix: 已有耗时矩阵文件路径（npy格式）
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 读取原始数据
    if raw_csv_path.endswith(".xlsx") or raw_csv_path.endswith(".xls"):
        raw = pd.read_excel(raw_csv_path)
    else:
        raw = pd.read_csv(raw_csv_path, encoding="utf-8")

    print(f"读取 {len(raw)} 条原始 POI 数据")

    # 清洗
    pois = clean_poi_data(raw)
    n_pois = len(pois)
    print(f"清洗后 {n_pois} 条 POI")

    # 截断到 max_pois
    if n_pois > 150:
        pois = pois.head(150).reset_index(drop=True)
        n_pois = 150
        print(f"截断到 {n_pois} 条 POI (max_pois=150)")

    # 特征提取
    features = extract_poi_features(pois, d_model)
    print(f"特征矩阵: {features.shape}")

    # 距离矩阵
    if existing_distance_matrix:
        dist = np.load(existing_distance_matrix).astype(np.float32)
        # 裁剪/填充到 n_pois
        if dist.shape[0] > n_pois:
            dist = dist[:n_pois, :n_pois]
        print(f"使用已有距离矩阵: {dist.shape}")
    else:
        dist = build_distance_matrix(pois)
        print(f"距离矩阵: {dist.shape}")

    # 耗时矩阵
    if existing_time_matrix:
        time_mat = np.load(existing_time_matrix).astype(np.float32)
        if time_mat.shape[0] > n_pois:
            time_mat = time_mat[:n_pois, :n_pois]
        print(f"使用已有耗时矩阵: {time_mat.shape}")
    else:
        time_mat = build_time_matrix(dist)
        print(f"耗时矩阵（估算）: {time_mat.shape}")

    # 邻接矩阵
    adj = build_adjacency(dist, max_distance_km)
    print(f"邻接矩阵: {adj.shape}, 连通边数: {(adj > 0).sum()}")

    # 路线数据
    routes_path = out / "routes.npy"
    if routes_path.exists():
        routes = np.load(routes_path, allow_pickle=True)
        print(f"使用已有路线数据: {len(routes)} 条")
    else:
        routes = generate_synthetic_routes(n_pois, n_synthetic_routes)
        print(f"生成随机路线: {len(routes)} 条")

    # 保存
    pois.to_csv(out / "poi_metadata.csv", index=False, encoding="utf-8")
    np.save(out / "poi_features.npy", features)
    np.save(out / "adjacency.npy", adj)
    np.save(out / "distance_matrix.npy", dist)
    np.save(out / "time_matrix.npy", time_mat)
    np.save(out / "routes.npy", routes)

    print(f"\n全部数据已保存至 {out}/")
    print(f"  poi_metadata.csv   — {n_pois} POI")
    print(f"  poi_features.npy   — {features.shape}")
    print(f"  adjacency.npy      — {adj.shape}")
    print(f"  distance_matrix.npy— {dist.shape}")
    print(f"  time_matrix.npy    — {time_mat.shape}")
    print(f"  routes.npy         — {len(routes)} 条路线")
