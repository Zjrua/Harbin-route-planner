"""天地图POI数据清洗与预处理.

将爬取的原始POI数据清洗、分类、筛选，生成模型所需的：
- poi_metadata.csv: POI元信息（名称、坐标、类别、评分）
- poi_features.npy: POI特征矩阵
- distance_matrix.npy: Haversine距离矩阵
- time_matrix.npy: 通行时间矩阵
- adjacency.npy: 路网邻接矩阵
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin
from pathlib import Path


# 搜索关键词 → 旅游类别映射
KEYWORD_CATEGORY_MAP = {
    # 景观类
    "风景区": "景点", "旅游景区": "景点",
    "冰雪大世界": "景点", "太阳岛": "景点",
    "中央大街": "景点", "索菲亚": "景点",
    "极地馆": "景点", "虎林园": "景点",
    "公园": "公园", "广场": "广场",
    # 文化场馆
    "博物馆": "文化场馆", "纪念馆": "文化场馆",
    "展览馆": "文化场馆", "美术馆": "文化场馆", "文化宫": "文化场馆",
    # 宗教建筑
    "教堂": "宗教建筑", "寺庙": "宗教建筑", "清真寺": "宗教建筑",
    # 冰雪特色
    "滑雪场": "冰雪运动", "冰雕": "冰雪运动",
    # 娱乐
    "游乐场": "娱乐", "动物园": "娱乐", "水上乐园": "娱乐", "温泉": "休闲",
    # 餐饮
    "火锅": "餐饮", "烧烤": "餐饮", "俄式餐厅": "餐饮",
    "老字号": "餐饮", "咖啡厅": "餐饮", "酒吧": "餐饮",
    # 住宿
    "酒店": "住宿", "宾馆": "住宿", "民宿": "住宿",
    # 购物
    "商场": "购物", "步行街": "购物", "特产": "购物",
    # 交通
    "火车站": "交通枢纽", "机场": "交通枢纽", "地铁站": "交通枢纽",
}

# 哈尔滨市区经纬度范围（过滤远郊）
URBAN_BOUNDS = {
    "min_lon": 126.30, "max_lon": 127.00,
    "min_lat": 45.55,  "max_lat": 45.95,
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine公式计算两点间距离(km)."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def clean_and_classify(df: pd.DataFrame) -> pd.DataFrame:
    """清洗原始数据并添加分类列."""
    df = df.copy()

    # 添加category列
    df["category"] = df["search_keyword"].map(KEYWORD_CATEGORY_MAP).fillna("其他")

    # 过滤掉无坐标的记录
    df = df.dropna(subset=["lon", "lat"])

    # 过滤到哈尔滨市区范围
    mask = (
        (df["lon"] >= URBAN_BOUNDS["min_lon"]) & (df["lon"] <= URBAN_BOUNDS["max_lon"]) &
        (df["lat"] >= URBAN_BOUNDS["min_lat"]) & (df["lat"] <= URBAN_BOUNDS["max_lat"])
    )
    df = df[mask].copy()

    # 去重（按hotPointID）
    df = df.drop_duplicates(subset=["hotPointID"], keep="first")

    return df.reset_index(drop=True)


def filter_tourism_pois(df: pd.DataFrame, max_pois: int = 150) -> pd.DataFrame:
    """按类别配额筛选旅游相关POI，确保类别多样性.

    配额分配（基于max_pois=150）：
    - 景点: 30, 公园: 15, 广场: 10
    - 文化场馆: 15, 宗教建筑: 10
    - 冰雪运动+娱乐: 10
    - 餐饮: 25, 住宿: 10, 购物: 15
    - 交通枢纽: 10
    """
    quota = {
        "景点": 30, "公园": 15, "广场": 10,
        "文化场馆": 15, "宗教建筑": 10,
        "冰雪运动": 5, "娱乐": 5, "休闲": 3,
        "餐饮": 25, "住宿": 10, "购物": 15,
        "交通枢纽": 10,
    }

    selected = []
    used_ids = set()

    for cat, n in quota.items():
        mask = (df["category"] == cat) & (~df["hotPointID"].isin(used_ids))
        candidates = df[mask].head(n)
        selected.append(candidates)
        used_ids.update(candidates["hotPointID"].tolist())

    result = pd.concat(selected, ignore_index=True)

    # 如果不足max_pois，从剩余POI中补充
    if len(result) < max_pois:
        remaining = df[~df["hotPointID"].isin(result["hotPointID"])]
        extra = remaining.head(max_pois - len(result))
        result = pd.concat([result, extra], ignore_index=True)

    # 如果超过max_pois，截断
    if len(result) > max_pois:
        result = result.head(max_pois)

    return result.reset_index(drop=True)


def assign_mock_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """为POI分配模拟评分（天地图API不提供评分）.

    策略：核心景点4.0-5.0，其他3.5-4.5，加随机噪声。
    """
    np.random.seed(42)
    df = df.copy()

    base_ratings = {
        "景点": 4.3, "公园": 4.2, "文化场馆": 4.4, "宗教建筑": 4.3,
        "冰雪运动": 4.6, "娱乐": 4.1, "广场": 4.0, "餐饮": 4.0,
        "住宿": 3.9, "购物": 3.8, "交通枢纽": 3.7, "休闲": 4.2,
    }

    df["rating"] = df["category"].map(base_ratings).fillna(3.8)
    noise = np.random.uniform(-0.3, 0.3, len(df))
    df["rating"] = (df["rating"] + noise).clip(3.0, 5.0).round(1)

    # 知名景点手动标注高分
    famous = {
        "冰雪大世界": 4.9, "中央大街": 4.8, "圣索菲亚教堂": 4.8,
        "太阳岛": 4.7, "哈尔滨极地馆": 4.6, "东北虎林园": 4.5,
        "防洪纪念塔": 4.5, "哈尔滨大剧院": 4.7, "龙塔": 4.4,
        "索菲亚广场": 4.6, "兆麟公园": 4.3, "哈尔滨游乐园": 4.2,
        "黑龙江省博物馆": 4.5, "哈尔滨极乐寺": 4.4,
        "东北烈士纪念馆": 4.5, "中华巴洛克": 4.5,
    }
    for name, score in famous.items():
        mask = df["name"].str.contains(name, na=False)
        df.loc[mask, "rating"] = score

    return df


def build_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """构建Haversine距离矩阵."""
    n = len(df)
    dist = np.zeros((n, n))
    lats = df["lat"].values
    lons = df["lon"].values

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(lats[i], lons[i], lats[j], lons[j])
            dist[i, j] = d
            dist[j, i] = d

    return dist


def build_adjacency_matrix(dist_matrix: np.ndarray,
                           max_distance_km: float = 50.0) -> np.ndarray:
    """构建邻接矩阵（距离阈值内为1）."""
    adj = (dist_matrix > 0) & (dist_matrix <= max_distance_km)
    return adj.astype(np.float32) * dist_matrix


def build_time_matrix(dist_matrix: np.ndarray,
                      avg_speed_kmh: float = 30.0) -> np.ndarray:
    """估算通行时间矩阵（分钟）."""
    return dist_matrix / avg_speed_kmh * 60.0


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """提取POI特征矩阵.

    特征: lon, lat, rating, category_onehot
    """
    # 连续特征
    cont = df[["lon", "lat", "rating"]].values

    # 类别one-hot
    categories = df["category"].unique()
    cat_to_idx = {c: i for i, c in enumerate(sorted(categories))}
    cat_onehot = np.zeros((len(df), len(categories)))
    for i, cat in enumerate(df["category"]):
        cat_onehot[i, cat_to_idx[cat]] = 1.0

    return np.hstack([cont, cat_onehot]).astype(np.float32)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="POI数据清洗与预处理")
    parser.add_argument("--input", default="data/raw/tianditu_pois_harbin.csv",
                        help="输入CSV路径")
    parser.add_argument("--output-dir", default="data/processed",
                        help="输出目录")
    parser.add_argument("--max-pois", type=int, default=150,
                        help="最大POI数量")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取原始数据
    print(f"读取 {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  原始数据: {len(df)} 条")

    # 2. 清洗分类
    df = clean_and_classify(df)
    print(f"  清洗后(市区+去重): {len(df)} 条")

    # 3. 筛选旅游POI
    df = filter_tourism_pois(df, max_pois=args.max_pois)
    print(f"  筛选后: {len(df)} 条")

    # 4. 模拟评分
    df = assign_mock_ratings(df)

    # 5. 保存元数据
    meta_cols = ["hotPointID", "name", "address", "phone", "lon", "lat",
                 "category", "rating", "search_keyword"]
    metadata = df[meta_cols].copy()
    metadata.index.name = "poi_id"
    metadata_path = output_dir / "poi_metadata.csv"
    metadata.to_csv(metadata_path, encoding="utf-8-sig")
    print(f"\n保存 {metadata_path} ({len(metadata)} 条)")

    # 6. 构建距离矩阵
    print("\n构建距离矩阵 ...")
    dist_matrix = build_distance_matrix(df)
    np.save(output_dir / "distance_matrix.npy", dist_matrix)
    print(f"  距离矩阵: {dist_matrix.shape}, "
          f"最大 {dist_matrix.max():.1f}km, 平均 {dist_matrix[dist_matrix > 0].mean():.2f}km")

    # 7. 邻接矩阵
    adj_matrix = build_adjacency_matrix(dist_matrix)
    np.save(output_dir / "adjacency.npy", adj_matrix)
    print(f"  邻接矩阵: {adj_matrix.shape}")

    # 8. 时间矩阵
    time_matrix = build_time_matrix(dist_matrix)
    np.save(output_dir / "time_matrix.npy", time_matrix)
    print(f"  时间矩阵: {time_matrix.shape}, "
          f"最大 {time_matrix.max():.0f}min, 平均 {time_matrix[time_matrix > 0].mean():.1f}min")

    # 9. 特征矩阵
    features = extract_features(df)
    np.save(output_dir / "poi_features.npy", features)
    print(f"  特征矩阵: {features.shape}")

    # 10. 打印统计
    print(f"\n=== POI类别分布 ===")
    for cat, cnt in df["category"].value_counts().items():
        avg_rating = df[df["category"] == cat]["rating"].mean()
        print(f"  {cat}: {cnt} 条, 平均评分 {avg_rating:.1f}")

    print(f"\n=== 前20个POI ===")
    for _, row in df.head(20).iterrows():
        print(f"  {row['name']:<20s} [{row['category']}] ★{row['rating']} "
              f"({row['lon']:.4f}, {row['lat']:.4f})")

    print(f"\n处理完成! 数据已保存到 {output_dir}/")


if __name__ == "__main__":
    main()
