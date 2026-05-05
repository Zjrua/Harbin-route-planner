"""处理真实路线数据：去重、名称对齐、矩阵修复.

将路线数据中的 POI 简称对齐到核心节点表，
修复距离/时间矩阵中的零距离 bug，生成最终训练数据。

用法:
    uv run python scripts/prepare_real_data.py
"""

import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

# 确保 src 包可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 活动类型映射（与 embeddings.py 中的 ACTIVITY_TYPES 保持一致）
ACTIVITY_TYPE_MAP = {
    "景点": 0,
    "餐饮": 1,
    "住宿": 2,
    "交通": 3,
    "购物": 4,
    "出发点": 5,  # 交通枢纽、机场、火车站等
}

def get_activity_type(category: str, name: str = "") -> int:
    """根据类别和名称判断活动类型."""
    category = str(category).strip()
    name = str(name).strip()

    # 交通枢纽 → 出发点
    if category == "交通":
        return ACTIVITY_TYPE_MAP["出发点"]

    # 住宿
    if category == "住宿":
        return ACTIVITY_TYPE_MAP["住宿"]

    # 餐饮
    if category == "餐饮":
        return ACTIVITY_TYPE_MAP["餐饮"]

    # 购物
    if category == "购物":
        return ACTIVITY_TYPE_MAP["购物"]

    # 景点（默认）
    return ACTIVITY_TYPE_MAP["景点"]


def clean_liked_count(val) -> int:
    """清洗点赞数（处理 '1.2万' 这种格式）."""
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if "万" in s:
        return int(float(s.replace("万", "")) * 10000)
    try:
        return int(float(s))
    except ValueError:
        return 0


def deduplicate_routes(df: pd.DataFrame) -> pd.DataFrame:
    """路线去重：按 note_id 去重，保留 liked_count 最高的."""
    if "liked_count" in df.columns:
        df["liked_count"] = df["liked_count"].apply(clean_liked_count)
        df = df.sort_values("liked_count", ascending=False)
    df = df.drop_duplicates(subset=["note_id"], keep="first")
    # 再按路线内容去重
    df = df.drop_duplicates(subset=["route"], keep="first")
    return df.reset_index(drop=True)


def build_name_index(poi_names: list[str]) -> dict[str, int]:
    """构建 POI 名称到索引的映射，支持别名."""
    name_to_idx = {}
    for i, name in enumerate(poi_names):
        name_to_idx[name.strip()] = i
    return name_to_idx


def normalize_name(name: str) -> str:
    """标准化 POI 名称：去标点、去空格、统一简繁体."""
    name = name.strip()
    # 去掉常见标点和修饰词
    name = re.sub(r"[·\-\(\)（）\s]", "", name)
    name = name.replace("（", "").replace("）", "")
    name = name.replace("哈尔滨市", "哈尔滨")
    name = name.replace("风景区", "")
    name = name.replace("历史文化街区", "")
    name = name.replace("纪念塔广场", "纪念塔")
    name = name.replace("纪念塔", "防洪纪念塔")
    return name


def match_poi(route_name: str, name_to_idx: dict[str, int],
              poi_names: list[str]) -> int | None:
    """将路线中的 POI 名称匹配到核心节点索引.

    策略：
    1. 精确匹配
    2. 标准化后匹配
    3. 包含匹配（路线名包含在核心名中，或反过来）
    4. 模糊匹配（SequenceMatcher >= 0.6）
    """
    route_name = route_name.strip()

    # 1. 精确匹配
    if route_name in name_to_idx:
        return name_to_idx[route_name]

    # 2. 标准化后匹配
    norm_route = normalize_name(route_name)
    for name, idx in name_to_idx.items():
        if normalize_name(name) == norm_route:
            return idx

    # 3. 包含匹配
    for name, idx in name_to_idx.items():
        norm_name = normalize_name(name)
        if norm_route in norm_name or norm_name in norm_route:
            return idx

    # 4. 模糊匹配
    best_score = 0
    best_idx = None
    for name, idx in name_to_idx.items():
        score = SequenceMatcher(None, norm_route, normalize_name(name)).ratio()
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_score >= 0.6:
        return best_idx

    return None


def parse_route(route_str: str, name_to_idx: dict[str, int],
                poi_names: list[str]) -> list[int]:
    """将路线字符串解析为 POI 索引列表."""
    parts = route_str.split("→")
    indices = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        idx = match_poi(part, name_to_idx, poi_names)
        if idx is not None:
            if idx not in indices:  # 去重
                indices.append(idx)
    return indices


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 公式计算球面距离（km）."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def fix_zero_distances(dist_matrix: np.ndarray, time_matrix: np.ndarray,
                       pois: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """修复距离/时间矩阵中的零距离 bug.

    对于非对角线上的零值，用 Haversine 距离替换。
    """
    n = dist_matrix.shape[0]
    lats = pois["lat"].values.astype(float)
    lngs = pois["lng"].values.astype(float)
    fixed_dist = dist_matrix.copy()
    fixed_time = time_matrix.copy()

    zero_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] == 0:
                d = haversine(lats[i], lngs[i], lats[j], lngs[j])
                if d > 0.1:  # 确实不在同一位置
                    zero_pairs.append((i, j, d))
                    fixed_dist[i, j] = d
                    fixed_dist[j, i] = d
                    # 时间按距离估算：城区 30km/h，郊区 60km/h
                    speed = 30.0 if d < 30 else 60.0
                    t = d / speed * 60
                    fixed_time[i, j] = t
                    fixed_time[j, i] = t

    print(f"  修复 {len(zero_pairs)} 对零距离（Haversine 补充）")
    return fixed_dist, fixed_time


def supplement_missing_categories(pois: pd.DataFrame, merged_path: Path,
                                   quotas: dict[str, int]) -> pd.DataFrame:
    """从合并数据中补充缺失类别（餐饮、购物等）."""
    merged = pd.read_csv(merged_path, encoding="utf-8-sig")
    merged.columns = [c.strip() for c in merged.columns]
    col_map = {"名称": "name", "经度": "lng", "纬度": "lat", "类别": "category",
               "地址": "address", "POI类型": "poi_type", "评分": "rating", "人均消费": "avg_cost"}
    merged = merged.rename(columns={k: v for k, v in col_map.items() if k in merged.columns})

    # 确保必要列存在
    for col in ["name", "lng", "lat", "category", "rating"]:
        if col not in merged.columns:
            return pois

    # 去掉已有的 POI（按名称去重）
    existing_names = set(pois["name"].astype(str))
    merged = merged[~merged["name"].astype(str).isin(existing_names)]

    # 清洗评分
    merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce").fillna(4.0)
    merged.loc[merged["rating"] == 0, "rating"] = 4.0

    new_rows = []
    for category, target_n in quotas.items():
        current_n = len(pois[pois["category"] == category])
        need = target_n - current_n
        if need <= 0:
            continue
        candidates = merged[merged["category"] == category].copy()
        if len(candidates) == 0:
            continue
        # 按评分排序，取 top N
        candidates = candidates.sort_values("rating", ascending=False).head(need)
        print(f"  补充 {category}: {len(candidates)} 个")
        new_rows.append(candidates)

    if new_rows:
        supplement = pd.concat(new_rows, ignore_index=True)
        # 确保列对齐
        for col in pois.columns:
            if col not in supplement.columns:
                supplement[col] = ""
        supplement = supplement[pois.columns]
        pois = pd.concat([pois, supplement], ignore_index=True)

    return pois


def augment_routes_with_dining_and_hotel(matched_routes: list[dict],
                                          pois: pd.DataFrame,
                                          dist_matrix: np.ndarray) -> list[dict]:
    """路线增强：在景点之间插入餐饮，末尾添加住宿.

    规则：
    - 每2-3个景点后插入一个餐饮POI
    - 路线末尾添加住宿POI
    - 选择距离当前位置最近的餐饮/住宿
    """
    # 获取餐饮和住宿POI的索引
    dining_pois = pois[pois["category"] == "餐饮"].index.tolist()
    hotel_pois = pois[pois["category"] == "住宿"].index.tolist()
    has_popularity = "xhs_popularity" in pois.columns

    if not dining_pois and not hotel_pois:
        print("  警告：没有餐饮或住宿POI，跳过路线增强")
        return matched_routes

    augmented = []
    for route_dict in matched_routes:
        indices = route_dict["indices"]
        if len(indices) < 3:
            augmented.append(route_dict)
            continue

        # 清理：去掉原路线中出现在中间的住宿POI
        cleaned = []
        for j, poi_idx in enumerate(indices):
            cat = str(pois.iloc[poi_idx]["category"])
            if cat == "住宿" and j < len(indices) - 1:
                continue  # 中间住宿去掉
            cleaned.append(poi_idx)
        indices = cleaned

        new_route = []
        consecutive_scenic = 0

        for i, poi_idx in enumerate(indices):
            new_route.append(poi_idx)

            # 确定当前POI的活动类型
            cat = str(pois.iloc[poi_idx]["category"])
            if cat == "景点":
                consecutive_scenic += 1
            else:
                consecutive_scenic = 0

            # 每2-3个景点后插入餐饮
            if consecutive_scenic >= 2 and dining_pois and i < len(indices) - 1:
                # 选最佳餐饮：距离分 + 热度分
                best_dining = None
                best_score = float('-inf')
                for d_idx in dining_pois:
                    if d_idx in new_route or d_idx in indices:
                        continue
                    d = dist_matrix[poi_idx, d_idx]
                    if d <= 0:
                        continue
                    # 综合分 = -距离(km) + 热度*2（近+热更好）
                    pop = float(pois.iloc[d_idx].get('xhs_popularity', 0)) if has_popularity else 0
                    score = -d + pop * 2.0
                    if score > best_score:
                        best_score = score
                        best_dining = d_idx
                if best_dining is not None:
                    new_route.append(best_dining)
                    consecutive_scenic = 0

        # 末尾以住宿作为绝对终点（住宿后不再有任何POI）
        if hotel_pois and len(indices) >= 3:
            last_is_hotel = str(pois.iloc[indices[-1]]["category"]) == "住宿"
            if not last_is_hotel:
                last_poi = indices[-1]
                best_hotel = None
                best_score = float('-inf')
                for h_idx in hotel_pois:
                    if h_idx in new_route:
                        continue
                    d = dist_matrix[last_poi, h_idx]
                    if d <= 0:
                        continue
                    pop = float(pois.iloc[h_idx].get('xhs_popularity', 0)) if has_popularity else 0
                    score = -d + pop * 3.0
                    if score > best_score:
                        best_score = score
                        best_hotel = h_idx
                if best_hotel is not None:
                    new_route.append(best_hotel)

        augmented.append({
            "indices": new_route,
            "season": route_dict["season"],
            "source": route_dict["source"],
            "liked_count": route_dict.get("liked_count", 0),
        })

    return augmented


def main():
    raw_dir = Path("data/raw")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载核心节点
    print("=== 加载核心节点 ===")
    pois = pd.read_csv(raw_dir / "哈尔滨POI_核心节点.csv", encoding="utf-8-sig")
    pois.columns = [c.strip() for c in pois.columns]
    col_map = {"名称": "name", "经度": "lng", "纬度": "lat", "类别": "category",
               "地址": "address", "POI类型": "poi_type", "评分": "rating", "人均消费": "avg_cost"}
    pois = pois.rename(columns={k: v for k, v in col_map.items() if k in pois.columns})
    print(f"  原始核心节点: {len(pois)} 个")
    print(f"  类别: {pois['category'].value_counts().to_dict()}")

    # 补充缺失类别：餐饮 30、购物 15
    merged_path = raw_dir / "merged_pois.csv"
    if merged_path.exists():
        print("\n=== 补充餐饮/购物 POI ===")
        pois = supplement_missing_categories(pois, merged_path, {"餐饮": 30, "购物": 15})
        print(f"  补充后: {len(pois)} 个")
        print(f"  类别: {pois['category'].value_counts().to_dict()}")

    n_pois = len(pois)

    # 2. 加载并去重路线
    print("\n=== 加载路线数据 ===")
    routes_df = pd.read_csv(raw_dir / "哈尔滨旅游路线数据.csv", encoding="utf-8-sig")
    routes_df.columns = [c.strip() for c in routes_df.columns]
    print(f"  原始路线: {len(routes_df)} 条")
    routes_df = deduplicate_routes(routes_df)
    print(f"  去重后: {len(routes_df)} 条")
    print(f"  季节: {routes_df['season'].value_counts().to_dict()}")

    # 3. POI 名称对齐
    print("\n=== POI 名称对齐 ===")
    poi_names = pois["name"].tolist()
    name_to_idx = build_name_index(poi_names)

    matched_routes = []
    unmatched_names = set()
    match_count = 0
    total_names = 0

    for _, row in routes_df.iterrows():
        indices = parse_route(row["route"], name_to_idx, poi_names)
        total_names += len(row["route"].split("→"))
        if len(indices) >= 2:  # 至少 2 个 POI 才是有效路线
            matched_routes.append({
                "indices": indices,
                "season": row["season"],
                "source": row["source"],
                "liked_count": row.get("liked_count", 0),
            })
            match_count += sum(1 for p in row["route"].split("→") if match_poi(p.strip(), name_to_idx, poi_names) is not None)
        # 收集未匹配名称
        for part in row["route"].split("→"):
            part = part.strip()
            if part and match_poi(part, name_to_idx, poi_names) is None:
                unmatched_names.add(part)

    print(f"  有效路线（>=2 POI）: {len(matched_routes)} 条")
    print(f"  名称匹配率: {match_count}/{total_names} ({match_count/max(total_names,1)*100:.1f}%)")
    if unmatched_names:
        print(f"  未匹配 POI ({len(unmatched_names)}): {list(unmatched_names)[:10]}...")

    # 4. 加载并修复矩阵
    print("\n=== 修复距离/时间矩阵 ===")
    dist_df = pd.read_csv(raw_dir / "距离矩阵_公里.csv", encoding="utf-8-sig", index_col=0)
    time_df = pd.read_csv(raw_dir / "耗时矩阵_分钟.csv", encoding="utf-8-sig", index_col=0)

    # 原始矩阵（135x135）
    orig_dist = dist_df.values.astype(np.float32)
    orig_time = time_df.values.astype(np.float32)
    orig_n = orig_dist.shape[0]

    if n_pois > orig_n:
        # 扩展矩阵：用 Haversine 计算新增 POI 的距离
        print(f"  扩展矩阵: {orig_n}x{orig_n} → {n_pois}x{n_pois}")
        dist_matrix = np.zeros((n_pois, n_pois), dtype=np.float32)
        time_matrix = np.zeros((n_pois, n_pois), dtype=np.float32)
        # 填充原始部分
        dist_matrix[:orig_n, :orig_n] = orig_dist
        time_matrix[:orig_n, :orig_n] = orig_time

        lats = pois["lat"].values.astype(float)
        lngs = pois["lng"].values.astype(float)

        # 计算新增 POI 与所有 POI 的距离
        for i in range(orig_n, n_pois):
            for j in range(n_pois):
                if i == j:
                    continue
                d = haversine(lats[i], lngs[i], lats[j], lngs[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                speed = 30.0 if d < 30 else 60.0
                t = d / speed * 60
                time_matrix[i, j] = t
                time_matrix[j, i] = t
    else:
        dist_matrix = orig_dist
        time_matrix = orig_time

    # 检查形状
    assert dist_matrix.shape == (n_pois, n_pois), f"距离矩阵形状 {dist_matrix.shape} != ({n_pois}, {n_pois})"

    n_zero = (dist_matrix == 0).sum() - n_pois
    print(f"  非对角线零距离: {n_zero} 对")
    dist_matrix, time_matrix = fix_zero_distances(dist_matrix, time_matrix, pois)

    # 5. 构建邻接矩阵
    max_dist = 50.0
    connected = (dist_matrix > 0) & (dist_matrix < max_dist)
    adj = np.where(connected, 1.0 - dist_matrix / max_dist, 0.0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    # 6. 构建概率分布矩阵
    dist_std = np.zeros_like(dist_matrix)
    dist_std[dist_matrix <= 10] = dist_matrix[dist_matrix <= 10] * 0.20
    mask_mid = (dist_matrix > 10) & (dist_matrix <= 50)
    dist_std[mask_mid] = dist_matrix[mask_mid] * 0.15
    dist_std[dist_matrix > 50] = dist_matrix[dist_matrix > 50] * 0.10
    np.fill_diagonal(dist_std, 0.0)

    time_std = np.zeros_like(time_matrix)
    time_std[dist_matrix <= 10] = time_matrix[dist_matrix <= 10] * 0.25
    time_std[mask_mid] = time_matrix[mask_mid] * 0.18
    time_std[dist_matrix > 50] = time_matrix[dist_matrix > 50] * 0.12
    np.fill_diagonal(time_std, 0.0)

    # 7. 构建特征矩阵
    from src.data.preprocess import extract_poi_features, normalize_features

    # 补全评分
    pois["rating"] = pd.to_numeric(pois["rating"].replace("[]", np.nan), errors="coerce")
    pois["rating"] = pois["rating"].fillna(pois["rating"].median())
    pois.loc[pois["rating"] == 0, "rating"] = np.nan
    pois["rating"] = pois["rating"].fillna(pois["rating"].median())

    # 热度
    if "liked_count" in pois.columns:
        pois["popularity"] = pd.to_numeric(pois["liked_count"], errors="coerce").fillna(0)
    else:
        pois["popularity"] = pois["rating"] * 20

    # 季节权重
    from src.data.preprocess import _is_winter_poi, SEASON_WEIGHTS
    poi_type_col = pois.get("poi_type", pd.Series([""] * len(pois)))
    pois["season_winter"] = pois.apply(
        lambda r: 1.0 if _is_winter_poi(r["name"], str(poi_type_col.iloc[r.name] if r.name < len(poi_type_col) else ""))
        else SEASON_WEIGHTS.get(r["category"], SEASON_WEIGHTS["景点"])["winter"],
        axis=1,
    )
    pois["season_summer"] = pois.apply(
        lambda r: 0.3 if _is_winter_poi(r["name"], str(poi_type_col.iloc[r.name] if r.name < len(poi_type_col) else ""))
        else SEASON_WEIGHTS.get(r["category"], SEASON_WEIGHTS["景点"])["summer"],
        axis=1,
    )

    features = extract_poi_features(pois, d_model=128)

    # 7.5 为每个 POI 添加活动类型标签
    print("\n=== 添加活动类型标签 ===")
    pois["activity_type"] = pois.apply(
        lambda r: get_activity_type(r["category"], r["name"]),
        axis=1
    )
    activity_type_names = {v: k for k, v in ACTIVITY_TYPE_MAP.items()}
    activity_dist = pois["activity_type"].value_counts().to_dict()
    print(f"  活动类型分布: {activity_dist}")
    for atype, count in activity_dist.items():
        print(f"    {activity_type_names.get(atype, '未知')}: {count}")

    # 8. 保存所有数据
    print("\n=== 保存数据 ===")
    pois.to_csv(out_dir / "poi_metadata.csv", index=False, encoding="utf-8")
    np.save(out_dir / "poi_features.npy", features)
    np.save(out_dir / "adjacency.npy", adj)
    np.save(out_dir / "distance_matrix.npy", dist_matrix)
    np.save(out_dir / "distance_std.npy", dist_std)
    np.save(out_dir / "time_matrix.npy", time_matrix)
    np.save(out_dir / "time_std.npy", time_std)

    # 保存活动类型标签（用于推理时约束解码）
    activity_types = pois["activity_type"].values.astype(np.int64)
    np.save(out_dir / "poi_activity_types.npy", activity_types)

    # 路线增强：插入餐饮和住宿
    print("\n=== 路线增强：插入餐饮/住宿 ===")
    n_before = len(matched_routes)
    avg_before = np.mean([len(r["indices"]) for r in matched_routes])
    matched_routes = augment_routes_with_dining_and_hotel(matched_routes, pois, dist_matrix)
    avg_after = np.mean([len(r["indices"]) for r in matched_routes])
    print(f"  增强前: {n_before} 条, 平均 {avg_before:.1f} 站")
    print(f"  增强后: {len(matched_routes)} 条, 平均 {avg_after:.1f} 站")
    type_counts = {}
    for r in matched_routes:
        for idx in r["indices"]:
            atype = int(activity_types[idx])
            atype_name = {0: "景点", 1: "餐饮", 2: "住宿", 3: "交通", 4: "购物", 5: "出发点"}[atype]
            type_counts[atype_name] = type_counts.get(atype_name, 0) + 1
    total = sum(type_counts.values())
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v} ({v/total*100:.1f}%)")

    # 路线保存为 npy
    route_arrays = [np.array(r["indices"]) for r in matched_routes]
    np.save(out_dir / "routes.npy", np.array(route_arrays, dtype=object))

    print(f"  poi_metadata.csv   — {n_pois} POI")
    print(f"  poi_features.npy   — {features.shape}")
    print(f"  adjacency.npy      — {adj.shape}")
    print(f"  distance_matrix.npy— {dist_matrix.shape} (真实路网距离)")
    print(f"  distance_std.npy   — {dist_std.shape}")
    print(f"  time_matrix.npy    — {time_matrix.shape} (真实路网耗时)")
    print(f"  time_std.npy       — {time_std.shape}")
    print(f"  poi_activity_types — {activity_types.shape} (活动类型标签)")
    print(f"  routes.npy         — {len(matched_routes)} 条真实路线 (平均 {np.mean([len(r) for r in route_arrays]):.1f} 站)")
    print(f"\n完成！")


if __name__ == "__main__":
    main()
