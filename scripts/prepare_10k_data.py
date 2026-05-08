"""Prepare ~10K POI data for training.

Filters merged_pois.csv to ~10K quality POIs, computes Haversine distance/time
matrices, generates synthetic routes with category-aware walks, and applies
dining/hotel augmentation.

Usage:
    uv run python scripts/prepare_10k_data.py
"""

import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import (
    load_raw_poi_data,
    clean_poi_data,
    extract_poi_features,
    build_distance_matrix,
    build_time_matrix,
    build_adjacency,
)

ACTIVITY_TYPE_MAP = {
    "景点": 0, "餐饮": 1, "住宿": 2, "交通": 3, "购物": 4, "出发点": 5,
}


def get_activity_type(category: str, name: str = "") -> int:
    category = str(category).strip()
    name = str(name).strip()
    if category == "交通":
        return ACTIVITY_TYPE_MAP["出发点"]
    if category in ACTIVITY_TYPE_MAP:
        return ACTIVITY_TYPE_MAP[category]
    return ACTIVITY_TYPE_MAP["景点"]


def build_walking_clusters(pois: pd.DataFrame, dist_matrix: np.ndarray,
                           max_dist_km: float = 1.0):
    """Build walking clusters from scenic POIs within max_dist_km via Union-Find."""
    scenic_idx = pois.index[pois["category"] == "景点"].tolist()
    n_scenic = len(scenic_idx)

    parent = list(range(n_scenic))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n_scenic):
        for j in range(i + 1, n_scenic):
            if dist_matrix[scenic_idx[i], scenic_idx[j]] <= max_dist_km:
                union(i, j)

    groups = {}
    for i in range(n_scenic):
        root = find(i)
        groups.setdefault(root, []).append(scenic_idx[i])

    clusters = [sorted(m) for m in groups.values() if len(m) >= 2]
    cluster_id = np.full(len(pois), -1, dtype=np.int32)
    for cid, members in enumerate(clusters):
        for poi_idx in members:
            cluster_id[poi_idx] = cid

    return cluster_id, clusters


def generate_synthetic_routes(pois: pd.DataFrame, dist_matrix: np.ndarray,
                               n_routes: int = 5000,
                               min_len: int = 5, max_len: int = 15,
                               seed: int = 42):
    """Generate category-aware nearest-neighbor random walk routes.

    Routes are primarily scenic with some shopping. Dining/hotel will be added
    by the augmentation step afterwards.
    """
    rng = np.random.RandomState(seed)

    cat_indices = {}
    for cat in ["景点", "餐饮", "购物", "住宿"]:
        cat_indices[cat] = pois[pois["category"] == cat].index.tolist()

    scenic_indices = cat_indices["景点"]
    if not scenic_indices:
        return []

    scenic_ratings = pois.loc[scenic_indices, "rating"].values.astype(float)
    start_weights = scenic_ratings / scenic_ratings.sum()

    routes = []
    for _ in range(n_routes):
        route_len = rng.randint(min_len, max_len + 1)
        start_pos = rng.choice(len(scenic_indices), p=start_weights)
        start = scenic_indices[start_pos]
        route = [start]
        visited = {start}

        for _step in range(route_len - 1):
            current = route[-1]
            cat_choice = rng.choice(["景点", "购物", "餐饮"], p=[0.80, 0.15, 0.05])
            candidates = [i for i in cat_indices.get(cat_choice, []) if i not in visited]

            if not candidates:
                candidates = [i for i in scenic_indices if i not in visited]
                if not candidates:
                    break

            candidates_arr = np.array(candidates)
            dists = np.maximum(dist_matrix[current, candidates_arr], 0.1)
            ratings = pois.loc[candidates, "rating"].values.astype(float)
            weights = (1.0 / dists**2) * ratings
            wsum = weights.sum()
            if wsum == 0:
                break
            weights = weights / wsum

            next_pos = rng.choice(len(candidates), p=weights)
            route.append(candidates[next_pos])
            visited.add(candidates[next_pos])

        if len(route) >= 3:
            routes.append(np.array(route))

    return routes


def _normalize_name(name: str) -> str:
    name = re.sub(r"[·\-\(\)（）\s]", "", name.strip())
    return name.replace("风景区", "").replace("哈尔滨市", "哈尔滨")


def match_poi(route_name: str, name_to_idx: dict, poi_names: list) -> int | None:
    route_name = route_name.strip()
    if route_name in name_to_idx:
        return name_to_idx[route_name]

    norm_route = _normalize_name(route_name)
    for name, idx in name_to_idx.items():
        if _normalize_name(name) == norm_route:
            return idx

    for name, idx in name_to_idx.items():
        nn = _normalize_name(name)
        if norm_route in nn or nn in norm_route:
            return idx

    best_score, best_idx = 0, None
    for name, idx in name_to_idx.items():
        score = SequenceMatcher(None, norm_route, _normalize_name(name)).ratio()
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_score >= 0.6:
        return best_idx
    return None


def load_xhs_routes(pois: pd.DataFrame, raw_dir: Path):
    routes_path = raw_dir / "哈尔滨旅游路线数据.csv"
    if not routes_path.exists():
        print("  未找到XHS路线数据，跳过")
        return []

    routes_df = pd.read_csv(routes_path, encoding="utf-8-sig")
    routes_df.columns = [c.strip() for c in routes_df.columns]

    def clean_count(v):
        if pd.isna(v):
            return 0
        s = str(v).strip()
        if "万" in s:
            return int(float(s.replace("万", "")) * 10000)
        try:
            return int(float(s))
        except ValueError:
            return 0

    if "liked_count" in routes_df.columns:
        routes_df["liked_count"] = routes_df["liked_count"].apply(clean_count)
        routes_df = routes_df.sort_values("liked_count", ascending=False)
    routes_df = routes_df.drop_duplicates(subset=["note_id"], keep="first")
    routes_df = routes_df.drop_duplicates(subset=["route"], keep="first")

    poi_names = pois["name"].tolist()
    name_to_idx = {name.strip(): i for i, name in enumerate(poi_names)}

    matched = []
    for _, row in routes_df.iterrows():
        parts = row["route"].split("→")
        indices = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            idx = match_poi(part, name_to_idx, poi_names)
            if idx is not None and idx not in indices:
                indices.append(idx)
        if len(indices) >= 2:
            matched.append({
                "indices": indices,
                "season": row.get("season", "winter"),
                "source": row.get("source", "xhs"),
                "liked_count": row.get("liked_count", 0),
            })

    return matched


def augment_routes(routes_list: list, pois: pd.DataFrame, dist_matrix: np.ndarray):
    """Insert dining every 2-3 scenic stops, add hotel at end."""
    dining_pois = pois[pois["category"] == "餐饮"].index.tolist()
    hotel_pois = pois[pois["category"] == "住宿"].index.tolist()

    # Normalize popularity to [0, 1] for scoring
    if "popularity" in pois.columns:
        pop_vals = pois["popularity"].values.astype(float)
        pop_max = pop_vals.max() if pop_vals.max() > 0 else 1.0
        pop_norm = pop_vals / pop_max
    else:
        pop_norm = np.zeros(len(pois))

    # Pre-compute nearest dining/hotel for each POI (top-50)
    n_pois = len(pois)
    nearest_dining = {}
    if dining_pois:
        dining_arr = np.array(dining_pois)
        dists_to_dining = dist_matrix[:, dining_arr]
        top_k = min(50, len(dining_pois))
        nearest_idx = np.argsort(dists_to_dining, axis=1)[:, :top_k]
        for i in range(n_pois):
            nearest_dining[i] = dining_arr[nearest_idx[i]].tolist()

    nearest_hotel = {}
    if hotel_pois:
        hotel_arr = np.array(hotel_pois)
        dists_to_hotel = dist_matrix[:, hotel_arr]
        top_k = min(50, len(hotel_pois))
        nearest_idx = np.argsort(dists_to_hotel, axis=1)[:, :top_k]
        for i in range(n_pois):
            nearest_hotel[i] = hotel_arr[nearest_idx[i]].tolist()

    augmented = []
    for route_dict in routes_list:
        indices = route_dict["indices"]
        if len(indices) < 3:
            augmented.append(route_dict)
            continue

        # Remove mid-route hotels
        cleaned = []
        for j, poi_idx in enumerate(indices):
            cat = str(pois.iloc[poi_idx]["category"])
            if cat == "住宿" and j < len(indices) - 1:
                continue
            cleaned.append(poi_idx)
        indices = cleaned

        new_route = []
        consecutive_scenic = 0

        for i, poi_idx in enumerate(indices):
            new_route.append(poi_idx)
            cat = str(pois.iloc[poi_idx]["category"])
            if cat == "景点":
                consecutive_scenic += 1
            else:
                consecutive_scenic = 0

            if consecutive_scenic >= 2 and i < len(indices) - 1:
                visited_set = set(new_route) | set(indices)
                best_dining = None
                best_score = float('-inf')
                for d_idx in nearest_dining.get(poi_idx, []):
                    if d_idx in visited_set:
                        continue
                    d = dist_matrix[poi_idx, d_idx]
                    if d <= 0:
                        continue
                    score = -d + pop_norm[d_idx] * 2.0
                    if score > best_score:
                        best_score = score
                        best_dining = d_idx
                if best_dining is not None:
                    new_route.append(best_dining)
                    consecutive_scenic = 0

        # Hotel at end
        if hotel_pois and len(indices) >= 3:
            last_poi = indices[-1]
            if str(pois.iloc[last_poi]["category"]) != "住宿":
                visited_set = set(new_route)
                best_hotel = None
                best_score = float('-inf')
                for h_idx in nearest_hotel.get(last_poi, []):
                    if h_idx in visited_set:
                        continue
                    d = dist_matrix[last_poi, h_idx]
                    if d <= 0:
                        continue
                    score = -d + pop_norm[h_idx] * 3.0
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

    # === 1. Load and filter POIs ===
    print("=== 加载POI数据 ===")
    raw = load_raw_poi_data(str(raw_dir / "merged_pois.csv"))
    print(f"  原始POI: {len(raw)}")

    pois = clean_poi_data(raw, max_pois=10000)
    n_pois = len(pois)
    print(f"  筛选后: {n_pois} POIs")
    print(f"  类别分布: {pois['category'].value_counts().to_dict()}")

    # === 2. Add activity types ===
    print("\n=== 活动类型标签 ===")
    pois["activity_type"] = pois.apply(
        lambda r: get_activity_type(r["category"], r["name"]), axis=1,
    )
    type_names = {v: k for k, v in ACTIVITY_TYPE_MAP.items()}
    for atype, count in pois["activity_type"].value_counts().items():
        print(f"  {type_names[atype]}: {count}")

    # === 3. Compute matrices ===
    print("\n=== 计算距离/时间矩阵 ===")
    dist_mean, dist_std = build_distance_matrix(pois)
    print(f"  平均距离: {dist_mean[dist_mean > 0].mean():.1f}km")

    time_mean, time_std = build_time_matrix(dist_mean, dist_std)
    print(f"  平均耗时: {time_mean[time_mean > 0].mean():.1f}min")

    adj = build_adjacency(dist_mean, max_distance_km=30.0)
    print(f"  连通边: {(adj > 0).sum()}")

    # === 4. Extract features ===
    print("\n=== 特征提取 ===")
    features = extract_poi_features(pois, d_model=128)
    print(f"  特征矩阵: {features.shape}")

    # === 5. Walking clusters ===
    print("\n=== 步行聚类 ===")
    cluster_id, clusters = build_walking_clusters(pois, dist_mean, max_dist_km=1.0)
    print(f"  {len(clusters)} 个团, {int((cluster_id >= 0).sum())} 景入团")

    # === 6. Load XHS routes ===
    print("\n=== XHS路线匹配 ===")
    xhs_routes = load_xhs_routes(pois, raw_dir)
    print(f"  匹配路线: {len(xhs_routes)} 条")

    # === 7. Generate synthetic routes ===
    print("\n=== 合成路线生成 ===")
    synthetic = generate_synthetic_routes(pois, dist_mean, n_routes=5000, seed=42)
    print(f"  生成: {len(synthetic)} 条")

    synthetic_dicts = []
    rng = np.random.RandomState(123)
    for route in synthetic:
        synthetic_dicts.append({
            "indices": route.tolist(),
            "season": rng.choice(["winter", "summer"]),
            "source": "synthetic",
            "liked_count": 0,
        })

    # === 8. Combine and augment ===
    print("\n=== 路线增强（餐饮/住宿插入）===")
    all_routes = xhs_routes + synthetic_dicts
    avg_before = np.mean([len(r["indices"]) for r in all_routes])
    print(f"  增强前: {len(all_routes)} 条, 平均 {avg_before:.1f} 站")

    all_routes = augment_routes(all_routes, pois, dist_mean)
    avg_after = np.mean([len(r["indices"]) for r in all_routes])
    print(f"  增强后: {len(all_routes)} 条, 平均 {avg_after:.1f} 站")

    type_counts = {}
    for r in all_routes:
        for idx in r["indices"]:
            atype = int(pois.iloc[idx]["activity_type"])
            aname = type_names.get(atype, "未知")
            type_counts[aname] = type_counts.get(aname, 0) + 1
    total = sum(type_counts.values())
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v} ({v / total * 100:.1f}%)")

    # === 9. Save ===
    print("\n=== 保存数据 ===")
    pois.to_csv(out_dir / "poi_metadata.csv", index=False, encoding="utf-8")
    np.save(out_dir / "poi_features.npy", features)
    np.save(out_dir / "adjacency.npy", adj)
    np.save(out_dir / "distance_matrix.npy", dist_mean)
    np.save(out_dir / "distance_std.npy", dist_std)
    np.save(out_dir / "time_matrix.npy", time_mean)
    np.save(out_dir / "time_std.npy", time_std)

    activity_types = pois["activity_type"].values.astype(np.int64)
    np.save(out_dir / "poi_activity_types.npy", activity_types)
    np.save(out_dir / "cluster_id.npy", cluster_id)
    np.save(out_dir / "clusters.npy", np.array(clusters, dtype=object))

    route_arrays = [np.array(r["indices"]) for r in all_routes]
    np.save(out_dir / "routes.npy", np.array(route_arrays, dtype=object))

    print(f"  poi_metadata.csv   — {n_pois} POI")
    print(f"  poi_features.npy   — {features.shape}")
    print(f"  adjacency.npy      — {adj.shape}")
    print(f"  distance_matrix.npy— {dist_mean.shape}")
    print(f"  routes.npy         — {len(all_routes)} 条 (平均 {avg_after:.1f} 站)")
    print(f"\n完成！")


if __name__ == "__main__":
    main()
