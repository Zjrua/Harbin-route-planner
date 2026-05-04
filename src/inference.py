"""推理脚本：加载训练好的模型，根据用户条件生成推荐路线.

支持活动类型约束解码，生成符合真实旅游节奏的路线：
- 景点 → 餐饮 → 景点 → 住宿（不连续餐饮、不连续住宿）
- 多日游支持：通过住宿节点自然分割天数

用法:
    python -m src.inference --checkpoint checkpoints/best_model.pt --start "冰雪大世界" --season winter
    python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season summer --max_stops 8 --max_hours 6
    python -m src.inference --checkpoint checkpoints/best_model.pt --season winter --n_routes 3
"""

import argparse
import json
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.transformer import RouteTransformer
from src.evaluate import (
    route_distance, route_time, satisfaction_score,
    diversity_score, composite_score,
)
from src.visualize import plot_route_on_map, plot_route_comparison, plot_route_on_map_with_roads


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_poi_by_name(pois: pd.DataFrame, query: str) -> list[int]:
    name_col = pois["name"] if "name" in pois.columns else pois.iloc[:, 0]
    matches = pois[name_col.astype(str).str.contains(query, case=False, na=False)]
    if matches.empty:
        return []
    return matches.index.tolist()


def print_route_detail(route: list[int], pois: pd.DataFrame,
                       dist_matrix: np.ndarray, time_matrix: np.ndarray):
    print(f"\n  {'='*58}")
    print(f"  {'序号':>4s}  {'POI名称':<24s}  {'类别':<6s}  {'评分':>4s}  {'距上一站':>8s}  {'耗时':>6s}")
    print(f"  {'----':>4s}  {'------------------------':<24s}  {'------':<6s}  {'----':>4s}  {'--------':>8s}  {'------':>6s}")

    total_dist = 0.0
    total_time = 0.0
    for i, idx in enumerate(route):
        row = pois.iloc[idx]
        name = str(row["name"])[:24] if "name" in pois.columns else str(row.iloc[0])[:24]
        cat = str(row.get("category", "-"))
        rating = float(row.get("rating", 0))
        seg_dist = float(dist_matrix[route[i - 1], idx]) if i > 0 else 0.0
        seg_time = float(time_matrix[route[i - 1], idx]) if i > 0 else 0.0
        total_dist += seg_dist
        total_time += seg_time
        dist_str = f"{seg_dist:.1f}km" if i > 0 else "-"
        time_str = f"{seg_time:.0f}min" if i > 0 else "-"
        print(f"  {i+1:>4d}  {name:<24s}  {cat:<6s}  {rating:>4.1f}  {dist_str:>8s}  {time_str:>6s}")

    print(f"  {'─'*58}")
    print(f"  总计: {len(route)} 个游览点 | {total_dist:.1f}km | {total_time:.0f}分钟 ({total_time/60:.1f}小时)")


def generate_with_constraints(model: RouteTransformer, encoder_output: torch.Tensor,
                              start_id: int, beam_size: int, max_len: int,
                              device: torch.device,
                              poi_activity_types: torch.Tensor) -> list[int]:
    """带活动类型约束的 beam search."""
    model.eval()
    with torch.no_grad():
        route = [start_id]

        for step in range(max_len - 1):
            route_t = torch.tensor([route], dtype=torch.long, device=device)

            # 获取活动类型嵌入
            activity_type_ids = poi_activity_types[route_t]
            target_emb = model.poi_embedding(route_t, activity_types=activity_type_ids)

            seq_len = target_emb.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()

            # Engram 记忆
            engram_memory = None
            if model.use_engram:
                query = encoder_output.mean(dim=1)
                retrieved, _ = model.engram.retrieve(query)
                engram_memory = retrieved

            dec_out = model.decode(engram_memory, encoder_output, target_emb, causal_mask)
            logits = model.output_proj(dec_out)[:, -1, :]  # [1, n_pois]

            # 应用活动类型约束
            last_poi = route[-1]
            last_activity = poi_activity_types[last_poi].item()
            constraints = model.activity_constraints  # [6, 6]

            for poi_idx in range(logits.size(-1)):
                activity = poi_activity_types[poi_idx].item()
                logits[0, poi_idx] += constraints[last_activity, activity]

            # 屏蔽已访问的 POI
            visited = set(route)
            for v in visited:
                if v < logits.size(-1):
                    logits[0, v] = -1e9

            log_probs = torch.log_softmax(logits, dim=-1)
            topk_probs, topk_idx = log_probs[0].topk(beam_size)

            # 选择最佳候选
            best_idx = topk_idx[0].item()
            route.append(best_idx)

    return route


def generate_diverse_routes(model: RouteTransformer, encoder_output: torch.Tensor,
                            beam_size: int, max_len: int, n_routes: int,
                            start_id: int | None, pois: pd.DataFrame,
                            device: torch.device,
                            poi_activity_types: torch.Tensor) -> list[list[int]]:
    """生成多条多样化路线：不同起点 + 约束 beam search."""
    routes = []
    used_starts = set()

    if start_id is not None:
        route = generate_with_constraints(
            model, encoder_output, start_id, beam_size, max_len, device, poi_activity_types
        )
        routes.append(route)
        used_starts.add(start_id)

    # 后续路线选不同高评分 POI 作起点
    if "rating" in pois.columns:
        top_pois = pois.sort_values("rating", ascending=False).index.tolist()
    else:
        top_pois = list(range(len(pois)))

    for poi_idx in top_pois:
        if len(routes) >= n_routes:
            break
        if poi_idx in used_starts:
            continue
        route = generate_with_constraints(
            model, encoder_output, poi_idx, beam_size, max_len, device, poi_activity_types
        )
        routes.append(route)
        used_starts.add(poi_idx)

    return routes


def filter_route(route: list[int], max_stops: int | None = None,
                 time_matrix: np.ndarray | None = None,
                 max_minutes: float | None = None) -> list[int]:
    """过滤路线：去重、去 padding、截断."""
    filtered = []
    seen = set()
    for poi_id in route:
        if poi_id == 0 and len(filtered) > 0:
            break
        if poi_id in seen:
            continue
        seen.add(poi_id)
        filtered.append(poi_id)

    if max_stops:
        filtered = filtered[:max_stops]

    if max_minutes and time_matrix is not None:
        total = 0.0
        for i in range(1, len(filtered)):
            seg = time_matrix[filtered[i - 1], filtered[i]]
            if total + seg > max_minutes:
                filtered = filtered[:i]
                break
            total += seg

    return filtered


def optimize_route_order(route: list[int], pois: pd.DataFrame,
                         dist_matrix: np.ndarray) -> list[int]:
    """优化路线顺序：使用最近邻算法，避免走回头路.

    保持起点不变，按照距离最近的原则重新排列后续景点。
    """
    if len(route) <= 2:
        return route

    # 获取POI坐标
    coords = []
    for idx in route:
        row = pois.iloc[idx]
        lat = float(row["lat"]) if "lat" in pois.columns else 45.80
        lng = float(row["lng"]) if "lng" in pois.columns else 126.53
        coords.append((lat, lng))

    # 最近邻算法：从起点开始，每次选择最近的未访问POI
    optimized = [route[0]]  # 保持起点
    remaining = set(range(1, len(route)))

    current = 0
    while remaining:
        # 找到距离当前POI最近的下一个POI
        min_dist = float('inf')
        nearest = None
        for next_idx in remaining:
            # 使用路线中的实际距离
            d = dist_matrix[route[current], route[next_idx]]
            if d < min_dist:
                min_dist = d
                nearest = next_idx

        if nearest is not None:
            optimized.append(route[nearest])
            remaining.remove(nearest)
            current = nearest

    return optimized


def optimize_route_2opt(route: list[int], dist_matrix: np.ndarray,
                        iterations: int = 100) -> list[int]:
    """2-opt 优化：消除路线中的交叉，进一步优化路线顺序.

    保持起点不变。
    """
    if len(route) <= 3:
        return route

    def route_cost(r):
        return sum(dist_matrix[r[i], r[i+1]] for i in range(len(r)-1))

    best = route[:]
    best_cost = route_cost(best)

    for _ in range(iterations):
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                # 尝试反转 i 到 j 的路径
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_cost = route_cost(new_route)
                if new_cost < best_cost:
                    best = new_route
                    best_cost = new_cost
                    improved = True
        if not improved:
            break

    return best


def main():
    parser = argparse.ArgumentParser(description="哈尔滨文旅路线推荐")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start", type=str, default=None, help="起点 POI 名称")
    parser.add_argument("--start_id", type=int, default=None, help="起点 POI 编号")
    parser.add_argument("--season", type=str, default="winter", choices=["winter", "summer"])
    parser.add_argument("--max_stops", type=int, default=None)
    parser.add_argument("--max_hours", type=float, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--n_routes", type=int, default=1, help="生成候选路线数量")
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    beam_size = args.beam_size or config["training"].get("beam_size", 5)
    max_stops = args.max_stops or config["model"]["max_route_len"]

    # 加载数据
    data_dir = Path(args.data_dir)
    poi_features = torch.from_numpy(np.load(data_dir / "poi_features.npy")).float()
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy")).float()
    dist_matrix = np.load(data_dir / "distance_matrix.npy")
    time_matrix = np.load(data_dir / "time_matrix.npy")

    # 加载活动类型标签
    if (data_dir / "poi_activity_types.npy").exists():
        poi_activity_types = torch.from_numpy(
            np.load(data_dir / "poi_activity_types.npy")
        ).long().to(device)
        print(f"已加载活动类型标签: {poi_activity_types.shape}")
    else:
        poi_activity_types = None
        print("警告: 未找到活动类型标签，将使用无约束解码")

    if (data_dir / "poi_metadata.csv").exists():
        pois = pd.read_csv(data_dir / "poi_metadata.csv", encoding="utf-8")
    else:
        pois = pd.DataFrame({"name": [f"POI-{i}" for i in range(len(poi_features))],
                              "lat": 45.80, "lng": 126.53,
                              "category": "未知", "rating": 4.0})

    n_pois = len(poi_features)
    ratings = pois["rating"].values if "rating" in pois.columns else np.full(n_pois, 4.0)
    categories = pois["category"].values if "category" in pois.columns else np.zeros(n_pois)

    # 确定起点
    start_id = args.start_id
    if args.start is not None:
        matches = find_poi_by_name(pois, args.start)
        if not matches:
            print(f"未找到包含 '{args.start}' 的 POI，可用 POI:")
            for i, row in pois.iterrows():
                pname = str(row["name"]) if "name" in pois.columns else str(row.iloc[0])
                print(f"  #{i}: {pname} ({row.get('category', '')})")
            return
        start_id = matches[0]
        poi_name = str(pois.iloc[start_id]["name"]) if "name" in pois.columns else str(pois.iloc[start_id].iloc[0])
        if len(matches) > 1:
            print(f"找到 {len(matches)} 个匹配项，使用第一个: #{start_id} {poi_name}")
        else:
            print(f"起点: #{start_id} {poi_name}")

    # 加载模型
    model = RouteTransformer(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"模型加载完成 (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('best_val_loss', '?')})")

    # 编码
    poi_feat_dev = poi_features.unsqueeze(0).to(device)
    adj_dev = adjacency.unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_output = model.encode(poi_feat_dev, adj_dev)

    # 生成路线
    print(f"\n生成条件: 季节={args.season}, 最大游览点={max_stops}, beam_size={beam_size}")
    print("=" * 60)

    max_minutes = args.max_hours * 60 if args.max_hours else None
    raw_routes = generate_diverse_routes(
        model, encoder_output, beam_size, max_stops, args.n_routes,
        start_id, pois, device, poi_activity_types
    )

    all_routes = []
    for raw in raw_routes:
        filtered = filter_route(raw, max_stops, time_matrix, max_minutes)
        if len(filtered) >= 2:
            # 优化路线顺序：避免走回头路
            optimized = optimize_route_order(filtered, pois, dist_matrix)
            # 2-opt 进一步优化
            optimized = optimize_route_2opt(optimized, dist_matrix, iterations=50)
            all_routes.append(optimized)

    # 评估
    metrics_cfg = config.get("metrics", {})
    weights = {
        "distance": metrics_cfg.get("distance_weight", 0.30),
        "time": metrics_cfg.get("time_weight", 0.25),
        "satisfaction": metrics_cfg.get("satisfaction_weight", 0.25),
        "diversity": metrics_cfg.get("diversity_weight", 0.20),
    }

    results = []
    for i, route in enumerate(all_routes):
        d = route_distance(route, dist_matrix)
        t = route_time(route, time_matrix)
        s = satisfaction_score(route, ratings)
        div = diversity_score(route, categories)
        max_d = dist_matrix.max() * len(route) if len(route) > 1 else 1.0
        max_t = time_matrix.max() * len(route) if len(route) > 1 else 1.0
        m = {"distance": d, "time": t, "satisfaction": s, "diversity": div,
             "max_distance": max_d, "max_time": max_t}
        score = composite_score(m, weights)

        # 获取活动类型序列
        activity_sequence = []
        if poi_activity_types is not None:
            activity_names = {0: "景点", 1: "餐饮", 2: "住宿", 3: "交通", 4: "购物", 5: "出发点"}
            for idx in route:
                activity_sequence.append(activity_names.get(poi_activity_types[idx].item(), "未知"))

        results.append({
            "route_indices": [int(x) for x in route],
            "route_names": [str(pois.iloc[idx]["name"]) if "name" in pois.columns else f"POI-{idx}" for idx in route],
            "activity_sequence": activity_sequence,
            "distance_km": round(float(d), 1),
            "time_min": round(float(t), 0),
            "satisfaction": round(float(s), 3),
            "diversity": round(float(div), 3),
            "composite_score": round(float(score), 4),
        })

        label = f"路线 {i+1}" if len(all_routes) > 1 else "推荐路线"
        if len(all_routes) > 1:
            print(f"\n--- {label} ---")
        print_route_detail(route, pois, dist_matrix, time_matrix)

        # 打印活动类型序列
        if activity_sequence:
            print(f"  活动节奏: {' → '.join(activity_sequence)}")

        print(f"  评估: 距离={d:.1f}km | 耗时={t:.0f}min | 满意度={s:.2f} | 多样性={div:.2f} | 综合={score:.3f}")

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # JSON 结果
    with open(output_dir / "routes_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_dir / 'routes_result.json'}")

    # 地图：所有路线对比
    if len(all_routes) > 1:
        labels = [f"路线{i+1}" for i in range(len(all_routes))]
        plot_route_comparison(all_routes, pois, labels,
                              output_path=str(output_dir / "routes_comparison.html"),
                              title=f"哈尔滨旅游路线对比 ({args.season})")
        print(f"路线对比图: {output_dir / 'routes_comparison.html'}")

    # 地图：最优路线（使用沿道路绘制的版本）
    best_idx = max(range(len(results)), key=lambda i: results[i]["composite_score"])
    best_route = all_routes[best_idx]

    # 尝试使用高德地图API获取实际道路路径
    amap_key = config.get("data", {}).get("amap_key", None)
    if amap_key:
        from src.visualize import plot_route_on_map_with_roads
        plot_route_on_map_with_roads(
            pois, best_route,
            output_path=str(output_dir / "best_route_map.html"),
            center_lat=config["data"]["center_lat"],
            center_lng=config["data"]["center_lng"],
            title=f"最优路线 ({args.season})",
            amap_key=amap_key
        )
    else:
        plot_route_on_map(
            pois, best_route,
            output_path=str(output_dir / "best_route_map.html"),
            center_lat=config["data"]["center_lat"],
            center_lng=config["data"]["center_lng"],
            title=f"最优路线 ({args.season})"
        )
    print(f"最优路线地图: {output_dir / 'best_route_map.html'}")


if __name__ == "__main__":
    main()
