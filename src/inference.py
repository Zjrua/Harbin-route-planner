"""推理脚本：加载训练好的模型，根据用户条件生成推荐路线.

用法:
    # 基本用法：指定起点和季节
    python -m src.inference --checkpoint checkpoints/best_model.pt --start "冰雪大世界" --season winter

    # 指定最大游览点数和时间预算
    python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season summer --max_stops 8 --max_hours 6

    # 指定起点 POI 编号
    python -m src.inference --checkpoint checkpoints/best_model.pt --start_id 0 --season winter

    # 生成多条候选路线对比
    python -m src.inference --checkpoint checkpoints/best_model.pt --start "中央大街" --season winter --n_routes 3
"""

import argparse
import os
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
from src.visualize import plot_route_on_map


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_poi_by_name(pois: pd.DataFrame, query: str) -> list[int]:
    """模糊匹配 POI 名称，返回匹配的索引列表."""
    matches = pois[pois["name"].str.contains(query, case=False, na=False)]
    if matches.empty:
        return []
    return matches.index.tolist()


def print_route_detail(route: list[int], pois: pd.DataFrame,
                       dist_matrix: np.ndarray, time_matrix: np.ndarray):
    """终端打印路线详情."""
    print(f"\n{'=' * 60}")
    print(f"  {'序号':>4s}  {'POI名称':<16s}  {'类别':<6s}  {'评分':>4s}  {'距上一站':>8s}  {'耗时':>6s}")
    print(f"  {'-'*4:>4s}  {'-'*16:<16s}  {'-'*6:<6s}  {'-'*4:>4s}  {'-'*8:>8s}  {'-'*6:>6s}")

    total_dist = 0.0
    total_time = 0.0
    for i, idx in enumerate(route):
        row = pois.iloc[idx]
        name = row.get("name", f"POI-{idx}")
        cat = row.get("category", "-")
        rating = row.get("rating", 0)
        seg_dist = dist_matrix[route[i - 1], idx] if i > 0 else 0.0
        seg_time = time_matrix[route[i - 1], idx] if i > 0 else 0.0
        total_dist += seg_dist
        total_time += seg_time
        dist_str = f"{seg_dist:.1f}km" if i > 0 else "-"
        time_str = f"{seg_time:.0f}min" if i > 0 else "-"
        print(f"  {i+1:>4d}  {name:<16s}  {str(cat):<6s}  {rating:>4.1f}  {dist_str:>8s}  {time_str:>6s}")

    print(f"  {'─' * 56}")
    print(f"  总计: {len(route)} 个游览点 | {total_dist:.1f}km | {total_time:.0f}分钟 ({total_time/60:.1f}小时)")


def generate_route(model: RouteTransformer, poi_features: torch.Tensor,
                   adjacency: torch.Tensor, config: dict,
                   device: torch.device, beam_size: int = 5) -> list[int]:
    """模型生成一条路线."""
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(poi_features, adjacency)
        routes = model.generate(encoder_output, beam_size=beam_size)
    return routes[0].cpu().tolist()


def filter_route_by_time(route: list[int], time_matrix: np.ndarray,
                         max_minutes: float) -> list[int]:
    """按时间预算截断路线."""
    total = 0.0
    for i in range(1, len(route)):
        seg = time_matrix[route[i - 1], route[i]]
        if total + seg > max_minutes:
            return route[:i]
        total += seg
    return route


def main():
    parser = argparse.ArgumentParser(description="哈尔滨文旅路线推荐")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="数据目录")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备 (cuda/cpu)")

    # 输入条件
    parser.add_argument("--start", type=str, default=None,
                        help="起点 POI 名称（模糊匹配）")
    parser.add_argument("--start_id", type=int, default=None,
                        help="起点 POI 编号")
    parser.add_argument("--season", type=str, default="winter",
                        choices=["winter", "summer"],
                        help="季节 (winter/summer)")
    parser.add_argument("--max_stops", type=int, default=None,
                        help="最大游览点数（默认使用配置文件 max_route_len）")
    parser.add_argument("--max_hours", type=float, default=None,
                        help="时间预算（小时），超出则截断路线")
    parser.add_argument("--beam_size", type=int, default=None,
                        help="Beam Search 宽度")
    parser.add_argument("--n_routes", type=int, default=1,
                        help="生成候选路线数量")

    args = parser.parse_args()

    # 加载配置
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

    if (data_dir / "poi_metadata.csv").exists():
        pois = pd.read_csv(data_dir / "poi_metadata.csv")
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
            for _, row in pois.iterrows():
                print(f"  #{row.name}: {row.get('name', '?')} ({row.get('category', '')})")
            return
        start_id = matches[0]
        if len(matches) > 1:
            print(f"找到 {len(matches)} 个匹配项，使用第一个: #{start_id} {pois.iloc[start_id]['name']}")

    if start_id is not None:
        print(f"起点: #{start_id} {pois.iloc[start_id].get('name', '?')}")

    # 加载模型
    model = RouteTransformer(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"模型加载完成 (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('best_val_loss', '?')})")

    # 准备输入
    poi_feat_dev = poi_features.unsqueeze(0).to(device)
    adj_dev = adjacency.unsqueeze(0).to(device)

    # 生成路线
    print(f"\n生成条件: 季节={args.season}, 最大游览点={max_stops}, beam_size={beam_size}")
    print(f"{'=' * 60}")

    all_routes = []
    for i in range(args.n_routes):
        with torch.no_grad():
            encoder_output = model.encode(poi_feat_dev, adj_dev)
            routes = model.generate(encoder_output, beam_size=beam_size)
        route = routes[0].cpu().tolist()

        # 过滤 padding (0) 和重复
        filtered = []
        seen = set()
        for poi_id in route:
            if poi_id == 0 and len(filtered) > 0:
                break
            if poi_id in seen:
                continue
            seen.add(poi_id)
            filtered.append(poi_id)
        route = filtered[:max_stops]

        # 按时间预算截断
        if args.max_hours is not None:
            route = filter_route_by_time(route, time_matrix, args.max_hours * 60)

        all_routes.append(route)

        if args.n_routes > 1:
            print(f"\n--- 候选路线 {i + 1} ---")

        print_route_detail(route, pois, dist_matrix, time_matrix)

    # 综合评分
    metrics_cfg = config.get("metrics", {})
    weights = {
        "distance": metrics_cfg.get("distance_weight", 0.30),
        "time": metrics_cfg.get("time_weight", 0.25),
        "satisfaction": metrics_cfg.get("satisfaction_weight", 0.25),
        "diversity": metrics_cfg.get("diversity_weight", 0.20),
    }

    print(f"\n{'=' * 60}")
    print("路线评估:")
    for i, route in enumerate(all_routes):
        if len(route) < 2:
            continue
        d = route_distance(route, dist_matrix)
        t = route_time(route, time_matrix)
        s = satisfaction_score(route, ratings)
        div = diversity_score(route, categories)
        max_d = route_distance(list(range(n_pois)), dist_matrix) if n_pois > 1 else 1.0
        max_t = route_time(list(range(n_pois)), time_matrix) if n_pois > 1 else 1.0
        m = {"distance": d, "time": t, "satisfaction": s, "diversity": div,
             "max_distance": max_d, "max_time": max_t}
        score = composite_score(m, weights)
        label = f"路线 {i + 1}" if args.n_routes > 1 else "推荐路线"
        print(f"  {label}: 距离={d:.1f}km | 耗时={t:.0f}min | 满意度={s:.2f} | 多样性={div:.2f} | 综合={score:.3f}")

    # 生成地图
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    best_route = all_routes[0]
    map_path = str(output_dir / "route_map.html")
    plot_route_on_map(pois, best_route, output_path=map_path,
                      center_lat=config["data"]["center_lat"],
                      center_lng=config["data"]["center_lng"])
    print(f"\n地图已保存: {map_path}")


if __name__ == "__main__":
    main()
