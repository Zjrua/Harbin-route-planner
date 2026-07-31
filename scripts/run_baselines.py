"""基线对比实验：Transformer vs 启发式/OR 方法（方向 C）.

在同一组测试起点上，对比以下方法的路线质量：
1. Random：随机选 POI（下界）
2. Nearest Neighbor：贪心最近邻
3. 2-opt：NN + 2-opt 精修
4. OR-Tools：VRP 求解器（距离最优上界）
5. Transformer：ItineraryTransformer（约束 Beam Search）

所有方法用同一套评估（src.evaluate 的 distance/time/satisfaction/diversity/composite）。

用法:
    ./.venv/Scripts/python.exe -m scripts.run_baselines
    ./.venv/Scripts/python.exe -m scripts.run_baselines --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import random_route, nearest_neighbor_route, two_opt_route, ortools_route, has_ortools
from src.evaluate import route_distance, route_time, satisfaction_score, diversity_score, composite_score


def evaluate_route(route, dist_matrix, time_matrix, ratings, categories,
                   max_distance, max_time):
    """评估单条路线的 4 个指标 + composite."""
    if len(route) < 2:
        return None
    d = route_distance(route, dist_matrix)
    t = route_time(route, time_matrix)
    s = satisfaction_score(route, ratings)
    div = diversity_score(route, categories)
    metrics = {
        "distance": d, "time": t, "satisfaction": s, "diversity": div,
        "max_distance": max_distance, "max_time": max_time,
    }
    weights = {"distance": 0.30, "time": 0.25, "satisfaction": 0.25, "diversity": 0.20}
    metrics["composite"] = composite_score(metrics, weights)
    return metrics


def generate_transformer_routes(model, device, config, starts, length):
    """用 Transformer 生成路线（需要 checkpoint）."""
    import torch
    data_dir = Path("data/processed")
    poi_features = torch.from_numpy(np.load(data_dir / "poi_features.npy")).float()
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy")).float()
    poi_activity_types = torch.from_numpy(np.load(data_dir / "poi_activity_types.npy")).long().to(device)

    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(poi_features.unsqueeze(0).to(device),
                                      adjacency.unsqueeze(0).to(device))

    routes = []
    from src.inference import generate_with_constraints, filter_route
    cluster_id = np.load(data_dir / "cluster_id.npy") if (data_dir / "cluster_id.npy").exists() else None
    for start_id in starts:
        raw = generate_with_constraints(model, encoder_output, start_id,
                                        config["training"]["beam_size"], length,
                                        device, poi_activity_types, cluster_id)
        filtered = filter_route(raw, length)
        routes.append(filtered)
    return routes


def main():
    parser = argparse.ArgumentParser(description="基线对比：Transformer vs 启发式/OR")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                        help="Transformer checkpoint 路径（留空则跳过 Transformer）")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-starts", type=int, default=5, help="测试起点数")
    parser.add_argument("--length", type=int, default=10, help="路线长度")
    parser.add_argument("--ortools-time", type=float, default=5.0, help="OR-Tools 单次求解时间(秒)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data_dir = Path("data/processed")
    print("=" * 64)
    print("  基线对比实验：Transformer vs 启发式/OR 方法（方向 C）")
    print(f"  OR-Tools 可用: {has_ortools()}")
    print(f"  路线长度: {args.length}, 起点数: {args.n_starts}")
    print("=" * 64)

    # === 加载数据 ===
    dist_matrix = np.load(data_dir / "distance_matrix.npy")
    time_matrix = np.load(data_dir / "time_matrix.npy")
    pois = pd.read_csv(data_dir / "poi_metadata.csv", encoding="utf-8")
    ratings = pois["rating"].values
    categories = pois["category"].values
    n_pois = len(pois)
    max_distance = float(dist_matrix.max() * args.length)
    max_time = float(time_matrix.max() * args.length)

    # 选起点：高评分景点（与 run_ablation.py 一致，保证可比）
    scenic = pois[pois["category"] == "景点"].sort_values("rating", ascending=False)
    starts = scenic.index.tolist()[:args.n_starts]
    print(f"  起点: {starts}（top-{args.n_starts} 高评分景点）")

    # === 各方法生成路线 ===
    methods_routes = {}

    # Random
    methods_routes["random"] = [random_route(s, n_pois, args.length, rng) for s in starts]
    print(f"  random: {len(methods_routes['random'])} 条")

    # NN
    methods_routes["NN"] = [nearest_neighbor_route(s, dist_matrix, args.length) for s in starts]
    print(f"  NN: {len(methods_routes['NN'])} 条")

    # 2-opt
    methods_routes["2-opt"] = [two_opt_route(s, dist_matrix, args.length, iterations=100)
                               for s in starts]
    print(f"  2-opt: {len(methods_routes['2-opt'])} 条")

    # OR-Tools
    if has_ortools():
        print(f"  OR-Tools 求解中（每条 {args.ortools_time}s）...")
        methods_routes["OR-Tools"] = [ortools_route(s, dist_matrix, args.length,
                                                     time_limit_sec=args.ortools_time)
                                      for s in starts]
        print(f"  OR-Tools: {len(methods_routes['OR-Tools'])} 条")

    # Transformer（可选）
    run_transformer = args.checkpoint and Path(args.checkpoint).exists()
    if run_transformer:
        import torch
        import yaml
        from src.models.transformer import ItineraryTransformer
        config = yaml.safe_load(open(args.config, encoding="utf-8"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ItineraryTransformer(config).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Transformer checkpoint 加载: {args.checkpoint} (ep{ckpt.get('epoch')})")
        methods_routes["Transformer"] = generate_transformer_routes(
            model, device, config, starts, args.length)
        print(f"  Transformer: {len(methods_routes['Transformer'])} 条")
    else:
        print(f"  ⚠️  跳过 Transformer（checkpoint 不存在: {args.checkpoint}）")

    # === 评估 ===
    print("\n" + "=" * 64)
    print("  评估结果")
    print("=" * 64)
    print(f"{'方法':<14} {'avg_dist':<10} {'avg_time':<10} {'satisf':<8} {'divers':<8} {'composite':<10}")
    print("-" * 64)

    results = {}
    for method, routes in methods_routes.items():
        all_metrics = []
        for route in routes:
            m = evaluate_route(route, dist_matrix, time_matrix, ratings, categories,
                               max_distance, max_time)
            if m is not None:
                all_metrics.append(m)
        if not all_metrics:
            continue
        avg = {
            "avg_distance_km": round(float(np.mean([m["distance"] for m in all_metrics])), 2),
            "avg_time_min": round(float(np.mean([m["time"] for m in all_metrics])), 2),
            "avg_satisfaction": round(float(np.mean([m["satisfaction"] for m in all_metrics])), 3),
            "avg_diversity": round(float(np.mean([m["diversity"] for m in all_metrics])), 3),
            "composite": round(float(np.mean([m["composite"] for m in all_metrics])), 4),
        }
        results[method] = avg
        print(f"{method:<14} {avg['avg_distance_km']:<10} {avg['avg_time_min']:<10} "
              f"{avg['avg_satisfaction']:<8} {avg['avg_diversity']:<8} {avg['composite']:<10}")

    # === 保存 ===
    out_path = Path("output/baseline_comparison.json")
    out_path.parent.mkdir(exist_ok=True)
    summary = {
        "config": {"length": args.length, "n_starts": args.n_starts, "starts": starts,
                   "ortools_time_sec": args.ortools_time},
        "results": results,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
