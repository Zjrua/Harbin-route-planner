"""用打分 v3 重新评估各方法（真实路线 / Transformer / NN / OR-Tools）.

验证 v3 打分是否更合理：真实路线应得高分，启发式"连续同类"应被罚。

用法:
    ./.venv/Scripts/python.exe scripts/evaluate_v3_scoring.py [--n-starts 5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import composite_score_v3, compute_route_metrics


def evaluate_routes_v3(routes, dist_matrix, time_matrix, ratings, categories,
                       activity_types, n_days=None):
    """批量评估路线，返回 v4 得分列表（含硬约束判负信息）.

    n_days=None 时按站数自动推断天数（1日≤10站, 2日11-16, 3日≥17），
    保证不同方法的路线用统一规则校准时间预算。
    """
    scores = []
    for route in routes:
        if len(route) < 2:
            continue
        if n_days is None:
            n_days = 1 if len(route) <= 10 else (2 if len(route) <= 16 else 3)
        result = composite_score_v3(route, dist_matrix, time_matrix, ratings,
                                    categories, n_days=n_days,
                                    activity_types=activity_types)
        scores.append(result)
    return scores


def summarize(scores, name):
    """汇总一组路线的平均得分（含可行性统计）.

    软指标只在可行路线上求平均（不可行路线没有这些分量）。
    """
    if not scores:
        return None
    avg = {}
    feasible_scores = [s for s in scores if s.get("feasible")]
    for key in ["proximity", "area_density", "rhythm", "satisfaction", "diversity"]:
        vals = [s[key] for s in feasible_scores] if feasible_scores else [0.0]
        avg[key] = round(float(np.mean(vals)), 4)
    # 原始指标（所有路线，含不可行的）
    for key in ["total_dist_km", "total_time_min", "hop_p50_km", "hop_p90_km"]:
        vals = [s["metrics"][key] for s in scores]
        avg[key] = round(float(np.mean(vals)), 1)
    # 可行性统计
    avg["score"] = round(float(np.mean([s["score"] for s in scores])), 4)
    feasible = len(feasible_scores)
    reasons = {}
    for s in scores:
        if not s.get("feasible"):
            r = s.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
    avg["feasible_rate"] = round(feasible / len(scores), 2)
    avg["infeasible_reasons"] = reasons
    print(f"=== {name} ===")
    print(f"  v4得分: {avg['score']} | 可行性 {avg['feasible_rate']:.0%} | 就近性 {avg['proximity']} "
          f"| 区域密度 {avg['area_density']}")
    if avg["infeasible_reasons"]:
        print(f"  判负原因: {avg['infeasible_reasons']}")
    print(f"  节奏 {avg['rhythm']} | 满意度 {avg['satisfaction']} | 多样性 {avg['diversity']}")
    print(f"  总距离 {avg['total_dist_km']}km | 总耗时 {avg['total_time_min']}min "
          f"| 跳转p50 {avg['hop_p50_km']}km | p90 {avg['hop_p90_km']}km")
    print()
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-starts", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path("data/processed")
    pois = pd.read_csv(data_dir / "poi_metadata.csv", encoding="utf-8")
    dist_matrix = np.load(data_dir / "distance_matrix.npy")
    time_matrix = np.load(data_dir / "time_matrix.npy")
    ratings = pois["rating"].values
    categories = pois["category"].values
    # activity_types 用于停留时间计算（v4 时间约束的关键）
    act_path = data_dir / "poi_activity_types.npy"
    activity_types = np.load(act_path) if act_path.exists() else None

    results = {}

    # 天数标签（v4 时间约束按天数校准）：加载 routes_days.npy 或按站数估算
    days_path = data_dir / "routes_days.npy"
    if days_path.exists():
        routes_days_all = np.load(days_path)
    else:
        # 按站数估算（与 prepare_data.py 一致）
        routes_all = np.load(data_dir / "routes.npy", allow_pickle=True)
        routes_days_all = np.array([
            1 if len(r) <= 10 else (2 if len(r) <= 16 else 3)
            for r in routes_all
        ])

    # 1. 真实 XHS 路线（每条用其估算天数，而非默认1天）
    real = np.load(data_dir / "routes_xhs_holdout.npy", allow_pickle=True)
    real_routes = [r for r in real if len(r) >= 3]
    # holdout 是真实路线，天数按站数估算
    real_scores = []
    for r in real_routes[:50]:
        n_days = 1 if len(r) <= 10 else (2 if len(r) <= 16 else 3)
        result = composite_score_v3(r, dist_matrix, time_matrix, ratings,
                                    categories, n_days=n_days, activity_types=activity_types)
        real_scores.append(result)
    results["real_xhs"] = summarize(real_scores, "真实 XHS 路线")

    # 2. Transformer 生成（从真实起点）
    from src.models.transformer import ItineraryTransformer
    from src.inference import generate_with_constraints, filter_route
    import yaml

    config = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ItineraryTransformer(config).to(device)
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    pf = torch.from_numpy(np.load(data_dir / "poi_features.npy")).float().to(device)
    adj = torch.from_numpy(np.load(data_dir / "adjacency.npy")).float().to(device)
    poi_act = torch.from_numpy(np.load(data_dir / "poi_activity_types.npy")).long().to(device)
    cluster = np.load(data_dir / "cluster_id.npy")
    with torch.no_grad():
        enc = model.encode(pf.unsqueeze(0), adj.unsqueeze(0))

    transformer_routes = []
    for r in real_routes[:args.n_starts]:
        start = int(r[0])
        length = min(len(r), 12)
        raw = generate_with_constraints(model, enc, start, 5, length, device, poi_act, cluster)
        transformer_routes.append(filter_route(raw, length))
    transformer_scores = evaluate_routes_v3(transformer_routes, dist_matrix,
                                            time_matrix, ratings, categories,
                                            activity_types)
    results["transformer"] = summarize(transformer_scores, "Transformer 生成")

    # 3. 启发式（NN + 2-opt）
    from src.baselines import nearest_neighbor_route, two_opt_route, ortools_route
    nn_routes = [nearest_neighbor_route(int(r[0]), dist_matrix, min(len(r), 12))
                 for r in real_routes[:args.n_starts]]
    results["nn"] = summarize(evaluate_routes_v3(nn_routes, dist_matrix, time_matrix,
                                                 ratings, categories, activity_types),
                              "NN 最近邻")

    ortools_routes = [ortools_route(int(r[0]), dist_matrix, min(len(r), 12), time_limit_sec=3)
                      for r in real_routes[:args.n_starts]]
    results["ortools"] = summarize(evaluate_routes_v3(ortools_routes, dist_matrix,
                                                      time_matrix, ratings, categories,
                                                      activity_types),
                                   "OR-Tools")

    # 4. 随机
    from src.baselines import random_route
    rng = np.random.default_rng(42)
    random_routes = [random_route(int(r[0]), len(pois), min(len(r), 12), rng)
                     for r in real_routes[:args.n_starts]]
    results["random"] = summarize(evaluate_routes_v3(random_routes, dist_matrix,
                                                     time_matrix, ratings, categories,
                                                     activity_types),
                                  "随机")

    # 保存
    out_path = Path("output/v3_scoring_comparison.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out_path}")

    # 排名
    print("\n>>> v3 得分排名:")
    ranking = sorted(results.items(), key=lambda kv: -kv[1]["score"] if kv[1] else -1)
    for name, r in ranking:
        print(f"  {name}: {r['score']}")


if __name__ == "__main__":
    main()
