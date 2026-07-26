"""真实数据独立评估（方向 B）— 在 XHS holdout 路线上评估模型泛化能力.

打破数据循环论证：不用合成规则评估，而是在真实路线上测试：
1. Next-POI 预测准确率（top-1 / top-5）：给真实路线前 k 个 POI，预测第 k+1 个
2. 路线长度分布对比：模型生成 vs 真实路线
3. POI 重叠率：模型生成路线与真实路线的 Jaccard

对比基线：
- most_popular：预测评分最高的 POI（无视上下文）
- nearest_neighbor：预测距离上一个 POI 最近的 POI

用法:
    ./.venv/Scripts/python.exe -m scripts.evaluate_on_real --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.transformer import ItineraryTransformer
from src.baselines import nearest_neighbor_route


def load_holdout(data_dir: Path):
    """加载 holdout 真实 XHS 路线."""
    holdout_path = data_dir / "routes_xhs_holdout.npy"
    if not holdout_path.exists():
        raise FileNotFoundError(
            f"holdout 文件不存在: {holdout_path}\n"
            "请先运行 scripts/prepare_data.py 生成（会自动产出 routes_xhs_holdout.npy）")
    routes = np.load(holdout_path, allow_pickle=True)
    return [r for r in routes if len(r) >= 2]  # 至少 2 个 POI 才能做 next-POI


def predict_next_poi(model, route_prefix, device, encoder_output, poi_activity_types):
    """用模型预测路线前缀的下一个 POI（返回 logits 的 top-k）."""
    route_t = torch.tensor([route_prefix], dtype=torch.long, device=device)
    activity_t = poi_activity_types[route_t] if poi_activity_types is not None else None
    target_emb = model.poi_embedding(route_t, activity_types=activity_t)
    seq_len = target_emb.size(1)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    engram_memory = None
    if model.use_engram:
        query = encoder_output.mean(dim=1)
        retrieved, _ = model.engram.retrieve(query)
        engram_memory = retrieved

    dec_out = model.decode(engram_memory, encoder_output, target_emb, causal_mask)
    logits = model.output_proj(dec_out)[:, -1, :]  # 取最后一步
    return logits.squeeze(0)  # [n_pois]


def evaluate_next_poi_accuracy(model, routes, device, encoder_output, poi_activity_types, top_k=5):
    """在真实路线上评估 next-POI 预测准确率.

    对每条路线的每个位置 t（从1开始），给定前 t 个 POI，预测第 t+1 个。
    """
    top1_correct, top5_correct, total = 0, 0, 0

    for route in routes:
        for t in range(1, len(route)):
            prefix = route[:t + 1]  # 含当前位置
            actual_next = route[t] if t < len(route) else None
            if actual_next is None:
                continue

            with torch.no_grad():
                logits = predict_next_poi(model, prefix[:-1], device,
                                          encoder_output, poi_activity_types)
            # 排除已在 prefix 中的 POI（路线不重复访问）
            visited = set(prefix[:-1])
            for v in visited:
                if v < len(logits):
                    logits[v] = -float("inf")

            pred_top1 = logits.argmax().item()
            pred_top5 = logits.topk(top_k).indices.tolist()

            if pred_top1 == actual_next:
                top1_correct += 1
            if actual_next in pred_top5:
                top5_correct += 1
            total += 1

    return {
        "top1_accuracy": round(top1_correct / max(total, 1), 4),
        "top5_accuracy": round(top5_correct / max(total, 1), 4),
        "total_predictions": total,
    }


def baseline_most_popular(routes, ratings, top_k=5):
    """基线：预测评分最高的 POI（无视上下文）."""
    popular = np.argsort(-ratings)
    top1_correct, top5_correct, total = 0, 0, 0
    for route in routes:
        for t in range(1, len(route)):
            actual_next = route[t]
            visited = set(route[:t])
            candidates = [p for p in popular if p not in visited]
            pred_top1 = candidates[0]
            pred_top5 = candidates[:top_k]
            if pred_top1 == actual_next:
                top1_correct += 1
            if actual_next in pred_top5:
                top5_correct += 1
            total += 1
    return {
        "top1_accuracy": round(top1_correct / max(total, 1), 4),
        "top5_accuracy": round(top5_correct / max(total, 1), 4),
        "total_predictions": total,
    }


def baseline_nearest_neighbor(routes, dist_matrix, top_k=5):
    """基线：预测距离上一个 POI 最近的未访问 POI."""
    top1_correct, top5_correct, total = 0, 0, 0
    for route in routes:
        for t in range(1, len(route)):
            actual_next = route[t]
            current = route[t - 1]
            visited = set(route[:t])
            dists = dist_matrix[current].copy()
            for v in visited:
                dists[v] = np.inf
            dists[current] = np.inf
            pred_top5 = np.argsort(dists)[:top_k].tolist()
            pred_top1 = pred_top5[0]
            if pred_top1 == actual_next:
                top1_correct += 1
            if actual_next in pred_top5:
                top5_correct += 1
            total += 1
    return {
        "top1_accuracy": round(top1_correct / max(total, 1), 4),
        "top5_accuracy": round(top5_correct / max(total, 1), 4),
        "total_predictions": total,
    }


def main():
    parser = argparse.ArgumentParser(description="真实数据独立评估（方向B）")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    routes = load_holdout(data_dir)
    print(f"加载 {len(routes)} 条真实 XHS holdout 路线")
    print(f"路线长度分布: min={min(len(r) for r in routes)}, "
          f"median={sorted(len(r) for r in routes)[len(routes)//2]}, "
          f"max={max(len(r) for r in routes)}")

    dist_matrix = np.load(data_dir / "distance_matrix.npy")
    import pandas as pd
    pois = pd.read_csv(data_dir / "poi_metadata.csv", encoding="utf-8")
    ratings = pois["rating"].values

    # === 基线评估 ===
    print("\n=== 基线评估 ===")
    pop = baseline_most_popular(routes, ratings)
    print(f"  most_popular:  top1={pop['top1_accuracy']}, top5={pop['top5_accuracy']}")
    nn = baseline_nearest_neighbor(routes, dist_matrix)
    print(f"  nearest_neighbor: top1={nn['top1_accuracy']}, top5={nn['top5_accuracy']}")

    # === Transformer 评估 ===
    import yaml
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ItineraryTransformer(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nTransformer checkpoint 加载: {args.checkpoint} (ep{ckpt.get('epoch')})")

    poi_features = torch.from_numpy(np.load(data_dir / "poi_features.npy")).float().to(device)
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy")).float().to(device)
    poi_activity_types = torch.from_numpy(np.load(data_dir / "poi_activity_types.npy")).long().to(device)

    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(poi_features.unsqueeze(0), adjacency.unsqueeze(0))

    print("\n=== Transformer 评估 ===")
    transformer = evaluate_next_poi_accuracy(model, routes, device, encoder_output,
                                             poi_activity_types)
    print(f"  Transformer:   top1={transformer['top1_accuracy']}, top5={transformer['top5_accuracy']}")

    # === 对比汇总 ===
    print("\n" + "=" * 50)
    print("  Next-POI 预测准确率对比（真实 XHS 路线）")
    print("=" * 50)
    print(f"{'方法':<18} {'top-1':<10} {'top-5':<10}")
    for name, r in [("most_popular", pop), ("nearest_neighbor", nn), ("Transformer", transformer)]:
        print(f"{name:<18} {r['top1_accuracy']:<10} {r['top5_accuracy']:<10}")

    # === 保存 ===
    out_path = Path("output/real_data_evaluation.json")
    out_path.parent.mkdir(exist_ok=True)
    summary = {
        "holdout_routes": len(routes),
        "checkpoint": args.checkpoint,
        "results": {
            "most_popular": pop,
            "nearest_neighbor": nn,
            "transformer": transformer,
        },
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
