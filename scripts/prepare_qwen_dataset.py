"""将 routes.npy（POI 索引路线）转换为 Qwen 微调数据集.

范式：Multiple-Choice Next-POI
- 输入：已选 POI 列表（名称/类别/评分/与当前点距离）+ K 个候选 POI
- 输出：从候选中选择正确的下一个 POI

这是可控的 next-POI 预测：干扰项从全空间采样（优先近距离/高分），
让 Qwen 学会"就近 + 评分 + 类别"综合决策。

输出格式（JSONL，每行一条）：
{
  "instruction": "你在规划哈尔滨旅游路线，已选 POI：...",
  "candidates": "候选 A. ... B. ...",
  "output": "C"  (或候选的 POI 名)
}

用法:
    ./.venv/Scripts/python.exe scripts/prepare_qwen_dataset.py [--max-samples 5000] [--candidates 10]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ACT_NAMES = {0: "景点", 1: "餐饮", 2: "住宿", 3: "交通", 4: "购物", 5: "出发点"}


def build_candidates(pois, dist_matrix, current_idx, n_candidates, rng, true_next=None):
    """构造候选集：正确答案 + (n-1) 个干扰项.

    干扰项优先从"近距离 + 高评分"的 POI 中采样（更符合真实选择难度）。
    """
    n_pois = len(pois)
    candidates = [true_next] if true_next is not None else []

    # 候选池：按"距离当前点近 + 高评分"排序
    dists = dist_matrix[current_idx]
    scores = pois["rating"].values
    # 综合打分：距离归一化 + 评分，距离权重更高
    pool_score = 1.0 / (dists + 1.0) + scores * 0.1
    pool_order = np.argsort(-pool_score)

    for idx in pool_order:
        if len(candidates) >= n_candidates:
            break
        if idx not in candidates and idx != current_idx:
            candidates.append(idx)

    # 如果池子不够（极端情况），随机补
    while len(candidates) < n_candidates:
        idx = rng.randint(0, n_pois - 1)
        if idx not in candidates and idx != current_idx:
            candidates.append(idx)

    rng.shuffle(candidates)  # 打乱，避免正确答案总在第一位
    return candidates


def build_prompt(pois, dist_matrix, route_prefix, candidates, rng):
    """构造 instruction + candidates 文本."""
    # 已选 POI 描述
    lines = []
    for i, idx in enumerate(route_prefix):
        p = pois.iloc[idx]
        act = ACT_NAMES.get(int(p.get("activity_type", 0)), "景点")
        lines.append(f"{i+1}. {p['name']}（{act}，评分{p['rating']:.1f}）")

    # 当前点（最后访问的 POI）
    current_idx = route_prefix[-1]
    dists = dist_matrix[current_idx]

    cand_lines = []
    for tag, idx in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", candidates):
        p = pois.iloc[idx]
        act = ACT_NAMES.get(int(p.get("activity_type", 0)), "景点")
        d = dists[idx] if idx < len(dists) else -1
        dist_str = f"，距离当前点{d:.1f}km" if d >= 0 else ""
        cand_lines.append(f"{tag}. {p['name']}（{act}，评分{p['rating']:.1f}{dist_str}）")

    instruction = (
        "你在规划哈尔滨的旅游路线。以下是已游览的 POI：\n"
        + "\n".join(lines)
        + "\n\n请从以下候选中选择下一个最合适的 POI（考虑就近性、评分和活动节奏）：\n"
        + "\n".join(cand_lines)
        + "\n\n请只回答候选的字母编号。"
    )
    return instruction, current_idx, dists


def convert_routes(routes, pois, dist_matrix, max_samples, n_candidates, seed=42):
    """将路线转换为 (instruction, output_letter) 样本."""
    rng = random.Random(seed)
    samples = []
    used = 0

    for route in routes:
        if len(route) < 3:
            continue
        # 遍历路线中的每个位置作为预测目标（teacher forcing）
        for t in range(1, len(route) - 1):
            prefix = route[:t + 1]  # 已访问的前 t+1 个 POI
            true_next = route[t + 1] if t + 1 < len(route) else None
            if true_next is None or true_next >= len(pois):
                continue

            current_idx = prefix[-1]
            candidates = build_candidates(pois, dist_matrix, current_idx,
                                          n_candidates, rng, true_next)
            # 确保正确答案在候选里
            if true_next not in candidates:
                continue

            instruction, _, _ = build_prompt(pois, dist_matrix, prefix, candidates, rng)
            true_tag = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[candidates.index(true_next)]

            samples.append({
                "instruction": instruction,
                "candidates_count": len(candidates),
                "output": true_tag,
                "true_next": int(true_next),
                "prefix_len": len(prefix),
            })
            used += 1
            if used >= max_samples:
                return samples
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=8000)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path("data/processed")
    pois = pd.read_csv(data_dir / "poi_metadata.csv", encoding="utf-8")
    dist_matrix = np.load(data_dir / "distance_matrix.npy")
    routes = np.load(data_dir / "routes.npy", allow_pickle=True)

    print(f"POI: {len(pois)}, 路线: {len(routes)}")
    print(f"候选数: {args.candidates}, 目标样本: {args.max_samples}")

    samples = convert_routes(routes, pois, dist_matrix,
                             args.max_samples, args.candidates, args.seed)
    print(f"生成样本: {len(samples)}")

    out_path = Path("data/qwen_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 打印样例
    print("\n=== 样例 ===")
    print(samples[0]["instruction"][:600])
    print(f"\n输出: {samples[0]['output']}（真值 POI: {samples[0]['true_next']}）")
    print(f"\n数据集已保存: {out_path}")


if __name__ == "__main__":
    main()
