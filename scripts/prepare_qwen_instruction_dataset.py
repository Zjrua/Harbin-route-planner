"""将 routes.npy 转换为 Qwen 指令跟随数据集（自然语言约束 → 路线序列）.

设计（按用户思路）：
- 自然语言输入：包含时间（季节/天数）、预算、偏好等旅游约束
- 结构化输出：POI 名称序列（路线）

从现有路线的元数据合成指令：
- 季节：routes 的 season 字段（冬季/夏季/四季）
- 预算：从 POI 的 avg_cost 估算路线总预算
- 天数：从路线长度和站点数估算
- 偏好：从路线类别构成提取（景点为主/美食为主/购物）
- 出发地：路线第一个 POI

输出格式（JSONL）：
{"instruction": "帮我规划一条哈尔滨旅游路线。...", "output": "中央大街→圣索菲亚教堂→..."}

用法:
    ./.venv/Scripts/python.exe scripts/prepare_qwen_instruction_dataset.py [--max-samples 5000]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEASON_MAP = {
    "winter": ("冰雪季", "冬季", "12月-次年2月", "哈尔滨冰雪大世界开放期间"),
    "summer": ("夏季", "暑假", "7-8月", "避暑"),
    "spring": ("春季", "3-5月"),
    "autumn": ("秋季", "9-10月"),
    "winter/summer": ("冬季或夏季", "淡季或旺季"),
    "四季皆宜": ("四季", "任何季节"),
}

PREFERENCE_TEMPLATES = [
    "我比较喜欢景点和美食，希望多去一些经典景点。",
    "希望兼顾购物和美食，行程不要太赶。",
    "以经典景点为主，可以适当安排住宿和餐饮。",
    "亲子游，希望景点多样化，节奏适中。",
    "文化爱好者，喜欢历史建筑和博物馆。",
]

START_TEMPLATES = [
    "从{start}出发",
    "以{start}为起点",
    "第一站想去{start}",
]

def infer_season(route_meta):
    """从路线的 season 元数据推断季节（routes_source 里没有，用 poi 或默认）."""
    return "winter" if random.random() < 0.5 else "summer"


def infer_budget(route, pois):
    """从 POI avg_cost 估算路线总预算（如果数据里有 avg_cost 列）."""
    costs = []
    for idx in route:
        if idx < len(pois):
            c = pois.iloc[idx].get("avg_cost", np.nan)
            if pd.notna(c) and c > 0:
                costs.append(float(c))
    if costs:
        total = sum(costs)
        return max(500, int(round(total, -2)))  # 四舍五入到百
    return None


def infer_days(route):
    """从路线长度估算天数（短路线=1天，长路线=2-3天）."""
    n = len(route)
    if n <= 8:
        return 1
    elif n <= 14:
        return random.choice([1, 2])
    else:
        return random.choice([2, 3])


def build_instruction(route, pois, route_meta, rng):
    """构造自然语言指令."""
    start = pois.iloc[route[0]]["name"] if route[0] < len(pois) else "中央大街"
    days = infer_days(route)
    budget = infer_budget(route, pois)
    season = infer_season(route_meta)

    season_names = {
        "winter": "冰雪季（冬季）",
        "summer": "夏季（暑假）",
    }.get(season, "四季皆宜")

    parts = []
    # 时间
    if days == 1:
        parts.append(f"帮我规划一条哈尔滨一日游路线")
    else:
        parts.append(f"帮我规划一条哈尔滨{days}日游路线")
    # 季节
    parts.append(f"{season_names}出行")
    # 预算（合理的哈尔滨旅游预算区间）
    if budget:
        # 至少 300 元，最高 5000 元，避免不合理低价
        budget = max(300, min(budget, 5000))
        parts.append(f"总预算约{budget}元")
    # 偏好
    pref = rng.choice(PREFERENCE_TEMPLATES).rstrip("。")
    parts.append(pref)
    # 起点
    parts.append(rng.choice(START_TEMPLATES).format(start=start))

    return "，".join(parts) + "。"


def build_output(route, pois):
    """构造结构化输出：POI 名称序列."""
    names = []
    for idx in route:
        if idx < len(pois):
            names.append(str(pois.iloc[idx]["name"]))
    return "→".join(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path("data/processed")
    pois = pd.read_csv(data_dir / "poi_metadata.csv", encoding="utf-8")
    routes = np.load(data_dir / "routes.npy", allow_pickle=True)

    rng = random.Random(args.seed)
    samples = []

    for i, route in enumerate(routes):
        if len(route) < 3:
            continue
        instruction = build_instruction(route, pois, None, rng)
        output = build_output(route, pois)
        if not output:
            continue
        samples.append({
            "instruction": instruction,
            "output": output,
            "route_len": len(route),
            "source": "xhs" if i < 168 else "synthetic",
        })
        if len(samples) >= args.max_samples:
            break

    out_path = Path("data/qwen_instruction_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"生成样本: {len(samples)}")
    print("\n=== 样例 ===")
    for s in samples[:3]:
        print(f"指令: {s['instruction']}")
        print(f"输出: {s['output'][:80]}...")
        print()
    print(f"保存至 {out_path}")


if __name__ == "__main__":
    main()
