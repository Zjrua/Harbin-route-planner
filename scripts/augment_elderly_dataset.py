"""数据增强：注入"带父母/老年人/慢节奏"指令变体.

背景：训练数据 5149 条里没有"带父母/老年人/慢节奏"类指令，
模型对此零样本泛化，把"慢节奏"误解为"少去景点"（三日游只出 4-6 站）。

做法：
- 从现有指令里挑"3日游"且路线较长的样本
- 把指令改写成带父母/老年人/慢节奏的版本（保留天数/季节/预算/起点）
- 路线保持原样 → 让模型学到"带父母 = 节奏舒缓但天数/站数不变"

输出：data/qwen_instruction_dataset_aug.jsonl（原样本 + 增强样本）

用法:
    ./.venv/Scripts/python.exe scripts/augment_elderly_dataset.py [--n 400]
"""

import argparse
import json
import random
import re
from pathlib import Path

SRC = "data/qwen_instruction_dataset.jsonl"
OUT = "data/qwen_instruction_dataset_aug.jsonl"

# 改写模板（多样化，避免模型过拟合单一措辞）
TEMPLATES = [
    "带父母去哈尔滨玩{days}天，{season}出行，预算约{budget}元，节奏要慢，适合老年人的景点优先，从{start}出发。",
    "陪老人游览哈尔滨{days}日，{season}出行，预算充足，希望行程舒缓，以经典景点为主，从{start}出发。",
    "带父母去哈尔滨{days}日游，{season}，预算约{budget}元，父母年纪大走不快，每天安排适量景点即可，从{start}出发。",
    "带爸妈去哈尔滨玩{days}天，{season}，预算约{budget}元，节奏放慢，少走路多休息，老年人友好的景点优先，从{start}出发。",
    "带父母去哈尔滨{days}日游，{season}出行，预算充足，父母喜欢安静不赶路，适合老年人的景点优先，从{start}出发。",
]

SEASON_EXAMPLES = [
    "冰雪季（冬季）",
    "冬季",
    "夏季",
    "夏季（暑假）",
]


def parse_instr(instr: str):
    """从现有指令提取 天数 / 季节 / 预算 / 起点."""
    days_m = re.search(r"(\d+)日", instr)
    days = days_m.group(1) if days_m else "3"
    season = "冰雪季（冬季）" if ("冬" in instr) else ("夏季" if "夏" in instr else "四季")
    budget_m = re.search(r"预算约?(\d+)元", instr)
    budget = budget_m.group(1) if budget_m else "2000"
    start_m = re.search(r"从(.+?)出发", instr)
    start = start_m.group(1) if start_m else "哈尔滨中央大街"
    return days, season, budget, start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400, help="增强样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-stops", type=int, default=14,
                        help="只选路线 >= 此站数的样本（保证三日游路线够长）")
    args = parser.parse_args()
    random.seed(args.seed)

    src = list(map(json.loads, open(SRC, encoding="utf-8")))

    # 挑"3日"且路线较长的样本
    candidates = []
    for d in src:
        instr, out = d["instruction"], d["output"]
        pois = [p.strip() for p in re.split(r"\s*(?:→|->)\s*", out) if p.strip()]
        if "3日" in instr and len(pois) >= args.min_stops:
            candidates.append((instr, out, len(pois)))
    print(f"候选样本（3日游且 ≥{args.min_stops}站）: {len(candidates)}")

    if not candidates:
        print("没有候选样本，退出"); return
    random.shuffle(candidates)
    chosen = candidates[: args.n]

    aug = []
    for instr, out, n_stops in chosen:
        days, season, budget, start = parse_instr(instr)
        tpl = random.choice(TEMPLATES)
        new_instr = tpl.format(days=days, season=season, budget=budget, start=start)
        aug.append({"instruction": new_instr, "output": out})

    # 合并原样本 + 增强
    merged = src + aug
    with open(OUT, "w", encoding="utf-8") as f:
        for d in merged:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"增强样本: {len(aug)} 条（路线站数 {min(c[2] for c in chosen)}-{max(c[2] for c in chosen)}）")
    print(f"合并后总量: {len(merged)} → {OUT}")
    print("\n=== 增强样例 ===")
    for d in aug[:3]:
        print(f"  指令: {d['instruction'][:70]}")
        print(f"  路线: {d['output'][:60]}...")


if __name__ == "__main__":
    main()
