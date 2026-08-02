"""逐日候选编号数据增强：为 Qwen3.5 新基模生成 SFT 训练数据.

背景（Phase 1 验证结论）：
- 模型只有 4B 参数，记不住 10K 个真实 POI 名 → 长序列生成会编造名字
- 正确形态：RAG 检索候选 → 模型从候选编号里选（编号输出 + 后端映射）

本脚本把现有路线数据改造成"候选编号"训练格式：
1. 取长路线（≥8 站）→ 按每天 5-6 站拆分成"天"级子路线
2. 每天：该天的 POI 作为"黄金答案"进入候选（保证编号可选到），
   再用检索补足到 8 个候选（全部真实 POI）
3. prompt: 候选编号列表 + 要求输出编号路线
   output: 黄金答案对应的编号序列

输出：data/qwen35_sft_dataset.jsonl（候选编号格式）

用法:
    ./.venv/Scripts/python.exe scripts/prepare_qwen35_dataset.py [--max-samples 4000]
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import retrieve_candidates, resolve_poi_index

SRC = "data/qwen_instruction_dataset_aug.jsonl"
OUT = "data/qwen35_sft_dataset.jsonl"
DATA_DIR = Path("data/processed")

# 每天目标站数（全天）
DAY_STOPS = 6
# 候选总数
N_CAND = 8

SYSTEM_PROMPT = (
    "你是一位哈尔滨旅游规划专家。你会收到候选景点编号列表，"
    "只能输出候选编号组成的路线，编号用 → 连接，不要输出任何其他文字。"
)


def parse_route_names(text: str):
    return [p.strip() for p in re.split(r"\s*(?:→|->)\s*", text.strip()) if p.strip()]


def split_route(names: list, per_day: int = 6):
    """把一条路线拆成逐日子路线（每天 per_day 个）. 返回 [day1_names, day2_names, ...]"""
    days = []
    for i in range(0, len(names), per_day):
        chunk = names[i:i + per_day]
        if len(chunk) >= 3:  # 少于 3 站的天丢弃（太短无意义）
            days.append(chunk)
    return days


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cand", type=int, default=N_CAND)
    parser.add_argument("--per-day", type=int, default=DAY_STOPS)
    args = parser.parse_args()
    random.seed(args.seed)

    pois = pd.read_csv(DATA_DIR / "poi_metadata.csv", encoding="utf-8")
    dist = np.load(DATA_DIR / "distance_matrix.npy")

    # 名称 → 索引
    name2idx = {}
    for i, nm in enumerate(pois["name"].tolist()):
        name2idx[nm] = i

    samples = []
    skipped = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            names = parse_route_names(d["output"])
            if len(names) < 8:
                skipped += 1
                continue
            # 转索引（能匹配的）
            idxs = [name2idx[n] for n in names if n in name2idx]
            if len(idxs) < 8:
                skipped += 1
                continue
            # 拆成逐日
            day_chunks = split_route(idxs, args.per_day)
            for day_idx, chunk in enumerate(day_chunks):
                # 黄金答案 = 该天 POI（按路线顺序）
                gold = chunk
                # 检索补足候选：以第一个 POI 为中心，排除已用，补到 n_cand
                center = gold[0]
                used = set()
                cands = list(gold)  # 黄金答案先占位（保证可选到）
                used.update(gold)
                # 检索补足
                extra = retrieve_candidates(
                    pois, dist, center_idx=center, used_indices=used,
                    n=args.n_cand - len(gold),
                    type_budget={0: 2, 1: 2, 4: 1, 2: 1},
                )
                cands.extend(extra)
                cands = list(dict.fromkeys(cands))[: args.n_cand]
                if len(cands) < args.n_cand:
                    continue
                # 打乱候选：黄金答案随机散落在 1..N（模型需"从候选中挑"，
                # 而非"输出前 N 个编号"——避免学到 1→2→3→... 的捷径）
                random.shuffle(cands)
                # 编号：黄金答案在候选任意位置（保持都在）
                idx_to_num = {idx: i + 1 for i, idx in enumerate(cands)}
                if any(g not in idx_to_num for g in gold):
                    skipped += 1
                    continue
                # output = 黄金答案的编号序列
                out_nums = [str(idx_to_num[g]) for g in gold]
                cand_lines = "\n".join(
                    f"{i+1}.{pois.loc[idx, 'name']}" for i, idx in enumerate(cands))
                instr = (
                    f"请为第{day_idx + 1}天安排{len(gold)}个左右景点的路线。候选景点：\n"
                    f"{cand_lines}\n"
                    f"输出{len(gold)}个左右景点编号的路线，编号用 → 连接"
                    f"（如 1→3→5→2→8→6），只能从上述 {len(cands)} 个候选中选。"
                )
                samples.append({
                    "system": SYSTEM_PROMPT,
                    "instruction": instr,
                    "output": "→".join(out_nums),
                    "day": day_idx + 1,
                    "gold_count": len(gold),
                })
                if len(samples) >= args.max_samples:
                    break
            if len(samples) >= args.max_samples:
                break

    with open(OUT, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"生成训练样本: {len(samples)}（跳过 {skipped}）")
    print(f"保存至: {OUT}")
    if samples:
        print("\n=== 样例 ===")
        s = samples[0]
        print(f"system: {s['system'][:40]}...")
        print(f"instruction: {s['instruction'][:120]}...")
        print(f"output: {s['output']}")


if __name__ == "__main__":
    main()
