"""用 v4 打分生成 DPO 偏好对数据集.

流程：
1. 用微调后的 Qwen 对每条指令采样 N 条路线（temperature 高，生成多样）
2. 用 v4 打分（就近/节奏/时间/满意度/多样性）对每条路线评分
3. 高分路线 → chosen, 低分路线 → rejected，构造 DPO 偏好对

输出：data/qwen_dpo_dataset.jsonl
  每行: {"prompt": 指令, "chosen": 路线A, "rejected": 路线B}

用法:
    ./.venv/Scripts/python.exe scripts/prepare_dpo_dataset.py [--n-instructions 200] [--samples 8]
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import composite_score_v3

MODEL_PATH = str(Path("data/external/modelscope_cache/models/Qwen--Qwen3-4B/snapshots/master"))
LORA_DIR = "output/qwen_route_lora"
DATA_DIR = Path("data/processed")

# 指令池（从现有指令数据集采样）
INSTRUCTION_PATH = "data/qwen_instruction_dataset.jsonl"

# 路线停留时间（与 v4 打分一致）
STAY = {0: 45, 1: 60, 2: 0, 3: 0, 4: 40, 5: 0}


def load_instructions(path, n):
    """从指令数据集采样指令."""
    instrs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            instrs.append(d["instruction"])
    random.seed(42)
    random.shuffle(instrs)
    return instrs[:n]


def generate_route(model, tokenizer, instruction, max_new_tokens=200, temperature=0.9):
    """生成路线文本（temperature 高，多样）."""
    msgs = [{"role": "system",
             "content": "你是一位哈尔滨旅游规划专家，根据用户的需求生成合理的旅游路线。路线用 POI 名称以 → 连接。"},
            {"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=temperature,
                             top_p=0.9, num_return_sequences=1)
    resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return resp


def parse_route(text):
    """从生成文本解析 POI 名称序列."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?assistant", "", text, flags=re.DOTALL)
    matches = re.findall(
        r"([\u4e00-\u9fa5A-Za-z0-9·（）()]+(?:\s*(?:→|->)\s*[\u4e00-\u9fa5A-Za-z0-9·（）()]+){2,})",
        text,
    )
    if matches:
        longest = max(matches, key=len)
        return [p.strip() for p in re.split(r"\s*(?:→|->)\s*", longest) if p.strip()]
    return []


def match_route_to_indices(names, pois):
    """POI 名称 → 索引（尽量匹配，未匹配丢弃）."""
    matched = []
    for name in names:
        exact = pois[pois["name"] == name]
        if len(exact) > 0:
            matched.append(int(exact.index[0]))
            continue
        # POI 名含括号等正则特殊字符，用 re.escape 防止正则解析错误
        contains = pois[pois["name"].str.contains(re.escape(name), na=False, regex=True)]
        if len(contains) > 0:
            matched.append(int(contains.index[0]))
    return matched


def score_route(route_indices, dist_matrix, time_matrix, ratings, categories,
                activity_types, n_days):
    """用 v4 打分评估路线，返回分数 + 是否可行."""
    if len(route_indices) < 3:
        return None
    result = composite_score_v3(route_indices, dist_matrix, time_matrix, ratings,
                                categories, n_days=n_days, activity_types=activity_types)
    if not result.get("feasible"):
        return None
    return result["score"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-instructions", type=int, default=100)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--min-gap", type=float, default=0.05,
                        help="chosen 与 rejected 的最小分数差")
    args = parser.parse_args()

    # === 加载数据 ===
    pois = pd.read_csv(DATA_DIR / "poi_metadata.csv", encoding="utf-8")
    dist_matrix = np.load(DATA_DIR / "distance_matrix.npy")
    time_matrix = np.load(DATA_DIR / "time_matrix.npy")
    ratings = pois["rating"].values
    categories = pois["category"].values
    activity_types = np.load(DATA_DIR / "poi_activity_types.npy")

    # === 加载微调模型 ===
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, LORA_DIR)
    print("微调模型加载完成")

    # === 采样指令 ===
    instructions = load_instructions(INSTRUCTION_PATH, args.n_instructions)
    print(f"指令数: {len(instructions)}")

    # === 生成偏好对 ===
    pairs = []
    no_score = 0
    for i, instr in enumerate(instructions):
        # 采样多条路线
        routes_scores = []
        for _ in range(args.samples):
            raw = generate_route(model, tokenizer, instr,
                                 temperature=args.temperature)
            names = parse_route(raw)
            if len(names) < 3:
                continue
            indices = match_route_to_indices(names, pois)
            if len(indices) < 3:
                continue
            n_days = 1 if len(indices) <= 10 else (2 if len(indices) <= 16 else 3)
            sc = score_route(indices, dist_matrix, time_matrix, ratings, categories,
                             activity_types, n_days)
            if sc is not None:
                routes_scores.append((names, sc))

        if len(routes_scores) < 2:
            no_score += 1
            continue

        # 取最高分和最低分作为 chosen/rejected（差距够大才算）
        routes_scores.sort(key=lambda x: -x[1])
        best_names, best_score = routes_scores[0]
        worst_names, worst_score = routes_scores[-1]
        if best_score - worst_score < args.min_gap:
            continue

        chosen = "→".join(best_names)
        rejected = "→".join(worst_names)
        pairs.append({
            "prompt": instr,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": round(float(best_score), 4),
            "rejected_score": round(float(worst_score), 4),
        })

        if (i + 1) % 20 == 0:
            print(f"  进度 {i+1}/{len(instructions)}, 已生成 {len(pairs)} 对")

    # === 保存 ===
    out_path = Path("data/qwen_dpo_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n生成偏好对: {len(pairs)}")
    print(f"跳过（无法打分/差距不足）: {no_score}")
    print(f"保存至 {out_path}")

    # 样例
    if pairs:
        print("\n=== 样例 ===")
        p = pairs[0]
        print(f"指令: {p['prompt'][:60]}...")
        print(f"chosen ({p['chosen_score']}): {p['chosen'][:70]}...")
        print(f"rejected ({p['rejected_score']}): {p['rejected'][:70]}...")


if __name__ == "__main__":
    main()
