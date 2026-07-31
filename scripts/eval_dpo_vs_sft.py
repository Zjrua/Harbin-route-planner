"""评估 DPO 微调效果：对比 SFT vs DPO 生成的路线质量.

用相同的测试指令，分别用 SFT 模型和 DPO 模型生成路线，
用 v4 打分对比质量（就近/节奏/时间/满意度/多样性）。

用法:
    ./.venv/Scripts/python.exe scripts/eval_dpo_vs_sft.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import composite_score_v3

MODEL_PATH = str(Path("data/external/modelscope_cache/models/Qwen--Qwen3-4B/snapshots/master"))
SFT_LORA = "output/qwen_route_lora"
DPO_LORA = "output/qwen_route_dpo"

TEST_INSTRUCTIONS = [
    "帮我规划一条哈尔滨一日游路线，冰雪季出行，总预算约500元，从中央大街出发，希望多去经典景点。",
    "帮我规划一条哈尔滨两日游路线，夏季出行，预算约1500元，喜欢美食和购物，节奏不要太赶。",
    "帮我规划一条哈尔滨三日游路线，冬季出行，预算约3000元，以冰雪大世界和中央大街为核心。",
    "我在哈尔滨只有半天时间，想从圣索菲亚教堂附近出发，看看冰雪大世界，怎么安排？",
    "带父母去哈尔滨玩三天，预算充足，节奏要慢，适合老年人的景点优先。",
]


def load_model(lora_dir, device):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, lora_dir)
    return model, tokenizer


def generate_route(model, tokenizer, instruction, max_new_tokens=250):
    msgs = [{"role": "system",
             "content": "你是一位哈尔滨旅游规划专家，根据用户的需求生成合理的旅游路线。路线用 POI 名称以 → 连接。"},
            {"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return resp


def parse_route(text):
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


def match_and_score(names, pois, dist_matrix, time_matrix, ratings, categories, activity_types):
    """POI 名称 → 索引 → v4 打分."""
    matched = []
    for name in names:
        exact = pois[pois["name"] == name]
        if len(exact) > 0:
            matched.append(int(exact.index[0])); continue
        contains = pois[pois["name"].str.contains(re.escape(name), na=False, regex=True)]
        if len(contains) > 0:
            matched.append(int(contains.index[0]))
    if len(matched) < 3:
        return None, len(matched) / max(len(names), 1)
    n_days = 1 if len(matched) <= 10 else (2 if len(matched) <= 16 else 3)
    result = composite_score_v3(matched, dist_matrix, time_matrix, ratings,
                                categories, n_days=n_days, activity_types=activity_types)
    return result, len(matched) / max(len(names), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo-lora", default=DPO_LORA)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("data/processed")
    pois = pd.read_csv(data_dir / "poi_metadata.csv", encoding="utf-8")
    dist_matrix = np.load(data_dir / "distance_matrix.npy")
    time_matrix = np.load(data_dir / "time_matrix.npy")
    ratings = pois["rating"].values
    categories = pois["category"].values
    activity_types = np.load(data_dir / "poi_activity_types.npy")

    # 加载 SFT 和 DPO 模型
    sft_model, sft_tok = load_model(SFT_LORA, device)
    print("SFT 模型加载完成")
    dpo_model, dpo_tok = load_model(args.dpo_lora, device)
    print("DPO 模型加载完成")

    results = {}
    for name, model, tok in [("SFT", sft_model, sft_tok), ("DPO", dpo_model, dpo_tok)]:
        print(f"\n=== {name} 模型 ===")
        scores = []
        for i, instr in enumerate(TEST_INSTRUCTIONS):
            raw = generate_route(model, tok, instr)
            names = parse_route(raw)
            result, match_rate = match_and_score(names, pois, dist_matrix, time_matrix,
                                                 ratings, categories, activity_types)
            if result is not None:
                scores.append(result["score"])
                feasible = "✓" if result.get("feasible") else f"✗({result.get('reason')})"
                print(f"  [{i+1}] v4={result['score']:.4f} {feasible} 匹配率{match_rate:.0%} 距离{result['metrics']['total_dist_km']:.0f}km")
            else:
                print(f"  [{i+1}] ⚠️ 无法匹配/评估 (匹配率{match_rate:.0%})")
        avg = np.mean(scores) if scores else 0
        print(f"  平均 v4 得分: {avg:.4f} ({len(scores)}/5 可评估)")
        results[name] = {"avg_score": round(float(avg), 4), "n_eval": len(scores),
                         "scores": [round(s, 4) for s in scores]}

    # 对比
    print("\n" + "=" * 50)
    print("SFT vs DPO 对比")
    print("=" * 50)
    for name, r in results.items():
        print(f"  {name}: 平均v4={r['avg_score']}, 可评估{r['n_eval']}/5")

    out_path = Path("output/dpo_vs_sft.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
