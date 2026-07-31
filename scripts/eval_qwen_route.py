"""Qwen 微调后推理评估：自然语言 → 路线.

评估内容：
1. 未微调 Qwen 生成路线
2. 微调后 Qwen 生成路线
3. 用 v3 打分对比生成质量
4. 解析路线中的 POI 是否真实存在（POI 名称匹配率）

用法:
    ./.venv/Scripts/python.exe scripts/eval_qwen_route.py [--lora-dir output/qwen_route_lora]
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

# 测试指令（覆盖多日/预算/季节/偏好）
TEST_INSTRUCTIONS = [
    "帮我规划一条哈尔滨一日游路线，冰雪季出行，总预算约500元，从中央大街出发，希望多去经典景点。",
    "帮我规划一条哈尔滨两日游路线，夏季出行，预算约1500元，喜欢美食和购物，节奏不要太赶。",
    "帮我规划一条哈尔滨三日游路线，冬季出行，预算约3000元，以冰雪大世界和中央大街为核心。",
    "我在哈尔滨只有半天时间，想从圣索菲亚教堂附近出发，看看冰雪大世界，怎么安排？",
    "带父母去哈尔滨玩三天，预算充足，节奏要慢，适合老年人的景点优先。",
]


def load_lora_model(lora_dir: str, device):
    """加载微调后的 LoRA 模型."""
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


def generate_route(model, tokenizer, instruction: str, max_new_tokens=300) -> str:
    """生成路线文本."""
    msgs = [{"role": "system",
             "content": "你是一位哈尔滨旅游规划专家，根据用户的需求生成合理的旅游路线。"
                        "路线用 POI 名称以 → 连接。"},
            {"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return resp


def parse_route_text(text: str) -> list[str]:
    """从生成文本中提取 POI 名称序列（→ 分隔）."""
    # 去掉思考部分（Qwen3 的 <think> 或思维链）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?assistant", "", text, flags=re.DOTALL)

    # 1. 优先找 → 连接的序列（允许箭头前后有空格）
    matches = re.findall(
        r"([\u4e00-\u9fa5A-Za-z0-9·（）()]+(?:\s*(?:→|->|-\s*>)\s*[\u4e00-\u9fa5A-Za-z0-9·（）()]+){2,})",
        text,
    )
    if matches:
        longest = max(matches, key=len)
        parts = re.split(r"\s*(?:→|->|-\s*>)\s*", longest)
        return [p.strip() for p in parts if p.strip()]

    # 2. 退而求其次：找顿号/逗号分隔的地名序列
    matches = re.findall(
        r"([\u4e00-\u9fa5A-Za-z0-9·（）()]+(?:[、,，][\u4e00-\u9fa5A-Za-z0-9·（）()]+){2,})",
        text,
    )
    if matches:
        longest = max(matches, key=len)
        return [p.strip() for p in re.split(r"[、,，]", longest) if p.strip()]

    # 3. 再退一步：按行找"第X站/第X天"开头的 POI（口语化输出）
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    names = []
    for line in lines:
        m = re.match(r"(?:第\s*[一二三四五六七八九十\d]+\s*[站天步]|[1-9]\s*[\.、．])[\s:：]*([\u4e00-\u9fa5A-Za-z0-9·（）()]+)", line)
        if m:
            names.append(m.group(1))
    if len(names) >= 2:
        return names

    # 完全无法解析
    return []


def match_poi_names(names: list[str], pois: pd.DataFrame) -> tuple[list[int], float]:
    """把 POI 名称匹配回索引（用包含匹配）."""
    matched = []
    for name in names:
        # 精确/包含匹配
        exact = pois[pois["name"] == name]
        if len(exact) > 0:
            matched.append(int(exact.index[0]))
            continue
        contains = pois[pois["name"].str.contains(re.escape(name), na=False, regex=True)]
        if len(contains) > 0:
            matched.append(int(contains.index[0]))
            continue
        # 反向包含（POI 名包含地名）
        reverse = pois[pois["name"].str.contains(re.escape(name[:4]), na=False)]
        if len(reverse) > 0:
            matched.append(int(reverse.index[0]))
            continue
        matched.append(-1)  # 未匹配
    rate = sum(1 for m in matched if m >= 0) / len(matched) if matched else 0
    return matched, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-dir", default="output/qwen_route_lora")
    parser.add_argument("--model", choices=["base", "lora"], default="lora")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据（用于 POI 匹配和 v3 打分）
    data_dir = Path("data/processed")
    pois = pd.read_csv(data_dir / "poi_metadata.csv", encoding="utf-8")
    dist_matrix = np.load(data_dir / "distance_matrix.npy")
    time_matrix = np.load(data_dir / "time_matrix.npy")
    ratings = pois["rating"].values
    categories = pois["category"].values

    # 加载模型
    print(f"加载模型: {args.model} (lora: {args.lora_dir if args.model == 'lora' else 'base'})")
    if args.model == "lora":
        model, tokenizer = load_lora_model(args.lora_dir, device)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16)

    print("开始评估...")
    print("=" * 70)

    all_results = []
    for i, instruction in enumerate(TEST_INSTRUCTIONS):
        print(f"\n[{i+1}] {instruction}")
        raw = generate_route(model, tokenizer, instruction)
        print(f"  raw: {raw[:150]}...")
        names = parse_route_text(raw)
        print(f"  解析出 {len(names)} 个 POI: {names[:8]}...")
        if len(names) < 2:
            print("  ⚠️ 无法解析路线，跳过")
            continue

        matched, match_rate = match_poi_names(names, pois)
        valid_route = [m for m in matched if m >= 0]
        if len(valid_route) >= 2:
            result = composite_score_v3(valid_route, dist_matrix, time_matrix,
                                        ratings, categories)
            print(f"  POI匹配率: {match_rate:.0%} | v3得分: {result['score']}")
            print(f"  总距离: {result['metrics']['total_dist_km']}km, "
                  f"总耗时: {result['metrics']['total_time_min']}min, "
                  f"跳转p50: {result['metrics']['hop_p50_km']}km")
            all_results.append({"instruction": instruction, "match_rate": match_rate,
                                "score": result["score"], "raw": raw[:200]})
        else:
            print(f"  ⚠️ 有效 POI 不足，POI匹配率: {match_rate:.0%}")
            all_results.append({"instruction": instruction, "match_rate": match_rate,
                                "score": None, "raw": raw[:200]})

    # 汇总
    print("\n" + "=" * 70)
    valid = [r for r in all_results if r["score"] is not None]
    if valid:
        avg_match = np.mean([r["match_rate"] for r in all_results])
        avg_score = np.mean([r["score"] for r in valid])
        print(f"平均 POI 匹配率: {avg_match:.0%}")
        print(f"平均 v3 得分: {avg_score:.4f}")
    print(f"模型: {args.model} | 结果数: {len(all_results)}")


if __name__ == "__main__":
    main()
