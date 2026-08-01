"""GRPO 训练：用 v4 硬约束打分直接做奖励（不需要偏好对/奖励模型）.

对比 DPO：DPO 依赖固定偏好对（150 对），偏好数据是"离线"的；
GRPO 是"在线"强化学习：每步从当前策略采样 8 条路线 → v4 打分 →
组内归一化后提升高分路线的概率。奖励可自动计算，无需训练奖励模型。

数据：从增强指令数据集采样（含带父母/老人样本）
奖励：v4 综合分（就近/区域密度/节奏/满意度/多样性 + 硬约束判负）
基线：SFT v3 LoRA（PeftModel，GRPO 会自动禁用 adapter 作 reference，省显存）

用法:
    ./.venv/Scripts/python.exe scripts/train_grpo.py [--max-steps 40] [--output output/qwen_route_grpo]
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

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.scoring import composite_score_v3, composite_score_v5

MODEL_PATH = str(Path("data/external/modelscope_cache/models/Qwen--Qwen3-4B/snapshots/master"))
SFT_LORA = "output/qwen_route_lora"
DATA_PATH = "data/qwen_instruction_dataset_aug.jsonl"
DATA_DIR = Path("data/processed")

SYSTEM_PROMPT = ("你是一位哈尔滨旅游规划专家，根据用户的需求生成合理的旅游路线。"
                 "路线用 POI 名称以 → 连接。路线不得重复景点，禁止中途折返。")
MAX_LEN = 1024


def build_prompt(instruction: str) -> str:
    """构造 SFT 格式完整 prompt（GRPO 只做字符串 tokenize，不套 chat template）."""
    return (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


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


# 数据单例（reward func 高频调用）
_DATA = None


def get_data():
    global _DATA
    if _DATA is None:
        pois = pd.read_csv(DATA_DIR / "poi_metadata.csv", encoding="utf-8")
        _DATA = {
            "pois": pois,
            "dist": np.load(DATA_DIR / "distance_matrix.npy"),
            "time": np.load(DATA_DIR / "time_matrix.npy"),
            "ratings": pois["rating"].values,
            "categories": pois["category"].values,
            "activity_types": np.load(DATA_DIR / "poi_activity_types.npy"),
        }
    return _DATA


def _score_one(names, d):
    """单条路线 v5 打分（质量 + 需求匹配），返回 score 或 None."""
    matched, unmatched = [], []
    pois = d["pois"]
    for name in names:
        exact = pois[pois["name"] == name]
        if len(exact) > 0:
            matched.append(int(exact.index[0])); continue
        contains = pois[pois["name"].str.contains(re.escape(name), na=False, regex=True)]
        if len(contains) > 0:
            matched.append(int(contains.index[0]))
    if len(matched) < 3:
        return None
    # 按索引去重（保留首次）
    seen, deduped = set(), []
    for idx in matched:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    if len(deduped) < 3:
        return None
    n_days = 1 if len(deduped) <= 10 else (2 if len(deduped) <= 16 else 3)
    result = composite_score_v5(
        deduped, d["dist"], d["time"], d["ratings"], d["categories"],
        n_days=n_days, activity_types=d["activity_types"],
        instruction=d.get("instruction"),  # 由 make_reward_func 注入当前指令
        poi_names=pois["name"].tolist(),
        avg_costs=pois["avg_cost"].values,
        season_winter=pois["season_winter"].values,
        season_summer=pois["season_summer"].values,
    )
    if not result.get("feasible"):
        return 0.0  # 硬约束违反（含需求硬约束）：判负
    return float(result["score"])


def make_reward_func(d):
    """构造 GRPO reward 函数：v5 打分（质量 + 需求匹配）.

    reward func 的 kwargs["inputs"] 是 train_dataset 的 example（含 raw_instruction），
    用它定位当前指令，传入 composite_score_v5 做需求匹配。
    """
    def reward_func(prompts, completions, **kwargs):
        inputs = kwargs.get("inputs") or [{} for _ in prompts]
        scores = []
        for i, comp in enumerate(completions):
            names = parse_route(comp)
            if not names:
                scores.append(0.0); continue
            d["instruction"] = inputs[i].get("raw_instruction", "") if i < len(inputs) else ""
            sc = _score_one(names, d)
            scores.append(sc if sc is not None else 0.0)
        return scores
    return reward_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--n-prompts", type=int, default=120, help="参与训练的指令数")
    parser.add_argument("--num-generations", type=int, default=8, help="rollout 每组采样数")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04, help="KL 惩罚系数")
    parser.add_argument("--output", default="output/qwen_route_grpo")
    args = parser.parse_args()

    print(f"GRPO 训练 | steps={args.max_steps} | gen/组={args.num_generations} | lr={args.lr} | beta={args.beta}")

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 4bit 加载 + SFT LoRA ===
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, SFT_LORA)
    model.train()
    print("SFT v3 LoRA 已加载（GRPO 在其上继续）")

    # === 训练指令 ===
    instrs = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            instrs.append(json.loads(line)["instruction"])
    random.seed(42)
    random.shuffle(instrs)
    prompts = [build_prompt(i) for i in instrs[: args.n_prompts]]
    ds = Dataset.from_list([{"prompt": p, "raw_instruction": i}
                            for p, i in zip(prompts, instrs[: args.n_prompts])])
    print(f"训练指令: {len(ds)}（含带父母/慢节奏增强样本）")

    # === 奖励函数 ===
    d = get_data()
    reward_func = make_reward_func(d)

    # === GRPO 配置 ===
    grpo_config = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=args.num_generations,
        max_completion_length=300,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=0.6,          # rollout 采样温度：过高会剪短路线，0.6 平衡
        top_p=0.9,
        generation_kwargs={"no_repeat_ngram_size": 4},  # 防 POI 折返/打转
        bf16=True,
        logging_steps=2,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        # 关闭 gradient checkpointing：与 torch 2.6 use_reentrant 不兼容
        gradient_checkpointing=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("开始 GRPO 训练...")
    trainer.train()

    # 保存 LoRA adapter（不 merge 4bit base）
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"GRPO LoRA 已保存: {args.output}")


if __name__ == "__main__":
    main()
