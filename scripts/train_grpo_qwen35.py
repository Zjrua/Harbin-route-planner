"""Qwen3.5-4B GRPO：候选编号任务的在线强化学习.

背景：Qwen3.5-SFT 已学会"从候选编号里选路线"（候选编号格式，防编造名）。
GRPO 在此基础上做在线强化：每个候选 prompt 采样 8 条编号路线 → 映射回真实
POI 名 → v3 质量打分（候选编号任务天然"每天 1 日"，用质量五维）→ 组内归一化
提升高分路线的概率。让模型学会"挑更优的子集和顺序"，而非 SFT 的模仿。

数据：data/qwen35_sft_dataset.jsonl（候选编号格式，含候选名列表）
奖励：输出编号 → 候选名 → POI 索引 → composite_score_v3（质量五维）
      × 匹配率惩罚（越界编号 = 灌水，压 reward）
基线：Qwen3.5-SFT LoRA（GRPOTrainer 自动禁用 adapter 当 reference，省显存）

用法:
    ./.venv/Scripts/python.exe scripts/train_grpo_qwen35.py [--max-steps 100] [--output output/qwen35_route_grpo]
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

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.scoring import composite_score_v3

MODEL_PATH = str(Path("data/external/modelscope_cache/models/Qwen--Qwen3.5-4B"))
SFT_LORA = "output/qwen35_route_lora"
DATA_PATH = "data/qwen35_sft_dataset.jsonl"
DATA_DIR = Path("data/processed")

MAX_LEN = 1024


def build_prompt(instruction: str, system: str) -> str:
    """候选编号格式 prompt（手动 <|im_start|>，防 chat template 注入 think）."""
    return (
        "<|im_start|>system\n" + system + "\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_candidates(instruction: str):
    """从候选编号 prompt 提取候选名列表（"N.名字"）."""
    return re.findall(r"\d+\.(.+)", instruction)


def pick_names(nums_str: str, cand_names):
    """编号输出 → 候选名列表（越界/重复丢弃）."""
    nums = [int(n) for n in re.findall(r"\d+", nums_str)]
    seen = set()
    out = []
    for v in nums:
        if 1 <= v <= len(cand_names) and v not in seen:
            seen.add(v)
            out.append(cand_names[v - 1])
    return out


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


def make_reward_func(d, cands_list):
    """GRPO reward：编号路线 → 真实 POI → v3 质量分 × 匹配率惩罚."""
    pois = d["pois"]
    name2idx = {}
    for i, nm in enumerate(pois["name"].tolist()):
        if nm not in name2idx:
            name2idx[nm] = i

    def _score(names):
        if len(names) < 3:
            return 0.0
        idxs = []
        for n in names:
            if n in name2idx:
                idxs.append(name2idx[n])
                continue
            # contains 兜底
            for nm in name2idx:
                if n in nm or nm in n:
                    idxs.append(name2idx[nm])
                    break
        if len(idxs) < 3:
            return 0.0
        seen, deduped = set(), []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        if len(deduped) < 3:
            return 0.0
        r = composite_score_v3(deduped, d["dist"], d["time"], d["ratings"],
                               d["categories"], n_days=1,
                               activity_types=d["activity_types"])
        if not r.get("feasible"):
            return 0.0
        return float(r["score"])

    def reward_func(prompts, completions, **kwargs):
        inputs = kwargs.get("inputs") or [{} for _ in prompts]
        scores = []
        for i, comp in enumerate(completions):
            cands = cands_list[i] if i < len(cands_list) else []
            names = pick_names(comp, cands)
            if len(names) < 3:
                scores.append(0.0)
                continue
            raw_nums = len(re.findall(r"\d+", comp))
            # 匹配率惩罚：模型输出大量越界编号（灌水）→ 压 reward
            penalty = min(1.0, len(names) / max(raw_nums, 1))
            scores.append(_score(names) * penalty)
        return scores
    return reward_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--n-prompts", type=int, default=300, help="参与训练的指令数")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--output", default="output/qwen35_route_grpo")
    args = parser.parse_args()

    print(f"Qwen3.5 GRPO | steps={args.max_steps} | gen/组={args.num_generations} | lr={args.lr}")

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 4bit 加载 + Qwen3.5-SFT LoRA ===
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, SFT_LORA)
    model.train()
    print(f"Qwen3.5-SFT LoRA 已加载（未合并）: {SFT_LORA}")

    # === 训练指令（候选编号格式） ===
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[: args.n_prompts]
    cands_list = [parse_candidates(r["instruction"]) for r in rows]
    prompts = [build_prompt(r["instruction"], r.get("system", "")) for r in rows]
    ds = Dataset.from_list([{"prompt": p, "raw_instruction": r["instruction"]}
                            for p, r in zip(prompts, rows)])
    print(f"训练指令: {len(ds)}（候选编号格式）")

    # === 奖励 ===
    d = get_data()
    reward_func = make_reward_func(d, cands_list)

    # === GRPO 配置 ===
    grpo_config = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=80,  # 编号输出很短
        max_steps=args.max_steps,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=0.6,
        top_p=0.9,
        bf16=True,
        logging_steps=2,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("开始 Qwen3.5 GRPO 训练...")
    trainer.train()

    # 保存 LoRA（不 merge 4bit base）
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Qwen3.5 GRPO LoRA 已保存: {args.output}")


if __name__ == "__main__":
    main()
