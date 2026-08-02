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


def make_reward_func(d, instr_list, constraints_list):
    """构造 GRPO reward 函数：v5 打分（质量 + 需求匹配）.

    Args:
        d: 数据字典
        instr_list: 训练指令列表（顺序与 ds 一致）
        constraints_list: 每条指令预解析的 Constraints（避免 reward 内重复解析）
    """
    # 预构建名称 → 索引 精确映射（加速匹配）
    pois = d["pois"]
    name2idx = {}
    for i, nm in enumerate(pois["name"].tolist()):
        if nm not in name2idx:
            name2idx[nm] = i
    names_all = list(name2idx.keys())

    def _match_idx(name):
        """名称 → POI 索引（精确优先，contains 兜底）."""
        if name in name2idx:
            return name2idx[name]
        for nm in names_all:  # 10K 线性扫描，reward 每批 ~300 名可接受
            if name in nm or nm in name:
                return name2idx[nm]
        return None

    def _score_one_cached(names, constraints):
        """v5 打分 × 匹配率惩罚.

        匹配率 = 有效匹配索引数 / 原始名称数。
        惩罚防止 reward hacking：模型若生成一堆编造/变形 POI 名凑长度
        （匹配不到 POI 库），有效匹配少 → reward 被大幅压低。
        """
        if not names:
            return 0.0
        n_raw = len(names)
        matched = []
        for name in names:
            idx = _match_idx(name)
            if idx is not None:
                matched.append(idx)
        n_matched = len(matched)
        if len(matched) < 3:
            return 0.0
        seen, deduped = set(), []
        for idx in matched:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        if len(deduped) < 3:
            return 0.0
        n_days = 1 if len(deduped) <= 10 else (2 if len(deduped) <= 16 else 3)
        result = composite_score_v5(
            deduped, d["dist"], d["time"], d["ratings"], d["categories"],
            n_days=n_days, activity_types=d["activity_types"],
            constraints=constraints,
            poi_names=names_all,
            avg_costs=pois["avg_cost"].values,
            season_winter=pois["season_winter"].values,
            season_summer=pois["season_summer"].values,
        )
        if not result.get("feasible"):
            return 0.0
        score = float(result["score"])
        # 匹配率惩罚：灌水（编造名凑长度）直接压 reward
        match_penalty = min(1.0, n_matched / max(n_raw, 1))
        return score * match_penalty

    def reward_func(prompts, completions, **kwargs):
        inputs = kwargs.get("inputs") or [{} for _ in prompts]
        scores = []
        for i, comp in enumerate(completions):
            names = parse_route(comp)
            # 用 inputs 定位指令 → 取预解析约束
            raw = inputs[i].get("raw_instruction", "") if i < len(inputs) else ""
            constraints = None
            for j, instr in enumerate(instr_list):
                if instr == raw:
                    constraints = constraints_list[j]
                    break
            scores.append(_score_one_cached(names, constraints))
        return scores
    return reward_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--n-prompts", type=int, default=120, help="参与训练的指令数")
    parser.add_argument("--num-generations", type=int, default=8, help="rollout 每组采样数")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04, help="KL 惩罚系数")
    parser.add_argument("--init-lora", default=None,
                        help="初始 LoRA（默认 SFT；传 GRPO 输出可继续训练）")
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
    init_lora = args.init_lora or SFT_LORA
    model = PeftModel.from_pretrained(model, init_lora)
    model.train()
    print(f"LoRA 已加载（未合并）: {init_lora}")

    # === 训练指令 ===
    instrs = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            instrs.append(json.loads(line)["instruction"])
    random.seed(42)
    random.shuffle(instrs)
    instr_list = instrs[: args.n_prompts]
    prompts = [build_prompt(i) for i in instr_list]
    ds = Dataset.from_list([{"prompt": p, "raw_instruction": i}
                            for p, i in zip(prompts, instr_list)])
    print(f"训练指令: {len(ds)}（含带父母/慢节奏增强样本）")

    # === 奖励函数（预解析约束，避免 reward 内重复解析） ===
    from src.constraint_parser import parse_constraints
    d = get_data()
    constraints_list = [parse_constraints(i, use_llm=False) for i in instr_list]
    reward_func = make_reward_func(d, instr_list, constraints_list)

    # === GRPO 配置 ===
    grpo_config = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,  # 必须整除 num_generations
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
