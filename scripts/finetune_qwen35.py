"""Qwen3.5-4B SFT：候选编号格式（多日游逐日拆分 + RAG 候选）.

Phase 2：把基模换成 Qwen3.5-4B（Qwen3.5-4B 是多模态，这里只用纯文本能力）。

数据格式（prepare_qwen35_dataset.py 生成）：
    {"system": "...", "instruction": "候选编号列表...", "output": "1→3→5→2→8→6"}

训练：模型学会"从候选编号里挑路线"，推理时 RAG 检索候选 + 编号输出 + 后端映射真实名。
这样模型不背 10K 个 POI 名（编造名根因），只需做"8 选 N"的排序任务。

用法:
    ./.venv/Scripts/python.exe scripts/finetune_qwen35.py [--epochs 3] [--output output/qwen35_route_lora]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = str(Path("data/external/modelscope_cache/models/Qwen--Qwen3.5-4B"))
DATASET_PATH = "data/qwen35_sft_dataset.jsonl"
MAX_LEN = 1024

SYSTEM_PROMPT = (
    "你是一位哈尔滨旅游规划专家。你会收到候选景点编号列表，"
    "只能输出候选编号组成的路线，编号用 → 连接，不要输出任何其他文字。"
)


def build_prompt(instruction: str, system: str) -> str:
    """构造 chat 格式 prompt（Qwen chat template）."""
    return (
        "<|im_start|>system\n" + system + "\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_samples(path: str, max_samples: int = None):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            prompt = build_prompt(d["instruction"], d.get("system", SYSTEM_PROMPT))
            samples.append({"prompt": prompt, "full": prompt + d["output"] + "<|im_end|>"})
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def tokenize_function(samples, tokenizer):
    enc = tokenizer(samples["full"], truncation=True, max_length=MAX_LEN, padding=False)
    enc["labels"] = enc["input_ids"].copy()
    return enc


def find_lora_targets(model) -> list:
    """探测可挂 LoRA 的注意力线性层名（Qwen3.5 混合注意力：self_attn + linear_attn）.

    Qwen3.5 是混合架构：部分层用标准 self_attn（q/k/v/o_proj），
    部分层用 linear_attn（in_proj_qkv 等）。只挂注意力层，排除 mlp 与 vision。
    """
    targets = []
    for name, module in model.named_modules():
        if "layers" not in name or "mlp" in name:
            continue
        cls = module.__class__.__name__
        if "Linear" in cls:  # Linear / Linear4bit / bnb.nn.Linear4bit
            leaf = name.split(".")[-1]
            if leaf in ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv"):
                targets.append(leaf)
    return list(dict.fromkeys(targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="output/qwen35_route_lora")
    args = parser.parse_args()

    print(f"模型: {MODEL_PATH}")
    print(f"QLoRA SFT（候选编号格式） | epochs={args.epochs} | lr={args.lr}")

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 4bit 量化加载 ===
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    print(f"模型加载完成: {type(model).__name__}")
    model = prepare_model_for_kbit_training(model)

    # === LoRA（探测目标层） ===
    targets = find_lora_targets(model)
    print(f"LoRA target modules: {targets}")
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=targets,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA 可训练参数: {trainable:,} ({trainable / total * 100:.2f}%)")

    # === 数据集 ===
    samples = load_samples(DATASET_PATH, args.max_samples)
    ds = Dataset.from_list(samples)
    ds = ds.map(lambda s: tokenize_function(s, tokenizer), batched=True,
                remove_columns=["prompt", "full"])
    split = ds.train_test_split(test_size=0.05, seed=42)
    print(f"train={len(split['train'])} eval={len(split['test'])}")

    # === 训练 ===
    training_args = TrainingArguments(
        output_dir=args.output + "_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
    )

    def collate_fn(batch):
        """Pad 到 batch 内最大长度，labels 用 -100 对齐（忽略 padding loss）."""
        max_len = max(len(d["input_ids"]) for d in batch)
        input_ids, attention_mask, labels = [], [], []
        for d in batch:
            n = len(d["input_ids"])
            pad_n = max_len - n
            input_ids.append(torch.tensor(d["input_ids"] + [tokenizer.pad_token_id] * pad_n))
            attention_mask.append(torch.tensor([1] * n + [0] * pad_n))
            labels.append(torch.tensor(d["labels"] + [-100] * pad_n))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        data_collator=collate_fn,
    )
    print("开始 SFT 训练...")
    trainer.train()

    # === 保存 LoRA ===
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Qwen3.5 LoRA 已保存: {args.output}")


if __name__ == "__main__":
    main()
