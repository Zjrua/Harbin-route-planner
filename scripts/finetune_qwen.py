"""Qwen3-4B QLoRA 微调：自然语言约束 → 哈尔滨旅游路线.

范式（用户设计）：
- 输入：自然语言（时间/季节/预算/偏好/起点）
- 输出：POI 名称序列（结构化路线，→ 分隔）

技术：
- QLoRA 4bit 量化（bitsandbytes）适配 24GB 显存
- LoRA rank=8, alpha=16，只训 attention 投影层
- 用 transformers.Trainer（比 trl.SFTTrainer 更稳定）

用法:
    ./.venv/Scripts/python.exe scripts/finetune_qwen.py [--epochs 3] [--batch-size 2]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = str(Path("data/external/modelscope_cache/models/Qwen--Qwen3-4B/snapshots/master"))
DATASET_PATH = "data/qwen_instruction_dataset.jsonl"
MAX_LEN = 1024


def build_prompt(instruction: str) -> str:
    """构造 prompt（system + user 指令），assistant 部分留空待模型生成."""
    return (
        "<|im_start|>system\n你是一位哈尔滨旅游规划专家，根据用户的需求生成合理的旅游路线。"
        "路线用 POI 名称以 → 连接。路线不得重复景点，禁止中途折返。\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_samples(path: str, max_samples: int = None):
    """加载 JSONL，返回 (prompt_text, full_text) 列表."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            prompt = build_prompt(d["instruction"])
            full = prompt + d["output"] + "<|im_end|>"
            samples.append({"prompt": prompt, "full": full})
            if max_samples and len(samples) >= max_samples:
                break
    print(f"加载样本: {len(samples)}")
    return samples


def tokenize_function(samples, tokenizer):
    """对 prompt + output 做 tokenize，labels 是完整序列（只对 output 部分算 loss 用 mask 逻辑略）.

    简化：labels = input_ids（全部参与 loss），对路线生成任务足够。
    """
    enc = tokenizer(
        samples["full"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dataset", default=DATASET_PATH, help="训练数据 JSONL 路径")
    parser.add_argument("--output", default="output/qwen_route_lora")
    args = parser.parse_args()

    print(f"模型: {MODEL_PATH}")
    print(f"QLoRA 微调 | epochs={args.epochs} | lr={args.lr} | batch={args.batch_size}")

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 4bit 量化加载 ===
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # === LoRA ===
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA 可训练参数: {trainable:,} ({trainable / sum(p.numel() for p in model.parameters()) * 100:.2f}%)")

    # === 数据集 ===
    samples = load_samples(args.dataset, args.max_samples)
    ds = Dataset.from_list(samples)
    ds = ds.map(lambda s: tokenize_function(s, tokenizer), batched=True,
                remove_columns=["prompt", "full"])
    split = ds.train_test_split(test_size=0.05, seed=42)
    print(f"train={len(split['train'])} eval={len(split['test'])}")

    # === 训练 ===
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
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

    print("开始训练...")
    trainer.train()

    # === 保存 ===
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"微调模型已保存: {args.output}")


if __name__ == "__main__":
    main()
