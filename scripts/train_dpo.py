"""Qwen3-4B DPO 训练：让模型偏好"v4 高分路线".

数据：data/qwen_dpo_dataset.jsonl（由 prepare_dpo_dataset.py 用 v4 打分生成）
方法：DPO（Direct Preference Optimization），在 SFT 微调基础上继续
技术：QLoRA 4bit + LoRA，只用 preference loss

用法:
    ./.venv/Scripts/python.exe scripts/train_dpo.py [--epochs 2] [--batch-size 2]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = str(Path("data/external/modelscope_cache/models/Qwen--Qwen3-4B/snapshots/master"))
SFT_LORA = "output/qwen_route_lora"          # SFT 阶段 LoRA（作为 DPO 初始）
DPO_DATASET = "data/qwen_dpo_dataset.jsonl"
MAX_LEN = 1024


def build_messages(instruction, route):
    """构造 chat 消息，chosen/rejected 都是完整对话."""
    system = "你是一位哈尔滨旅游规划专家，根据用户的需求生成合理的旅游路线。路线用 POI 名称以 → 连接。"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": route},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO 温度参数")
    parser.add_argument("--output", default="output/qwen_route_dpo")
    args = parser.parse_args()

    print(f"DPO 训练 | epochs={args.epochs} | lr={args.lr} | beta={args.beta}")
    print(f"初始 LoRA: {SFT_LORA}")

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 4bit 量化 + SFT LoRA ===
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    # 加载 SFT 阶段的 LoRA 权重（DPO 在这个基础上继续）
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, SFT_LORA)
    # 重置为新 LoRA（DPO 单独训练一套，避免污染 SFT 权重）
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))

    # === 加载 DPO 数据集 ===
    samples = []
    with open(DPO_DATASET, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            samples.append({
                "chosen": build_messages(d["prompt"], d["chosen"]),
                "rejected": build_messages(d["prompt"], d["rejected"]),
            })
            if args.max_samples and len(samples) >= args.max_samples:
                break
    print(f"加载偏好对: {len(samples)}")
    ds = Dataset.from_list(samples)

    def tokenize_fn(examples):
        chosen_ids = [tokenizer.apply_chat_template(c, tokenize=True, max_length=MAX_LEN)
                      for c in examples["chosen"]]
        rejected_ids = [tokenizer.apply_chat_template(r, tokenize=True, max_length=MAX_LEN)
                        for r in examples["rejected"]]
        return {"prompt_ids": [c[:1] for c in chosen_ids],  # 占位，实际用 chat 模板
                "chosen_input_ids": chosen_ids,
                "rejected_input_ids": rejected_ids}

    # DPOTrainer 需要 prompt/chosen/rejected 三个字段的文本
    ds = ds.map(lambda ex: {
        "prompt": ex["chosen"][1]["content"],  # user 指令
        "chosen": ex["chosen"][2]["content"],  # assistant 路线
        "rejected": ex["rejected"][2]["content"],
    })

    # === DPO 训练 ===
    dpo_config = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
        max_length=MAX_LEN,
        beta=args.beta,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 用 model 自身做 reference（DPO 简化）
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("开始 DPO 训练...")
    trainer.train()

    trainer.save_model(args.output)
    print(f"DPO 模型已保存: {args.output}")


if __name__ == "__main__":
    main()
