"""统计化评估：无 LLM 规则基线 vs Qwen3.5-SFT（配对设计 + Wilcoxon 检验）.

回答的问题：模型在候选编号任务上的真实边际价值是多少？
（检索/类型配额/节奏整理都在规则层，模型只做"10 选 6 排序"——
如果规则基线打平，说明投入应该转向检索与数据。）

设计：
- 指令集：60 条合成指令（种子固定），覆盖天数(1/2/3/3.5)/季节/预算/人群/
  偏好/出发地/核心景点/半日需求的组合
- 方法：rule_score（检索分数序取前N）、rule_near（贪心最近）、
  llm（Qwen3.5-SFT 候选编号，3 个采样种子）
- 指标：逐日 v5 均值（每条指令）、需求匹配、总 POI 数
- 统计：均值±标准差；同一指令下 LLM(3seed均值) vs 规则的配对 Wilcoxon 符号秩检验

用法:
    ./.venv/Scripts/python.exe scripts/eval_llm_vs_rule.py [--n-instr 60] [--seeds 3]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.itinerary_planner import plan_itinerary, load_semantic
from src.scoring import composite_score_v5
from src.constraint_parser import parse_constraints

MODEL_BASE = "data/external/modelscope_cache/models/Qwen--Qwen3.5-4B"
SFT_LORA = "output/qwen35_route_lora"

# ==== 指令合成维度 ====
DAYS_POOL = ["一日", "两日", "三日", "三天半"]
SEASON_POOL = ["冰雪季（冬季）出行", "夏季出行", ""]
BUDGET_POOL = ["总预算约500元", "总预算约1500元", "总预算约3000元", "预算充足", ""]
AUDIENCE_POOL = ["带父母，节奏要慢，适合老年人的景点优先", "带娃亲子游", ""]
PREF_POOL = ["喜欢美食", "喜欢购物", "文化爱好者，喜欢历史建筑和博物馆", "喜欢自然风光", ""]
START_POOL = ["从中央大街出发", "从圣索菲亚教堂出发", "从哈尔滨站出发", ""]
CORE_POOL = ["以冰雪大世界和中央大街为核心", "以太阳岛为核心", ""]
HALF_POOL = ["第二天下午5点走", "最后一天中午离开", ""]

CLASSIC_5 = [
    "帮我规划一条哈尔滨一日游路线，冰雪季出行，总预算约500元，从中央大街出发，希望多去经典景点。",
    "帮我规划一条哈尔滨两日游路线，夏季出行，预算约1500元，喜欢美食和购物，节奏不要太赶。",
    "帮我规划一条哈尔滨三日游路线，冬季出行，预算约3000元，以冰雪大世界和中央大街为核心。",
    "我在哈尔滨只有半天时间，想从圣索菲亚教堂附近出发，看看冰雪大世界，怎么安排？",
    "带父母去哈尔滨玩三天，预算充足，节奏要慢，适合老年人的景点优先。",
]


def synth_instructions(n: int, seed: int = 7) -> list:
    """种子固定的组合采样，生成多样测试指令."""
    rng = random.Random(seed)
    instrs = list(CLASSIC_5)
    while len(instrs) < n:
        days = rng.choice(DAYS_POOL)
        parts = [f"帮我规划一条哈尔滨{days}游路线"]
        for pool in (SEASON_POOL, BUDGET_POOL, AUDIENCE_POOL, PREF_POOL,
                     START_POOL, CORE_POOL, HALF_POOL):
            if pool and rng.random() < 0.55:
                parts.append(rng.choice(pool))
        instrs.append("，".join(parts) + "。")
    return instrs[:n]


def run_method(instruction, d, semantic, selector, model=None, tok=None, seed=None):
    """跑一次规划，返回汇总指标（逐日v5均值/需求匹配/总POI）."""
    if seed is not None:
        torch.manual_seed(seed)
    res = plan_itinerary(model, tok, instruction, d, selector=selector)
    days = res.get("days", [])
    day_scores = [float((x.get("score_detail") or {}).get("score", 0.0)) for x in days]
    overall = res.get("overall", {})
    return {
        "day_v5_mean": float(np.mean(day_scores)) if day_scores else 0.0,
        "req_match": float(overall.get("requirement_match", 0.0)),
        "overall_v5": float(overall.get("score", 0.0)),
        "n_pois": int(res.get("total_pois", 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-instr", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="output/eval_llm_vs_rule.json")
    args = parser.parse_args()

    # 数据
    pois = pd.read_csv("data/processed/poi_metadata.csv", encoding="utf-8")
    d = {
        "pois": pois,
        "dist_matrix": np.load("data/processed/distance_matrix.npy"),
        "time_matrix": np.load("data/processed/time_matrix.npy"),
        "ratings": pois["rating"].values,
        "categories": pois["category"].values,
        "activity_types": np.load("data/processed/poi_activity_types.npy"),
        "season_winter": pois["season_winter"].values,
        "season_summer": pois["season_summer"].values,
    }
    semantic = load_semantic()

    # 模型（仅 LLM 方法需要）
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, SFT_LORA)
    model.eval()
    print("模型加载完成", flush=True)

    instrs = synth_instructions(args.n_instr)
    print(f"指令数: {len(instrs)}，LLM seeds: {list(range(args.seeds))}", flush=True)

    records = []
    for i, instr in enumerate(instrs):
        row = {"instr": instr}
        # 规则基线（确定性，1 次）
        for sel in ("rule_score", "rule_near"):
            row[sel] = run_method(instr, d, semantic, sel)
        # LLM（3 seeds 取均值，同时记录组内标准差）
        llm_runs = [run_method(instr, d, semantic, "llm", model, tok, seed=s)
                    for s in range(args.seeds)]
        row["llm"] = {
            k: float(np.mean([r[k] for r in llm_runs]))
            for k in llm_runs[0]
        }
        row["llm_seed_std"] = float(np.std([r["day_v5_mean"] for r in llm_runs]))
        records.append(row)
        if (i + 1) % 10 == 0:
            print(f"  进度 {i+1}/{len(instrs)}", flush=True)

    # ==== 汇总统计 ====
    def agg(key, field="day_v5_mean"):
        vals = [r[key][field] for r in records]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {}
    for key in ("rule_score", "rule_near", "llm"):
        m, s = agg(key)
        summary[key] = {"day_v5_mean": round(m, 4), "std": round(s, 4)}
        summary[key]["req_match"] = round(agg(key, "req_match")[0], 4)
        summary[key]["n_pois"] = round(agg(key, "n_pois")[0], 1)

    # 配对 Wilcoxon：LLM(3seed均值) vs 各规则基线
    from scipy.stats import wilcoxon
    tests = {}
    for rule in ("rule_score", "rule_near"):
        llm_vals = [r["llm"]["day_v5_mean"] for r in records]
        rule_vals = [r[rule]["day_v5_mean"] for r in records]
        diff = [a - b for a, b in zip(llm_vals, rule_vals)]
        try:
            stat, p = wilcoxon(llm_vals, rule_vals)
        except Exception:
            stat, p = float("nan"), float("nan")
        tests[f"llm_vs_{rule}"] = {
            "mean_diff": round(float(np.mean(diff)), 4),
            "win_rate": round(float(np.mean([x > 0 for x in diff])), 3),
            "wilcoxon_stat": round(float(stat), 3) if stat == stat else None,
            "p_value": round(float(p), 5) if p == p else None,
        }

    result = {
        "n_instructions": len(records),
        "llm_seeds": args.seeds,
        "summary": summary,
        "paired_tests": tests,
        "llm_seed_std_mean": round(float(np.mean([r["llm_seed_std"] for r in records])), 4),
        "records": records,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_tests": tests,
                      "llm_seed_std_mean": result["llm_seed_std_mean"]},
                     ensure_ascii=False, indent=2))
    print(f"保存至: {args.out}")


if __name__ == "__main__":
    main()
