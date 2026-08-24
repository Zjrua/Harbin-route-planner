"""混合选择器验证：α×就近分 + (1-α)×LLM概率，能否同时拿下近/中远两个桶.

背景（双峰发现）：真实游客 52% 转移就近闲逛（规则 66% vs LLM 29%），
48% 中远距跨区（规则 0% vs LLM 19.7%）。若混合分数在总命中和分桶
都超过单方法，则"规则+模型互补"可落地为架构。

设计（复用 eval_next_poi 的 447 个预测点，seed 相同精确重建）：
- near_score(cand)  = exp(-d/3) 归一化（d = 距前缀末站距离）
- llm_score(cand)   = first-token logits 在候选编号 token 上的 softmax
                      （SFT 模型输出即编号开头，一次 forward 得真实概率）
- hybrid(α)         = α·near + (1-α)·llm，α∈{0,0.1,...,1} 扫描
- 统计：α 曲线（总命中+分桶）+ 最优 α vs 纯规则/纯LLM 的 McNemar

用法:
    ./.venv/Scripts/python.exe scripts/eval_hybrid_selector.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_next_poi import build_points, mcnemar_exact  # 同目录复用
from src.itinerary_planner import load_semantic

MODEL_BASE = "data/external/modelscope_cache/models/Qwen--Qwen3.5-4B"
SFT_LORA = "output/qwen35_route_lora"

SYSTEM_PROMPT = (
    "你是一位哈尔滨旅游规划专家。根据已游览的路线和候选列表，"
    "选出下一个最应该去的地点，只输出其编号，不要输出任何其他文字。"
)

BUCKETS = [("near_lt2km", 0, 2), ("mid_2_8km", 2, 8), ("far_gt8km", 8, 9999)]


def llm_first_token_probs(model, tok, pt, pois):
    """一次 forward 拿候选编号的概率分布（SFT 模型输出即编号开头）."""
    prefix_names = [str(pois.iloc[i]["name"]) for i in pt["prefix"]]
    cand_lines = "\n".join(f"{j+1}.{pois.iloc[c]['name']}"
                           for j, c in enumerate(pt["cands"]))
    instr = (f"已游览路线：{' → '.join(prefix_names)}。\n候选地点：\n{cand_lines}\n"
             f"选出下一个最应该去的地点，只输出其编号（1-{len(pt['cands'])}）。")
    text = ("<|im_start|>system\n" + SYSTEM_PROMPT + "\n<|im_end|>\n"
            f"<|im_start|>user\n{instr}\n<|im_end|>\n<|im_start|>assistant\n")
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    cand_p = []
    for j in range(len(pt["cands"])):
        ids = tok.encode(str(j + 1), add_special_tokens=False)
        cand_p.append(float(probs[ids[0]]))
    cand_p = np.array(cand_p)
    return cand_p / max(cand_p.sum(), 1e-9)  # 归一化（编号外的概率丢弃）


def main():
    routes = np.load("data/processed/routes_xhs_holdout.npy", allow_pickle=True)
    pois = pd.read_csv("data/processed/poi_metadata.csv", encoding="utf-8")
    dist = np.load("data/processed/distance_matrix.npy")
    semantic = load_semantic()

    points = build_points(routes, pois, dist, semantic, n_points=500)
    print(f"预测点: {len(points)}", flush=True)

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

    # 每个点：near 分数向量 + llm 概率向量 + 真实下一站索引 + 距离
    rows = []
    for i, pt in enumerate(points):
        true_next = pt["true_next"]
        last = pt["prefix"][-1]
        d_true = float(dist[last][true_next])

        d = np.array([float(dist[last][c]) for c in pt["cands"]])
        near = np.exp(-d / 3.0)
        near /= near.sum()
        llm_p = llm_first_token_probs(model, tok, pt, pois)

        pos = pt["cands"].index(true_next)
        rows.append({"near": near, "llm": llm_p, "pos": pos, "d_true": d_true,
                     "near_pick": int(np.argmax(near)), "llm_pick": int(np.argmax(llm_p))})
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(points)}", flush=True)

    n = len(rows)
    # === α 扫描 ===
    curve = []
    for ai in range(11):
        a = ai / 10
        hits = [(a * r["near"] + (1 - a) * r["llm"]).argmax() == r["pos"]
                for r in rows]
        bucket_hit = {}
        for bname, lo, hi in BUCKETS:
            sub = [h for h, r in zip(hits, rows) if lo <= r["d_true"] < hi]
            bucket_hit[bname] = round(float(np.mean(sub)), 3) if sub else None
        curve.append({"alpha": round(a, 1), "total": round(float(np.mean(hits)), 4),
                      **bucket_hit})

    # 最优 α（总命中）
    best = max(curve, key=lambda c: c["total"])

    # McNemar: 最优α vs 纯规则(α=1) / 纯LLM(α=0)
    def picks_at(a):
        return [int((a * r["near"] + (1 - a) * r["llm"]).argmax()) for r in rows]
    best_p = [p == r["pos"] for p, r in zip(picks_at(best["alpha"]), rows)]
    rule_p = [p == r["pos"] for p, r in zip(picks_at(1.0), rows)]
    llm0_p = [p == r["pos"] for p, r in zip(picks_at(0.0), rows)]

    def mcn(m1, m2):
        b = sum(1 for a, c in zip(m1, m2) if a and not c)
        c_ = sum(1 for a, c in zip(m1, m2) if not a and c)
        return {"win_only": b, "lose_only": c_, "p": round(mcnemar_exact(b, c_), 6)}

    out = {
        "n_points": n,
        "alpha_curve": curve,
        "best_alpha": best,
        "sanity": {
            "rule_near_total": round(float(np.mean(rule_p)), 4),
            "llm_firsttoken_total": round(float(np.mean(llm0_p)), 4),
        },
        "mcnemar_best_vs_rule": mcn(best_p, rule_p),
        "mcnemar_best_vs_llm": mcn(best_p, llm0_p),
    }
    Path("output/eval_hybrid_selector.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("保存至: output/eval_hybrid_selector.json")


if __name__ == "__main__":
    main()
