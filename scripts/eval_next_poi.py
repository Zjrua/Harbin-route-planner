"""下一站预测：真实路线上的效度检验（打分器盲区里的对比）.

背景：v5 打分器与贪心最近基线同源（都奖励就近），LLM vs rule_near 打平
不能区分"模型没用"和"打分器测不到"。本实验用 XHS 真实游记路线的
下一站选择作为 ground truth——模型若在真实人类选择上显著赢过规则，
说明模型学到了打分器测不到的东西。

设计：
- 数据：routes_xhs_holdout.npy（168 条真实路线），每条随机抽 3 个位置
- 任务：给定前缀 + 8 个候选（真实下一站 + 7 个干扰），选下一站
- 干扰项构造：前缀末站附近检索池中采样，优先与真实下一站同类型（有迷惑性）
- 方法：
    random      随机选（下限）
    rule_near   距前缀末站最近的候选（v5 实验的冠军基线）
    rule_markov 真实数据一阶类型转移概率 × 距离衰减（更强规则基线）
    llm         Qwen3.5-SFT 前缀引导选编号
- 指标：top-1 命中率（+类型命中率）+ Wilson 95% CI
- 统计：同一预测点配对 McNemar 检验（精确二项）

用法:
    ./.venv/Scripts/python.exe scripts/eval_next_poi.py [--n-points 500] [--seeds 2]
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

from src.retrieval import retrieve_candidates, resolve_poi_index
from src.itinerary_planner import load_semantic

MODEL_BASE = "data/external/modelscope_cache/models/Qwen--Qwen3.5-4B"
SFT_LORA = "output/qwen35_route_lora"
T = {0: "景点", 1: "餐饮", 2: "住宿", 3: "交通", 4: "购物", 5: "出发点"}

SYSTEM_PROMPT = (
    "你是一位哈尔滨旅游规划专家。根据已游览的路线和候选列表，"
    "选出下一个最应该去的地点，只输出其编号，不要输出任何其他文字。"
)


def build_transition(routes, pois, n_types=6):
    """真实路线的一阶活动类型转移计数（加 1 平滑）."""
    counts = np.ones((n_types, n_types)) * 0.5  # 平滑
    for r in routes:
        idxs = list(r)
        types = [int(pois.iloc[i]["activity_type"]) for i in idxs]
        for a, b in zip(types, types[1:]):
            counts[a][b] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def build_points(routes, pois, dist, semantic, n_points, seed=7):
    """构造预测点：(前缀索引, 真实下一站, 候选索引列表[8, 含真实])."""
    rng = random.Random(seed)
    points = []
    for ri, r in enumerate(routes):
        idxs = [int(i) for i in r]
        if len(idxs) < 4:
            continue
        ts = rng.sample(range(2, len(idxs)), k=min(3, len(idxs) - 2))
        for t in ts:
            prefix, true_next = idxs[:t], idxs[t]
            used = set(prefix) | {true_next}
            # 干扰池：前缀末站附近检索（语义过滤）
            pool = retrieve_candidates(
                pois, dist, center_idx=prefix[-1], used_indices=used,
                n=25, semantic=semantic)
            if len(pool) < 7:
                continue
            true_type = int(pois.iloc[true_next]["activity_type"])
            same_type = [c for c in pool
                         if int(pois.iloc[c]["activity_type"]) == true_type]
            others = [c for c in pool if c not in same_type]
            rng.shuffle(same_type)
            rng.shuffle(others)
            # 最多 4 个同类型干扰 + 其余异类型（有迷惑但不全是同类）
            n_same = min(4, len(same_type))
            distract = same_type[:n_same] + others[: 7 - n_same]
            if len(distract) < 7:
                continue
            cands = [true_next] + distract[:7]
            rng.shuffle(cands)
            points.append({"route_i": ri, "t": t, "prefix": prefix,
                           "true_next": true_next, "cands": cands})
            if len(points) >= n_points:
                return points
    return points


def rule_near_pred(pt, pois, dist):
    last = pt["prefix"][-1]
    return pt["cands"][int(np.argmin([dist[last][c] for c in pt["cands"]]))]


def rule_markov_pred(pt, pois, dist, trans):
    last = pt["prefix"][-1]
    lt = int(pois.iloc[last]["activity_type"])
    scores = []
    for c in pt["cands"]:
        ct = int(pois.iloc[c]["activity_type"])
        d = float(dist[last][c])
        scores.append(trans[lt][ct] * np.exp(-d / 3.0))  # 转移概率×距离衰减
    return pt["cands"][int(np.argmax(scores))]


def llm_pred(model, tok, pt, pois):
    prefix_names = [str(pois.iloc[i]["name"]) for i in pt["prefix"]]
    cand_lines = "\n".join(f"{j+1}.{pois.iloc[c]['name']}"
                           for j, c in enumerate(pt["cands"]))
    instr = (f"已游览路线：{' → '.join(prefix_names)}。\n候选地点：\n{cand_lines}\n"
             f"选出下一个最应该去的地点，只输出其编号（1-{len(pt['cands'])}）。")
    text = ("<|im_start|>system\n" + SYSTEM_PROMPT + "\n<|im_end|>\n"
            f"<|im_start|>user\n{instr}\n<|im_end|>\n<|im_start|>assistant\n")
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=8, do_sample=True,
                             temperature=0.4, top_p=0.9)
    raw = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    for token in __import__("re").findall(r"\d+", raw):
        v = int(token)
        if 1 <= v <= len(pt["cands"]):
            return pt["cands"][v - 1]
    return pt["cands"][0]  # 解析失败兜底（记为错）


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, center - half), min(1, center + half)


def mcnemar_exact(b, c):
    """配对二元结果精确 McNemar（二项检验）."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--out", default="output/eval_next_poi.json")
    args = parser.parse_args()

    routes = np.load("data/processed/routes_xhs_holdout.npy", allow_pickle=True)
    pois = pd.read_csv("data/processed/poi_metadata.csv", encoding="utf-8")
    dist = np.load("data/processed/distance_matrix.npy")
    semantic = load_semantic()
    print(f"真实路线: {len(routes)} 条", flush=True)

    trans = build_transition(routes, pois)
    print("=== 真实一阶类型转移（行=当前类型, 列=下一类型概率）===")
    header = "      " + " ".join(f"{T[c]:>4s}" for c in range(6))
    print(header)
    for r_ in range(6):
        row = "  ".join(f"{trans[r_][c]:.2f}" for c in range(6))
        print(f"  {T[r_]:>3s} | {row}")

    points = build_points(routes, pois, dist, semantic, args.n_points)
    print(f"预测点: {len(points)}", flush=True)

    # 模型
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

    rng = random.Random(11)
    results = []  # 每个点的各方法命中（bool）
    for i, pt in enumerate(points):
        true_next = pt["true_next"]
        tn_type = int(pois.iloc[true_next]["activity_type"])
        row = {"route_i": pt["route_i"], "t": pt["t"]}

        rnd = rng.choice(pt["cands"])
        rn = rule_near_pred(pt, pois, dist)
        rm = rule_markov_pred(pt, pois, dist, trans)
        for name, pred in (("random", rnd), ("rule_near", rn), ("rule_markov", rm)):
            row[name] = (pred == true_next)
            row[name + "_type"] = (int(pois.iloc[pred]["activity_type"]) == tn_type)

        # LLM 多 seed 投票（多数票，平票取第一seed）
        votes = []
        for s in range(args.seeds):
            torch.manual_seed(s * 977 + i)
            votes.append(llm_pred(model, tok, pt, pois))
        counts = {v: votes.count(v) for v in set(votes)}
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        row["llm"] = (best == true_next)
        row["llm_type"] = (int(pois.iloc[best]["activity_type"]) == tn_type)

        results.append(row)
        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(points)}", flush=True)

    # ==== 汇总 ====
    n = len(results)
    summary = {}
    for m in ("random", "rule_near", "rule_markov", "llm"):
        k = sum(r[m] for r in results)
        kt = sum(r[m + "_type"] for r in results)
        lo, hi = wilson_ci(k, n)
        summary[m] = {"hit": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
                      "type_hit": round(kt / n, 4), "k": k, "n": n}

    # 配对 McNemar
    def mcnemar(m1, m2):
        b = sum(1 for r in results if r[m1] and not r[m2])  # m1对m2错
        c = sum(1 for r in results if not r[m1] and r[m2])  # m2对m1错
        return {"m1_only_win": b, "m2_only_win": c,
                "p_value": round(mcnemar_exact(b, c), 5)}

    tests = {f"{a}_vs_{b}": mcnemar(a, b)
             for a, b in [("llm", "rule_near"), ("llm", "rule_markov"),
                          ("llm", "random"), ("rule_markov", "rule_near")]}

    out = {"n_points": n, "llm_seeds": args.seeds,
           "transition_matrix": trans.round(3).tolist(),
           "summary": summary, "paired_mcnemar": tests, "records": results}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_mcnemar": tests},
                     ensure_ascii=False, indent=2))
    print(f"保存至: {args.out}")


if __name__ == "__main__":
    main()
