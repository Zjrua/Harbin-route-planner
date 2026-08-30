"""三层统计评测：逐日生成层实验（评测标准 v1.0，层 2 + 层 3 + 附录 v5）.

设计文档: docs/superpowers/specs/2026-08-30-behavioral-eval-design.md

协议：
- 指令集：60 条合成指令（seed=7，与 eval_llm_vs_rule 同源可对比）
- 方法：rule_score / rule_near / markov / llm(3 seed 均值) / hybrid(确定性)
- 行为先验：拟合份 84 条估计（compute_artifacts 注入，数据纪律不变）
- 层2（个体，描述性）：每条日路线的行为对数似然（nat/转移），
  同指令配对 Wilcoxon（hybrid/markov vs rule_near）
- 层3（群体，裁决）：各方法全部日路线 vs 真实测试份路线的
  转移距离 KS+Wasserstein（bootstrap 95% CI）+ 节奏分项 KS + 跨区占比
- 参照基线：真实拟合份 vs 测试份的 KS/W（"真实对真实"期望水平）
- sanity check：真实测试份路线自身的层2似然应高于全部生成方法
- 附录：v5 逐日分项（quality/req_match 分列，不再合成裁决）

用法:
    ./.venv/Scripts/python.exe scripts/run_eval_generation.py [--n-instr 60]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_llm_vs_rule import synth_instructions
from src.behavioral_eval import (BehaviorModel, dist_report, far_share,
                                 null_reference, rhythm_report, route_loglik)
from src.itinerary_planner import plan_itinerary
from run_fusion_eval import compute_artifacts

MODEL_BASE = "data/external/modelscope_cache/models/Qwen--Qwen3.5-4B"
SFT_LORA = "output/qwen35_route_lora"
OUT_JSON = "output/eval_generation_behavioral.json"


def run_once(instr, d, selector, model=None, tok=None, seed=None):
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    res = plan_itinerary(model, tok, instr, d, selector=selector)
    days = res.get("days", [])
    routes = [day.get("poi_idxs", []) for day in days]
    v5 = [float((day.get("score_detail") or {}).get("score", 0.0)) for day in days]
    qual = [float((day.get("score_detail") or {}).get("quality", 0.0))
            for day in days]
    req = float((res.get("overall", {}) or {}).get("requirement_match", 0.0))
    return {"routes": [r for r in routes if len(r) >= 2],
            "v5_day_mean": float(np.mean(v5)) if v5 else 0.0,
            "v5_quality_mean": float(np.mean(qual)) if qual else 0.0,
            "req_match": req}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-instr", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args()

    # ==== 行为先验（拟合份）与数据 ====
    print("[1] 行为先验（拟合份 84 条）", flush=True)
    art = compute_artifacts(skip_llm=True)
    pois = pd.read_csv("data/processed/poi_metadata.csv", encoding="utf-8")
    type_of = np.load("data/processed/poi_activity_types.npy")
    bm = BehaviorModel(region_of_poi=art["region_of_poi"], mk=art["mk"],
                       decay=art["decay"], T_type=art["T_type"],
                       dist=art["dist"], type_of=type_of)
    splits = art["splits"]
    real_routes = art["routes"]
    real_test = [[int(x) for x in real_routes[i]] for i in splits["test"]]
    real_fit = [[int(x) for x in real_routes[i]] for i in splits["fit"]]

    d = {"pois": pois, "dist_matrix": art["dist"],
         "time_matrix": np.load("data/processed/time_matrix.npy"),
         "ratings": pois["rating"].values,
         "categories": pois["category"].values,
         "activity_types": type_of,
         "season_winter": pois["season_winter"].values,
         "season_summer": pois["season_summer"].values,
         "behavior_model": bm}

    # ==== 模型 ====
    print("[2] 加载 Qwen3.5-4B + SFT LoRA", flush=True)
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
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

    instrs = synth_instructions(args.n_instr)
    methods = ["rule_score", "rule_near", "markov", "llm", "hybrid"]
    print(f"[3] 生成：{len(instrs)} 指令 × {methods}", flush=True)

    per_instr = {m: [] for m in methods}     # 每指令的日路线集合
    ll_dat = {m: [] for m in methods}        # 层2: 每日路线 nat/转移
    v5_dat = {m: [] for m in methods}
    for i, instr in enumerate(instrs):
        for m in methods:
            if m == "llm":
                runs = [run_once(instr, d, m, model, tok, seed=s)
                        for s in range(args.seeds)]
            else:
                runs = [run_once(instr, d, m, model, tok)]
            routes = [r for run in runs for r in run["routes"]]
            per_instr[m].append(routes)
            lls = [route_loglik(r, bm)[0] for r in routes]
            ll_dat[m] += [x for x in lls if x is not None]
            v5_dat[m].append(float(np.mean(
                [run["v5_day_mean"] for run in runs])))
        if (i + 1) % 10 == 0:
            print(f"  进度 {i+1}/{len(instrs)}", flush=True)

    # ==== 层 2：个体似然（描述性）====
    print("[4] 层2 个体行为似然", flush=True)
    from scipy.stats import wilcoxon
    layer2 = {}
    for m in methods:
        layer2[m] = {"ll_mean": round(float(np.mean(ll_dat[m])), 4),
                     "ll_std": round(float(np.std(ll_dat[m])), 4),
                     "n_routes": len(ll_dat[m])}
    # 真实参照
    real_ll = [route_loglik(r, bm)[0] for r in real_test]
    layer2["real_test_reference"] = {
        "ll_mean": round(float(np.mean(real_ll)), 4),
        "ll_std": round(float(np.std(real_ll)), 4), "n_routes": len(real_ll)}
    # 配对 Wilcoxon（指令级均值）
    l2_tests = {}
    for m in ("markov", "hybrid", "llm"):
        a = [float(np.mean([route_loglik(r, bm)[0] or 0.0
                            for r in per_instr[m][i]]))
             for i in range(len(instrs))]
        b = [float(np.mean([route_loglik(r, bm)[0] or 0.0
                            for r in per_instr["rule_near"][i]]))
             for i in range(len(instrs))]
        try:
            _, p = wilcoxon(a, b)
        except Exception:
            p = float("nan")
        l2_tests[f"{m}_vs_rule_near"] = {
            "mean_diff": round(float(np.mean(np.array(a) - np.array(b))), 4),
            "p_value": round(float(p), 5) if p == p else None}

    # ==== 层 3：群体分布（裁决）====
    print("[5] 层3 群体分布距离", flush=True)
    layer3 = {}
    for m in methods:
        gen_routes = [r for routes in per_instr[m] for r in routes]
        layer3[m] = {"dist": dist_report(gen_routes, real_test, art["dist"],
                                         B=args.boot),
                     "rhythm": rhythm_report(gen_routes, real_test, type_of),
                     "far_share": far_share(gen_routes, art["dist"])}
    layer3["null_real_vs_real"] = null_reference(real_fit, real_test,
                                                 art["dist"])
    layer3["real_test_reference"] = {
        "far_share": far_share(real_test, art["dist"]),
        "n_routes": len(real_test)}

    # ==== 附录：v5 分项（不合成裁决）====
    appendix_v5 = {m: {"v5_day_mean": round(float(np.mean(v5_dat[m])), 4)}
                   for m in methods}

    out = {"protocol": {"n_instr": len(instrs), "llm_seeds": args.seeds,
                        "boot": args.boot,
                        "prior_fit": "fit split (84 routes)"},
           "layer2_individual": layer2, "layer2_tests": l2_tests,
           "layer3_population": layer3,
           "appendix_v5_descriptive": appendix_v5,
           "sanity": {"real_ll_higher_than_all_methods":
                          bool(all(layer2["real_test_reference"]["ll_mean"]
                                   > layer2[m]["ll_mean"] for m in methods))}}
    Path(OUT_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps({"layer2": layer2, "layer2_tests": l2_tests,
                      "far_share": {m: layer3[m]["far_share"]
                                    for m in methods},
                      "ks": {m: layer3[m]["dist"]["ks"] for m in methods},
                      "null": layer3["null_real_vs_real"],
                      "sanity": out["sanity"]},
                     ensure_ascii=False, indent=2))
    print(f"保存至: {OUT_JSON}")


if __name__ == "__main__":
    main()
