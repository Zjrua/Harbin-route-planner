"""门控选择器验证：按候选池距离结构切换 rule_near / LLM（双峰的模式切换兑现）.

背景链：v5 打平(同源疑云) → 总分规则赢 → 分桶反转(LLM 赢中远桶, 辛普森悖论)
→ α 线性混合失败(双峰=模式切换非连续偏好)。
门控假设：真实游客行为是「就近闲逛 / 跨区奔袭」两种模式，由**当前环境的
选项结构**决定——候选池中位距离大（近处没得选）→ 跨区模式。

诊断（447 点）：cand_med_d AUC=0.808, n_near_cand 反向 AUC=0.808，
惯性问题特征 last_hop_d 仅 0.577 —— 选项结构 >> 行为惯性。

设计（防过拟合）：
- 447 点分半（固定 seed）：train 上扫阈值 τ（cand_med_d），目标最大化
  gate 选择器命中（gate: cand_med_d>τ → LLM，否则 → rule_near）；
- test 上报告最终数字 + 分桶 + McNemar（vs 纯规则 / 纯 LLM / 最优α混合）。

用法:
    ./.venv/Scripts/python.exe scripts/eval_gated_selector.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_next_poi import build_points, mcnemar_exact
from eval_hybrid_selector import llm_first_token_probs
from src.itinerary_planner import load_semantic

MODEL_BASE = "data/external/modelscope_cache/models/Qwen--Qwen3.5-4B"
SFT_LORA = "output/qwen35_route_lora"
BUCKETS = [("near_lt2km", 0, 2), ("mid_2_8km", 2, 8), ("far_gt8km", 8, 9999)]


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

    # 每点：门控特征 + 两方法的预测位置 + 真实位置 + 真实距离
    rows = []
    for i, pt in enumerate(points):
        last = pt["prefix"][-1]
        true_next = pt["true_next"]
        cand_d = np.array([float(dist[last][c]) for c in pt["cands"]])
        near_pick = int(np.argmin(cand_d))                     # rule_near
        llm_p = llm_first_token_probs(model, tok, pt, pois)
        llm_pick = int(np.argmax(llm_p))                       # LLM first-token
        rows.append({
            "cand_med_d": float(np.median(cand_d)),
            "d_true": float(cand_d[pt["cands"].index(true_next)]),
            "pos": pt["cands"].index(true_next),
            "near_pick": near_pick, "llm_pick": llm_pick,
        })
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(points)}", flush=True)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(rows))
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]

    def gate_hits(sub, tau):
        return [r["llm_pick"] == r["pos"] if r["cand_med_d"] > tau
                else r["near_pick"] == r["pos"] for r in sub]

    # train 上扫 τ
    grid = np.arange(0.5, 12.1, 0.25)
    tr_curve = [(round(float(t), 2), float(np.mean(gate_hits([rows[i] for i in tr], t))))
                for t in grid]
    best_tau, best_tr = max(tr_curve, key=lambda x: x[1])

    # test 评估
    sub_te = [rows[i] for i in te]
    gate_te = gate_hits(sub_te, best_tau)
    rule_te = [r["near_pick"] == r["pos"] for r in sub_te]
    llm_te = [r["llm_pick"] == r["pos"] for r in sub_te]

    def bucket_stat(hits):
        out = {}
        for bname, lo, hi in BUCKETS:
            sel = [h for h, r in zip(hits, sub_te) if lo <= r["d_true"] < hi]
            out[bname] = round(float(np.mean(sel)), 3) if sel else None
        return out

    def mcn(m1, m2):
        b = sum(1 for a, c in zip(m1, m2) if a and not c)
        c_ = sum(1 for a, c in zip(m1, m2) if not a and c)
        return {"win_only": b, "lose_only": c_, "p": round(mcnemar_exact(b, c_), 6)}

    # 门控分布（多少点路由给了 LLM）
    routed_llm = sum(1 for r in sub_te if r["cand_med_d"] > best_tau)

    out = {
        "n_points": len(rows), "n_train": len(tr), "n_test": len(te),
        "best_tau": best_tau,
        "train_gate_hit": round(best_tr, 4),
        "test": {
            "gate": {"total": round(float(np.mean(gate_te)), 4),
                     **bucket_stat(gate_te)},
            "rule_near": {"total": round(float(np.mean(rule_te)), 4),
                          **bucket_stat(rule_te)},
            "llm": {"total": round(float(np.mean(llm_te)), 4),
                    **bucket_stat(llm_te)},
        },
        "routed_to_llm": {"n": routed_llm, "share": round(routed_llm / len(te), 3)},
        "mcnemar_gate_vs_rule": mcn(gate_te, rule_te),
        "mcnemar_gate_vs_llm": mcn(gate_te, llm_te),
    }
    Path("output/eval_gated_selector.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("保存至: output/eval_gated_selector.json")


if __name__ == "__main__":
    main()
