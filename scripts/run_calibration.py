"""LLM 候选概率校准实验（开题报告 §8 下一优先项）.

背景：LLM 对齐后普遍过度自信，未校准概率进入 log-linear 池化可能
扭曲权重语义（文献：Guo 2017 温度缩放；对齐后过度自信）。

设计（数据纪律不变：λ 份选参、测试份一次性评估）：
1. 校准诊断：P_llm 置信度（argmax 概率）vs 实际命中率的期望校准误差
   (ECE)，分 10 桶；比较原分布与温度缩放后分布。
2. 联合参数搜索：在 λ 份上 2D grid 搜索 (温度 T, 固定 λ) 最小化
   融合 log-loss；输出最优 (T*, λ*)。
3. 测试份一次性评估：未校准固定λ（对照）vs 校准后固定λ vs
   校准后状态λ(s)（λ(s) 在 λ 份上以校准后 P_llm 重新学习）。

用法:
    ./.venv/Scripts/python.exe scripts/run_calibration.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_fusion_eval import (apply_lambda, compute_artifacts, fuse,
                             learn_state_lambda, mcnemar_exact, score_split,
                             wilson_ci)

OUT_JSON = "output/eval_calibration.json"


def ece(probs, y_true, n_bins=10):
    """期望校准误差：argmax 置信度分桶，比较平均置信度 vs 桶内命中率."""
    conf = probs.max(axis=1)
    hit = probs.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    bin_low = np.digitize(conf, edges[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_low == b
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(conf[mask].mean() - hit[mask].mean())
    return float(ece)


def temp_scale(log_p, T):
    return (log_p / T) - (log_p / T).max(axis=-1, keepdims=True)


def main():
    art = compute_artifacts(skip_llm=True)  # LLM 分数已缓存
    llm_probs = art["llm_probs"]
    lam_res, lam_lp, lam_X, lam_y = score_split(art, "lambda")
    test_res, test_lp, test_X, test_y = score_split(art, "test")

    lam_ll = np.log(np.clip(llm_probs["lambda"], 1e-12, 1))
    test_ll = np.log(np.clip(llm_probs["test"], 1e-12, 1))

    # ==== 1. 校准诊断（λ 份） ====
    ece_raw = ece(llm_probs["lambda"], lam_y)
    print(f"[1] λ份校准诊断: ECE(原始)={ece_raw:.4f}")
    # 每个 T 的 ECE
    ece_scan = {}
    for T in (1.0, 1.5, 2.0, 3.0, 4.0):
        P = np.exp(temp_scale(lam_ll, T))
        P = P / P.sum(axis=-1, keepdims=True)
        ece_scan[T] = round(ece(P, lam_y), 4)
    print(f"    ECE(T)={ece_scan}")

    # ==== 2. 联合 (T, λ) 搜索（λ 份融合 log-loss） ====
    best = None
    for T in np.linspace(0.5, 5.0, 19):
        llT = temp_scale(lam_ll, T)
        for g in np.linspace(0, 1, 21):
            P = fuse(lam_lp, llT, g)
            ll = -np.log(np.clip(P[np.arange(len(lam_y)), lam_y],
                                 1e-12, 1)).mean()
            if best is None or ll < best[0]:
                best = (float(ll), float(T), float(g))
    ll_best, T_star, lam_star = best
    # 对照：不校准
    P0 = fuse(lam_lp, lam_ll, 0.55)
    ll_nocal = -np.log(np.clip(P0[np.arange(len(lam_y)), lam_y],
                               1e-12, 1)).mean()
    print(f"[2] (T,λ)搜索: 最优 T*={T_star:.2f} λ*={lam_star:.2f} "
          f"log-loss={ll_best:.4f} | 不校准对照(λ=0.55)={ll_nocal:.4f}")

    # ==== 3. 测试份一次性评估 ====
    # 3a. 校准后固定 λ
    P_fix = fuse(test_lp, temp_scale(test_ll, T_star), lam_star)
    test_res["cal_fixed"] = P_fix.argmax(axis=1) == test_y
    # 3b. 校准后状态 λ(s)：在 λ 份上用校准后 P_llm 重学 λ(s)
    theta = learn_state_lambda(lam_X, lam_lp, temp_scale(lam_ll, T_star),
                               lam_y, l2=0.1)
    lam_test = apply_lambda(theta, test_X)
    P_st = fuse(test_lp, temp_scale(test_ll, T_star), lam_test[:, None])
    test_res["cal_state"] = P_st.argmax(axis=1) == test_y
    # 3c. 对照：未校准 fixed（沿用原 0.55）已在 test_res["fixed"]
    print(f"[3] λ(s) 分布(校准后测试份): mean={lam_test.mean():.3f} "
          f"p10={np.percentile(lam_test,10):.3f} p90={np.percentile(lam_test,90):.3f}")

    # ==== 汇总 ====
    summary = {}
    for m, hits in test_res.items():
        k = int(hits.sum())
        lo, hi = wilson_ci(k, len(hits))
        summary[m] = {"hit": round(k / len(hits), 4),
                      "ci95": [round(lo, 4), round(hi, 4)], "k": k,
                      "n": len(hits)}

    def mcnemar(m1, m2):
        b = int(np.sum(test_res[m1] & ~test_res[m2]))
        c = int(np.sum(~test_res[m1] & test_res[m2]))
        return {"m1_only_win": b, "m2_only_win": c,
                "p_value": round(mcnemar_exact(b, c), 5)}

    tests = {f"{a}_vs_{b}": mcnemar(a, b) for a, b in [
        ("cal_fixed", "fixed"), ("cal_state", "cal_fixed"),
        ("cal_state", "fixed"), ("cal_fixed", "rule_near"),
        ("cal_state", "rule_near")] if a in test_res and b in test_res}

    out = {
        "ece": {"raw_lambda": round(ece_raw, 4), "by_T": ece_scan},
        "joint_search": {"T_star": round(T_star, 2), "lambda_star": round(lam_star, 2),
                         "logloss_cal": round(ll_best, 4),
                         "logloss_nocal": round(ll_nocal, 4),
                         "delta_logloss": round(ll_nocal - ll_best, 4)},
        "lambda_dist_cal_test": {"mean": round(float(lam_test.mean()), 3),
                                 "p10": round(float(np.percentile(lam_test, 10)), 3),
                                 "p90": round(float(np.percentile(lam_test, 90)), 3)},
        "summary": summary, "paired_mcnemar": tests,
    }
    Path(OUT_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_mcnemar": tests},
                     ensure_ascii=False, indent=2))
    print(f"保存至: {OUT_JSON}")


if __name__ == "__main__":
    main()
