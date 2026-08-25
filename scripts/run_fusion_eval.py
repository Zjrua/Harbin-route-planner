"""毕设主实验：行为先验 × LLM 候选概率的状态加权融合（§3.1–§3.3 完整建模）.

数据纪律（前置实验 #2 的三分，索引存 output/test_split_power.json）：
- 拟合份 84 条：区域划分检验外的全部统计估计（二阶转移矩阵、α、距离衰减、混合分布）
- λ 份 42 条：衰减选型确认 + λ(s) stacking 学习（嵌套 CV 估泛化）
- 测试份 42 条：只做最终评估，全部 ~414 个转移点（功效粗算裁定）

评估任务：下一站 8 选 1（真实下一站 + 7 干扰，池构造同 eval_next_poi 管线，
干扰只用前缀与真实下一站信息，无泄漏）。

方法（消融链）：
    random / rule_near / rule_markov(类型级,拟合份估计)
    prior   = P2(region|state) × decay(d)         ← §3.1 统计先验
    llm     = Qwen3.5-SFT 候选编号首 token 概率    ← 意图模型
    fixed   = log-linear 池化, 全局固定 λ
    state   = λ(s) 逻辑回归 stacking               ← 核心贡献
统计：top-1 命中率 + Wilson CI + 配对精确 McNemar。

用法:
    ./.venv/Scripts/python.exe scripts/run_fusion_eval.py [--skip-llm]
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.behavior_prior import (DecayModel, HopMixture, RegionMap,
                                SecondOrderMarkov, region_sequences)
from src.itinerary_planner import load_semantic
from src.retrieval import retrieve_candidates

MODEL_BASE = "data/external/modelscope_cache/models/Qwen--Qwen3.5-4B"
SFT_LORA = "output/qwen35_route_lora"
CACHE_POOLS = "output/fusion_pools.pkl"
CACHE_LLM = "output/fusion_llm_scores.npz"
OUT_JSON = "output/eval_fusion.json"


# ==== 候选池 ====
def build_pools_for_split(routes, pois, dist, semantic, indices, seed):
    """一个分割的全部转移点（每条路线 t=2..len-1 全取）."""
    rng = random.Random(seed)
    points = []
    for ri in indices:
        idxs = [int(i) for i in routes[ri]]
        if len(idxs) < 4:
            continue
        for t in range(2, len(idxs)):
            prefix, true_next = idxs[:t], idxs[t]
            used = set(prefix) | {true_next}
            pool = retrieve_candidates(pois, dist, center_idx=prefix[-1],
                                       used_indices=used, n=25,
                                       semantic=semantic)
            if len(pool) < 7:
                continue
            true_type = int(pois.iloc[true_next]["activity_type"])
            same_type = [c for c in pool
                         if int(pois.iloc[c]["activity_type"]) == true_type]
            others = [c for c in pool if c not in same_type]
            rng.shuffle(same_type)
            rng.shuffle(others)
            n_same = min(4, len(same_type))
            distract = same_type[:n_same] + others[: 7 - n_same]
            if len(distract) < 7:
                continue
            cands = [true_next] + distract[:7]
            rng.shuffle(cands)
            points.append({"route_i": int(ri), "t": t, "prefix": prefix,
                           "true_next": true_next, "cands": cands})
    return points


def get_pools(routes, pois, dist, semantic, splits):
    if Path(CACHE_POOLS).exists():
        return pickle.load(open(CACHE_POOLS, "rb"))
    pools = {}
    for name, idx in splits.items():
        pools[name] = build_pools_for_split(routes, pois, dist, semantic, idx,
                                            seed={"fit": 1, "lambda": 2,
                                                  "test": 3}[name])
        print(f"  池[{name}]: {len(pools[name])} 个预测点", flush=True)
    pickle.dump(pools, open(CACHE_POOLS, "wb"))
    return pools


# ==== LLM 候选概率 ====
SYSTEM_PROMPT = ("你是一位哈尔滨旅游规划专家。根据已游览的路线和候选列表，"
                 "选出下一个最应该去的地点，只输出其编号，不要输出任何其他文字。")


def build_prompt(tok, pt, pois):
    prefix_names = [str(pois.iloc[i]["name"]) for i in pt["prefix"]]
    cand_lines = "\n".join(f"{j+1}.{pois.iloc[c]['name']}"
                           for j, c in enumerate(pt["cands"]))
    instr = (f"已游览路线：{' → '.join(prefix_names)}。\n候选地点：\n{cand_lines}\n"
             f"选出下一个最应该去的地点，只输出其编号（1-{len(pt['cands'])}）。")
    return ("<|im_start|>system\n" + SYSTEM_PROMPT + "\n<|im_end|>\n"
            f"<|im_start|>user\n{instr}\n<|im_end|>\n<|im_start|>assistant\n")


def llm_scores(model, tok, points, pois, batch_first_ids):
    """每点一次前向：assistant 首 token 上编号 1-8 的 logprob → 归一化概率."""
    import torch
    probs = np.zeros((len(points), 8))
    for i, pt in enumerate(points):
        text = build_prompt(tok, pt, pois)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1]  # 下一 token 分布
        lp = torch.log_softmax(logits.float(), dim=-1)
        cand_lp = np.array([float(lp[t]) for t in batch_first_ids])
        cand_lp -= cand_lp.max()
        p = np.exp(cand_lp)
        probs[i] = p / p.sum()
        if (i + 1) % 100 == 0:
            print(f"  LLM 进度 {i+1}/{len(points)}", flush=True)
    return probs


def get_llm_probs(pools, pois, skip_llm=False):
    cached = Path(CACHE_LLM).exists()
    if skip_llm and not cached:
        print("  [skip-llm] 无缓存，跳过 LLM 打分（仅评估规则/先验链）",
              flush=True)
        return {}
    if cached:
        z = np.load(CACHE_LLM)
        return {k: z[k] for k in z.files}
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
    first_ids = [tok(str(j), add_special_tokens=False)["input_ids"][0]
                 for j in range(1, 9)]
    out = {}
    for name, pts in pools.items():
        print(f"  LLM 打分 [{name}]: {len(pts)} 点", flush=True)
        out[name] = llm_scores(model, tok, pts, pois, first_ids)
    np.savez(CACHE_LLM, **out)
    return out


# ==== 特征与 λ(s) ====
def point_features(pt, pois, dist, region_of_poi, mk, mix):
    """λ(s) 的推理时可得特征（§3.2，全部不含真实下一站信息）."""
    last = pt["prefix"][-1]
    prev = pt["prefix"][-2] if len(pt["prefix"]) >= 2 else last
    ds = np.array([float(dist[last][c]) for c in pt["cands"]])
    cand_regions = np.array([region_of_poi[c] for c in pt["cands"]])
    P2 = mk.probs()
    row = P2[region_of_poi[prev], region_of_poi[last], cand_regions]
    p_norm = row / (row.sum() + 1e-12)
    return [
        float(mix.posterior_near(np.median(ds))[0]),   # 模式后验(池距离结构)
        float(mix.posterior_near(ds.min())[0]),
        mk.entropy(region_of_poi[prev], region_of_poi[last]),  # 转移熵
        float(p_norm.max()), float(p_norm.std()),         # 候选池区域分布结构
        float(np.log1p(len(pt["prefix"]))),               # 前缀长度
        float(np.log(ds.min() + 0.1)), float(np.log(np.median(ds) + 0.1)),
    ]


def fuse(log_prior, log_llm, lam):
    """log-linear 池化: log P = λ·logP_prior + (1-λ)·logP_llm (逐点 λ)."""
    lp = lam * log_prior + (1 - lam) * log_llm
    lp -= lp.max(axis=-1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(axis=-1, keepdims=True)


def learn_state_lambda(X, log_prior, log_llm, y_true, l2=0.1):
    """逻辑回归 stacking: λ(s)=sigmoid(w·f)，目标最小化池化 log loss."""
    from scipy.optimize import minimize
    n, d = X.shape

    def nll(theta):
        w = theta[:-1]
        b = theta[-1]
        lam = 1 / (1 + np.exp(-(X @ w + b)))[:, None]
        P = fuse(log_prior, log_llm, lam)
        p_true = np.clip(P[np.arange(n), y_true], 1e-12, 1)
        return -np.log(p_true).mean() + l2 * np.sum(w ** 2)

    res = minimize(nll, np.zeros(d + 1), method="L-BFGS-B")
    return res.x


def apply_lambda(theta, X):
    w, b = theta[:-1], theta[-1]
    return 1 / (1 + np.exp(-(X @ w + b)))


# ==== 统计工具 ====
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, center - half), min(1, center + half)


def mcnemar_exact(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def compute_artifacts(skip_llm=False):
    """数据 + 先验拟合 + 衰减选型 + LLM 概率（池/LLM 分数走缓存）.

    返回 dict，供本脚本与 run_calibration.py 复用。
    """
    routes = np.load("data/processed/routes_xhs_holdout.npy", allow_pickle=True)
    pois = pd.read_csv("data/processed/poi_metadata.csv", encoding="utf-8")
    dist = np.load("data/processed/distance_matrix.npy")
    semantic = load_semantic()
    splits = {k: np.array(v) for k, v in
              json.load(open("output/test_split_power.json",
                             encoding="utf-8"))["split"]["indices"].items()}

    # ==== §3.1 先验模型（只在拟合份上） ====
    print("[1] 拟合行为先验（拟合份 84 条）", flush=True)
    meta = pd.read_csv("data/processed/poi_metadata.csv")
    clusters = np.load("data/processed/clusters.npy", allow_pickle=True)
    cluster_id = np.load("data/processed/cluster_id.npy")
    rmap = RegionMap().fit(meta, clusters)
    region_of_poi = rmap.transform(cluster_id)
    fit_seqs = region_sequences(routes, splits["fit"], region_of_poi)
    lam_seqs = region_sequences(routes, splits["lambda"], region_of_poi)

    # α 敏感性（λ 份 held-out log-loss 选 α）
    alpha_scan = {}
    for a in (0.1, 0.5, 1.0, 2.0, 5.0):
        mk_a = SecondOrderMarkov(alpha=a).fit(fit_seqs)
        n_tr = sum(len(s) - 2 for s in lam_seqs)
        alpha_scan[a] = round(-mk_a.heldout_loglik(lam_seqs) / n_tr, 4)
    best_alpha = min(alpha_scan, key=alpha_scan.get)
    print(f"  α 敏感性(λ份 log-loss/转移): {alpha_scan} → α={best_alpha}",
          flush=True)
    mk = SecondOrderMarkov(alpha=best_alpha).fit(fit_seqs)

    # 混合距离分布
    hops_fit = np.array([h for s in
                         [[float(dist[a][b]) for a, b in
                           zip([int(x) for x in routes[i]],
                               [int(x) for x in routes[i]][1:])]
                          for i in splits["fit"]] for h in s])
    mix = HopMixture().fit(hops_fit)
    print(f"  距离混合: w_near={mix.w_:.2f} logN(median={np.exp(mix.mu_):.2f}km) "
          f"+ logGamma(a={mix.a_:.2f}, median={np.exp(mix.a_+0.577):.2f}km)",
          flush=True)

    # 类型转移（rule_markov 用，拟合份）
    T_counts = np.ones((6, 6)) * 0.5
    for i in splits["fit"]:
        idxs = [int(x) for x in routes[i]]
        for a, b in zip(idxs, idxs[1:]):
            T_counts[int(pois.iloc[a]["activity_type"]),
                     int(pois.iloc[b]["activity_type"])] += 1
    T_type = T_counts / T_counts.sum(axis=1, keepdims=True)

    # ==== 候选池 ====
    print("[2] 构建候选池", flush=True)
    pools = get_pools(routes, pois, dist, semantic, splits)

    # ==== 距离衰减选型（拟合份池 MLE，λ 份池确认） ====
    print("[3] 距离衰减选型", flush=True)

    def decay_pools(name):
        return [([float(dist[p["prefix"][-1]][c]) for c in p["cands"]],
                 p["cands"].index(p["true_next"])) for p in pools[name]]

    decay_out = {}
    models = {}
    for form in ("exp", "power", "mix"):
        m = DecayModel(form=form).fit(decay_pools("fit"))
        decay_out[form] = {"params": [round(p, 3) for p in m.params_],
                           "fit_loglik": round(m.loglik_, 1),
                           "aic": round(m.aic(), 1),
                           "lambda_logloss": round(m.heldout_logloss(
                               decay_pools("lambda")), 4)}
        models[form] = m
    best_form = min(decay_out, key=lambda f: decay_out[f]["lambda_logloss"])
    print(f"  {json.dumps(decay_out, ensure_ascii=False)} → {best_form}",
          flush=True)
    decay = models[best_form]

    # ==== LLM 概率 ====
    print("[4] LLM 候选概率", flush=True)
    llm_probs = get_llm_probs(pools, pois, skip_llm=skip_llm)

    return {"routes": routes, "pois": pois, "dist": dist,
            "splits": splits, "region_of_poi": region_of_poi,
            "fit_seqs": fit_seqs, "mk": mk, "mix": mix, "T_type": T_type,
            "decay": decay, "decay_out": decay_out, "best_form": best_form,
            "alpha_scan": alpha_scan, "best_alpha": best_alpha,
            "pools": pools, "llm_probs": llm_probs}


def score_split(art, name):
    """一个分割的逐方法命中 / 先验 log 分 / λ(s) 特征 / 真值下标."""
    pois, dist = art["pois"], art["dist"]
    region_of_poi, mk, mix, T_type = (art["region_of_poi"], art["mk"],
                                      art["mix"], art["T_type"])
    pts = art["pools"][name]
    n = len(pts)
    res = {m: np.zeros(n, bool) for m in
           ("random", "rule_near", "rule_markov", "prior")}
    log_prior = np.zeros((n, 8))
    X = np.zeros((n, 8))
    y_true = np.zeros(n, int)
    rng = random.Random(5)
    for i, pt in enumerate(pts):
        last, prev = pt["prefix"][-1], pt["prefix"][-2]
        ds = np.array([float(dist[last][c]) for c in pt["cands"]])
        res["random"][i] = rng.choice(pt["cands"]) == pt["true_next"]
        res["rule_near"][i] = (pt["cands"][int(np.argmin(ds))]
                               == pt["true_next"])
        lt = int(pois.iloc[last]["activity_type"])
        rm_sc = [T_type[lt][int(pois.iloc[c]["activity_type"])]
                 * np.exp(-float(dist[last][c]) / 3.0) for c in pt["cands"]]
        res["rule_markov"][i] = (pt["cands"][int(np.argmax(rm_sc))]
                                 == pt["true_next"])
        cr = np.array([region_of_poi[c] for c in pt["cands"]])
        lp = mk.log_score(region_of_poi[prev], region_of_poi[last], cr) \
            + decay_log_score(art, ds)
        log_prior[i] = lp - lp.max()
        res["prior"][i] = (pt["cands"][int(np.argmax(lp))]
                           == pt["true_next"])
        X[i] = point_features(pt, pois, dist, region_of_poi, mk, mix)
        y_true[i] = pt["cands"].index(pt["true_next"])
    return res, log_prior, X, y_true


def decay_log_score(art, ds):
    return art["decay"].log_score(ds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true",
                        help="使用缓存的 LLM 分数（无缓存时评估先验/规则链）")
    args = parser.parse_args()

    art = compute_artifacts(skip_llm=args.skip_llm)
    alpha_scan, best_alpha = art["alpha_scan"], art["best_alpha"]
    decay_out, best_form, mix = art["decay_out"], art["best_form"], art["mix"]
    pools, llm_probs = art["pools"], art["llm_probs"]

    # ==== 逐方法预测（λ 份 + 测试份） ====
    print("[5] 评估", flush=True)
    lam_res, lam_lp, lam_X, lam_y = score_split(art, "lambda")
    test_res, test_lp, test_X, test_y = score_split(art, "test")

    # 固定 λ 扫描（λ 份上选）
    if "lambda" in llm_probs:
        lam_ll = np.log(np.clip(llm_probs["lambda"], 1e-12, 1))
        grid = np.linspace(0, 1, 21)
        ll_scan = []
        for g in grid:
            P = fuse(lam_lp, lam_ll, g)
            ll_scan.append(-np.log(np.clip(
                P[np.arange(len(lam_y)), lam_y], 1e-12, 1)).mean())
        best_g = float(grid[int(np.argmin(ll_scan))])

        # λ(s) 嵌套 CV（外层 5 折估泛化 log-loss）
        from scipy.optimize import minimize
        n_lam = len(lam_y)
        fold = np.arange(n_lam) % 5
        cv_ll = {("fixed", best_g): [], ("state_l2", 0.1): []}
        for f in range(5):
            tr, te = fold != f, fold == f
            theta = learn_state_lambda(lam_X[tr], lam_lp[tr], lam_ll[tr],
                                       lam_y[tr], l2=0.1)
            P = fuse(lam_lp[te], lam_ll[te],
                     apply_lambda(theta, lam_X[te])[:, None])
            cv_ll[("state_l2", 0.1)].append(-np.log(np.clip(
                P[np.arange(te.sum()), lam_y[te]], 1e-12, 1)).mean())
            Pf = fuse(lam_lp[te], lam_ll[te], best_g)
            cv_ll[("fixed", best_g)].append(-np.log(np.clip(
                Pf[np.arange(te.sum()), lam_y[te]], 1e-12, 1)).mean())
        cv_summary = {str(k): round(float(np.mean(v)), 4)
                      for k, v in cv_ll.items()}
        print(f"  嵌套CV log-loss: {cv_summary}", flush=True)

        # 全 λ 份训练最终 λ(s)，测试份评估
        theta = learn_state_lambda(lam_X, lam_lp, lam_ll, lam_y, l2=0.1)
        test_ll = np.log(np.clip(llm_probs["test"], 1e-12, 1))
        lam_test = apply_lambda(theta, test_X)
        for name, lam_vec in (("llm", 0.0), ("fixed", best_g),
                              ("state", lam_test)):
            lv = np.full(len(test_y), lam_vec) if np.isscalar(lam_vec) \
                else np.asarray(lam_vec)
            P = fuse(test_lp, test_ll, lv[:, None])
            pred = np.argmax(P, axis=1)
            test_res[name] = pred == test_y
        lam_dist = {"mean": round(float(lam_test.mean()), 3),
                    "p10": round(float(np.percentile(lam_test, 10)), 3),
                    "p90": round(float(np.percentile(lam_test, 90)), 3)}
        print(f"  λ(s) 分布(测试份): {lam_dist}", flush=True)
    else:
        best_g, cv_summary, lam_dist = None, None, None

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
        ("prior", "rule_near"), ("llm", "rule_near"),
        ("state", "rule_near"), ("state", "prior"), ("state", "llm"),
        ("state", "fixed"), ("fixed", "rule_near"),
        ("rule_markov", "rule_near")] if a in test_res and b in test_res}

    out = {
        "prior_model": {
            "K": 8, "alpha_selected": best_alpha,
            "alpha_sensitivity": alpha_scan,
            "decay_selection": decay_out, "decay_form": best_form,
            "mixture": {"w_near": round(float(mix.w_), 3),
                        "near_median_km": round(float(np.exp(mix.mu_)), 2),
                        "far_loggamma_a": round(float(mix.a_), 3),
                        "far_median_km": round(float(np.exp(mix.a_ + 0.577)), 2)}},
        "n_points": {"lambda": len(lam_y), "test": len(test_y)},
        "fixed_lambda": best_g, "cv_logloss": cv_summary,
        "lambda_dist_test": lam_dist,
        "summary": summary, "paired_mcnemar": tests,
    }
    Path(OUT_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_mcnemar": tests},
                     ensure_ascii=False, indent=2))
    print(f"保存至: {OUT_JSON}")


if __name__ == "__main__":
    main()
