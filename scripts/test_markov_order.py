"""前置实验 #3：马尔可夫模型阶数——一阶 vs 二阶（回应导师质疑#3）.

设计文档：docs/superpowers/specs/2026-08-02-thesis-markov-transformer-fusion-design.md §3.1

两层裁决：
1. 参数 bootstrap LRT（拟合份 84 条，K=8 区域）：H0=一阶 vs H1=二阶。
   二阶在单次出现的 (i,j) 上下文上退化（概率 1），常规卡方不可用，
   用一阶模型模拟数据重算 LRT 分布求 p 值。
2. 样本外预测（λ 份 42 条，410 转移）：两阶模型均用 Dirichlet 平滑
   (α=0.5) 在拟合份上估计，比较 λ 份逐转移 log-loss；路线级 cluster
   bootstrap 检验差异。这是实际规划用途下的裁决（稀疏上下文靠平滑兜底）。

用法:
    ./.venv/Scripts/python.exe scripts/test_markov_order.py [--boot 200]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

K = 8
ALPHA = 0.5  # Dirichlet 平滑
SPLIT_JSON = json.load(open("output/test_split_power.json", encoding="utf-8"))
SPLIT = {k: np.array(v) for k, v in SPLIT_JSON["split"]["indices"].items()}


def load_region_of_poi(K):
    meta = pd.read_csv("data/processed/poi_metadata.csv")
    clusters = np.load("data/processed/clusters.npy", allow_pickle=True)
    cluster_id = np.load("data/processed/cluster_id.npy")
    centroids = np.array([[meta.iloc[cl]["lng"].mean(), meta.iloc[cl]["lat"].mean()]
                          for cl in clusters])
    Z = linkage(centroids, method="ward")
    roc = fcluster(Z, t=K, criterion="maxclust") - 1
    return np.array([roc[c] for c in cluster_id])


def region_routes(routes, indices, region_of_poi):
    """每条路线 → 区域序列（保留路线分组）."""
    out = []
    for i in indices:
        r = [region_of_poi[int(x)] for x in routes[i]]
        if len(r) >= 2:
            out.append(r)
    return out


# ==== 计数与似然 ====
def count1(seqs):
    C = np.zeros((K, K))
    for s in seqs:
        for a, b in zip(s, s[1:]):
            C[a, b] += 1
    return C


def count2(seqs):
    C = np.zeros((K, K, K))
    for s in seqs:
        for a, b, c in zip(s, s[1:], s[2:]):
            C[a, b, c] += 1
    return C


def probs_from_counts(counts, alpha):
    """行向 Dirichlet 平滑到概率（counts 可为 2D 或 3D，最后一维为下一站）."""
    P = counts + alpha
    tot = P.sum(axis=-1, keepdims=True)
    bad = (tot == 0).squeeze(-1)          # 从未见过的行 → 均匀分布兜底
    P = np.where(tot == 0, 1.0 / counts.shape[-1], P / np.where(tot == 0, 1, tot))
    return P


def ll_first(seqs, C1, alpha=0.0):
    P = probs_from_counts(C1, alpha)
    return sum(np.log(P[a, b]) for s in seqs
               for a, b in zip(s, s[1:]))


def ll_second(seqs, C2, alpha=0.0):
    P = probs_from_counts(C2, alpha)
    return sum(np.log(P[a, b, c]) for s in seqs
               for a, b, c in zip(s, s[1:], s[2:]))


def lrt_stat(seqs):
    """一阶 vs 二阶 LRT 统计量（MLE，无平滑；单次上下文退化由 bootstrap 吸收）."""
    C1, C2 = count1(seqs), count2(seqs)
    return 2 * (ll_second(seqs, C2) - ll_first(seqs, C1))


def simulate_first(C1, n_seq_lens, rng):
    """H0（一阶 MLE）下模拟区域路线序列，长度取自真实路线."""
    P = probs_from_counts(C1, 0.0)
    seqs = []
    for L in n_seq_lens:
        s = [rng.randint(K)]
        for _ in range(L - 1):
            s.append(rng.choice(K, p=P[s[-1]]))
        seqs.append([int(x) for x in s])
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=200)
    args = parser.parse_args()

    routes = np.load("data/processed/routes_xhs_holdout.npy", allow_pickle=True)
    region_of_poi = load_region_of_poi(K)
    fit = region_routes(routes, SPLIT["fit"], region_of_poi)
    lam = region_routes(routes, SPLIT["lambda"], region_of_poi)
    n_fit = sum(len(s) - 1 for s in fit)
    print(f"拟合份: {len(fit)} 条区域序列, 转移 {n_fit} | λ份: {len(lam)} 条, "
          f"转移 {sum(len(s)-1 for s in lam)}")

    # ---- 1. 参数 bootstrap LRT ----
    C1 = count1(fit)
    t_obs = lrt_stat(fit)
    # 有效上下文数（出现≥2次的 (i,j)），仅作参考 df
    C2 = count2(fit)
    ctx = C2.sum(axis=2)
    df_eff = int(sum((ctx[m] - 1) * (K - 1) for m in zip(*np.where(ctx >= 2))))
    rng = np.random.RandomState(3)
    lens = [len(s) for s in fit]
    sims = []
    for _ in range(args.boot):
        seqs = simulate_first(C1, lens, rng)
        sims.append(lrt_stat(seqs))
    sims = np.array(sims)
    p_boot = (np.sum(sims >= t_obs) + 1) / (len(sims) + 1)
    print(f"\n[1] LRT: 观测={t_obs:.1f} | H0模拟 p50={np.median(sims):.1f} "
          f"p95={np.percentile(sims, 95):.1f} | bootstrap p={p_boot:.3f} "
          f"(参考名义 df={df_eff})")

    # ---- 2. 样本外 log-loss（λ 份，Dirichlet α=0.5） ----
    ll1 = ll_first(lam, C1, ALPHA)
    ll2 = ll_second(lam, C2, ALPHA)
    n_lam_1 = sum(len(s) - 1 for s in lam)
    n_lam_2 = sum(len(s) - 2 for s in lam if len(s) >= 3)
    print(f"\n[2] λ份样本外(α={ALPHA}): 一阶 log-loss/转移 = {-ll1/n_lam_1:.4f} | "
          f"二阶 = {-ll2/n_lam_2:.4f}")

    # 路线级 cluster bootstrap 检验 log-loss 差（同一路线下逐转移差之和）
    P1 = probs_from_counts(C1, ALPHA)
    P2 = probs_from_counts(C2, ALPHA)
    per_route_diff = []
    for s in lam:
        d1 = sum(np.log(P1[a, b]) for a, b in zip(s, s[1:]))
        d2 = sum(np.log(P2[a, b, c])
                 for a, b, c in zip(s, s[1:], s[2:])) if len(s) >= 3 else 0.0
        per_route_diff.append(d2 - d1)
    per_route_diff = np.array(per_route_diff)
    rng = np.random.RandomState(11)
    boots = [np.mean(per_route_diff[rng.choice(len(per_route_diff),
                                               len(per_route_diff), replace=True)])
             for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two = 2 * min(np.mean(np.array(boots) <= 0), np.mean(np.array(boots) >= 0))
    print(f"    二阶-一阶 总loglik差 = {per_route_diff.sum():.1f} | "
          f"路线bootstrap 95%CI [{lo:.2f}, {hi:.2f}] | p≈{min(p_two,1):.3f}")

    verdict = "second_order" if p_boot < 0.05 and lo > 0 else "first_order"
    print(f"\n=== 裁决: {verdict} ===")

    out = {
        "K": K, "alpha": ALPHA, "boot": args.boot,
        "n_fit_transitions": n_fit,
        "lrt": {"t_obs": round(float(t_obs), 1),
                "h0_p95": round(float(np.percentile(sims, 95)), 1),
                "bootstrap_p": round(float(p_boot), 4),
                "nominal_df_ref": df_eff},
        "heldout": {"first_order_loss_per_trans": round(float(-ll1 / n_lam_1), 4),
                    "second_order_loss_per_trans": round(float(-ll2 / n_lam_2), 4),
                    "total_ll_diff": round(float(per_route_diff.sum()), 1),
                    "route_bootstrap_ci95": [round(float(lo), 2), round(float(hi), 2)],
                    "p": round(float(min(p_two, 1)), 4)},
        "verdict": verdict,
    }
    Path("output/test_markov_order.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("保存至 output/test_markov_order.json")


if __name__ == "__main__":
    main()
