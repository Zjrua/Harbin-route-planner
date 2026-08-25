"""前置实验 #2：数据三分方案 + 功效粗算（回应导师质疑#2）.

设计文档：docs/superpowers/specs/2026-08-02-thesis-markov-transformer-fusion-design.md §3.3

三件事：
1. 三分方案：168 条路线 → 拟合马尔可夫(84) / 学 λ(42) / 最终测试(42)，
   按路线长度分层随机（各份转移距离分布一致性用 KS 检验）；种子稳定性扫描。
2. 粒度回退：90 簇 → K 个区域（质心 Ward 合并），在拟合份上算 K×K 转移
   计数矩阵的稀疏度/覆盖曲线，按决策规则选 K（若太稀疏 → 降为 4-5 大区）。
3. 功效粗算：最终测试份 McNemar 可检测最小差异（MDE），关联结构取自
   eval_next_poi.json 的真实配对记录（rule_near vs llm / rule_markov）。

用法:
    ./.venv/Scripts/python.exe scripts/test_split_power.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED = 20260802
SPLIT = {"fit": 84, "lambda": 42, "test": 42}


# ==== 1. 三分 ====
def stratified_split(routes, seed):
    """按路线长度三分位分层，层内随机按 84/42/42 分配（比例分配）."""
    lens = np.array([len(r) for r in routes])
    q = np.quantile(lens, [1 / 3, 2 / 3])
    strata = np.digitize(lens, q)
    rng = np.random.RandomState(seed)
    idx = {"fit": [], "lambda": [], "test": []}
    for s in range(3):
        members = np.where(strata == s)[0]
        rng.shuffle(members)
        n = len(members)
        n_fit = round(n * SPLIT["fit"] / 168)
        n_lam = round(n * SPLIT["lambda"] / 168)
        idx["fit"] += members[:n_fit].tolist()
        idx["lambda"] += members[n_fit:n_fit + n_lam].tolist()
        idx["test"] += members[n_fit + n_lam:].tolist()
    return {k: np.array(sorted(v)) for k, v in idx.items()}


_DIST = None


def hops_of(routes, indices):
    global _DIST
    dist = _DIST if _DIST is not None else np.load(
        "data/processed/distance_matrix.npy")
    _DIST = dist
    out = []
    for i in indices:
        r = [int(x) for x in routes[i]]
        if len(r) < 2:
            continue
        out += [float(dist[a][b]) for a, b in zip(r, r[1:])]
    return np.array(out)


# ==== 2. 区域粒度 ====
def build_regions(K, meta, clusters):
    """90 簇质心 Ward 合并到 K 区域，返回 poi_id -> region 映射."""
    from scipy.cluster.hierarchy import fcluster, linkage
    centroids = []
    for cl in clusters:
        sub = meta.iloc[cl]
        centroids.append([sub["lng"].mean(), sub["lat"].mean()])
    centroids = np.array(centroids)
    Z = linkage(centroids, method="ward")
    region_of_cluster = fcluster(Z, t=K, criterion="maxclust")
    cluster_id = np.load("data/processed/cluster_id.npy")
    return cluster_id.map(lambda c: int(region_of_cluster[c]) - 1) if hasattr(
        cluster_id, "map") else np.array([region_of_cluster[c] for c in cluster_id])


def region_stats(routes, indices, region_of_poi, K):
    """K×K 转移计数矩阵的稀疏度与覆盖曲线."""
    C = np.zeros((K, K))
    for i in indices:
        r = [int(x) for x in routes[i]]
        for a, b in zip(r, r[1:]):
            C[region_of_poi[a], region_of_poi[b]] += 1
    total = C.sum()
    nz = C > 0
    diag = np.diag(C).sum()
    return {
        "n_transitions": int(total),
        "zero_cell_rate": round(float(1 - nz.mean()), 3),
        "diag_share": round(float(diag / total), 3),
        "cov_ge1": round(float(C[C >= 1].sum() / total), 3),
        "cov_ge3": round(float(C[C >= 3].sum() / total), 3),
        "cov_ge5": round(float(C[C >= 5].sum() / total), 3),
        "median_nonzero_count": round(float(np.median(C[nz])), 1),
    }


# ==== 3. McNemar 功效 ====
def mcnemar_power(n, b, c, alpha=0.05):
    """精确二项 McNemar 功效：给定真实 discordant (b,c)，缩放到样本量 n."""
    scale = n / (b + c) if b + c else 0
    b_s, c_s = b * scale, c * scale
    n_d = b_s + c_s
    p = b_s / n_d  # H1 下 discordant 归属概率
    # 双侧精确二项检验在 H0 p=0.5 的功效
    lo = stats.binom.ppf(alpha / 2, n_d, 0.5) + 1
    hi = stats.binom.ppf(1 - alpha / 2, n_d, 0.5)
    power = (stats.binom.cdf(lo - 1, n_d, p)
             + 1 - stats.binom.cdf(hi, n_d, p))
    return power, n_d


def mde(n, b, c, alpha=0.05, target=0.8):
    """达到 target 功效所需的最小 discordant 失衡（扫描 b-c 失衡，b+c 固定）."""
    n_d = (b + c) * n / (b + c + 0)  # discordant 数按样本量线性缩放
    n_d = (b + c) * n / 447  # 基准来自 447 点记录
    if n_d < 2:
        return None
    for delta_pts in range(0, 40):  # 整体命中率差（百分点）
        d = delta_pts / 100 * n
        b_s = (n_d + d) / 2
        c_s = (n_d - d) / 2
        if c_s < 0.5:
            continue
        pw, _ = mcnemar_power(n, b_s, c_s, alpha)
        if pw >= target:
            return delta_pts, round(pw, 3)
    return None


def main():
    routes = np.load("data/processed/routes_xhs_holdout.npy", allow_pickle=True)
    meta = pd.read_csv("data/processed/poi_metadata.csv")

    # ---- 1. 三分 + 一致性 ----
    split = stratified_split(routes, SEED)
    hops = {k: hops_of(routes, v) for k, v in split.items()}
    print("[1] 三分方案（分层: 路线长度三分位, seed=%d）" % SEED)
    for k in ["fit", "lambda", "test"]:
        h = hops[k]
        print(f"    {k:6s}: 路线 {len(split[k]):3d} | 转移 {len(h):4d} | "
              f"p50={np.median(h):.2f}km 跨区(≥2km)占比={np.mean(h >= 2):.2f}")
    for k in ["lambda", "test"]:
        ks = stats.ks_2samp(hops["fit"], hops[k])
        print(f"    KS(fit vs {k}): D={ks.statistic:.3f} p={ks.pvalue:.3f}")

    # 种子稳定性
    trans_test = []
    for s in range(5):
        sp = stratified_split(routes, s)
        trans_test.append(len(hops_of(routes, sp["test"])))
    print(f"    种子稳定性(5 seeds): test 转移数 {trans_test} "
          f"(均值 {np.mean(trans_test):.0f})")

    # ---- 2. 区域粒度 ----
    clusters = np.load("data/processed/clusters.npy", allow_pickle=True)
    cluster_id = np.load("data/processed/cluster_id.npy")
    print("\n[2] 区域粒度回退扫描（在拟合份 84 条路线上）")
    granularity = {}
    for K in range(4, 13):
        region_of_poi = np.array([0] * len(cluster_id))
        from scipy.cluster.hierarchy import fcluster, linkage
        centroids = np.array([[meta.iloc[cl]["lng"].mean(),
                               meta.iloc[cl]["lat"].mean()] for cl in clusters])
        Z = linkage(centroids, method="ward")
        roc = fcluster(Z, t=K, criterion="maxclust") - 1
        for poi, c in enumerate(cluster_id):
            region_of_poi[poi] = roc[c]
        st = region_stats(routes, split["fit"], region_of_poi, K)
        granularity[K] = st
        print(f"    K={K:2d}: 转移 {st['n_transitions']:4d} | 零格率 {st['zero_cell_rate']:.2f} | "
              f"对角占比 {st['diag_share']:.2f} | 计数≥3覆盖 {st['cov_ge3']:.2f} | "
              f"非零格中位计数 {st['median_nonzero_count']}")

    # 决策规则：满足 cov_ge3 ≥ 0.98 且非零格中位计数 ≥ 3 的最大 K（零格=从未共现的
    # 区域对，由 Dirichlet 平滑兜底，不构成估计障碍）
    ok = [K for K, st in granularity.items()
          if st["cov_ge3"] >= 0.98 and st["median_nonzero_count"] >= 3]
    rec_K = max(ok) if ok else 4
    print(f"    => 推荐 K={rec_K}" + ("" if ok else "（规则未满足，回退到粗大区）"))

    # ---- 3. McNemar 功效 ----
    rec = json.load(open("output/eval_next_poi.json", encoding="utf-8"))["records"]
    rn = np.array([r["rule_near"] for r in rec])
    pairs = {
        "hybrid_vs_rule_near(以llm为代理)": ("llm", "rule_near"),
        "markov_vs_rule_near": ("rule_markov", "rule_near"),
    }
    print("\n[3] 测试份 McNemar 功效（n_test 按 447×42/168≈112 预测点, 另给全转移 415）")
    power_out = {}
    for name, (a, b_) in pairs.items():
        xa = np.array([r[a] for r in rec]).astype(int)
        xb = np.array([r[b_] for r in rec]).astype(int)
        b_only = int(np.sum(xa & ~xb))   # a 赢 b 输
        c_only = int(np.sum(~xa & xb))
        base_rate = xb.mean()
        row = {"baseline_hit": round(float(base_rate), 3),
               "b_only": b_only, "c_only": c_only}
        for n_test, tag in [(112, "n112"), (415, "n415")]:
            pw, n_d = mcnemar_power(n_test, b_only, c_only)
            m = mde(n_test, b_only, c_only)
            row[tag] = {"achieved_power_if真实差异同现观测": round(pw, 3),
                        "n_discordant": round(n_d, 0),
                        "mde_pts_at80power": m[0] if m else None}
        power_out[name] = row
        print(f"    {name}: 基线命中 {base_rate:.2f} | 现有配对 b={b_only} c={c_only}")
        for tag in ["n112", "n415"]:
            r = row[tag]
            print(f"      {tag}: MDE(80%功效)={r['mde_pts_at80power']}pt")
    # 反向：若融合真实提升 8pt，测试份能检出吗
    print("    敏感性: 若融合真实提升 8pt(近/远分桶反转规模), test=112 时 "
          "discordant 失衡需求见上 MDE 对比")

    out = {
        "split": {"seed": SEED,
                  "indices": {k: v.tolist() for k, v in split.items()},
                  "n_transitions": {k: len(hops[k]) for k in hops},
                  "far_share": {k: round(float(np.mean(hops[k] >= 2)), 3) for k in hops},
                  "ks_fit_vs": {k: {"D": round(float(stats.ks_2samp(hops['fit'], hops[k]).statistic), 3),
                                    "p": round(float(stats.ks_2samp(hops['fit'], hops[k]).pvalue), 3)}
                                for k in ["lambda", "test"]},
                  "seed_stability_test_transitions": trans_test},
        "granularity": granularity,
        "recommended_K": rec_K,
        "power": power_out,
    }
    Path("output/test_split_power.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n保存至 output/test_split_power.json")


if __name__ == "__main__":
    main()
