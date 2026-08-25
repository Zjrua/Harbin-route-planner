"""双峰性检验：真实游记转移距离是双峰混合还是重尾单峰？（毕设叙事裁决实验）.

背景：分桶分析用 2km 拍定阈值观察到"双峰"（52% 近/48% 远），但它可能是
binning artifact——重尾单峰（对数正态）同样能产生该现象。本实验正式裁决：
若单峰胜出，"模式切换"叙事需改写为"连续状态调权"。

四层检验：
1. Hartigan dip test（原始距离 + log 距离）——直接检验单峰性偏离
2. 参数 bootstrap LRT：H0=单峰对数正态 vs H1=两分量对数正态混合。
   混合模型 LRT 不满足常规卡方（参数在边界），用参数自助法求 p 值（标准做法）
3. 路线级 cluster bootstrap（重采样路线，处理转移间序列相关）——稳健性
4. 阈值敏感性：跨区占比随阈值(0.5-10km)变化曲线——量化 binning 依赖

用法:
    ./.venv/Scripts/python.exe scripts/test_bimodality.py [--boot 200] [--cluster 100]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EPS = 1e-3  # log 安全项


def extract_hops(routes, dist):
    """全量相邻转移距离（按路线分组保留，供 cluster bootstrap）."""
    hops_by_route = []
    for r in routes:
        idxs = [int(i) for i in r]
        if len(idxs) < 2:
            continue
        hops_by_route.append([float(dist[a][b]) for a, b in zip(idxs, idxs[1:])])
    return hops_by_route


# ==== 单峰对数正态 MLE（闭式） ====
def fit_lognormal(x):
    lx = np.log(x)
    return lx.mean(), lx.std(ddof=0)


def loglik_lognormal(x, mu, sigma):
    lx = np.log(x)
    return float(np.sum(-0.5 * ((lx - mu) / sigma) ** 2 - np.log(sigma)
                        - 0.5 * np.log(2 * np.pi) - lx))


# ==== 两分量对数正态混合 EM ====
def fit_mix2_lognormal(x, n_init=8, max_iter=300, tol=1e-7, seed=0):
    lx = np.log(x)
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(n_init):
        # 随机双初值（均值分位散布）
        m1, m2 = np.quantile(lx, rng.uniform(0.1, 0.45)), np.quantile(lx, rng.uniform(0.55, 0.9))
        s1 = s2 = lx.std() / 2 + 1e-3
        w = 0.5
        for _ in range(max_iter):
            # E 步
            p1 = w * _dnorm(lx, m1, s1)
            p2 = (1 - w) * _dnorm(lx, m2, s2)
            tot = p1 + p2 + 1e-300
            r1 = p1 / tot
            # M 步
            n1 = r1.sum()
            w = n1 / len(lx)
            if w < 1e-6 or w > 1 - 1e-6:  # 退化保护
                break
            m1 = (r1 * lx).sum() / n1
            m2 = ((1 - r1) * lx).sum() / (len(lx) - n1)
            s1 = np.sqrt((r1 * (lx - m1) ** 2).sum() / n1) + 1e-6
            s2 = np.sqrt(((1 - r1) * (lx - m2) ** 2).sum() / (len(lx) - n1)) + 1e-6
            ll = _mix_ll(lx, w, m1, s1, m2, s2)
            if best is None or ll > best[0]:
                pass
            if abs(ll - (_mix_ll(lx, w, m1, s1, m2, s2))) < tol:
                break
        ll = _mix_ll(lx, w, m1, s1, m2, s2)
        if best is None or ll > best[0]:
            best = (ll, w, m1, s1, m2, s2)
    return best


def _dnorm(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _mix_ll(lx, w, m1, s1, m2, s2):
    return float(np.sum(np.log(w * _dnorm(lx, m1, s1) + (1 - w) * _dnorm(lx, m2, s2) + 1e-300)))


def loglik_mix2(x, params):
    _, w, m1, s1, m2, s2 = params
    lx = np.log(x)
    return _mix_ll(lx, w, m1, s1, m2, s2)


def lrt(x):
    """观测 LRT 统计量 + 拟合参数."""
    mu, sigma = fit_lognormal(x)
    ll0 = loglik_lognormal(x, mu, sigma)
    params = fit_mix2_lognormal(x)
    ll1 = loglik_mix2(x, params)
    return 2 * (ll1 - ll0), (mu, sigma), params


def bootstrap_lrt(x, mu, sigma, B, seed=1):
    """参数 bootstrap：H0 下模拟样本重算 LRT 分布."""
    rng = np.random.RandomState(seed)
    n = len(x)
    sims = []
    for _ in range(B):
        xs = np.exp(rng.normal(mu, sigma, n))
        try:
            t, _, _ = lrt(xs)
            sims.append(t)
        except Exception:
            continue
    return np.array(sims)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=200)
    parser.add_argument("--cluster", type=int, default=100)
    parser.add_argument("--out", default="output/test_bimodality.json")
    args = parser.parse_args()

    import pandas as pd
    from diptest import diptest as dip

    routes = np.load("data/processed/routes_xhs_holdout.npy", allow_pickle=True)
    dist = np.load("data/processed/distance_matrix.npy")
    hops_by_route = extract_hops(routes, dist)
    all_hops = np.array([h for hs in hops_by_route for h in hs])
    print(f"路线 {len(hops_by_route)} 条，转移 {len(all_hops)} 个", flush=True)
    print(f"距离: p25={np.percentile(all_hops,25):.2f} p50={np.percentile(all_hops,50):.2f} "
          f"p75={np.percentile(all_hops,75):.2f} p90={np.percentile(all_hops,90):.2f} max={all_hops.max():.0f}km",
          flush=True)

    # ==== 1. dip test ====
    dip_raw_stat, dip_raw_p = dip(all_hops)
    dip_log_stat, dip_log_p = dip(np.log(all_hops + EPS))
    print(f"\n[1] Hartigan dip test: 原始距离 p={dip_raw_p:.4f} (stat={dip_raw_stat:.4f}) | "
          f"log距离 p={dip_log_p:.4f} (stat={dip_log_stat:.4f})", flush=True)

    # ==== 2. 参数 bootstrap LRT ====
    t_obs, (mu, sig), mix_params = lrt(all_hops)
    sims = bootstrap_lrt(all_hops, mu, sig, args.boot)
    p_boot = (np.sum(sims >= t_obs) + 1) / (len(sims) + 1)
    print(f"[2] LRT 观测={t_obs:.1f} | H0模拟 p95={np.percentile(sims,95):.1f} | "
          f"bootstrap p={p_boot:.4f}", flush=True)
    _, w, m1, s1, m2, s2 = mix_params
    print(f"    混合分量: w={w:.2f} logN({np.exp(m1):.2f}km) + logN({np.exp(m2):.2f}km)", flush=True)

    # ==== 3. 路线级 cluster bootstrap 稳健性 ====
    rng = np.random.RandomState(7)
    p_dips, t_obs_list = [], []
    for _ in range(args.cluster):
        sel = rng.choice(len(hops_by_route), len(hops_by_route), replace=True)
        xs = np.array([h for i in sel for h in hops_by_route[i]])
        p_dips.append(dip(np.log(xs + EPS))[1])
        try:
            t, (mu_, sig_), _ = lrt(xs)
            t_obs_list.append((t, mu_, sig_))
        except Exception:
            pass
    p_dips = np.array(p_dips)
    print(f"[3] cluster bootstrap({args.cluster}): log-dip p<0.05 的比例 = "
          f"{np.mean(p_dips < 0.05):.2f} | p 中位 = {np.median(p_dips):.3f}", flush=True)

    # LRT 的 cluster 稳健 p（每次用其自身 H0 的少量模拟，B=50 省时）
    robust_ps = []
    for t, mu_, sig_ in t_obs_list[:: max(1, len(t_obs_list) // 20)]:
        s_ = bootstrap_lrt(all_hops, mu_, sig_, B=50, seed=2)
        robust_ps.append((np.sum(s_ >= t) + 1) / (len(s_) + 1))
    print(f"    LRT cluster 稳健 p(抽样20次): 中位={np.median(robust_ps):.3f} "
          f"<0.05比例={np.mean(np.array(robust_ps) < 0.05):.2f}", flush=True)

    # ==== 4. 阈值敏感性 ====
    thresholds = np.arange(0.5, 10.5, 0.5)
    share_far = {round(float(t), 1): round(float(np.mean(all_hops >= t)), 3) for t in thresholds}
    print(f"[4] 跨区占比随阈值: 1km={share_far[1.0]:.2f} 2km={share_far[2.0]:.2f} "
          f"3km={share_far[3.0]:.2f} 5km={share_far[5.0]:.2f}", flush=True)

    # ==== 裁决 ====
    bimodal_evidence = (dip_log_p < 0.05 or p_boot < 0.05) and np.median(p_dips) < 0.05
    verdict = "bimodal" if bimodal_evidence else "unimodal_heavy_tail"
    print(f"\n=== 裁决: {verdict} ===", flush=True)

    out = {
        "n_routes": len(hops_by_route), "n_transitions": int(len(all_hops)),
        "dip_test": {"raw_p": round(float(dip_raw_p), 4),
                     "log_p": round(float(dip_log_p), 4),
                     "raw_stat": round(float(dip_raw_stat), 4),
                     "log_stat": round(float(dip_log_stat), 4)},
        "bootstrap_lrt": {"lrt_obs": round(float(t_obs), 2), "B": args.boot,
                          "p_value": round(float(p_boot), 4),
                          "mixture": {"w": round(float(w), 3),
                                      "comp1_median_km": round(float(np.exp(m1)), 2),
                                      "comp2_median_km": round(float(np.exp(m2)), 2)}},
        "cluster_bootstrap": {"n": args.cluster,
                              "dip_p_lt005_share": round(float(np.mean(p_dips < 0.05)), 3),
                              "lrt_robust_p_median": round(float(np.median(robust_ps)), 3)},
        "threshold_sensitivity": share_far,
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"保存至: {args.out}")


if __name__ == "__main__":
    main()
