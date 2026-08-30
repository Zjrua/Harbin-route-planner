"""三层统计评测体系（评测标准 v1.0）：层2 路线似然 + 层3 分布距离.

设计文档: docs/superpowers/specs/2026-08-30-behavioral-eval-design.md

构念：行为符合度（behavioral fidelity）。裁决规则（预注册）：
- 层1（点层，McNemar）在 run_fusion_eval 已有，本模块不涉及
- 层2 个体似然 = 描述性（依赖行为模型设定，不进结论句）
- 层3 群体分布距离 = 群体裁决（模型无关，经验分布比较）

所有行为模型组件（二阶转移/衰减/类型转移）一律在拟合份 84 条上估计，
由调用方注入（数据纪律）。
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


@dataclass
class BehaviorModel:
    """拟合份估计的行为模型包（route_loglik 与选择器共用）."""
    region_of_poi: np.ndarray      # int[10000]
    mk: object                     # SecondOrderMarkov（二阶）
    decay: object                  # DecayModel（log_score(d)）
    T_type: np.ndarray             # float[6,6] 一阶类型转移
    dist: np.ndarray               # 10K×10K 距离矩阵
    type_of: np.ndarray = None     # poi→activity_type


# ==== 层 2：个体路线行为似然 ====
def route_loglik(route, bm: BehaviorModel):
    """路线行为对数似然（nat/转移，按长度标准化）.

    LL = Σₜ [log f(dₜ) + log P2(regionₜ | 前两步区域) + log P(typeₜ|typeₜ₋₁)]
    首转移无二步状态：区域项退化为省略（只用距离+类型）。
    返回 (nat_per_transition, n_transitions)；路线<2 站返回 (None, 0)。
    """
    idxs = [int(i) for i in route if int(i) >= 0]
    if len(idxs) < 2:
        return None, 0
    regs = [bm.region_of_poi[i] for i in idxs]
    ll = 0.0
    n = len(idxs) - 1
    for t in range(1, len(idxs)):
        a, b = idxs[t - 1], idxs[t]
        d = float(bm.dist[a][b])
        # 距离衰减密度（截断到合理范围防 log(0)）
        lf = float(bm.decay.log_score(np.array([d]))[0])
        ll += lf
        # 区域转移
        if t >= 2:
            ll += float(np.log(bm.mk.probs()[regs[t - 2], regs[t - 1],
                                              regs[t]] + 1e-12))
        # 类型转移
        ta, tb = int(bm.type_of[a]), int(bm.type_of[b])
        ll += float(np.log(bm.T_type[ta, tb] + 1e-12))
    return ll / n, n


# ==== 层 3：群体分布统计 ====
def hops_of(route, dist):
    idxs = [int(i) for i in route if int(i) >= 0]
    return [float(dist[a][b]) for a, b in zip(idxs, idxs[1:])
            if len(idxs) >= 2]


def rhythm_stats(routes, type_of):
    """类型节奏统计量集合（餐饮间隔/首餐位置/住宿位置分布 + 跨区占比）.

    routes: List[List[int]]；type_of: poi→activity_type 映射（0景点1餐饮2住宿）。
    返回 dict of 数组，供 KS 与真实路线对比。
    """
    meal_gaps, first_meal_pos, hotel_pos, far_share = [], [], [], []
    for r in routes:
        idxs = [int(i) for i in r if int(i) >= 0]
        if len(idxs) < 2:
            continue
        types = [type_of[i] for i in idxs]
        meals = [j for j, t in enumerate(types) if t == 1]
        if meals:
            first_meal_pos.append(meals[0] / max(1, len(idxs) - 1))
            meal_gaps += [meals[k + 1] - meals[k] for k in
                          range(len(meals) - 1)]
        hotels = [j for j, t in enumerate(types) if t == 2]
        hotel_pos += [h / max(1, len(idxs) - 1) for h in hotels]
    return {"meal_gap": np.array(meal_gaps, float),
            "first_meal_pos": np.array(first_meal_pos, float),
            "hotel_pos": np.array(hotel_pos, float)}


def far_share(routes, dist, thresh=2.0):
    """跨区转移占比（双峰右模的重现性指标）."""
    all_h = np.array([h for r in routes for h in hops_of(r, dist)])
    if len(all_h) == 0:
        return None
    return float(np.mean(all_h >= thresh))


def dist_report(gen_routes, real_routes, dist, B=1000, seed=0):
    """层3 主报告：转移距离分布 KS + Wasserstein（真实测试份为参照）.

    bootstrap：重采样生成路线集合（B 次），给 KS/W 的 95% CI。
    另给就近峰占比（<2km）与 p50/p90，直读双峰形态重现。
    """
    gen_hops = [hops_of(r, dist) for r in gen_routes]
    real_hops = [hops_of(r, dist) for r in real_routes]
    gen_flat = np.array([h for hs in gen_hops for h in hs])
    real_flat = np.array([h for hs in real_hops for h in hs])

    def stats(hops_flat):
        return {"ks": float(ks_2samp(hops_flat, real_flat).statistic),
                "w": float(wasserstein_distance(hops_flat, real_flat)),
                "near_share": round(float(np.mean(hops_flat < 2.0)), 3),
                "p50": round(float(np.median(hops_flat)), 2),
                "p90": round(float(np.percentile(hops_flat, 90)), 2)}

    obs = stats(gen_flat)
    # bootstrap CI（生成侧重采样，参照侧固定）
    rng = np.random.RandomState(seed)
    n_rt = len(gen_hops)
    ks_b, w_b = [], []
    for _ in range(B):
        sel = rng.randint(0, n_rt, n_rt)
        hb = np.array([h for i in sel for h in gen_hops[i]])
        if len(hb) == 0:
            continue
        ks_b.append(ks_2samp(hb, real_flat).statistic)
        w_b.append(wasserstein_distance(hb, real_flat))
    obs["ks_ci95"] = [round(float(np.percentile(ks_b, 2.5)), 4),
                      round(float(np.percentile(ks_b, 97.5)), 4)]
    obs["w_ci95"] = [round(float(np.percentile(w_b, 2.5)), 4),
                     round(float(np.percentile(w_b, 97.5)), 4)]
    obs["n_transitions"] = int(len(gen_flat))
    return obs


def rhythm_report(gen_routes, real_routes, type_of):
    """层3 节奏分项：各统计量对真实参照的 KS（无 bootstrap，描述性）."""
    g = rhythm_stats(gen_routes, type_of)
    r = rhythm_stats(real_routes, type_of)
    out = {}
    for k in ("meal_gap", "first_meal_pos", "hotel_pos"):
        if len(g[k]) >= 2 and len(r[k]) >= 2:
            out[k] = round(float(ks_2samp(g[k], r[k]).statistic), 4)
        else:
            out[k] = None
    return out


def null_reference(real_a, real_b, dist, seed=1):
    """层3 的参照基线：两个真实子集互比的 KS/W（'真实对真实'的期望水平）."""
    ha = np.array([h for r in real_a for h in hops_of(r, dist)])
    hb = np.array([h for r in real_b for h in hops_of(r, dist)])
    return {"ks": round(float(ks_2samp(ha, hb).statistic), 4),
            "w": round(float(wasserstein_distance(ha, hb)), 4)}
