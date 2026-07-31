"""路线打分 v3：时间可行性 + 就近性 + 区域密度 + 节奏.

v1/v2 的问题（诊断结论）：
- v1: distance/time 权重过高且共线，奖励"同区域反复横跳"
- v2: 加了 rhythm，但仍无"时间可行性"硬约束、无"就近性"标准

v3 设计（基于真实 XHS 路线分布校准）：
- 时间可行性（硬约束）：总耗时（交通+游览）超过一天预算 → 强惩罚
- 就近性：每步跳转 p50 应在合理区间（真实路线 1.9km，容忍到 6km），极端长跳重罚
- 区域密度：奖励在一区域内密集游览（连续短跳），惩罚跨区域乱跳
- 节奏：惩罚连续同类（继承 v2）
- 满意度/多样性：保留

诊断数据（真实 vs 模型生成）：
真实: 跳转p50=1.9km, p90=28km, 总耗时137min, 连续同类29.4%
生成: 跳转p50=7.1km, p90=117km, 总耗时346min, 连续同类2.2%
"""

import numpy as np
from typing import List, Dict

# 一天旅游时间预算（分钟）：含交通+游览+餐饮休息
DAY_TIME_BUDGET_MIN = 720  # 12 小时


def compute_route_metrics(route: List[int], dist_matrix: np.ndarray,
                          time_matrix: np.ndarray, ratings: np.ndarray,
                          categories: np.ndarray) -> Dict[str, float]:
    """计算一条路线的全部原始指标."""
    n = len(route)
    if n < 2:
        return {}

    # 每步跳转距离
    hops = [dist_matrix[route[i]][route[i + 1]] for i in range(n - 1)]
    hop_times = [time_matrix[route[i]][route[i + 1]] for i in range(n - 1)]

    total_dist = sum(hops)
    total_time = sum(hop_times)

    # 每步跳转 p50 / p90
    hops_sorted = sorted(hops)
    hop_p50 = hops_sorted[len(hops_sorted) // 2]
    hop_p90 = hops_sorted[int(len(hops_sorted) * 0.9)] if len(hops_sorted) > 1 else hops_sorted[-1]

    # 连续同类比例
    cats = [categories[i] for i in route if i < len(categories)]
    if len(cats) > 1:
        same_type = sum(1 for i in range(1, len(cats)) if cats[i] == cats[i - 1])
        same_type_rate = same_type / (len(cats) - 1)
    else:
        same_type_rate = 0.0

    # 满意度（评分均值）
    valid_ratings = [ratings[i] for i in route if i < len(ratings) and ratings[i] > 0]
    satisfaction = float(np.mean(valid_ratings)) if valid_ratings else 0.0

    # 多样性（unique 类别比例）
    unique_cats = len(set(cats))
    diversity = unique_cats / max(len(cats), 1)

    return {
        "total_dist_km": total_dist,
        "total_time_min": total_time,
        "hop_p50_km": hop_p50,
        "hop_p90_km": hop_p90,
        "same_type_rate": same_type_rate,
        "satisfaction": satisfaction,
        "diversity": diversity,
        "n_pois": n,
    }


def score_feasibility(metrics: Dict[str, float], n_days: int = 1) -> float:
    """时间可行性得分：总耗时是否在预算内.

    超过预算 → 指数衰减（硬约束）。
    """
    budget = DAY_TIME_BUDGET_MIN * n_days
    total = metrics.get("total_time_min", 0)
    if total <= budget:
        return 1.0
    # 超预算惩罚：线性降到 0.3（不是全零，允许轻微超时）
    over = total / budget
    return max(0.3, 1.0 - (over - 1.0) * 0.7)


def score_proximity(metrics: Dict[str, float]) -> float:
    """就近性得分：每步跳转 p50 是否在合理区间.

    真实路线 p50=1.9km。合理区间 [1, 8]km，超出则衰减。
    """
    p50 = metrics.get("hop_p50_km", 0)
    if p50 <= 1.0:
        return 0.9  # 太近也可能意味着都在一个点（但通常合理）
    if p50 <= 8.0:
        return 1.0 - (p50 - 1.0) / 7.0 * 0.2  # 1-8km 之间轻微递减
    # >8km 显著惩罚
    return max(0.2, 1.0 - (p50 - 8.0) / 20.0)


def score_area_density(metrics: Dict[str, float]) -> float:
    """区域密度得分：短跳占比高则好.

    衡量"是否在一区域内密集游览"：
    - 跳转 ≤5km 的比例（真实路线多数跳转很短）
    """
    # 用 hop_p90 与 hop_p50 的比值近似：比值大 = 分布离散 = 乱跳
    p50 = max(metrics.get("hop_p50_km", 0), 0.1)
    p90 = metrics.get("hop_p90_km", 0)
    ratio = p90 / p50 if p50 > 0 else 10
    # 理想 3-8 倍（少数跨区 + 多数就近），>15 倍说明乱跳
    if ratio <= 15:
        return 1.0 - ratio / 50.0
    return max(0.3, 1.0 - (ratio - 15) / 40.0)


def composite_score_v3(route: List[int], dist_matrix: np.ndarray,
                       time_matrix: np.ndarray, ratings: np.ndarray,
                       categories: np.ndarray, n_days: int = 1,
                       weights: Dict[str, float] = None) -> Dict[str, float]:
    """路线综合打分 v3.

    Returns:
        {"score": 综合得分, **各分量得分, **原始指标}
    """
    if weights is None:
        weights = {
            "feasibility": 0.25,   # 时间可行性（硬约束）
            "proximity": 0.20,     # 就近性
            "area_density": 0.15,  # 区域密度
            "rhythm": 0.15,        # 活动节奏
            "satisfaction": 0.15,  # 满意度
            "diversity": 0.10,     # 多样性
        }

    metrics = compute_route_metrics(route, dist_matrix, time_matrix, ratings, categories)
    if not metrics:
        return {"score": 0.0}

    # 各分量得分
    feasibility = score_feasibility(metrics, n_days)
    proximity = score_proximity(metrics)
    area_density = score_area_density(metrics)
    rhythm = 1.0 - metrics["same_type_rate"]
    satisfaction = min(metrics["satisfaction"] / 5.0, 1.0)
    diversity = metrics["diversity"]

    components = {
        "feasibility": feasibility,
        "proximity": proximity,
        "area_density": area_density,
        "rhythm": rhythm,
        "satisfaction": satisfaction,
        "diversity": diversity,
    }

    score = sum(weights[k] * components[k] for k in weights)
    components["score"] = round(score, 4)
    components["metrics"] = {k: round(v, 2) for k, v in metrics.items()}
    return components
