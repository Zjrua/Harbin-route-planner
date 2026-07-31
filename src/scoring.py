"""路线打分 v4：硬约束 + 软指标分层.

v3 的问题（评估结论）：
- 软打分可 trade-off：random 靠 rhythm/diversity 补上距离短板，得分虚高
- 时间约束失效：只算交通时间，10 站路线 400min < 720min 预算，feasibility 全 1.0
- 无"结构合理性"判断：连续 5 个住宿的路线不会被判负

v4 核心思想：硬约束 + 软指标分层。
- 硬约束（不满足 → 直接判负，不可 trade-off）：
  1. 时间超预算（含每站停留时间）
  2. 路线内 POI 重复
  3. 路线过短
- 软指标（通过硬约束后才计分）：
  就近性 / 区域密度 / 节奏 / 满意度 / 多样性

时间模型修正：总耗时 = 交通时间 + 每站停留时间
（v3 只算交通，导致时间约束永远不触发）

返回结构：{"score": float, "feasible": bool, "reason": str|None, **分量}
"""

import numpy as np
from typing import List, Dict, Optional

# 一天旅游时间预算（分钟）
DAY_TIME_BUDGET_MIN = 720  # 12 小时

# 每类 POI 的停留时间（分钟）——活动类型编码：0景点 1餐饮 2住宿 3交通 4购物 5出发点
# 经敏感性分析校准：景点 45min（10站路线 5-6 个景点 ≈ 5h，合理）
STAY_TIME_MIN = {0: 45, 1: 60, 2: 0, 3: 0, 4: 40, 5: 0}

# 硬约束：路线最小有效长度
MIN_ROUTE_LEN = 3


def compute_route_metrics(route: List[int], dist_matrix: np.ndarray,
                          time_matrix: np.ndarray, ratings: np.ndarray,
                          categories: np.ndarray,
                          activity_types: Optional[np.ndarray] = None) -> Dict[str, float]:
    """计算一条路线的全部原始指标.

    时间模型：总耗时 = 交通时间 + 每站停留时间（activity_types 提供时）。
    """
    n = len(route)
    if n < 2:
        return {}

    hops = [dist_matrix[route[i]][route[i + 1]] for i in range(n - 1)]
    hop_times = [time_matrix[route[i]][route[i + 1]] for i in range(n - 1)]

    total_dist = sum(hops)
    travel_time = sum(hop_times)

    # 停留时间（按活动类型）
    stay_time = 0.0
    if activity_types is not None:
        for i in route:
            if i < len(activity_types):
                atype = int(activity_types[i])
                stay_time += STAY_TIME_MIN.get(atype, 30)
    total_time = travel_time + stay_time

    # 跳转 p50 / p90
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

    # 重复 POI 检测
    has_repeat = len(set(route)) != len(route)

    return {
        "total_dist_km": total_dist,
        "travel_time_min": travel_time,
        "stay_time_min": stay_time,
        "total_time_min": total_time,
        "hop_p50_km": hop_p50,
        "hop_p90_km": hop_p90,
        "same_type_rate": same_type_rate,
        "satisfaction": satisfaction,
        "diversity": diversity,
        "n_pois": n,
        "has_repeat": has_repeat,
    }


def check_hard_constraints(metrics: Dict[str, float], n_days: int = 1) -> Optional[str]:
    """硬约束检查。返回 None=通过，否则返回违反的约束名."""
    # 1. 时间超预算
    budget = DAY_TIME_BUDGET_MIN * n_days
    if metrics["total_time_min"] > budget:
        return "time_over_budget"

    # 2. POI 重复
    if metrics.get("has_repeat", False):
        return "repeat_poi"

    # 3. 路线过短
    if metrics["n_pois"] < MIN_ROUTE_LEN:
        return "too_short"

    return None


def score_proximity(metrics: Dict[str, float]) -> float:
    """就近性：每步跳转 p50 是否在合理区间."""
    p50 = metrics.get("hop_p50_km", 0)
    if p50 <= 1.0:
        return 0.9
    if p50 <= 8.0:
        return 1.0 - (p50 - 1.0) / 7.0 * 0.2
    return max(0.2, 1.0 - (p50 - 8.0) / 20.0)


def score_area_density(metrics: Dict[str, float]) -> float:
    """区域密度：短跳占比高则好（用 p90/p50 比值近似分布离散度）."""
    p50 = max(metrics.get("hop_p50_km", 0), 0.1)
    p90 = metrics.get("hop_p90_km", 0)
    ratio = p90 / p50 if p50 > 0 else 10
    if ratio <= 15:
        return 1.0 - ratio / 50.0
    return max(0.3, 1.0 - (ratio - 15) / 40.0)


def composite_score_v3(route: List[int], dist_matrix: np.ndarray,
                       time_matrix: np.ndarray, ratings: np.ndarray,
                       categories: np.ndarray, n_days: int = 1,
                       activity_types: Optional[np.ndarray] = None,
                       weights: Dict[str, float] = None) -> Dict[str, float]:
    """路线综合打分 v4（硬约束 + 软指标）.

    Args:
        route: POI 索引列表
        dist_matrix/time_matrix/ratings/categories: 基础数据
        n_days: 天数（时间预算 = 720 × n_days）
        activity_types: [n_pois] 活动类型（计算停留时间用，可选）

    Returns:
        {"score": float, "feasible": bool, "reason": str|None, **分量}
        score ∈ [0,1]；不可行路线 score=0, feasible=False, reason=约束名
    """
    if weights is None:
        weights = {
            "proximity": 0.25,     # 就近性
            "area_density": 0.20,  # 区域密度
            "rhythm": 0.20,        # 活动节奏
            "satisfaction": 0.20,  # 满意度
            "diversity": 0.15,     # 多样性
        }

    metrics = compute_route_metrics(route, dist_matrix, time_matrix,
                                    ratings, categories, activity_types)
    if not metrics:
        return {"score": 0.0, "feasible": False, "reason": "invalid_route"}

    # === 硬约束检查 ===
    reason = check_hard_constraints(metrics, n_days)
    if reason is not None:
        return {
            "score": 0.0,
            "feasible": False,
            "reason": reason,
            "metrics": {k: round(v, 2) for k, v in metrics.items()},
        }

    # === 软指标（通过硬约束后） ===
    proximity = score_proximity(metrics)
    area_density = score_area_density(metrics)
    rhythm = 1.0 - metrics["same_type_rate"]
    satisfaction = min(metrics["satisfaction"] / 5.0, 1.0)
    diversity = metrics["diversity"]

    components = {
        "proximity": proximity,
        "area_density": area_density,
        "rhythm": rhythm,
        "satisfaction": satisfaction,
        "diversity": diversity,
    }

    score = sum(weights[k] * components[k] for k in weights)
    components["score"] = round(score, 4)
    components["feasible"] = True
    components["reason"] = None
    components["metrics"] = {k: round(v, 2) for k, v in metrics.items()}
    return components


# 兼容别名：v4 就是当前实现（函数名保持 v3 以便不破坏调用方，但语义已升级）
composite_score_v4 = composite_score_v3
