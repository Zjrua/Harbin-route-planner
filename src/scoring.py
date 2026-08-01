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


# ==== v5：质量 + 需求匹配（指令约束） ====

def infer_days_from_len(n_pois: int) -> int:
    """按去重站数推断游览天数（与评估脚本一致）."""
    if n_pois <= 10:
        return 1
    elif n_pois <= 16:
        return 2
    return 3


def composite_score_v5(route: List[int], dist_matrix: np.ndarray,
                       time_matrix: np.ndarray, ratings: np.ndarray,
                       categories: np.ndarray, n_days: int = 1,
                       activity_types: Optional[np.ndarray] = None,
                       weights: Dict[str, float] = None,
                       instruction: Optional[str] = None,
                       constraints=None,
                       use_llm: bool = False,
                       poi_names: Optional[List[str]] = None,
                       avg_costs: Optional[np.ndarray] = None,
                       season_winter: Optional[np.ndarray] = None,
                       season_summer: Optional[np.ndarray] = None) -> Dict:
    """路线综合打分 v5：v4 质量分 + 需求匹配（指令约束）.

    在 v4 基础上新增（只对"解析出的约束"生效）：
    - 硬约束（判负）：天数严重不符 / 核心景点缺失 / 出发地不符
    - 软维度（加权）：需求匹配度 requirement_match（预算/偏好/节奏/季节/天数超出）
    - 分数 = 0.70 × 质量五维加权 + 0.30 × requirement_match

    Args:
        route/dist_matrix/time_matrix/ratings/categories/n_days/activity_types: 同 v4
        instruction: 用户指令；传入时解析约束并纳入打分
        constraints: 预解析的 Constraints（不传则从 instruction 解析）
        use_llm: 约束解析是否允许 LLM 兜底（训练固定 False）
        poi_names: [n_pois] 名称列表（核心景点/出发地匹配用）
        avg_costs: [n_pois] 人均花费（预算估算用）
        season_winter/season_summer: [n_pois] 季节分（0~1）

    Returns:
        同 v4 结构 + requirement_match / requirement_breakdown。
        不传 instruction/constraints → 完全退回 v4 行为。
    """
    if weights is None:
        weights = {
            "proximity": 0.25, "area_density": 0.20, "rhythm": 0.20,
            "satisfaction": 0.20, "diversity": 0.15,
        }

    # 约束解析（可选）
    if constraints is None:
        if instruction:
            from src.constraint_parser import parse_constraints
            constraints = parse_constraints(instruction, use_llm=use_llm)
        else:
            constraints = None

    # 无约束 → 退回 v4（纯质量分）
    if constraints is None:
        return composite_score_v3(route, dist_matrix, time_matrix, ratings,
                                  categories, n_days=n_days,
                                  activity_types=activity_types, weights=weights)

    metrics = compute_route_metrics(route, dist_matrix, time_matrix,
                                    ratings, categories, activity_types)
    if not metrics:
        return {"score": 0.0, "feasible": False, "reason": "invalid_route"}

    # === 质量硬约束（v4 原有） ===
    reason = check_hard_constraints(metrics, n_days)
    if reason is not None:
        return {"score": 0.0, "feasible": False, "reason": reason,
                "metrics": {k: round(v, 2) for k, v in metrics.items()}}

    # === 需求硬约束 ===
    inferred_days = infer_days_from_len(len(route))
    # 1. 天数严重不符：推断天数 ≤ 指令天数 - 2（如 3 日游只出 1 日量）→ 硬判负
    if (constraints.days is not None
            and inferred_days <= constraints.days - 2):
        return {"score": 0.0, "feasible": False, "reason": "days_mismatch",
                "metrics": {k: round(v, 2) for k, v in metrics.items()},
                "inferred_days": inferred_days, "asked_days": constraints.days}
    # 2. 核心景点缺失（核心景点在 POI 库存在才判负）
    if constraints.core_pois and poi_names is not None:
        route_names = [poi_names[i] for i in route if i < len(poi_names)]
        hit = [cp for cp in constraints.core_pois
               if any(cp in rn or rn in cp for rn in route_names)]
        if len(hit) < len(constraints.core_pois):
            all_in_db = all(
                any(cp in pn or pn in cp for pn in poi_names)
                for cp in constraints.core_pois)
            if all_in_db:
                return {"score": 0.0, "feasible": False, "reason": "missing_core_poi",
                        "metrics": {k: round(v, 2) for k, v in metrics.items()},
                        "missing": [cp for cp in constraints.core_pois if cp not in hit]}
    # 3. 出发地不符（前两站容错；出发地 POI 在库存在才判负）
    if constraints.start is not None and poi_names is not None:
        start_names = [poi_names[i] for i in route[:2] if i < len(poi_names)]
        hit_start = any(constraints.start in rn or rn in constraints.start
                        for rn in start_names)
        if not hit_start:
            start_in_db = any(constraints.start in pn or pn in constraints.start
                              for pn in poi_names)
            if start_in_db:
                return {"score": 0.0, "feasible": False, "reason": "start_mismatch",
                        "metrics": {k: round(v, 2) for k, v in metrics.items()}}

    # === 软指标（质量五维，v4 逻辑） ===
    proximity = score_proximity(metrics)
    area_density = score_area_density(metrics)
    rhythm = 1.0 - metrics["same_type_rate"]
    satisfaction = min(metrics["satisfaction"] / 5.0, 1.0)
    diversity = metrics["diversity"]
    components = {
        "proximity": proximity, "area_density": area_density,
        "rhythm": rhythm, "satisfaction": satisfaction, "diversity": diversity,
    }
    quality_score = sum(weights[k] * components[k] for k in weights)

    # === 软扣分：需求匹配度 ===
    deductions = {}
    # 预算
    if constraints.budget_max is not None and avg_costs is not None:
        cost = sum(float(avg_costs[i]) for i in route
                   if i < len(avg_costs) and avg_costs[i] > 0)
        if cost > constraints.budget_max:
            over = (cost - constraints.budget_max) / constraints.budget_max
            deductions["budget"] = 0.6 if over > 0.5 else (0.3 if over > 0.2 else 0.1)
    # 偏好
    if constraints.preferences and activity_types is not None:
        n = len(route)
        types = [int(activity_types[i]) for i in route if i < len(activity_types)]
        food_ratio = types.count(1) / n if n else 0.0
        shop_ratio = types.count(4) / n if n else 0.0
        worst = 0.0
        for pref in constraints.preferences:
            if pref == "food" and food_ratio < 0.25:
                worst = max(worst, min(0.45, (0.25 - food_ratio) / 0.1 * 0.15))
            elif pref == "shopping" and shop_ratio < 0.25:
                worst = max(worst, min(0.45, (0.25 - shop_ratio) / 0.1 * 0.15))
        if worst:
            deductions["preference"] = worst
    # 节奏
    if constraints.pace == "slow":
        per_day = len(route) / max(n_days, 1)
        if per_day > 9:
            deductions["pace"] = 0.6
        elif per_day > 7:
            deductions["pace"] = 0.3
    elif constraints.pace == "fast" and constraints.days is not None:
        per_day = len(route) / max(n_days, 1)
        if per_day < 3:
            deductions["pace"] = 0.3
    # 季节（用 winter−summer 差值识别"冬季特色"，避免四季皆宜被误判）
    if season_winter is not None and season_summer is not None:
        diff = [float(season_winter[i]) - float(season_summer[i])
                for i in route if i < len(season_winter)]
        n_spec = sum(1 for d in diff if d > 0.3)
        if constraints.season == "winter" and n_spec == 0:
            deductions["season"] = 0.2
        elif constraints.season == "summer" and len(diff) > 0 \
                and n_spec / len(diff) > 0.5:
            deductions["season"] = 0.2
    # 天数超出（路线过长，超出用户时间）
    if constraints.days is not None and inferred_days > constraints.days:
        over_days = inferred_days - constraints.days
        deductions["days_over"] = 0.15 if over_days == 1 else 0.3
    # 天数偏短（推断天数 = 指令天数 - 1，如 3 日游出 2 日量 11-16 站）→ 软扣
    if constraints.days is not None and inferred_days == constraints.days - 1:
        deductions["days_short"] = 0.5

    req_match = max(0.0, 1.0 - sum(deductions.values()))
    score = 0.70 * quality_score + 0.30 * req_match

    components.update({
        "score": round(score, 4),
        "feasible": True,
        "reason": None,
        "metrics": {k: round(v, 2) for k, v in metrics.items()},
        "quality_score": round(quality_score, 4),
        "requirement_match": round(req_match, 4),
        "requirement_breakdown": deductions,
        "inferred_days": inferred_days,
    })
    return components
