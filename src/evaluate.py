"""评估脚本：路线质量指标计算.

指标:
- route_distance: 路线总距离
- route_time: 路线总时间
- satisfaction_score: 路线满意度
- diversity_score: 路线多样性
- composite_score: 综合加权得分
"""

import numpy as np
from typing import List, Dict


def route_distance(route: List[int], dist_matrix: np.ndarray) -> float:
    """计算路线总距离.

    Args:
        route: POI 索引列表
        dist_matrix: 距离矩阵 [n_pois, n_pois]

    Returns:
        路线总距离 (km)
    """
    total = 0.0
    for i in range(len(route) - 1):
        total += dist_matrix[route[i], route[i + 1]]
    return total


def route_time(route: List[int], time_matrix: np.ndarray) -> float:
    """计算路线总通行时间.

    Args:
        route: POI 索引列表
        time_matrix: 时间矩阵 [n_pois, n_pois]

    Returns:
        路线总时间 (分钟)
    """
    total = 0.0
    for i in range(len(route) - 1):
        total += time_matrix[route[i], route[i + 1]]
    return total


def satisfaction_score(route: List[int], ratings: np.ndarray) -> float:
    """计算路线的游客满意度评分.

    Args:
        route: POI 索引列表
        ratings: 各 POI 评分数组 [n_pois]

    Returns:
        路线平均满意度评分
    """
    return float(np.mean([ratings[i] for i in route]))


def diversity_score(route: List[int], categories: np.ndarray) -> float:
    """计算路线的类别多样性.

    衡量路线中包含的 POI 类别的丰富程度。

    Args:
        route: POI 索引列表
        categories: 各 POI 类别数组 [n_pois]

    Returns:
        多样性得分（基于类别熵或 unique 比例）
    """
    route_cats = [categories[i] for i in route]
    unique = len(set(route_cats))
    total = len(route_cats)
    # 基于 unique 比例 + 熵的多样性度量
    if total == 0:
        return 0.0
    ratio = unique / total
    # 熵部分
    counts = {}
    for c in route_cats:
        counts[c] = counts.get(c, 0) + 1
    probs = [v / total for v in counts.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    return float(ratio * 0.5 + (entropy / np.log(len(probs) + 1)) * 0.5) if len(probs) > 1 else float(ratio)


def composite_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """计算综合加权得分.

    各指标先归一化到 [0, 1]，再按权重加权求和。

    Args:
        metrics: 指标字典 {"distance": ..., "time": ..., "satisfaction": ..., "diversity": ...}
        weights: 权重字典

    Returns:
        综合得分
    """
    # 各指标归一化到 [0, 1]（距离和时间越小越好，取反）
    normalized = {
        "distance": 1.0 - min(metrics.get("distance", 0) / max(metrics.get("max_distance", 1), 1e-10), 1.0),
        "time": 1.0 - min(metrics.get("time", 0) / max(metrics.get("max_time", 1), 1e-10), 1.0),
        "satisfaction": min(metrics.get("satisfaction", 0) / 5.0, 1.0),
        "diversity": min(metrics.get("diversity", 0), 1.0),
    }
    return sum(weights.get(k, 0) * v for k, v in normalized.items())


def evaluate_routes(routes: List[List[int]], dist_matrix: np.ndarray,
                    time_matrix: np.ndarray, ratings: np.ndarray,
                    categories: np.ndarray, weights: Dict[str, float]) -> Dict[str, float]:
    """批量评估多条路线.

    Args:
        routes: 路线列表
        dist_matrix: 距离矩阵
        time_matrix: 时间矩阵
        ratings: POI 评分
        categories: POI 类别
        weights: 指标权重

    Returns:
        平均指标字典
    """
    all_metrics = {"distance": [], "time": [], "satisfaction": [], "diversity": []}
    for route in routes:
        if len(route) < 2:
            continue
        all_metrics["distance"].append(route_distance(route, dist_matrix))
        all_metrics["time"].append(route_time(route, time_matrix))
        all_metrics["satisfaction"].append(satisfaction_score(route, ratings))
        all_metrics["diversity"].append(diversity_score(route, categories))

    avg = {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}
    avg["max_distance"] = float(np.max(all_metrics["distance"])) if all_metrics["distance"] else 1.0
    avg["max_time"] = float(np.max(all_metrics["time"])) if all_metrics["time"] else 1.0
    avg["composite"] = composite_score(avg, weights)
    return avg
