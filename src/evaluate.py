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
    raise NotImplementedError("需实现：沿路线累加相邻 POI 间距离")


def route_time(route: List[int], time_matrix: np.ndarray) -> float:
    """计算路线总通行时间.

    Args:
        route: POI 索引列表
        time_matrix: 时间矩阵 [n_pois, n_pois]

    Returns:
        路线总时间 (分钟)
    """
    raise NotImplementedError("需实现：沿路线累加通行时间")


def satisfaction_score(route: List[int], ratings: np.ndarray) -> float:
    """计算路线的游客满意度评分.

    Args:
        route: POI 索引列表
        ratings: 各 POI 评分数组 [n_pois]

    Returns:
        路线平均满意度评分
    """
    raise NotImplementedError("需实现：取路线中 POI 评分的加权平均")


def diversity_score(route: List[int], categories: np.ndarray) -> float:
    """计算路线的类别多样性.

    衡量路线中包含的 POI 类别的丰富程度。

    Args:
        route: POI 索引列表
        categories: 各 POI 类别数组 [n_pois]

    Returns:
        多样性得分（基于类别熵或 unique 比例）
    """
    raise NotImplementedError("需实现：计算路线中类别的多样性指标")


def composite_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """计算综合加权得分.

    各指标先归一化到 [0, 1]，再按权重加权求和。

    Args:
        metrics: 指标字典 {"distance": ..., "time": ..., "satisfaction": ..., "diversity": ...}
        weights: 权重字典

    Returns:
        综合得分
    """
    raise NotImplementedError("需实现：归一化 + 加权求和")


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
    raise NotImplementedError("需实现：遍历路线计算各指标，返回平均值")
