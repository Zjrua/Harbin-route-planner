"""启发式/运筹学基线方法，用于与 ItineraryTransformer 对比.

包含：
- random_route: 随机选 POI（下界基线）
- nearest_neighbor_route: 贪心最近邻
- two_opt_route: NN + 2-opt 精修
- ortools_route: Google OR-Tools VRP 求解器

所有方法返回 list[int]（POI 索引序列），与 Transformer 的 generate() 输出格式一致，
便于用同一套评估函数（src.evaluate）对比。
"""

from .heuristics import random_route, nearest_neighbor_route, two_opt_route
from .ortools_solver import ortools_route, has_ortools

__all__ = [
    "random_route",
    "nearest_neighbor_route",
    "two_opt_route",
    "ortools_route",
    "has_ortools",
]
