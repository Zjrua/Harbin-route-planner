"""Google OR-Tools VRP 求解器基线.

用 OR-Tools 的 RoutingIndexManager + pywrapcp 求解单车辆路线问题
（实质是带固定访问数的 TSP）。

设计：
- 单"车辆"（一条路线）
- 起点 = start_poi（depot）
- 访问数 = length（通过伪节点控制）
- 距离用整数（OR-Tools 要求整数代价）

注：OR-Tools 最小化总距离，不考虑评分/多样性/活动类型约束。
它代表"纯粹的距离最优"上界——如果 Transformer 连距离都比不过 OR-Tools，
说明 seq2seq 范式在该任务上的空间结构学习失败。
"""

import numpy as np

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False


def has_ortools() -> bool:
    """OR-Tools 是否可用（未安装时降级为 None）。"""
    return _HAS_ORTOOLS


def ortools_route(start_poi: int, dist_matrix: np.ndarray,
                  length: int = 10,
                  time_limit_sec: float = 5.0,
                  candidate_mask: np.ndarray = None) -> list[int]:
    """用 OR-Tools 求解近似最优路线（最小化总距离）.

    Args:
        start_poi: 起点 POI 索引（depot）
        dist_matrix: [n_pois, n_pois] 距离矩阵（km，float）
        length: 路线长度（含起点）
        time_limit_sec: 求解时间上限
        candidate_mask: [n_pois] bool，True 表示该 POI 可选（None=全部可选）

    Returns:
        route: list[int]，POI 索引序列，route[0] == start_poi
    """
    if not _HAS_ORTOOLS:
        raise ImportError("OR-Tools 未安装，无法运行 ortools_route。"
                          "安装：pip install ortools")

    n_pois = dist_matrix.shape[0]

    # 候选集：起点 + 可选 POI。OR-Tools 需要把候选子集映射到连续索引。
    if candidate_mask is None:
        candidate_mask = np.ones(n_pois, dtype=bool)
    candidate_mask = candidate_mask.copy()
    candidate_mask[start_poi] = True  # 起点必须在候选集

    # 候选 POI 列表（起点放第一个）
    candidates = [start_poi] + [i for i in range(n_pois)
                                if candidate_mask[i] and i != start_poi]
    n_candidates = len(candidates)
    poi_to_idx = {poi: idx for idx, poi in enumerate(candidates)}

    # 如果候选数不足 length，全用上
    visit_count = min(length, n_candidates)

    # OR-Tools 要求整数距离。km → 米（×1000），避免精度丢失。
    # 子距离矩阵：只包含候选 POI
    sub_dist = np.zeros((n_candidates, n_candidates), dtype=np.int64)
    for i, pi in enumerate(candidates):
        for j, pj in enumerate(candidates):
            if i == j:
                continue
            d = dist_matrix[pi, pj]
            sub_dist[i][j] = max(1, int(d * 1000))  # 至少1米，避免0代价循环

    # 管理：1辆车，depot = poi_to_idx[start_poi]
    manager = pywrapcp.RoutingIndexManager(n_candidates, 1, poi_to_idx[start_poi])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return sub_dist[from_node][to_node]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # 约束：恰好访问 visit_count 个节点（含 depot）
    # 用 "Count" dimension 累计访问数
    count_callback = lambda from_idx, to_idx: 1
    count_idx = routing.RegisterTransitCallback(count_callback)
    routing.AddDimension(count_idx, 0, visit_count, True, "Count")
    count_dim = routing.GetDimensionOrDie("Count")
    for node in range(n_candidates):
        idx = manager.NodeToIndex(node)
        if node != poi_to_idx[start_poi]:
            count_dim.CumulVar(idx).SetRange(1, visit_count)

    # 搜索参数
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = int(time_limit_sec)
    params.log_search = False

    solution = routing.SolveWithParameters(params)
    if solution is None:
        # 求解失败，退化为最近邻
        from .heuristics import nearest_neighbor_route
        return nearest_neighbor_route(start_poi, dist_matrix, length, candidate_mask)

    # 提取路线
    idx = routing.Start(0)
    route_nodes = []
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        route_nodes.append(candidates[node])
        idx = solution.Value(routing.NextVar(idx))

    # route_nodes 包含起点，可能含巡回回到起点的最后一段（IsEnd）
    # 去掉末尾如果重复了起点（OR-Tools 默认回到 depot，旅游路线不需要）
    result = route_nodes
    if len(result) > 1 and result[-1] == start_poi:
        result = result[:-1]

    # 截断到 length
    return result[:length] if len(result) > length else result
