"""简单启发式基线：随机、最近邻、2-opt.

这些方法从零生成路线（不依赖训练数据），作为 Transformer 的对比基线。
与 src/inference.py 的 optimize_route_order（后处理优化）不同，这里是从起点开始构建完整路线。
"""

import numpy as np


def random_route(start_poi: int, n_pois: int, length: int = 10,
                 rng: np.random.Generator = None) -> list[int]:
    """随机基线：从起点出发，随机选 POI（不重复）.

    作为下界基线，衡量"完全不学习"的路线质量。
    """
    if rng is None:
        rng = np.random.default_rng()
    candidates = [i for i in range(n_pois) if i != start_poi]
    chosen = rng.choice(candidates, size=min(length - 1, len(candidates)), replace=False)
    return [start_poi] + chosen.tolist()


def nearest_neighbor_route(start_poi: int, dist_matrix: np.ndarray,
                           length: int = 10,
                           candidate_mask: np.ndarray = None) -> list[int]:
    """贪心最近邻：每步选距离当前 POI 最近的未访问 POI.

    经典 TSP 启发式。不考虑活动类型/评分，纯粹最小化距离。
    """
    n_pois = dist_matrix.shape[0]
    visited = np.zeros(n_pois, dtype=bool)
    route = [start_poi]
    visited[start_poi] = True
    current = start_poi

    for _ in range(length - 1):
        # 距离当前点最近的未访问 POI
        dists = dist_matrix[current].copy()
        if candidate_mask is not None:
            dists[~candidate_mask] = np.inf
        dists[visited] = np.inf
        dists[current] = np.inf
        nxt = int(np.argmin(dists))
        if not np.isfinite(dists[nxt]):
            break  # 没有可达的 POI 了
        route.append(nxt)
        visited[nxt] = True
        current = nxt

    return route


def two_opt_route(start_poi: int, dist_matrix: np.ndarray,
                  length: int = 10,
                  iterations: int = 100,
                  candidate_mask: np.ndarray = None) -> list[int]:
    """NN 初始化 + 2-opt 精修.

    先用最近邻生成初始路线，再用 2-opt 消除交叉边。
    固定起点（start_poi 始终在 route[0]）。
    """
    init = nearest_neighbor_route(start_poi, dist_matrix, length, candidate_mask)
    if len(init) <= 3:
        return init

    def route_cost(r):
        return sum(dist_matrix[r[i], r[i + 1]] for i in range(len(r) - 1))

    best = init[:]
    best_cost = route_cost(best)

    for _ in range(iterations):
        improved = False
        # 固定起点 route[0]，对 [1:] 做 2-opt
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                new_cost = route_cost(new_route)
                if new_cost < best_cost:
                    best = new_route
                    best_cost = new_cost
                    improved = True
        if not improved:
            break

    return best
