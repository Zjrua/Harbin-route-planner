"""路网数据构建模块.

基于 POI 坐标构建路网邻接矩阵，包含距离、通行时间等信息。
"""

import numpy as np
import pandas as pd
from typing import Optional


def build_adjacency_matrix(pois: pd.DataFrame,
                           max_distance_km: float = 50.0,
                           distance_metric: str = "haversine") -> np.ndarray:
    """构建 POI 间邻接矩阵.

    Args:
        pois: POI 数据 DataFrame，需含 lat, lng 列
        max_distance_km: 最大连接距离阈值
        distance_metric: 距离度量，"haversine" 或 "euclidean"

    Returns:
        adjacency: [n_pois, n_pois] 的邻接矩阵，值为距离（km）
    """
    raise NotImplementedError("需实现：计算 POI 间距离矩阵")


def build_travel_time_matrix(adjacency: np.ndarray,
                             avg_speed_kmh: float = 30.0) -> np.ndarray:
    """基于距离矩阵估算通行时间矩阵.

    Args:
        adjacency: 距离矩阵 [n_pois, n_pois]
        avg_speed_kmh: 平均行驶速度 (km/h)

    Returns:
        time_matrix: [n_pois, n_pois] 时间矩阵（分钟）
    """
    raise NotImplementedError("需实现：距离 / 速度 -> 时间矩阵")
