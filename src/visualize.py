"""路线可视化模块.

使用 folium 生成交互式地图可视化，使用 matplotlib 生成静态统计图表。
"""

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Optional


def plot_route_on_map(pois: pd.DataFrame, route: List[int],
                      output_path: str = "route_map.html",
                      center_lat: float = 45.80,
                      center_lng: float = 126.53) -> folium.Map:
    """在地图上绘制旅游路线.

    Args:
        pois: POI 数据 DataFrame，需含 lat, lng, name 列
        route: POI 索引列表
        output_path: HTML 输出路径
        center_lat: 地图中心纬度
        center_lng: 地图中心经度

    Returns:
        folium.Map 对象
    """
    raise NotImplementedError("需实现：folium 地图 + 路线折线 + POI 标记")


def plot_training_curves(log_data: dict, output_path: str = "training_curves.png") -> None:
    """绘制训练曲线（loss / 指标 vs epoch）.

    Args:
        log_data: 训练日志数据字典
        output_path: 图片输出路径
    """
    raise NotImplementedError("需实现：matplotlib 绘制训练/验证 loss 曲线")


def plot_route_comparison(routes: List[List[int]], pois: pd.DataFrame,
                          labels: List[str],
                          output_path: str = "route_comparison.png") -> None:
    """对比不同方法生成的路线.

    Args:
        routes: 多条路线列表
        pois: POI 数据
        labels: 路线标签
        output_path: 输出路径
    """
    raise NotImplementedError("需实现：多条路线在同一地图上的对比可视化")


def plot_ablation_results(results: dict, output_path: str = "ablation_results.png") -> None:
    """绘制消融实验结果柱状图.

    Args:
        results: 消融实验结果字典 {"实验名": {"指标": 值}}
        output_path: 输出路径
    """
    raise NotImplementedError("需实现：分组柱状图展示消融实验对比")
