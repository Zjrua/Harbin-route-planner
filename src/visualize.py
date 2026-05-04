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
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)

    # 绘制路线上的 POI 标记和连线
    route_coords = []
    for idx in route:
        row = pois.iloc[idx]
        lat, lng = row["lat"], row["lng"]
        name = row["name"] if "name" in pois.columns else f"POI-{idx}"
        route_coords.append([lat, lng])
        folium.Marker(
            [lat, lng], popup=f"{name} (#{idx})",
            icon=folium.Icon(color="blue"),
        ).add_to(m)

    # 绘制路线折线
    if len(route_coords) >= 2:
        folium.PolyLine(route_coords, color="red", weight=3, opacity=0.8).add_to(m)

    m.save(output_path)
    return m


def plot_training_curves(log_data: dict, output_path: str = "training_curves.png") -> None:
    """绘制训练曲线（loss / 指标 vs epoch）.

    Args:
        log_data: 训练日志数据字典
        output_path: 图片输出路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if "train_loss" in log_data:
        ax.plot(log_data["train_loss"], label="Train Loss")
    if "val_loss" in log_data:
        ax.plot(log_data["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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
    m = folium.Map(location=[pois["lat"].mean(), pois["lng"].mean()], zoom_start=12)
    colors = ["red", "blue", "green", "purple", "orange"]

    for i, (route, label) in enumerate(zip(routes, labels)):
        color = colors[i % len(colors)]
        coords = []
        for idx in route:
            row = pois.iloc[idx]
            lat, lng = row["lat"], row["lng"]
            coords.append([lat, lng])
            pname = row["name"] if "name" in pois.columns else f"POI-{idx}"
            folium.Marker(
                [lat, lng], popup=f"{label}: {pname}",
                icon=folium.Icon(color=color),
            ).add_to(m)
        if len(coords) >= 2:
            folium.PolyLine(coords, color=color, weight=3, opacity=0.7, popup=label).add_to(m)

    m.save(output_path)


def plot_ablation_results(results: dict, output_path: str = "ablation_results.png") -> None:
    """绘制消融实验结果柱状图.

    Args:
        results: 消融实验结果字典 {"实验名": {"指标": 值}}
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    experiment_names = list(results.keys())
    metrics_names = list(results[experiment_names[0]].keys())
    n_groups = len(experiment_names)
    n_metrics = len(metrics_names)
    x = np.arange(n_groups)
    width = 0.8 / n_metrics

    for j, metric in enumerate(metrics_names):
        values = [results[exp].get(metric, 0) for exp in experiment_names]
        ax.bar(x + j * width, values, width, label=metric)

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study Results")
    ax.set_xticks(x + width * n_metrics / 2)
    ax.set_xticklabels(experiment_names, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
