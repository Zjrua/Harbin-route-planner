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
                      center_lng: float = 126.53,
                      title: str = "旅游路线") -> folium.Map:
    """在地图上绘制旅游路线，带序号标记和方向箭头.

    Args:
        pois: POI 数据 DataFrame，需含 lat, lng, name 列
        route: POI 索引列表
        output_path: HTML 输出路径
        center_lat: 地图中心纬度
        center_lng: 地图中心经度
        title: 地图标题

    Returns:
        folium.Map 对象
    """
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)

    # 添加标题
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: white; padding: 10px 20px;
                border-radius: 5px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                font-size: 16px; font-weight: bold;">
        {title}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # 绘制路线上的 POI 标记和连线
    route_coords = []
    for i, idx in enumerate(route):
        row = pois.iloc[idx]
        lat, lng = row["lat"], row["lng"]
        name = row["name"] if "name" in pois.columns else f"POI-{idx}"
        category = row.get("category", "")
        route_coords.append([lat, lng])

        # 起点用绿色，终点用红色，中间用蓝色
        if i == 0:
            color = "green"
            icon_text = "起"
        elif i == len(route) - 1:
            color = "red"
            icon_text = "终"
        else:
            color = "blue"
            icon_text = str(i + 1)

        # 使用带序号的图标
        folium.Marker(
            [lat, lng],
            popup=f"<b>第{i+1}站</b><br>{name}<br>类别: {category}",
            icon=folium.DivIcon(
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                html=f'<div style="background: {color}; color: white; '
                     f'border-radius: 50%; width: 24px; height: 24px; '
                     f'display: flex; align-items: center; justify-content: center; '
                     f'font-weight: bold; font-size: 12px; '
                     f'box-shadow: 0 2px 4px rgba(0,0,0,0.3);">'
                     f'{icon_text}</div>'
            ),
        ).add_to(m)

        # 添加名称标签（带背景）
        folium.Marker(
            [lat + 0.001, lng],  # 稍微偏移避免重叠
            icon=folium.DivIcon(
                icon_size=(150, 20),
                icon_anchor=(75, 0),
                html=f'<div style="background: white; padding: 2px 6px; '
                     f'border-radius: 3px; font-size: 11px; '
                     f'box-shadow: 0 1px 3px rgba(0,0,0,0.2); '
                     f'white-space: nowrap; overflow: hidden; '
                     f'text-overflow: ellipsis; max-width: 150px;">'
                     f'{name}</div>'
            ),
        ).add_to(m)

    # 绘制路线折线（带方向箭头）
    if len(route_coords) >= 2:
        # 主路线
        folium.PolyLine(
            route_coords,
            color="red",
            weight=4,
            opacity=0.8,
            dash_array="10"  # 虚线表示方向
        ).add_to(m)

        # 添加方向箭头（每隔几个点添加一个箭头）
        for i in range(0, len(route_coords) - 1, max(1, len(route_coords) // 5)):
            start = route_coords[i]
            end = route_coords[i + 1]
            # 计算中点
            mid_lat = (start[0] + end[0]) / 2
            mid_lng = (start[1] + end[1]) / 2
            # 添加箭头图标
            folium.Marker(
                [mid_lat, mid_lng],
                icon=folium.DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                    html='<div style="color: red; font-size: 16px; font-weight: bold;">→</div>'
                ),
            ).add_to(m)

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
                          output_path: str = "route_comparison.html",
                          title: str = "路线对比") -> None:
    """对比不同方法生成的路线，带序号标记.

    Args:
        routes: 多条路线列表
        pois: POI 数据
        labels: 路线标签
        output_path: 输出路径
        title: 地图标题
    """
    m = folium.Map(location=[pois["lat"].mean(), pois["lng"].mean()], zoom_start=12)
    colors = ["red", "blue", "green", "purple", "orange"]

    # 添加标题
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: white; padding: 10px 20px;
                border-radius: 5px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                font-size: 16px; font-weight: bold;">
        {title}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # 添加图例
    legend_html = '''
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000;
                background: white; padding: 15px; border-radius: 5px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 13px;">
        <b>路线图例</b><br>
    '''
    for i, label in enumerate(labels):
        color = colors[i % len(colors)]
        legend_html += f'<span style="color: {color};">● {label}</span><br>'
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    for i, (route, label) in enumerate(zip(routes, labels)):
        color = colors[i % len(colors)]
        coords = []

        for j, idx in enumerate(route):
            row = pois.iloc[idx]
            lat, lng = row["lat"], row["lng"]
            coords.append([lat, lng])
            pname = row["name"] if "name" in pois.columns else f"POI-{idx}"

            # 起点用特殊标记
            if j == 0:
                folium.Marker(
                    [lat, lng],
                    popup=f"<b>{label} 起点</b><br>{pname}",
                    icon=folium.DivIcon(
                        icon_size=(30, 30),
                        icon_anchor=(15, 15),
                        html=f'<div style="background: {color}; color: white; '
                             f'border-radius: 50%; width: 24px; height: 24px; '
                             f'display: flex; align-items: center; justify-content: center; '
                             f'font-weight: bold; font-size: 12px;">起</div>'
                    ),
                ).add_to(m)
            else:
                folium.Marker(
                    [lat, lng],
                    popup=f"{label} 第{j+1}站: {pname}",
                    icon=folium.DivIcon(
                        icon_size=(20, 20),
                        icon_anchor=(10, 10),
                        html=f'<div style="background: {color}; color: white; '
                             f'border-radius: 50%; width: 16px; height: 16px; '
                             f'display: flex; align-items: center; justify-content: center; '
                             f'font-size: 10px;">{j+1}</div>'
                    ),
                ).add_to(m)

        if len(coords) >= 2:
            folium.PolyLine(
                coords,
                color=color,
                weight=3,
                opacity=0.7,
                popup=label,
                dash_array="5"
            ).add_to(m)

    m.save(output_path)


def get_walking_route(start_lat: float, start_lng: float,
                      end_lat: float, end_lng: float,
                      api_key: str = None) -> List[List[float]]:
    """获取步行路线（沿实际道路）.

    使用高德地图步行路线API获取实际道路路径。
    如果没有API key，则返回直线路径。
    """
    if api_key is None:
        # 没有API key，返回直线路径
        return [[start_lat, start_lng], [end_lat, end_lng]]

    import requests
    url = "https://restapi.amap.com/v3/direction/walking"
    params = {
        "key": api_key,
        "origin": f"{start_lng},{start_lat}",
        "destination": f"{end_lng},{end_lat}",
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if data.get("status") == "1" and data.get("route"):
            path = data["route"]["paths"][0]["steps"]
            coords = []
            for step in path:
                # 解析polyline
                polyline = step.get("polyline", "")
                points = polyline.split(";")
                for point in points:
                    if point:
                        lng, lat = point.split(",")
                        coords.append([float(lat), float(lng)])
            return coords if coords else [[start_lat, start_lng], [end_lat, end_lng]]
    except Exception:
        pass

    return [[start_lat, start_lng], [end_lat, end_lng]]


def plot_route_on_map_with_roads(pois: pd.DataFrame, route: List[int],
                                  output_path: str = "route_map.html",
                                  center_lat: float = 45.80,
                                  center_lng: float = 126.53,
                                  title: str = "旅游路线",
                                  amap_key: str = None) -> folium.Map:
    """在地图上绘制旅游路线，沿实际道路.

    Args:
        pois: POI 数据 DataFrame
        route: POI 索引列表
        output_path: HTML 输出路径
        center_lat: 地图中心纬度
        center_lng: 地图中心经度
        title: 地图标题
        amap_key: 高德地图API key（可选）

    Returns:
        folium.Map 对象
    """
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)

    # 添加标题
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: white; padding: 10px 20px;
                border-radius: 5px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                font-size: 16px; font-weight: bold;">
        {title}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # 绘制路线上的 POI 标记
    for i, idx in enumerate(route):
        row = pois.iloc[idx]
        lat, lng = row["lat"], row["lng"]
        name = row["name"] if "name" in pois.columns else f"POI-{idx}"
        category = row.get("category", "")

        # 起点用绿色，终点用红色，中间用蓝色
        if i == 0:
            color = "green"
            icon_text = "起"
        elif i == len(route) - 1:
            color = "red"
            icon_text = "终"
        else:
            color = "blue"
            icon_text = str(i + 1)

        # 使用带序号的图标
        folium.Marker(
            [lat, lng],
            popup=f"<b>第{i+1}站</b><br>{name}<br>类别: {category}",
            icon=folium.DivIcon(
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                html=f'<div style="background: {color}; color: white; '
                     f'border-radius: 50%; width: 24px; height: 24px; '
                     f'display: flex; align-items: center; justify-content: center; '
                     f'font-weight: bold; font-size: 12px; '
                     f'box-shadow: 0 2px 4px rgba(0,0,0,0.3);">'
                     f'{icon_text}</div>'
            ),
        ).add_to(m)

        # 添加名称标签
        folium.Marker(
            [lat + 0.001, lng],
            icon=folium.DivIcon(
                icon_size=(150, 20),
                icon_anchor=(75, 0),
                html=f'<div style="background: white; padding: 2px 6px; '
                     f'border-radius: 3px; font-size: 11px; '
                     f'box-shadow: 0 1px 3px rgba(0,0,0,0.2); '
                     f'white-space: nowrap; overflow: hidden; '
                     f'text-overflow: ellipsis; max-width: 150px;">'
                     f'{name}</div>'
            ),
        ).add_to(m)

    # 绘制路线（沿实际道路）
    for i in range(len(route) - 1):
        start_idx = route[i]
        end_idx = route[i + 1]
        start_row = pois.iloc[start_idx]
        end_row = pois.iloc[end_idx]

        # 获取实际道路路径
        road_coords = get_walking_route(
            start_row["lat"], start_row["lng"],
            end_row["lat"], end_row["lng"],
            api_key=amap_key
        )

        # 绘制路线
        folium.PolyLine(
            road_coords,
            color="red",
            weight=4,
            opacity=0.8,
        ).add_to(m)

        # 在路线中点添加方向箭头
        if len(road_coords) >= 2:
            mid_idx = len(road_coords) // 2
            mid_lat, mid_lng = road_coords[mid_idx]
            folium.Marker(
                [mid_lat, mid_lng],
                icon=folium.DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                    html='<div style="color: red; font-size: 16px; font-weight: bold;">→</div>'
                ),
            ).add_to(m)

    m.save(output_path)
    return m


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
