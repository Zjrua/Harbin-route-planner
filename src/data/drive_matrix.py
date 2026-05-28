"""基于天地图驾车路径规划API计算POI间实际道路距离和时间矩阵.

替换 Haversine 直线距离为真实驾车距离，使模型更贴近实际路线规划场景。
"""

import time
import json
import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from urllib.parse import quote
from pathlib import Path
from typing import Tuple


BASE_URL = "http://api.tianditu.gov.cn/drive"


def query_drive(
    tk: str,
    orig_lon: float, orig_lat: float,
    dest_lon: float, dest_lat: float,
    style: str = "0",
    timeout: float = 15.0,
) -> Tuple[float, float]:
    """查询单条驾车路线的距离和时间.

    Args:
        tk: 天地图API密钥
        orig_lon, orig_lat: 起点经纬度
        dest_lon, dest_lat: 终点经纬度
        style: 路线类型 (0=最快, 1=最短, 2=避高速, 3=步行)
        timeout: 请求超时秒数

    Returns:
        (distance_km, duration_seconds) 距离和时间，失败返回 (None, None)
    """
    post_str = json.dumps({
        "orig": f"{orig_lon},{orig_lat}",
        "dest": f"{dest_lon},{dest_lat}",
        "style": style,
    }, ensure_ascii=False)

    url = f"{BASE_URL}?postStr={quote(post_str)}&type=search&tk={tk}"

    try:
        resp = requests.get(url, timeout=timeout)
        resp.encoding = "utf-8"
        root = ET.fromstring(resp.text)

        dist_elem = root.find("distance")
        dur_elem = root.find("duration")

        if dist_elem is not None and dist_elem.text:
            distance = float(dist_elem.text.strip())
        else:
            return None, None

        duration = float(dur_elem.text.strip()) if dur_elem is not None and dur_elem.text else None

        return distance, duration

    except Exception as e:
        return None, None


def build_drive_matrix(
    tk: str,
    metadata_path: str = "data/processed/poi_metadata.csv",
    output_dir: str = "data/processed",
    style: str = "0",
    delay: float = 0.15,
    batch_save: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算所有POI间的驾车距离和时间矩阵.

    策略：
    - 只计算上三角 (i < j)，下三角由对称性填充
    - 每对POI只请求一次（道路距离近似对称）
    - 定期保存中间结果防止中断丢失
    - 失败的对使用 Haversine 直线距离 * 1.3 作为兜底

    Args:
        tk: 天地图API密钥
        metadata_path: POI元数据CSV路径
        output_dir: 输出目录
        style: 驾车路线类型
        delay: 请求间隔秒数
        batch_save: 每N次请求保存一次中间结果

    Returns:
        (drive_distance_matrix, drive_time_matrix)
    """
    df = pd.read_csv(metadata_path)
    n = len(df)
    output_dir = Path(output_dir)

    # 尝试加载已有矩阵（断点续算）
    dist_path = output_dir / "distance_matrix.npy"
    time_path = output_dir / "time_matrix.npy"
    if dist_path.exists():
        dist_matrix = np.load(dist_path)
        time_matrix = np.load(time_path)
        print(f"加载已有矩阵，已有 {np.count_nonzero(dist_matrix)//2} 对", flush=True)
    else:
        dist_matrix = np.zeros((n, n))
        time_matrix = np.zeros((n, n))

    total_pairs = n * (n - 1) // 2
    pair_count = 0
    fail_count = 0

    print(f"计算 {n} 个POI间的驾车距离，共 {total_pairs} 对", flush=True)

    for i in range(n):
        for j in range(i + 1, n):
            # 跳过已计算的
            if dist_matrix[i, j] > 0:
                continue

            d, t = query_drive(
                tk,
                df.iloc[i]["lon"], df.iloc[i]["lat"],
                df.iloc[j]["lon"], df.iloc[j]["lat"],
                style=style,
            )

            if d is not None and d > 0:
                dist_matrix[i, j] = dist_matrix[j, i] = d
                if t is not None and t > 0:
                    time_matrix[i, j] = time_matrix[j, i] = t / 60.0  # 秒→分钟
                else:
                    time_matrix[i, j] = time_matrix[j, i] = d / 30.0 * 60.0
            else:
                fail_count += 1
                from src.data.clean_tianditu import haversine
                h = haversine(
                    df.iloc[i]["lat"], df.iloc[i]["lon"],
                    df.iloc[j]["lat"], df.iloc[j]["lon"],
                )
                dist_matrix[i, j] = dist_matrix[j, i] = h * 1.3
                time_matrix[i, j] = time_matrix[j, i] = h * 1.3 / 30.0 * 60.0

            pair_count += 1

            if pair_count % 50 == 0:
                done = np.count_nonzero(dist_matrix) // 2
                print(f"  进度: {done}/{total_pairs} "
                      f"({done/total_pairs*100:.1f}%) "
                      f"失败: {fail_count}", flush=True)

            # 定期保存
            if pair_count % batch_save == 0:
                np.save(output_dir / "distance_matrix.npy", dist_matrix)
                np.save(output_dir / "time_matrix.npy", time_matrix)

            time.sleep(delay)

    # 同时更新邻接矩阵
    adj = (dist_matrix > 0) & (dist_matrix <= 50.0)
    adj_matrix = adj.astype(np.float64) * dist_matrix

    # 最终保存
    np.save(output_dir / "distance_matrix.npy", dist_matrix)
    np.save(output_dir / "time_matrix.npy", time_matrix)
    np.save(output_dir / "adjacency.npy", adj_matrix)

    print(f"\n=== 驾车距离矩阵计算完成 ===")
    print(f"  总对数: {pair_count}, 失败: {fail_count} ({fail_count/pair_count*100:.1f}%)")
    print(f"  距离范围: {dist_matrix[dist_matrix>0].min():.2f} ~ {dist_matrix.max():.2f} km")
    print(f"  时间范围: {time_matrix[time_matrix>0].min():.1f} ~ {time_matrix.max():.0f} min")
    print(f"  平均距离: {dist_matrix[dist_matrix>0].mean():.2f} km")
    print(f"  平均时间: {time_matrix[time_matrix>0].mean():.1f} min")
    print(f"  已保存到 {output_dir}/")

    return dist_matrix, time_matrix


def main():
    import argparse

    parser = argparse.ArgumentParser(description="天地图驾车距离矩阵计算")
    parser.add_argument("--tk", type=str, required=True, help="天地图API密钥")
    parser.add_argument("--metadata", type=str, default="data/processed/poi_metadata.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--delay", type=float, default=0.15, help="请求间隔秒数")
    args = parser.parse_args()

    build_drive_matrix(
        tk=args.tk,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
