"""高德地图 POI 爬取模块."""

import requests
import pandas as pd
from typing import List, Optional


def crawl_pois(api_key: str, city: str = "哈尔滨",
               keywords: Optional[List[str]] = None,
               min_rating: float = 3.5) -> pd.DataFrame:
    """爬取高德地图 POI 数据.

    Args:
        api_key: 高德地图 API Key
        city: 城市名称
        keywords: POI 关键词列表（如 ["景点", "美食", "住宿"]）
        min_rating: 最低评分过滤阈值

    Returns:
        POI 数据 DataFrame，包含 id, name, lat, lng, category, rating 等列
    """
    raise NotImplementedError("需实现：调用高德 POI 搜索 API，分页获取数据")


def crawl_poi_detail(api_key: str, poi_id: str) -> dict:
    """获取单个 POI 的详细信息.

    Args:
        api_key: 高德地图 API Key
        poi_id: POI 唯一标识

    Returns:
        POI 详情字典
    """
    raise NotImplementedError("需实现：调用高德 POI 详情 API")
