"""天地图 API POI 爬取模块.

使用天地图 v2/search 接口，按关键词分页爬取哈尔滨文旅相关 POI 数据。
API文档: http://lbs.tianditu.gov.cn/server/search.html
"""

import time
import json
import requests
import pandas as pd
from urllib.parse import quote
from typing import List, Optional


BASE_URL = "http://api.tianditu.gov.cn/v2/search"

# 哈尔滨市边界（经纬度范围）
HARBIN_BOUND = "125.70,45.40,127.50,46.20"

# 哈尔滨市中心坐标
HARBIN_CENTER = "126.53,45.80"

# 旅游相关搜索关键词（排除模糊匹配噪音大的词如"景点"、"美食"、"餐饮"）
TOURISM_KEYWORDS = [
    # 核心景点（精确词）
    "风景区", "旅游景区", "公园", "广场", "冰雪大世界", "太阳岛",
    "中央大街", "索菲亚", "极地馆", "虎林园",
    # 文化场馆
    "博物馆", "纪念馆", "展览馆", "美术馆", "文化宫",
    # 宗教建筑
    "教堂", "寺庙", "清真寺",
    # 冰雪特色
    "滑雪场", "冰雕",
    # 娱乐休闲
    "游乐场", "动物园", "水上乐园", "温泉",
    # 餐饮美食（精确词）
    "火锅", "烧烤", "俄式餐厅", "老字号", "咖啡厅", "酒吧",
    # 住宿
    "酒店", "宾馆", "民宿",
    # 购物
    "商场", "步行街", "特产",
    # 交通枢纽
    "火车站", "机场", "地铁站",
]

# POI类型映射
POI_TYPE_MAP = {
    "101": "旅游景点",
    "102": "公交车站",
    "103": "地铁站",
    "104": "停车场",
    "105": "加油站",
    "106": "收费站",
    "107": "汽车站",
    "108": "火车站",
    "109": "机场",
    "110": "港口码头",
    "111": "长途汽车站",
    "112": "餐饮",
    "113": "购物",
    "114": "生活服务",
    "115": "住宿",
    "116": "风景名胜",
    "117": "科教文化",
    "118": "医疗",
    "119": "政府机构",
    "120": "运动健身",
    "121": "公园广场",
    "122": "休闲娱乐",
}


def search_pois(
    tk: str,
    keyword: str,
    bound: str = HARBIN_BOUND,
    level: str = "12",
    query_type: str = "1",
    count: int = 50,
    max_records: Optional[int] = None,
    delay: float = 0.3,
) -> List[dict]:
    """按关键词分页搜索POI.

    Args:
        tk: 天地图API密钥
        keyword: 搜索关键词
        bound: 搜索范围 "minLon,minLat,maxLon,maxLat"
        level: 缩放级别 1-18
        query_type: 搜索类型 (1=普通搜索)
        count: 每页返回数量 (max 300)
        max_records: 最大获取条数，None表示全部
        delay: 请求间隔秒数

    Returns:
        POI列表
    """
    all_pois = []
    start = 0

    while True:
        post_str = json.dumps({
            "keyWord": keyword,
            "level": level,
            "mapBound": bound,
            "queryType": query_type,
            "count": str(count),
            "start": str(start),
        }, ensure_ascii=False)

        url = f"{BASE_URL}?postStr={quote(post_str)}&type=query&tk={tk}"

        try:
            resp = requests.get(url, timeout=15)
            resp.encoding = "utf-8"
            data = resp.json()
        except Exception as e:
            print(f"  [ERROR] keyword={keyword}, start={start}: {e}")
            break

        status = data.get("status", {})
        if status.get("infocode") != 1000:
            print(f"  [WARN] API返回错误: {status}")
            break

        total = data.get("count", 0)
        pois = data.get("pois", [])

        if not pois:
            break

        all_pois.extend(pois)
        print(f"  [{keyword}] 获取 {len(pois)} 条 (累计 {len(all_pois)}/{total})")

        start += count
        if total <= start:
            break
        if max_records and len(all_pois) >= max_records:
            all_pois = all_pois[:max_records]
            break

        time.sleep(delay)

    return all_pois


def parse_pois(raw_pois: List[dict], keyword: str) -> List[dict]:
    """解析原始POI数据为标准化字典.

    Args:
        raw_pois: API返回的POI列表
        keyword: 搜索用的关键词（作为类别参考）

    Returns:
        标准化后的POI字典列表
    """
    parsed = []
    for poi in raw_pois:
        lonlat = poi.get("lonlat", "")
        parts = lonlat.split(",")
        lon = float(parts[0]) if len(parts) == 2 else None
        lat = float(parts[1]) if len(parts) == 2 else None

        if lon is None or lat is None:
            continue

        parsed.append({
            "hotPointID": poi.get("hotPointID", ""),
            "name": poi.get("name", ""),
            "address": poi.get("address", ""),
            "phone": poi.get("phone", ""),
            "lon": lon,
            "lat": lat,
            "poi_type_code": poi.get("poiType", ""),
            "poi_type_name": POI_TYPE_MAP.get(poi.get("poiType", ""), ""),
            "search_keyword": keyword,
            "source": poi.get("source", "0"),
        })
    return parsed


def crawl_all_pois(
    tk: str,
    keywords: Optional[List[str]] = None,
    bound: str = HARBIN_BOUND,
    delay: float = 0.3,
) -> pd.DataFrame:
    """爬取所有类别的POI数据并去重.

    Args:
        tk: 天地图API密钥
        keywords: 搜索关键词列表
        bound: 搜索范围
        delay: 请求间隔

    Returns:
        去重后的POI DataFrame
    """
    if keywords is None:
        keywords = TOURISM_KEYWORDS

    all_parsed = []

    for kw in keywords:
        print(f"\n>>> 搜索: {kw}")
        raw = search_pois(tk, kw, bound=bound, delay=delay)
        parsed = parse_pois(raw, kw)
        all_parsed.extend(parsed)
        print(f"    本轮 {len(parsed)} 条")

    if not all_parsed:
        print("\n未获取到任何数据")
        return pd.DataFrame()

    df = pd.DataFrame(all_parsed)
    print(f"\n=== 去重前总计 {len(df)} 条 ===")

    # 按hotPointID去重，保留第一条
    before = len(df)
    df = df.drop_duplicates(subset=["hotPointID"], keep="first").reset_index(drop=True)
    print(f"=== 去重后 {len(df)} 条 (去除 {before - len(df)} 条重复) ===")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="天地图POI数据爬取")
    parser.add_argument("--tk", type=str, required=True, help="天地图API密钥")
    parser.add_argument("--output", type=str, default="data/raw/tianditu_pois_harbin.csv",
                        help="输出CSV路径")
    parser.add_argument("--keywords", type=str, nargs="+", default=None,
                        help="自定义搜索关键词")
    parser.add_argument("--delay", type=float, default=0.3, help="请求间隔秒数")
    args = parser.parse_args()

    df = crawl_all_pois(tk=args.tk, keywords=args.keywords, delay=args.delay)

    if not df.empty:
        import os
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n数据已保存到 {args.output}")
        print(f"共 {len(df)} 个唯一POI")

        # 打印统计
        print("\n--- 按POI类型统计 ---")
        type_counts = df["poi_type_name"].value_counts()
        for t, c in type_counts.items():
            print(f"  {t or '未知'}: {c}")

        print("\n--- 按搜索关键词统计(去重前) ---")
        kw_counts = df["search_keyword"].value_counts().head(20)
        for k, c in kw_counts.items():
            print(f"  {k}: {c}")


if __name__ == "__main__":
    main()
