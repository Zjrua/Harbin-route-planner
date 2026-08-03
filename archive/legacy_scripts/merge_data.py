"""合并高德 + 百度 POI 数据，统一格式后输出到 data/raw/merged_pois.csv.

用法:
    python scripts/merge_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_amap_data(path: str) -> pd.DataFrame:
    """加载高德 POI 数据."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 列: 名称, 经度, 纬度, 类别, 地址, POI类型, 评分, 人均消费
    return pd.DataFrame({
        "name": df["名称"],
        "lng": df["经度"].astype(float),
        "lat": df["纬度"].astype(float),
        "category": df["类别"],
        "address": df.get("地址", ""),
        "poi_type": df.get("POI类型", ""),
        "rating": pd.to_numeric(df["评分"].replace("[]", np.nan), errors="coerce"),
        "avg_cost": pd.to_numeric(df.get("人均消费", pd.Series([np.nan]*len(df))).replace("[]", np.nan), errors="coerce"),
        "source": "amap",
    })


def load_baidu_data(path: str) -> pd.DataFrame:
    """加载百度 POI 数据."""
    df = pd.read_excel(path)
    # 列: uid, name, addr, showtag, srcname, stdtag, phone, wgslng, wgslat,
    #     overall_rating, image, price, comment_num, shop_hours, image_num

    # 从 stdtag 提取类别
    cat_map = {
        "美食": "餐饮", "餐饮": "餐饮",
        "酒店": "住宿", "公寓式酒店": "住宿",
        "旅游景点": "景点", "风景名胜": "景点",
        "运动健身": "景点", "休闲体育": "景点",
        "生活服务": "购物", "购物": "购物",
        "文化媒体": "景点", "教育培训": "其他",
    }

    def map_category(stdtag: str, srcname: str) -> str:
        tag = str(stdtag).split(";")[0]
        if tag in cat_map:
            return cat_map[tag]
        # fallback: 用 srcname
        src = str(srcname)
        if src == "cater":
            return "餐饮"
        if src == "shopping":
            return "购物"
        if src == "scope":
            return "景点"
        return "其他"

    categories = [
        map_category(row.get("stdtag", ""), row.get("srcname", ""))
        for _, row in df.iterrows()
    ]

    return pd.DataFrame({
        "name": df["name"],
        "lng": df["wgslng"].astype(float),
        "lat": df["wgslat"].astype(float),
        "category": categories,
        "address": df["addr"].fillna(""),
        "poi_type": df["stdtag"].fillna(""),
        "rating": df["overall_rating"],
        "avg_cost": df["price"],
        "comment_num": df["comment_num"],
        "source": "baidu",
    })


def deduplicate(amap: pd.DataFrame, baidu: pd.DataFrame) -> pd.DataFrame:
    """合并去重：高德数据优先，百度数据补充.

    去重策略：
    1. 同名 POI：高德优先（评分更可靠），用百度数据补充缺失字段
    2. 百度独有的 POI：保留，要求有评分
    3. 同名但来源不同时：取高德的评分和坐标，取百度的 comment_num 等
    """
    # 高德数据直接保留
    amap_valid = amap.dropna(subset=["lat", "lng"])

    # 百度数据：过滤无评分和 "其他" 类别
    baidu_valid = baidu.dropna(subset=["lat", "lng"])
    baidu_valid = baidu_valid[baidu_valid["rating"].notna() & (baidu_valid["rating"] > 0)]
    baidu_valid = baidu_valid[baidu_valid["category"] != "其他"]

    # 百度数据中与高德同名的：补充高德缺失字段（如 avg_cost）
    amap_names = set(amap_valid["name"].dropna().str.strip())
    baidu_overlap = baidu_valid[baidu_valid["name"].str.strip().isin(amap_names)]
    baidu_unique = baidu_valid[~baidu_valid["name"].str.strip().isin(amap_names)]

    # 对于高德数据中缺失 avg_cost 的，尝试从百度补充
    if "avg_cost" in amap_valid.columns:
        baidu_cost_map = {}
        for _, row in baidu_overlap.iterrows():
            if pd.notna(row.get("avg_cost")):
                baidu_cost_map[row["name"].strip()] = row["avg_cost"]

        amap_valid = amap_valid.copy()
        amap_valid["avg_cost"] = amap_valid.apply(
            lambda r: r["avg_cost"] if pd.notna(r.get("avg_cost")) else baidu_cost_map.get(r["name"].strip(), np.nan),
            axis=1,
        )

    # 高德中缺失评分的，尝试从百度补充
    if "rating" in amap_valid.columns:
        baidu_rating_map = {}
        for _, row in baidu_overlap.iterrows():
            if pd.notna(row.get("rating")):
                baidu_rating_map[row["name"].strip()] = row["rating"]

        amap_valid = amap_valid.copy()
        amap_valid["rating"] = amap_valid.apply(
            lambda r: r["rating"] if pd.notna(r.get("rating")) else baidu_rating_map.get(r["name"].strip(), np.nan),
            axis=1,
        )

    merged = pd.concat([amap_valid, baidu_unique], ignore_index=True)
    merged = merged.drop_duplicates(subset=["name"])

    return merged.reset_index(drop=True)


def main():
    raw_dir = Path("data/raw")

    print("加载高德数据...")
    amap = load_amap_data(str(raw_dir / "哈尔滨POI数据_完整版.csv"))
    print(f"  高德: {len(amap)} 条, 类别: {amap['category'].value_counts().to_dict()}")

    print("加载百度数据...")
    baidu = load_baidu_data(str(raw_dir / "百度poi(旅游相关) - 全集.xlsx"))
    print(f"  百度: {len(baidu)} 条, 类别: {baidu['category'].value_counts().to_dict()}")

    print("合并去重...")
    merged = deduplicate(amap, baidu)
    print(f"  合并后: {len(merged)} 条")
    print(f"  类别: {merged['category'].value_counts().to_dict()}")
    print(f"  来源: {merged['source'].value_counts().to_dict()}")

    output_path = raw_dir / "merged_pois.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n已保存: {output_path}")


if __name__ == "__main__":
    main()
