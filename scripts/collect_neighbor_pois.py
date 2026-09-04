#!/usr/bin/env python3
"""哈尔滨+ 周边城市景点摸底采集: 按 types=110000(风景名胜) 分页拉取全量.

目的: 给"哈尔滨+"(哈尔滨+伊春/大庆/牡丹江等近邻城市, 3~5 天以上线路)
摸清各市景点供给量, 产出统一 GCJ-02 坐标的候选 POI 池.
后续若扩展餐饮/住宿等类别, 加 --types 即可.

用法:
  python scripts/collect_neighbor_pois.py                    # 默认三市试点
  python scripts/collect_neighbor_pois.py --cities 伊春市,黑河市
产出: data/processed/neighbor_pois_pilot.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from verify_poi_amap import REPO, amap_text_search, load_key, parse_loc

SCENIC = "110000"  # 风景名胜(公园/旅游景点/纪念馆等)
OUT = REPO / "data/processed/neighbor_pois_pilot.csv"
FIELDS = ["city", "amap_uid", "name", "lng", "lat", "type", "typecode", "address", "adname"]

DEFAULT_CITIES = ["伊春市", "大庆市", "牡丹江市"]


def collect_city(key: str, city: str, max_pages: int = 50) -> list[dict]:
    """分页拉取一市风景名胜类 POI. 高德单查询上限 900 条(v3), 景点量足够."""
    seen: dict[str, dict] = {}
    for page in range(1, max_pages + 1):
        batch = amap_text_search(key, "", city=city, rows=25, types=SCENIC, page=page)
        if not batch:
            break
        for p in batch:
            xy = parse_loc(p)
            if xy is None or p.get("id") in seen:
                continue
            seen[p["id"]] = {
                "city": city,
                "amap_uid": p.get("id", ""),
                "name": p.get("name", ""),
                "lng": xy[0],
                "lat": xy[1],
                "type": p.get("type", ""),
                "typecode": str(p.get("typecode", "")),
                "address": p.get("address", "") if isinstance(p.get("address"), str) else "",
                "adname": p.get("adname", ""),
            }
        if len(batch) < 25:
            break
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--cities", default=",".join(DEFAULT_CITIES))
    ap.add_argument("--key", default=None)
    args = ap.parse_args()
    key = load_key(args.key)
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    all_rows: list[dict] = []
    for city in cities:
        rows = collect_city(key, city)
        all_rows += rows
        districts = Counter(r["adname"] for r in rows)
        print(f"{city}: {len(rows)} 个风景名胜类 POI, 分布 {dict(districts.most_common(6))}")
        print("   样例:", "、".join(r["name"] for r in rows[:5]))

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n产出 {OUT} ({len(all_rows)} 行, GCJ-02, uid 为实体键)")


if __name__ == "__main__":
    main()
