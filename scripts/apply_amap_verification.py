#!/usr/bin/env python3
"""把 verify_poi_amap 的回查结果修正回 poi_metadata.csv.

策略(按 poi_amap_verified.csv 的 match_status × drift):
  exact  + 任意漂移       -> 采用高德坐标+uid(名字精确命中, 高德为权威;
                             漂移大=旧坐标错, 保留 needs_review 供人工复核)
  contains + drift<=500m  -> 采用高德坐标+uid(邻近度证实同实体)
  contains + drift>500m   -> 不采用, 标 needs_review(跨实体误配风险)
  none                    -> 保留旧坐标
未覆盖的行(非景点类别等): 百度源做 WGS-84->GCJ-02 纯数学转换(全量景点已实证
  百度源=WGS-84, 转换后中位偏差 11m), 高德源原样保留.
全表统一输出 GCJ-02. 保持行序与既有列, 仅追加列.
注意: distance/time 矩阵与训练数据集基于旧坐标, 本修正后属已知滞后,
重建前不要把它们当作修正后坐标的精确反映(见 data/README.md).
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from verify_poi_amap import REPO, wgs84_to_gcj02

POI_CSV = REPO / "data/processed/poi_metadata.csv"
VERIFIED = REPO / "data/processed/poi_amap_verified.csv"

NEW_COLS = ["amap_uid", "coord_source", "match_status", "drift_m", "needs_review", "coord_fixed_on"]


def main() -> None:
    with open(POI_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields_in = list(reader.fieldnames or [])
        rows = list(reader)
    ver = {r["name"]: r for r in csv.DictReader(open(VERIFIED, encoding="utf-8"))}

    n = {"adopted": 0, "converted_only": 0, "kept": 0, "needs_review": 0}
    for row in rows:
        v = ver.get(row["name"])
        adopted = converted = review = False
        uid = status = drift = ""
        if v is not None:
            status = v["match_status"]
            drift = v.get("drift_m", "")
            uid = v.get("amap_uid", "")
            drift_val = float(drift) if drift != "" else None
            if status == "exact":
                adopted = True                      # 精确命中: 高德为权威
                review = v.get("drift_flag") == "True"
            elif status == "contains" and drift_val is not None and drift_val <= 500:
                adopted = True                      # 弱名+近坐标: 邻近证实同实体
            elif status == "contains":
                review = True                       # 弱名+远坐标: 误配风险, 不采用
        if adopted and v.get("amap_lng"):
            row["lng"], row["lat"] = v["amap_lng"], v["amap_lat"]
            row["coord_source"] = "amap"
        elif row["source"] == "baidu":
            lng, lat = wgs84_to_gcj02(float(row["lng"]), float(row["lat"]))
            row["lng"], row["lat"] = f"{lng:.7f}", f"{lat:.7f}"
            row["coord_source"] = "baidu_wgs2gcj"
            converted = True
        else:
            row["coord_source"] = "amap_orig"
        row["amap_uid"] = uid if adopted else ""
        row["match_status"] = status if v else "unverified"
        row["drift_m"] = drift if adopted else ""
        row["needs_review"] = str(review)
        row["coord_fixed_on"] = date.today().isoformat()
        n["adopted" if adopted else ("converted_only" if converted else "kept")] += 1
        if review:
            n["needs_review"] += 1

    with open(POI_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_in + NEW_COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"修正完成: 共 {len(rows)} 行 (行序未变)")
    print(f"  采用高德坐标: {n['adopted']} | 仅基准转换: {n['converted_only']} | 原样保留: {n['kept']}")
    print(f"  needs_review: {n['needs_review']} 个 -> 可在 poi_metadata.csv 按该列筛出复核")
    report = json.load(open(REPO / "data/processed/amap_verify_report.json"))
    print(f"  底层报告: 景点匹配 {report['match_status']}, 基准判定 {report['chosen_transform']}")


if __name__ == "__main__":
    main()
