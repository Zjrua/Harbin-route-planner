#!/usr/bin/env python3
"""P0 数据质量: 用高德 place/text 回查 POI, 拿权威 GCJ-02 坐标 + 高德 uid.

背景: poi_metadata.csv 由百度(WGS-84)与高德(GCJ-02)来源直接按名字合并,
全程无坐标基准转换, 且名字去重会合并同名连锁分店. 本脚本对每个 POI 回查
高德, 产出 uid + 权威坐标, 并做漂移检测(转换后偏差 >500m 视为可疑).

用法:
  # 试点: 抽样判定既有坐标的实际基准(identity / WGS-84→GCJ-02 / BD-09→GCJ-02)
  python scripts/verify_poi_amap.py --pilot 20
  # 全量跑某类别(默认景点), 断点续传
  python scripts/verify_poi_amap.py --category 景点
  # 只从断点汇总产出, 不再请求
  python scripts/verify_poi_amap.py --category 景点 --collect-only

产出:
  data/processed/amap_verify_checkpoint.jsonl  逐条断点(可删掉重跑)
  data/processed/poi_amap_verified.csv         回查结果宽表
  data/processed/amap_verify_report.json       质检报告

key 来源: --key > 环境变量 AMAP_KEY > .zcode/config.json(已 gitignore)
限速: 默认 3 QPS(用户账号限额), 命中 QPS 错误自动退避重试.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

REPO = Path(__file__).resolve().parents[1]
POI_CSV = REPO / "data/processed/poi_metadata.csv"
CKPT = REPO / "data/processed/amap_verify_checkpoint.jsonl"
OUT_CSV = REPO / "data/processed/poi_amap_verified.csv"
OUT_REPORT = REPO / "data/processed/amap_verify_report.json"

API = "https://restapi.amap.com/v3/place/text"
CITY = "哈尔滨"
QPS = 3
MIN_INTERVAL = 1.0 / QPS + 0.02
DRIFT_FLAG_M = 500.0

# ---------------------------------------------------------------- 坐标基准转换


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lng: float, lat: float) -> tuple[float, float]:
    a, ee = 6378245.0, 0.00669342162296594323
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = 1 - ee * math.sin(radlat) ** 2
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    return lng + dlng, lat + dlat


def bd09_to_gcj02(lng: float, lat: float) -> tuple[float, float]:
    x, y = lng - 0.0065, lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * math.pi * 3000.0 / 180.0)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * math.pi * 3000.0 / 180.0)
    return z * math.cos(theta), z * math.sin(theta)


TRANSFORMS = {
    "identity": lambda lng, lat: (lng, lat),  # 原坐标已是 GCJ-02
    "wgs84_to_gcj02": wgs84_to_gcj02,
    "bd09_to_gcj02": bd09_to_gcj02,
}


def haversine_m(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


# ---------------------------------------------------------------- 高德 API

_session = requests.Session()
_last_call = 0.0


def load_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    import os

    env = os.environ.get("AMAP_KEY")
    if env:
        return env
    cfg = REPO / ".zcode/config.json"  # gitignored, MCP 配置里已有同一把 key
    if cfg.exists():
        url = json.loads(cfg.read_text())["mcp"]["servers"]["amap"]["url"]
        return parse_qs(urlparse(url).query)["key"][0]
    raise SystemExit("找不到 AMAP key: 传 --key / 设 AMAP_KEY / 或先配好 .zcode/config.json")


def amap_text_search(key: str, keywords: str, city: str = CITY, rows: int = 10,
                     types: str | None = None, page: int = 1) -> list[dict]:
    """带 3QPS 限速 + QPS 超限退避重试. 返回候选 poi 列表(空列表=无结果)."""
    global _last_call
    params = {
        "key": key,
        "keywords": keywords,
        "city": city,
        "citylimit": "true",
        "offset": rows,
        "page": page,
        "extensions": "base",
    }
    if types:
        params["types"] = types
    for attempt in range(4):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            resp = _session.get(API, params=params, timeout=10)
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
            continue
        data = resp.json()
        if data.get("status") == "1":
            return data.get("pois", []) or []
        info = data.get("info", "")
        if "QPS" in info or "LIMIT" in info:  # 超限退避
            time.sleep(1.5 * (attempt + 1))
            continue
        raise SystemExit(f"高德 API 错误: infocode={data.get('infocode')} info={info} (keywords={keywords})")
    return []  # 重试耗尽, 记为无结果


# ---------------------------------------------------------------- 名字匹配

_PAREN = re.compile(r"[（(].*?[)）]")


def norm_name(name: str) -> str:
    """全角括号→半角, 去空白; 另给去括号版本用于模糊比较."""
    s = name.strip().replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", s)


def parse_loc(c: dict) -> tuple[float, float] | None:
    """高德 place/text 的坐标在 location 字段, 形如 '126.641608,45.773929'."""
    parts = str(c.get("location", "")).split(",")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def pick_candidate(poi_name: str, cands: list[dict], old_xy: tuple[float, float]) -> tuple[dict | None, str]:
    """返回 (候选, 匹配状态). 匹配层: 全名相等>去括号相等>包含(短名≥3字)>模糊≥0.80;
    道路名(19xxxx, 地名地址)降权, 组内按与旧坐标的原始距离最近决胜."""
    oname = norm_name(poi_name)
    obare = _PAREN.sub("", oname)
    scored = []
    for c in cands:
        cname = norm_name(c.get("name", ""))
        xy = parse_loc(c)
        if not cname or xy is None:
            continue
        cbare = _PAREN.sub("", cname)
        if cname == oname:
            tier = 3.0
        elif cbare == obare:
            tier = 2.5
        else:
            short, long = sorted((cbare, obare), key=len)
            # 包含层收紧: 短名≥4字, 且禁止撞上交通设施(15)/地名地址(19)——
            # "松花江"吞掉湿地公园、"博物馆"撞上地铁站这类跨实体误配就是它
            ok_contains = len(short) >= 4 and short in long \
                and not str(c.get("typecode", "")).startswith(("15", "19"))
            tier = 2.0 if ok_contains else difflib.SequenceMatcher(None, cname, oname).ratio()
        tier -= 0.05 if str(c.get("typecode", "")).startswith("19") else 0.0
        dist = haversine_m(*old_xy, *xy)
        scored.append((tier, -dist, {**c, "lng": xy[0], "lat": xy[1]}))
    if not scored:
        return None, "none"
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    tier, _, best = scored[0]
    if tier < 1.0:
        return None, "none"
    status = "exact" if tier >= 2.4 else ("contains" if tier >= 1.9 else "fuzzy")
    return best, status


# ---------------------------------------------------------------- 主流程


def iter_targets(category: str, pilot: int):
    with open(POI_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pool = [r for r in rows if r["category"] == category] if category else rows
    if pilot:
        by_src: dict[str, list] = {}
        for r in pool:
            by_src.setdefault(r["source"], []).append(r)
        rng = random.Random(42)
        picked = []
        for src, lst in sorted(by_src.items()):
            rng.shuffle(lst)
            picked += lst[:pilot]
        return picked
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--category", default="景点", help="只跑该类别; 传 all 跑全量")
    ap.add_argument("--pilot", type=int, default=0, help="每来源抽样 N 条做基准判定试点")
    ap.add_argument("--key", default=None)
    ap.add_argument("--collect-only", action="store_true", help="只用断点汇总, 不发请求")
    args = ap.parse_args()

    key = load_key(args.key)
    category = None if args.category == "all" else args.category
    targets = iter_targets(category, args.pilot)
    print(f"目标 POI: {len(targets)} 条 (category={args.category}, pilot={args.pilot or 'off'})")

    done: dict[str, dict] = {}
    if CKPT.exists():
        for line in CKPT.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["name"]] = rec
        print(f"断点: 已有 {len(done)} 条, 跳过")

    handle = open(CKPT, "a", encoding="utf-8")
    n_new = 0
    for i, row in enumerate(targets):
        if row["name"] in done:
            continue
        if args.collect_only:
            break
        rec = {
            "name": row["name"],
            "category": row["category"],
            "source": row["source"],
            "old_lng": row["lng"],
            "old_lat": row["lat"],
            "old_address": row["address"],
        }
        cands = amap_text_search(key, row["name"])
        best, status = pick_candidate(row["name"], cands, (float(row["lng"]), float(row["lat"])))
        rec["match_status"] = status
        rec["n_candidates"] = len(cands)
        if best is not None:
            rec.update(
                amap_uid=best.get("id", ""),
                amap_name=best.get("name", ""),
                amap_type=best.get("type", ""),
                amap_typecode=str(best.get("typecode", "")),
                amap_address=best.get("address", "") if isinstance(best.get("address"), str) else "",
                amap_adname=best.get("adname", ""),
                amap_lng=best["lng"],
                amap_lat=best["lat"],
                amap_rating=(best.get("biz_ext") or {}).get("rating") or "",
                amap_cost=(best.get("biz_ext") or {}).get("cost") or "",
            )
        else:
            rec.update(amap_uid="", amap_name="", amap_type="", amap_typecode="",
                       amap_address="", amap_adname="", amap_lng="", amap_lat="",
                       amap_rating="", amap_cost="")
        done[rec["name"]] = rec
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        handle.flush()
        n_new += 1
        if n_new % 50 == 0:
            print(f"  进度 {len(done)}/{len(targets)}")
    handle.close()
    print(f"新查询 {n_new} 条, 断点共 {len(done)} 条")

    collect(done, pilot=bool(args.pilot))


def collect(done: dict[str, dict], pilot: bool) -> None:
    """基准判定(试点) + 漂移统计 + 产出 CSV/报告."""
    recs = list(done.values())

    # 1) 基准判定: 对每个假设变换, 算"旧坐标(变换后)→高德坐标"的中位距离, 最小者胜
    verdict: dict[str, dict] = {}
    for src in ("amap", "baidu"):
        subset = [r for r in recs if r["source"] == src and r.get("amap_lng")]
        per_t: dict[str, list[float]] = {t: [] for t in TRANSFORMS}
        for r in subset:
            for t, fn in TRANSFORMS.items():
                lng0, lat0 = fn(float(r["old_lng"]), float(r["old_lat"]))
                per_t[t].append(haversine_m(lng0, lat0, float(r["amap_lng"]), float(r["amap_lat"])))
        verdict[src] = {t: sorted(ds)[len(ds) // 2] if ds else None for t, ds in per_t.items()}
    if pilot:
        print("\n基准判定(中位偏差, 米; amap 源应为 identity 最小):")
        for src, v in verdict.items():
            print(f"  {src}: " + ", ".join(f"{k}={vv:.0f}m" if vv is not None else f"{k}=NA"
                                           for k, vv in v.items()))
        for src, v in verdict.items():
            if all(x is not None for x in v.values()):
                best = min(v, key=v.get)
                print(f"  => {src} 源最优假设: {best}")

    # 2) 逐条: 按各源最优变换后的偏差做漂移检测
    def best_transform(src: str) -> str:
        v = verdict[src]
        return min(v, key=v.get) if v and all(x is not None for x in v.values()) else "identity"

    fields = ["name", "category", "source", "match_status", "amap_uid", "amap_name",
              "amap_type", "amap_typecode", "amap_address", "amap_adname",
              "old_lng", "old_lat", "amap_lng", "amap_lat",
              "coord_transform", "drift_m", "drift_flag", "amap_rating", "amap_cost",
              "old_address", "n_candidates"]
    rows_out = []
    for r in recs:
        t = best_transform(r["source"])
        r2 = {k: r.get(k, "") for k in fields}
        r2["coord_transform"] = t
        if r.get("amap_lng"):
            lng0, lat0 = TRANSFORMS[t](float(r["old_lng"]), float(r["old_lat"]))
            drift = haversine_m(lng0, lat0, float(r["amap_lng"]), float(r["amap_lat"]))
            r2["drift_m"] = round(drift, 1)
            r2["drift_flag"] = bool(drift > DRIFT_FLAG_M)
        else:
            r2["drift_m"] = ""
            r2["drift_flag"] = ""
        rows_out.append(r2)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    matched = [r for r in rows_out if r["match_status"] in ("exact", "contains", "fuzzy")]
    drifts = sorted(r["drift_m"] for r in matched if r["drift_m"] != "")
    def pct(p):
        return drifts[int(p * (len(drifts) - 1))] if drifts else None
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(rows_out),
        "match_status": Counter(r["match_status"] for r in rows_out),
        "datum_verdict_median_m": verdict,
        "chosen_transform": {src: best_transform(src) for src in ("amap", "baidu")},
        "drift_after_transform_m": {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
                                    "flag_gt_500m": sum(1 for r in rows_out if r["drift_flag"])},
        "flagged": [r["name"] for r in rows_out if r["drift_flag"]],
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n产出: {OUT_CSV.name} ({len(rows_out)} 行), {OUT_REPORT.name}")
    print(f"匹配: {dict(report['match_status'])}, 漂移 p50={pct(0.5)}m p90={pct(0.9)}m, "
          f"可疑(>500m): {report['drift_after_transform_m']['flag_gt_500m']}")


if __name__ == "__main__":
    main()
