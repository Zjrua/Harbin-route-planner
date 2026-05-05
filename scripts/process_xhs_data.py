"""处理小红书笔记数据：提取餐饮住宿热度 + 增强路线.

策略:
1. 从笔记中统计各POI的提及频率 → 热度权重
2. 从路线笔记中提取结构化路线（箭头分隔、序号列表等）
3. 用XHS热度数据改进现有路线增强

用法:
    uv run python scripts/process_xhs_data.py
"""

import json
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from difflib import SequenceMatcher


def clean_liked_count(val) -> int:
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().replace("+", "").replace(",", "")
    if not s or s == "0":
        return 0
    if "万" in s:
        try:
            return int(float(s.replace("万", "")) * 10000)
        except ValueError:
            return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def build_search_index(poi_names: list[str]) -> dict:
    """构建名称的快速模糊索引."""
    idx = defaultdict(set)
    for name in poi_names:
        clean = re.sub(r"[·\-\(\)（）\s]", "", str(name))
        # 提取所有2-5字的子串作为搜索键
        for k in range(2, min(len(clean) + 1, 6)):
            for j in range(len(clean) - k + 1):
                key = clean[j:j + k]
                if len(key) >= 2:
                    idx[key].add(name)
    return idx


def match_poi_mentions(text: str, index: dict, name_to_idx: dict) -> set[int]:
    """在文本中匹配POI，返回匹配到的POI索引集合."""
    clean_text = re.sub(r"[·\-\(\)（）\s]", "", text)
    matched = set()
    for key, names in index.items():
        if key in clean_text:
            for name in names:
                if name in name_to_idx:
                    matched.add(name_to_idx[name])
    return matched


def extract_structured_routes(notes: list[dict], pois: pd.DataFrame) -> list[dict]:
    """从结构化的路线笔记中提取路线序列.

    优先处理:
    1. 箭头分隔: "A→B→C"
    2. 序号列表: "1. A 2. B 3. C"
    3. 日分隔: "Day1: A, B; Day2: C, D"
    """
    poi_names = pois["name"].dropna().astype(str).tolist()
    name_to_idx = {name: i for i, name in enumerate(poi_names)}
    index = build_search_index(poi_names)

    routes = []
    for note in notes:
        text = str(note.get("title", "")) + "\n" + str(note.get("desc", ""))[:5000]

        indices = []

        # 1. 尝试箭头分隔
        arrow_parts = re.split(r"→|->|→|–>|—>", text)
        if len(arrow_parts) >= 3:
            for part in arrow_parts:
                part_clean = part.strip()[:40]
                matches = match_poi_mentions(part_clean, index, name_to_idx)
                for m in matches:
                    if m not in indices:
                        indices.append(m)

        # 2. 尝试序号列表
        if len(indices) < 3:
            # 找 "1. XXX 2. XXX" 或 "Day1: XXX" 模式
            items = re.findall(r'(?:^|\n)\s*(?:\d+[\.、）\)]|Day\s*\d+[：:])[ \t]*([^\n]{3,40})', text)
            for item in items:
                matches = match_poi_mentions(item.strip(), index, name_to_idx)
                for m in matches:
                    if m not in indices:
                        indices.append(m)

        # 3. 短横分隔 "A - B - C"
        if len(indices) < 3:
            dash_parts = re.split(r"\s+[-–—]\s+", text)
            if len(dash_parts) >= 3:
                for part in dash_parts:
                    part_clean = part.strip()[:40]
                    matches = match_poi_mentions(part_clean, index, name_to_idx)
                    for m in matches:
                        if m not in indices:
                            indices.append(m)

        if len(indices) >= 3:
            liked = clean_liked_count(note.get("liked_count", 0))
            routes.append({
                "indices": indices,
                "season": "winter",
                "source": "xhs",
                "liked_count": liked,
                "title": note.get("title", ""),
            })

    return routes


def compute_poi_popularity(notes: list[dict], pois: pd.DataFrame) -> np.ndarray:
    """从笔记中计算POI的热度权重.

    热度 = 提及次数 + 高赞笔记的额外权重
    """
    poi_names = pois["name"].dropna().astype(str).tolist()
    name_to_idx = {name: i for i, name in enumerate(poi_names)}
    index = build_search_index(poi_names)

    # 提及计数（加权）
    mention_weight = np.zeros(len(pois))
    mention_count = np.zeros(len(pois), dtype=int)

    for note in notes:
        text = str(note.get("title", "")) + "\n" + str(note.get("desc", ""))[:2000]
        likes = clean_liked_count(note.get("liked_count", 0))
        # 权重 = 1 + log(1+likes)/10 （点赞多的大V笔记权重更高）
        weight = 1.0 + np.log(1 + likes) / 10

        matched = match_poi_mentions(text, index, name_to_idx)
        for idx in matched:
            mention_count[idx] += 1
            mention_weight[idx] += weight

    # 只在有提及的POI中归一化
    if mention_weight.sum() > 0:
        mention_weight = mention_weight / mention_weight.max()

    return mention_weight, mention_count


def main():
    raw_dir = Path("data/raw")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载所有笔记
    print("=== 加载笔记 ===")
    notes = []
    for f in raw_dir.glob("search_*.jsonl"):
        print(f"  {f.name}")
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    notes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"  总计: {len(notes)} 条")

    # 2. 分类笔记
    route_kw = ["路线", "攻略", "行程", "打卡", "一日", "三天", "怎么玩", "安排", "暴走", "citywalk"]
    def is_route(note):
        text = str(note.get("title", "")) + str(note.get("desc", ""))[:500]
        return sum(1 for kw in route_kw if kw in text) >= 2

    route_notes = [n for n in notes if is_route(n)]
    print(f"\n  路线类笔记: {len(route_notes)}")

    # 3. 加载POI
    pois = pd.read_csv(out_dir / "poi_metadata.csv", encoding="utf-8")
    print(f"  POI: {len(pois)} 个")

    # 4. 计算POI热度
    print("\n=== 计算POI热度 ===")
    popularity, mention_count = compute_poi_popularity(notes, pois)

    # 添加到POI数据
    pois["xhs_mentions"] = mention_count
    pois["xhs_popularity"] = popularity

    # 展示热度最高的餐饮/住宿
    print("\n  热度最高餐饮 (XHS):")
    dining = pois[pois["category"] == "餐饮"].nlargest(10, "xhs_popularity")
    for _, r in dining.iterrows():
        name = str(r["name"])[:30]
        print(f"    {name:<30} 提及{r['xhs_mentions']:>4d} 热度{r['xhs_popularity']:.3f}")

    print("\n  热度最高住宿 (XHS):")
    hotels = pois[pois["category"] == "住宿"].nlargest(10, "xhs_popularity")
    for _, r in hotels.iterrows():
        name = str(r["name"])[:30]
        print(f"    {name:<30} 提及{r['xhs_mentions']:>4d} 热度{r['xhs_popularity']:.3f}")

    # 5. 提取结构化路线
    print("\n=== 提取结构化路线 ===")
    routes = extract_structured_routes(route_notes, pois)
    # 去重（按路线内容）
    unique_routes = []
    seen = set()
    for r in routes:
        key = tuple(r["indices"])
        if key not in seen and len(key) >= 3:
            seen.add(key)
            unique_routes.append(r)
    print(f"  提取: {len(routes)} 条, 去重后: {len(unique_routes)} 条")

    if unique_routes:
        # 统计
        cat_counter = Counter()
        for r in unique_routes:
            for idx in r["indices"]:
                cat_counter[str(pois.iloc[idx]["category"])] += 1
        total = sum(cat_counter.values())
        for cat, cnt in cat_counter.most_common():
            print(f"    {cat}: {cnt} ({cnt/total*100:.1f}%)")

        # 展示
        unique_routes.sort(key=lambda r: r["liked_count"], reverse=True)
        for r in unique_routes[:5]:
            names = " → ".join(str(pois.iloc[idx]["name"])[:12] for idx in r["indices"])
            print(f"  [{r['liked_count']:,}赞] {names}")

    # 6. 保存
    print("\n=== 保存 ===")
    # XHS热度增强的POI元数据
    pois.to_csv(out_dir / "poi_metadata.csv", index=False, encoding="utf-8")
    np.save(out_dir / "poi_xhs_popularity.npy", popularity)

    # 将新路线与现有路线合并
    existing_routes = list(np.load(out_dir / "routes.npy", allow_pickle=True))
    for r in unique_routes:
        existing_routes.append(np.array(r["indices"]))

    np.save(out_dir / "routes.npy", np.array(existing_routes, dtype=object))

    print(f"  poi_metadata.csv: +xhs_mentions, +xhs_popularity")
    print(f"  poi_xhs_popularity.npy: {popularity.shape}")
    print(f"  routes.npy: {len(existing_routes)} 条 (原+{len(unique_routes)}条XHS)")
    print(f"\n完成！")


if __name__ == "__main__":
    main()
