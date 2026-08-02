"""RAG 候选检索：从 POI 库为"某一天"检索候选景点（Phase 1）.

核心思路（解决模型编造 POI 名的根因）：
模型只有 4B 参数，记不住 10K 个真实 POI 名——长序列生成时会即兴编造
（"南岗区天主教堂""中央广场"等假名）。RAG 把"背 POI 名"变成"在给定
候选里排序"：检索出 30-40 个真实 POI 塞进 prompt，模型只能从真实候选里选。

检索信号：
- 就近：以出发地/核心景点为中心，用距离矩阵取最近候选
- 季节：冬季给滑雪场/冰上项目加权，夏季排除（或降低权重）
- 偏好：按指令偏好调类别比例（美食→餐饮、购物→购物、自然→景点）
- 质量：sort_score（POI 综合排序分）优先
- 跨日不重复：used_indices 排除已分配 POI

用法:
    from src.retrieval import retrieve_candidates
    cands = retrieve_candidates(pois, dist_matrix, constraints,
                                center_idx=750, used_indices=set(), n=35)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Set


def resolve_poi_index(name: str, pois: pd.DataFrame) -> Optional[int]:
    """指令中的 POI 名（如"中央大街"）→ POI 库索引（contains 匹配）."""
    exact = pois[pois["name"] == name]
    if len(exact) > 0:
        return int(exact.index[0])
    contains = pois[pois["name"].str.contains(name, na=False, regex=False)]
    if len(contains) > 0:
        return int(contains.index[0])
    return None


def _season_weight(pois: pd.DataFrame, idx: int, season: Optional[str]) -> float:
    """季节权重：冬季→滑雪场/冰上项目加权；夏季→降权."""
    if season is None:
        return 1.0
    sw = pois.loc[idx, "season_winter"] if idx in pois.index else 0.5
    ss = pois.loc[idx, "season_summer"] if idx in pois.index else 0.5
    diff = float(sw) - float(ss)  # >0.3 表示冬季特色
    if season == "winter":
        return 1.0 + diff  # 冬季特色 +1 倍权重
    else:
        return max(0.2, 1.0 - diff)  # 夏季排除滑雪场


def _preference_mask(pois: pd.DataFrame, idx: int, preferences: List[str]) -> float:
    """偏好权重：喜欢美食→餐饮类加权；喜欢景点→景点类加权."""
    atype = pois.loc[idx, "activity_type"] if idx in pois.index else -1
    cat = pois.loc[idx, "category"] if idx in pois.index else ""
    w = 1.0
    if "food" in preferences and atype == 1:
        w *= 1.5
    elif "attraction" in preferences and atype == 0:
        w *= 1.3
    elif "shopping" in preferences and atype == 4:
        w *= 1.4
    elif "culture" in preferences and atype == 0 and cat in ("博物馆", "文化场馆"):
        w *= 1.4
    elif "nature" in preferences and atype == 0 and cat in ("公园", "湿地", "风景区"):
        w *= 1.4
    elif "family" in preferences and atype == 0 and cat in ("主题乐园", "动物园"):
        w *= 1.4
    return w


def retrieve_candidates(
    pois: pd.DataFrame,
    dist_matrix: np.ndarray,
    constraints=None,
    center_idx: Optional[int] = None,
    used_indices: Optional[Set[int]] = None,
    n: int = 35,
    prefer_core: bool = True,
    type_budget: Optional[dict] = None,
) -> List[int]:
    """检索某一天的候选 POI 索引列表（按推荐度排序）.

    Args:
        pois: POI 元数据表（含 name/lat/lng/category/activity_type/sort_score/season_*）
        dist_matrix: [n_pois, n_pois] 距离矩阵（km）
        constraints: 解析出的 Constraints（含 core_pois/start/preferences/season）
        center_idx: 检索中心（出发地或核心景点索引）；None 时用核心景点或默认
        used_indices: 已分配 POI（跨日不重复）
        n: 候选数量
        prefer_core: 核心景点/出发地强制进入候选（前排）
        type_budget: 类别配额 {activity_type: 数量}，如 {0: 4, 1: 2, 2: 1, 4: 1}
                     保证每天有合理的景点/餐饮/住宿/购物构成

    Returns:
        候选 POI 索引列表（推荐度降序）
    """
    used = set(used_indices or set())
    constraints = constraints or type("C", (), {"core_pois": [], "start": None,
                                                "preferences": [], "season": None})()

    # 检索中心
    center = center_idx
    if center is None:
        for name in (constraints.core_pois + ([constraints.start] if constraints.start else [])):
            idx = resolve_poi_index(name, pois)
            if idx is not None:
                center = idx
                break
    if center is None:
        idx = resolve_poi_index("中央大街", pois)
        center = idx

    # 全部候选：按距离 + 季节 + 偏好 + 质量 加权
    all_idx = list(pois.index)
    dists = dist_matrix[center] if center is not None else np.zeros(len(pois))
    scored = []
    for i in all_idx:
        i = int(i)
        if i in used:
            continue
        d = float(dists[i])
        sw = _season_weight(pois, i, constraints.season)
        pw = _preference_mask(pois, i, constraints.preferences)
        quality = float(pois.loc[i, "sort_score"]) if i in pois.index else 0.5
        dist_score = max(0.1, 1.0 - (max(0, d - 3) / 3) * 0.15)
        score = dist_score * 0.5 + quality * 0.3 + min(sw, 1.5) * 0.1 + min(pw, 1.5) * 0.1
        atype = int(pois.loc[i, "activity_type"]) if i in pois.index else -1
        scored.append((score, i, atype))

    scored.sort(key=lambda x: -x[0])

    # 按类别配额分配（保证景点/餐饮/购物/住宿混合）
    if type_budget:
        candidates = []
        budget = dict(type_budget)
        # 先满足各类型配额
        for score, i, atype in scored:
            if len(candidates) >= n:
                break
            if atype in budget and budget[atype] > 0:
                candidates.append(i)
                budget[atype] -= 1
        # 配额未满足的类型用剩余排序池补足（优先景点）
        if len(candidates) < n:
            need = n - len(candidates)
            fill_types = [t for t, v in budget.items() if v > 0]
            for score, i, atype in scored:
                if len(candidates) >= n:
                    break
                if i in candidates:
                    continue
                if atype in budget and budget[atype] <= 0:
                    continue
                if atype in fill_types or atype == 0:  # 补配额或补景点
                    candidates.append(i)
                    if atype in budget:
                        budget[atype] -= 1
    else:
        candidates = [i for _, i, _ in scored[:n]]

    # 核心景点/出发地强制入候选（去重后排前）
    if prefer_core:
        forced = []
        for name in (constraints.core_pois + ([constraints.start] if constraints.start else [])):
            idx = resolve_poi_index(name, pois)
            if idx is not None and idx not in used and idx not in candidates:
                forced.append(idx)
        candidates = forced + [c for c in candidates if c not in forced]

    return candidates[:n]
