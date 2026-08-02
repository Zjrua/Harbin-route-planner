"""逐日拆分规划器（Phase 1 核心）.

把"一次生成整条路线"改造成"逐日生成"：
1. 解析指令 → 天数（含小数/半日）+ 约束
2. 把总行程拆成逐日计划（每天候选 POI 集，跨日不重复）
3. 每天用 RAG 候选集生成（模型只能从真实候选里选，治编造名）
4. 汇总每日路线 + v5 逐日打分

用法:
    from src.itinerary_planner import plan_itinerary
    result = plan_itinerary(model, tokenizer, instruction, d)
"""

import re
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from src.constraint_parser import parse_constraints
from src.retrieval import retrieve_candidates, resolve_poi_index
from src.scoring import composite_score_v5, infer_days_from_len

SYSTEM_PROMPT = (
    "你是一位哈尔滨旅游规划专家。你会收到候选景点列表（真实存在的哈尔滨 POI），"
    "只能从候选列表中选择景点规划路线，绝对禁止使用候选之外的名称。"
    "路线用 POI 名称以 → 连接。路线不得重复景点，禁止中途折返。"
)


def split_days(constraints) -> List[dict]:
    """把总天数拆成逐日计划.

    constraints.days 支持小数（3.5 = 3 天半）。
    half_days 标记第几天是半天（如"第2天下午5点走" → half_days=[2]）。

    Returns:
        [{"day": 1, "half": False, "n_stops_target": 7}, ...]
    """
    total = constraints.days or 1.0
    full_days = int(total)
    half_set = set(constraints.half_days or [])

    plan = []
    for d in range(1, full_days + 1):
        half = d in half_set
        # 半天目标 3-4 站，全天 6-8 站
        target = 3 if half else 6
        plan.append({"day": d, "half": half, "n_stops_target": target})
    # 半日需求超出整数天数（如"3天半"且 half_days 标记第4天）→ 补半日
    if half_set and len(plan) < max(half_set):
        for d in range(len(plan) + 1, max(half_set) + 1):
            plan.append({"day": d, "half": True, "n_stops_target": 3})
    # 小数部分（3.5 → 额外半天）
    if total > full_days and not any(p["half"] for p in plan if p["day"] > full_days):
        if not any(p["half"] and p["day"] <= full_days for p in plan):
            plan.append({"day": full_days + 1, "half": True, "n_stops_target": 3})
    return plan


def _candidate_names(pois: pd.DataFrame, idxs: List[int]) -> str:
    names = [str(pois.loc[i, "name"]) for i in idxs if i in pois.index]
    return "、".join(names)


def plan_itinerary(model, tokenizer, instruction: str, d: dict) -> dict:
    """逐日规划主入口.

    Args:
        model/tokenizer: Qwen 模型
        instruction: 用户指令
        d: 数据字典 {pois, dist_matrix, time_matrix, ratings, categories,
                    activity_types, season_winter, season_summer}

    Returns:
        {"ok": bool, "days": [{"day":1, "half":bool, "pois":[...], "score_detail":...}],
         "warnings": [...], "raw_per_day": {...}}
    """
    pois = d["pois"]
    constraints = parse_constraints(instruction, use_llm=False)
    plan = split_days(constraints)
    warnings = []
    if constraints.half_days:
        warnings.append(f"已按半日拆分：第 {constraints.half_days} 天为半天")

    # 起始检索中心
    center_idx = None
    for name in (constraints.core_pois + ([constraints.start] if constraints.start else [])):
        idx = resolve_poi_index(name, pois)
        if idx is not None:
            center_idx = idx
            break

    used = set()  # 跨日不重复
    days_result = []
    all_raw = []
    for i, day_info in enumerate(plan):
        # 每天候选：全天 8 个（景点4/餐饮2/购物1/住宿1），半天 5 个（景点2/餐饮1/购物1/住宿1）
        is_half = day_info["half"]
        budget = {0: 2, 1: 1, 4: 1, 2: 1} if is_half else {0: 4, 1: 2, 4: 1, 2: 1}
        n_cand = 5 if is_half else 8
        candidates = retrieve_candidates(
            pois, d["dist_matrix"], constraints,
            center_idx=center_idx, used_indices=used,
            n=n_cand, type_budget=budget,
        )
        cand_names = [str(pois.loc[i, "name"]) for i in candidates if i in pois.index]
        used.update(candidates)

        # 构造当天 prompt：候选编号 + 半天/全天提示
        half_note = "（半天，安排 2-3 个即可）" if is_half else ""
        cand_lines = "\n".join(f"{j+1}.{n}" for j, n in enumerate(cand_names))
        prompt_instr = (
            f"请为第{day_info['day']}天安排路线{half_note}。候选景点：\n{cand_lines}\n"
            f"输出{day_info['n_stops_target']}个左右景点的编号路线，编号用 → 连接"
            f"（如 1→3→5→2→8→6），只能从上述 {len(cand_names)} 个候选中选，不要输出编号以外的任何文字。"
        )
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_instr}]
        # 手动 <|im_start|> 格式（Qwen3/Qwen3.5 通用）：
        # apply_chat_template 会注入 <think>，破坏候选编号格式（Qwen3.5 尤其明显）
        text = (
            "<|im_start|>system\n" + SYSTEM_PROMPT + "\n<|im_end|>\n"
            f"<|im_start|>user\n{prompt_instr}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=100, do_sample=True,
                                 temperature=0.4, top_p=0.9, no_repeat_ngram_size=4)
        raw = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        all_raw.append(raw)

        # 解析编号（丢弃越界 0 或 >len(cand_names) 的编号，即模型灌水部分）
        picked_nums = []
        for token in re.findall(r"\d+", raw):
            v = int(token)
            if 1 <= v <= len(cand_names) and v not in picked_nums:
                picked_nums.append(v)
        # 编号 → 候选名
        day_names = [cand_names[v - 1] for v in picked_nums]
        # 不足目标数：用候选池按检索序补足（保持真实、够长）
        missing = day_info["n_stops_target"] - len(day_names)
        if missing > 0:
            for cn in cand_names:
                if len(day_names) >= day_info["n_stops_target"]:
                    break
                if cn not in day_names:
                    day_names.append(cn)
        days_result.append({"day": day_info["day"], "half": is_half,
                            "pois": day_names, "candidates": cand_names})

    # === 汇总 + 逐日打分 ===
    return _assemble(d, constraints, days_result, warnings, all_raw)


def _parse_route_names(text: str) -> List[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?assistant", "", text, flags=re.DOTALL)
    matches = re.findall(
        r"([\u4e00-\u9fa5A-Za-z0-9·（）()]+(?:\s*(?:→|->)\s*[\u4e00-\u9fa5A-Za-z0-9·（）()]+){1,})",
        text,
    )
    if matches:
        longest = max(matches, key=len)
        return [p.strip() for p in re.split(r"\s*(?:→|->)\s*", longest) if p.strip()]
    return []


def _assemble(d, constraints, days_result, warnings, all_raw) -> dict:
    """汇总逐日结果 + 打分."""
    from src.constraint_parser import Constraints
    pois = d["pois"]
    # 逐日约束副本：天数=该日（全天1日），保留预算/偏好/季节等
    def day_constraints(day: dict):
        return Constraints(
            days=1.0,  # 单日按 1 日游评估（半天也按 1 日评估时长，站数少自然软扣）
            budget_min=constraints.budget_min, budget_max=constraints.budget_max,
            start=None,  # 出发地只约束总行程第 1 天
            core_pois=constraints.core_pois,
            preferences=constraints.preferences,
            pace=constraints.pace,
            season=constraints.season,
        )
    total_route = []
    day_scores = []
    for day in days_result:
        idxs = []
        for name in day["pois"]:
            exact = pois[pois["name"] == name]
            if len(exact) > 0:
                idxs.append(int(exact.index[0])); continue
            contains = pois[pois["name"].str.contains(name, na=False, regex=False)]
            if len(contains) > 0:
                idxs.append(int(contains.index[0]))
        day["indices"] = idxs
        total_route.extend(idxs)
        # 逐日打分（该日按 1 日游评估）
        if len(idxs) >= 3:
            result = composite_score_v5(
                idxs, d["dist_matrix"], d["time_matrix"], d["ratings"],
                d["categories"], n_days=1, activity_types=d["activity_types"],
                constraints=day_constraints(day),
                poi_names=pois["name"].tolist(),
                avg_costs=pois["avg_cost"].values,
                season_winter=pois["season_winter"].values,
                season_summer=pois["season_summer"].values,
            )
            day["score_detail"] = result if result.get("feasible") else \
                {"score": 0.0, "feasible": False, "reason": result.get("reason"),
                 "metrics": result.get("metrics", {})}
        else:
            day["score_detail"] = {"score": 0.0, "feasible": False, "reason": "too_short"}

    # 总体路线 = 每日拼接
    n_days_total = infer_days_from_len(len(total_route))
    overall = composite_score_v5(
        total_route, d["dist_matrix"], d["time_matrix"], d["ratings"],
        d["categories"], n_days=n_days_total, activity_types=d["activity_types"],
        constraints=constraints,
        poi_names=pois["name"].tolist(),
        avg_costs=pois["avg_cost"].values,
        season_winter=pois["season_winter"].values,
        season_summer=pois["season_summer"].values,
    ) if len(total_route) >= 3 else {"score": 0.0, "feasible": False, "reason": "too_short"}

    return {
        "ok": True,
        "days": [{"day": x["day"], "half": x["half"], "pois": x["pois"],
                  "score_detail": x.get("score_detail")} for x in days_result],
        "overall": overall,
        "total_pois": len(total_route),
        "warnings": warnings,
        "raw_per_day": all_raw,
    }
