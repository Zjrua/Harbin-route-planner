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
from pathlib import Path
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

# 语义标注缓存（全量 48,961 POI，由 label_poi_semantics.py 生成）
_SEMANTIC = None


def load_semantic():
    """加载语义标注（name → {is_tourism, suitable_elderly, suitable_family}）."""
    global _SEMANTIC
    if _SEMANTIC is not None:
        return _SEMANTIC
    label_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "merged_pois_labeled.csv"
    _SEMANTIC = {}
    if label_path.exists():
        try:
            labels = pd.read_csv(label_path, encoding="utf-8")
            for _, r in labels.iterrows():
                _SEMANTIC[str(r["name"])] = {
                    "is_tourism": bool(r.get("is_tourism", True)),
                    "suitable_elderly": bool(r.get("suitable_elderly", True)),
                    "suitable_family": bool(r.get("suitable_family", True)),
                }
        except Exception as e:
            print(f"语义标注加载失败: {e}")
            _SEMANTIC = {}
    return _SEMANTIC


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


def _llm_cand_logprobs(model, tok, picked_names, remaining_names, pois):
    """LLM 对剩余候选的编号首 token log 概率（确定性，无采样）.

    编号 1-9 单 token；10 为两 token（"1"+"0"），需第二段前向。
    返回与 remaining_names 等长的 log 概率数组（未归一）。
    """
    sel_lines = "\n".join(f"{j+1}.{n}" for j, n in enumerate(remaining_names))
    ctx = f"已游览路线：{' → '.join(picked_names)}。\n" if picked_names else ""
    instr = (f"{ctx}候选地点：\n{sel_lines}\n"
             f"选出下一个最应该去的地点，只输出其编号（1-{len(remaining_names)}）。")
    text = ("<|im_start|>system\n你是一位哈尔滨旅游规划专家。根据已游览的路线和"
            "候选列表，选出下一个最应该去的地点，只输出其编号。\n<|im_end|>\n"
            f"<|im_start|>user\n{instr}\n<|im_end|>\n<|im_start|>assistant\n")
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1]
    lp = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for j in range(len(remaining_names)):
        num = str(j + 1)
        ids = tok(num, add_special_tokens=False)["input_ids"]
        v = float(lp[ids[0]])
        if len(ids) > 1:  # 两位编号：条件化第一段后再取
            inp2 = tok(text + num[0], return_tensors="pt").to(model.device)
            with torch.no_grad():
                lg2 = model(**inp2).logits[0, -1]
            lp2 = torch.log_softmax(lg2.float(), dim=-1)
            v += float(lp2[ids[1]])
        out.append(v)
    return np.array(out)


def _prior_greedy_select(bm, candidates, cand_names, d, n_target,
                         center_idx, pois, model=None, tok=None,
                         use_llm=False, lam=0.55):
    """行为先验贪心选择（markov）/ log-linear 融合贪心选择（hybrid）.

    每步对剩余候选打分：log f(d) + log P2(区域|两步状态) + log T(类型)；
    hybrid 在此之上与 LLM 首 token 概率做池内 softmax 后的 log-linear 融合
    （与 run_fusion_eval.fuse 同构）。逐步更新两步状态。
    """
    cur = center_idx if center_idx is not None else candidates[0]
    prev = None
    remaining = list(candidates)
    picked = []
    while remaining and len(picked) < n_target:
        scores = []
        for c in remaining:
            dd = float(bm.dist[cur][c])
            lp = float(bm.decay.log_score(np.array([dd]))[0])
            if prev is not None:
                lp += float(np.log(bm.mk.probs()[
                    bm.region_of_poi[prev], bm.region_of_poi[cur],
                    bm.region_of_poi[c]] + 1e-12))
            ta, tb = int(bm.type_of[cur]), int(bm.type_of[c])
            lp += float(np.log(bm.T_type[ta, tb] + 1e-12))
            scores.append(lp)
        scores = np.array(scores)
        if use_llm:
            picked_names = [str(pois.loc[i, "name"]) for i in picked]
            rem_names = [str(pois.loc[i, "name"]) for i in remaining]
            lpl = _llm_cand_logprobs(model, tok, picked_names, rem_names, pois)
            # 池内归一（与 fuse() 同构：先 softmax 再对数池化）
            p_prior = np.exp(scores - scores.max())
            p_prior /= p_prior.sum()
            p_llm = np.exp(lpl - lpl.max())
            p_llm /= p_llm.sum()
            fused = lam * np.log(p_prior + 1e-12) + (1 - lam) * np.log(p_llm + 1e-12)
            best_i = int(np.argmax(fused))
        else:
            best_i = int(np.argmax(scores))
        picked.append(remaining.pop(best_i))
        prev, cur = cur, picked[-1]
    return [str(pois.loc[i, "name"]) for i in picked if i in pois.index]


def plan_itinerary(model, tokenizer, instruction: str, d: dict,
                   selector: str = "llm") -> dict:
    """逐日规划主入口.

    Args:
        model/tokenizer: Qwen 模型（selector="llm" 时必需，规则模式传 None）
        instruction: 用户指令
        d: 数据字典 {pois, dist_matrix, time_matrix, ratings, categories,
                    activity_types, season_winter, season_summer}
        selector: 候选选择方式
            - "llm"        模型候选编号输出（默认）
            - "rule_score" 规则基线：按检索分数直接取前 N（无 LLM）
            - "rule_near"  规则基线：从中心贪心选最近候选（无 LLM）
            - "markov"     行为先验：逐步 logP2(区域|两步状态)+logf(d)+logT(类型)
                           （需 d["behavior_model"]，拟合份估计，无 LLM）
            - "hybrid"     log-linear 融合：0.55×先验 + 0.45×LLM 首 token 概率
                           （需 d["behavior_model"] + model/tokenizer，确定性打分无采样）

    Returns:
        {"ok": bool, "days": [{"day":1, "half":bool, "pois":[...], "poi_idxs":[...],
                               "candidates":[...], "score_detail":...}],
         "warnings": [...], "raw_per_day": {...}}
    """
    pois = d["pois"]
    constraints = parse_constraints(instruction, use_llm=False)
    plan = split_days(constraints)
    warnings = []
    if constraints.half_days:
        warnings.append(f"已按半日拆分：第 {constraints.half_days} 天为半天")
    # 语义标注（过滤非旅游 POI + 适合人群）
    semantic = load_semantic()

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
        # 每天候选：全天 10 个（景点4/餐饮3/购物1/住宿2），半天 6 个（景点2/餐饮2/购物1/住宿1）
        # 提高餐饮/住宿配额：SFT 数据只有 26% 的天含住宿，模型偏好景点把住宿挤掉
        is_half = day_info["half"]
        budget = {0: 2, 1: 2, 4: 1, 2: 1} if is_half else {0: 4, 1: 3, 4: 1, 2: 2}
        n_cand = 6 if is_half else 10
        candidates = retrieve_candidates(
            pois, d["dist_matrix"], constraints,
            center_idx=center_idx, used_indices=used,
            n=n_cand, type_budget=budget, semantic=semantic,
        )
        cand_names = [str(pois.loc[i, "name"]) for i in candidates if i in pois.index]
        used.update(candidates)

        # === 候选选择（模型编号 or 规则基线 or 行为先验/融合） ===
        if selector in ("markov", "hybrid"):
            bm = d["behavior_model"]
            day_names = _prior_greedy_select(
                bm, candidates, cand_names, d, day_info["n_stops_target"],
                center_idx, pois,
                model=model, tok=tokenizer, use_llm=(selector == "hybrid"))
            all_raw.append("")
        elif selector == "llm":
            # 构造当天 prompt：候选编号 + 半天/全天提示
            half_note = "（半天，安排 2-3 个即可）" if is_half else ""
            cand_lines = "\n".join(f"{j+1}.{n}" for j, n in enumerate(cand_names))
            prompt_instr = (
                f"请为第{day_info['day']}天安排路线{half_note}。候选景点：\n{cand_lines}\n"
                f"输出{day_info['n_stops_target']}个左右景点的编号路线，编号用 → 连接"
                f"（如 1→3→5→2→8→6），只能从上述 {len(cand_names)} 个候选中选，不要输出编号以外的任何文字。"
            )
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
        else:
            # 规则基线（无 LLM，确定性）
            all_raw.append("")
            if selector == "rule_near":
                # 贪心最近：从当日中心出发，每步选最近未访问候选
                cur = center_idx if center_idx is not None else candidates[0]
                remaining = list(candidates)
                picked = []
                for _ in range(day_info["n_stops_target"]):
                    if not remaining:
                        break
                    dists = [float(d["dist_matrix"][cur][c]) for c in remaining]
                    best = remaining[dists.index(min(dists))]
                    picked.append(best)
                    remaining.remove(best)
                    cur = best
                day_names = [str(pois.loc[i, "name"]) for i in picked if i in pois.index]
            else:  # rule_score：检索分数序即 candidates 顺序，直接取前 N
                day_names = cand_names[: day_info["n_stops_target"]]
        # 不足目标数：用候选池按检索序补足（保持真实、够长）
        missing = day_info["n_stops_target"] - len(day_names)
        if missing > 0:
            for cn in cand_names:
                if len(day_names) >= day_info["n_stops_target"]:
                    break
                if cn not in day_names:
                    day_names.append(cn)

        # 类型校验补足：全天路线必须含餐饮+住宿，半天至少含餐饮。
        # 模型偏好景点（SFT 数据只有 26% 天含住宿），用候选里未选的类型补上。
        atype_of = {str(pois.loc[i, "name"]): int(pois.loc[i, "activity_type"])
                    for i in candidates if i in pois.index}
        # 找一个餐饮候选（未选）补入
        def _ensure_type(need_type: int):
            for cn in cand_names:
                if cn not in day_names and atype_of.get(cn) == need_type:
                    day_names.append(cn)
                    return
        if not is_half:
            if not any(atype_of.get(n) == 1 for n in day_names):
                _ensure_type(1)  # 餐饮
            if not any(atype_of.get(n) == 2 for n in day_names):
                _ensure_type(2)  # 住宿
        else:
            if not any(atype_of.get(n) == 1 for n in day_names):
                _ensure_type(1)  # 半天至少餐饮

        # 人类旅游节奏整理（当天的 atype_of）：
        # 住宿移到末尾（入住后不再移动），半天不安排住宿，连续餐饮错开
        day_names = organize_day_rhythm(day_names, atype_of, cand_names,
                                        is_half=is_half)

        # 名字→索引（供行为评测层用；节奏整理只重排/替换候选，映射完整）
        name2idx = {str(pois.loc[i, "name"]): i
                    for i in candidates if i in pois.index}
        day_idxs = [name2idx.get(n, -1) for n in day_names]

        days_result.append({"day": day_info["day"], "half": is_half,
                            "pois": day_names, "poi_idxs": day_idxs,
                            "candidates": cand_names})

    # === 汇总 + 逐日打分 ===
    return _assemble(d, constraints, days_result, warnings, all_raw)


def organize_day_rhythm(day_names: List[str], atype_of: dict,
                        cand_names: List[str], is_half: bool = False) -> List[str]:
    """人类旅游节奏整理（纯规则，不重训）.

    规律（人类旅游节奏）：
    1. 住宿 = 一天的终点站 → 移到末尾（入住后不再移动）；半天不安排住宿
    2. 以景点开头（上午从景点/活动开始，不先吃饭）
    3. 餐饮 = 节奏调节器 → 两个餐饮之间至少隔 1 个非餐饮（避免连吃两顿）
    4. 半天：2-3 个景点 + 1 次餐饮，无住宿

    策略：
    - 住宿全部移到末尾
    - 连续餐饮（相邻或隔一个）用后续非餐饮交换，或用候选池未选景点替换
    """
    if not day_names:
        return day_names
    # 分离住宿
    hotels = [n for n in day_names if atype_of.get(n) == 2]
    rest = [n for n in day_names if atype_of.get(n) != 2]
    if is_half:
        hotels = []  # 半天不安排住宿（下午离开）

    # 以景点/活动开头（避免以餐饮开头）
    if rest and atype_of.get(rest[0]) == 1:
        # 找第一个非餐饮换到开头
        swap_j = next((j for j in range(1, len(rest)) if atype_of.get(rest[j]) != 1), None)
        if swap_j is not None:
            rest[0], rest[swap_j] = rest[swap_j], rest[0]

    # 餐饮间隔：两个餐饮之间至少隔 1 个非餐饮
    used = set(rest)
    spare = [cn for cn in cand_names if atype_of.get(cn) == 0 and cn not in used]
    i = 0
    while i < len(rest):
        if atype_of.get(rest[i]) == 1:
            # 找下一个餐饮位置
            j = next((j for j in range(i + 1, len(rest))
                      if atype_of.get(rest[j]) == 1), None)
            if j is not None and j - i < 2:  # 中间隔不足 1 个非餐饮
                # 尝试用 j 与后面非餐饮交换
                swap_j = next((k for k in range(j + 1, len(rest))
                               if atype_of.get(rest[k]) != 1), None)
                if swap_j is not None:
                    rest[j], rest[swap_j] = rest[swap_j], rest[j]
                    continue  # 重查
                # 否则用候选池景点替换 j（保持数量）
                if spare:
                    rest[j] = spare.pop(0)
                    continue
        i += 1
    # 住宿放末尾（餐饮→住宿 = 晚餐后回酒店，合理）
    # 但全天最多 1 个住宿（2 个酒店连排不合理——同一晚只需一家）
    if len(hotels) > 1:
        extra = hotels[1:]
        spare_att = [cn for cn in cand_names
                     if atype_of.get(cn) == 0 and cn not in day_names and cn not in extra]
        # 用候选池景点替换多余的酒店（保持数量）
        for i, h in enumerate(extra):
            if spare_att:
                hotels = hotels[:1] + [spare_att.pop(0)]
            else:
                hotels = hotels[:1]
    return rest + hotels


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
                  "poi_idxs": x.get("indices", []),
                  "score_detail": x.get("score_detail")} for x in days_result],
        "overall": overall,
        "total_pois": len(total_route),
        "warnings": warnings,
        "raw_per_day": all_raw,
    }
