"""指令约束抽取器：从自然语言旅游指令提取结构化约束（v5 打分用）.

规则为主 + LLM 兜底（use_llm=True 时，评估/Demo 可开）：
- 训练路径（GRPO/DPO 奖励）固定 use_llm=False —— 奖励必须确定可复现，
  LLM 解析的随机性会污染梯度信号；解析失败的指令降级为"无约束"或跳过。
- 评估 / Demo 可传 use_llm=True 兜底规则覆盖不到的复杂措辞。

用法:
    from src.constraint_parser import parse_constraints
    c = parse_constraints("帮我规划一条哈尔滨三日游路线，冬季出行，预算约3000元，以冰雪大世界和中央大街为核心。")
    # c.days == 3, c.season == "winter", c.budget_max == 3000.0,
    # c.core_pois == ["冰雪大世界", "中央大街"], c.confidence == "high"
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional

# 汉字天数映射
_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


@dataclass
class Constraints:
    days: Optional[float] = None           # 天数（支持小数，如 3.5 = 3 天半）
    budget_min: Optional[float] = None      # 预算下界
    budget_max: Optional[float] = None      # 预算上界（"预算充足"→None 无上限）
    start: Optional[str] = None             # 出发地 POI 名
    core_pois: List[str] = field(default_factory=list)   # 核心景点名
    preferences: List[str] = field(default_factory=list)  # 归一化偏好键
    pace: Optional[str] = None              # slow / normal / fast
    season: Optional[str] = None            # winter / summer / None
    half_days: List[int] = field(default_factory=list)  # 半日索引（1-based），如第2天下午走 → [2]
    confidence: str = "high"                # high（规则完整解析）/ low

    def to_dict(self):
        return {
            "days": self.days, "budget_min": self.budget_min,
            "budget_max": self.budget_max, "start": self.start,
            "core_pois": self.core_pois, "preferences": self.preferences,
            "pace": self.pace, "season": self.season,
            "half_days": self.half_days,
            "confidence": self.confidence,
        }


# ---- 正则规则 ----
_DAYS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[日天]")
_CN_DAYS_RE = re.compile(r"([一二两三四五六七八九十]+)\s*[日天]")
# 半日需求："3天半" 的小数部分 / "第2天下午5点走" / "第二天中午走" / "最后一天下午离开"
_HALF_EXTRA_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[日天]\s*(?:半|余)")
_HALF_DAY_RE = re.compile(r"(?:第|到)?([一二两三四五六七八九十\d]+)\s*[日天]?[的]?(?:下午|中午|傍晚)\s*[0-9]*(?:点)?\s*(?:走|离开|出发|回|结束)|(?:最后|末尾)\s*[一二两三四五六七八九十\d]?\s*[日天][的]?(?:下午|中午)\s*(?:走|离开)")
_CN_NUM_FULL = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_BUDGET_RE = re.compile(r"预算[约]?(\d+(?:\.\d+)?)\s*元")
_START_RE = re.compile(r"从(.{2,12}?)出发|以(.{2,12}?)为起点|第一站想去(.{2,12}?)")
_CORE_RE = re.compile(r"以(.{2,15}?)为核心|重点(?:游览)?(.{2,15}?)(?:。|，|,|$)|必须去(.{2,15}?)(?:。|，|,|$)|一定要去(.{2,15}?)(?:。|，|,|$)")
_PREF_RE = re.compile(r"喜欢(.{2,24}?)(?:。|，|,|$)|偏好(.{2,24}?)(?:。|，|,|$)")

_PACE_SLOW = ["慢节奏", "节奏要慢", "不要太赶", "走不快", "适合老人", "慢一点", "父母", "老年人", "长辈", "舒缓"]
_PACE_FAST = ["赶时间", "紧凑", "尽快", "半天"]
_SEASON_WINTER = ["冰雪季", "冬季", "冬天", "12月", "1月", "2月", "雪乡", "冰灯", "冰雪"]
_SEASON_SUMMER = ["夏季", "暑假", "夏天", "7月", "8月", "避暑"]

# 偏好关键词 → 归一化键
_PREF_MAP = {
    "food": ["美食", "吃", "餐饮", "吃饭", "餐厅"],
    "shopping": ["购物", "买", "逛街", "商场"],
    "attraction": ["景点", "名胜", "打卡", "经典"],
    "culture": ["文化", "博物馆", "历史", "建筑", "展"],
    "nature": ["自然", "风景", "山水", "湿地", "公园"],
    "family": ["亲子", "儿童", "孩子", "带娃"],
}


def _cn_to_int(s: str) -> int:
    total = 0
    for ch in s:
        total = total * 10 + _CN_NUM.get(ch, 0)
    return total


def _norm_poi(raw: str) -> str:
    """清洗抽取出的 POI 名（去掉停顿词/标点尾部）."""
    raw = re.sub(r"[。，,、；;！!？? ]+$", "", raw.strip())
    raw = re.sub(r"^去|^到|^在|^去趟", "", raw)
    return raw


def parse_constraints(instruction: str, use_llm: bool = False) -> Constraints:
    """从指令抽取结构化约束（规则主路径）.

    Args:
        instruction: 用户自然语言指令
        use_llm: 规则解析 confidence=low 时是否用 LLM 兜底
                 （训练奖励固定 False；评估/Demo 可开）

    Returns:
        Constraints：解析失败的关键字段为 None，调用方按"无约束"处理
    """
    c = Constraints()

    # --- 天数（支持小数：3天半 → 3.5） ---
    m = _DAYS_RE.search(instruction)
    if m:
        c.days = float(m.group(1))
        # "3天半" → 0.5
        if _HALF_EXTRA_RE.search(instruction):
            c.days += 0.5
    else:
        m2 = _CN_DAYS_RE.search(instruction)
        if m2:
            c.days = float(_cn_to_int(m2.group(1)))
            if "半" in instruction:
                c.days += 0.5
    # 半日需求："第2天下午5点走" → 标记该日为半天
    last_day_half = False
    for m in _HALF_DAY_RE.finditer(instruction):
        # "最后/末尾一天" → 指总天数的最后一天
        if m.group(0).startswith("最后") or m.group(0).startswith("末尾"):
            last_day_half = True
            continue
        day_str = m.group(1)
        if day_str:
            if day_str.isdigit():
                c.half_days.append(int(day_str))
            else:
                c.half_days.append(_CN_NUM_FULL.get(day_str, 0))
    if last_day_half and c.days is not None and c.days >= 1:
        c.half_days.append(int(c.days))  # 最后一天 = days 整数部分
    if "半天" in instruction:
        # "只有半天" → 半天总数 0.5，标记为第 1 天半天
        if c.days is None or c.days >= 1:
            c.days = 0.5
        elif c.days == 0.5 and not c.half_days:
            c.half_days.append(1)
    c.half_days = list(dict.fromkeys(c.half_days))

    # --- 预算 ---
    if "预算充足" in instruction or "预算无上限" in instruction or "不差钱" in instruction:
        c.budget_min = None
        c.budget_max = None  # 无上限
    else:
        m = _BUDGET_RE.search(instruction)
        if m:
            c.budget_min = float(m.group(1))
            c.budget_max = float(m.group(1))

    # --- 出发地 ---
    m = _START_RE.search(instruction)
    if m:
        c.start = _norm_poi(next(g for g in m.groups() if g))

    # --- 核心景点 ---
    for m in _CORE_RE.finditer(instruction):
        g = next((x for x in m.groups() if x), None)
        if g:
            # 拆分"和/与/及/、/，"连接的多景点（"以冰雪大世界和中央大街为核心"）
            for part in re.split(r"[和与及、，,]+", g):
                name = _norm_poi(part)
                if name:
                    c.core_pois.append(name)
    # 去重保留顺序
    c.core_pois = list(dict.fromkeys(c.core_pois))

    # --- 偏好 ---
    raw_prefs = []
    for m in _PREF_RE.finditer(instruction):
        g = next((x for x in m.groups() if x), None)
        if g:
            raw_prefs.append(g)
    for rp in raw_prefs:
        for key, kws in _PREF_MAP.items():
            if any(kw in rp for kw in kws) and key not in c.preferences:
                c.preferences.append(key)
    # 未用"喜欢"句式但直接含关键词（如"美食爱好者"）
    if not c.preferences:
        for key, kws in _PREF_MAP.items():
            if any(kw in instruction for kw in kws):
                c.preferences.append(key)
                break

    # --- 节奏 ---
    if any(k in instruction for k in _PACE_SLOW):
        c.pace = "slow"
    elif any(k in instruction for k in _PACE_FAST):
        c.pace = "fast"
    elif "节奏适中" in instruction:
        c.pace = "normal"

    # --- 季节 ---
    if any(k in instruction for k in _SEASON_WINTER):
        c.season = "winter"
    elif any(k in instruction for k in _SEASON_SUMMER):
        c.season = "summer"

    # --- confidence：天数/季节/预算都解析失败才 low ---
    if c.days is None and c.season is None and c.budget_max is None:
        c.confidence = "low"
        if use_llm:
            _llm_fallback(c, instruction)

    return c


def _llm_fallback(c: Constraints, instruction: str):
    """LLM 兜底：规则解析失败时用 Qwen 补全关键字段.

    仅评估/Demo 调用（低频）；训练路径 use_llm=False 不会走到这里。
    """
    # 简单实现：复用本地 Qwen3-4B 把指令转结构化 JSON。
    # 当前规则覆盖已足够（天数/季节/预算正则基本命中），此兜底保持最小实现，
    # 需要的场景再扩展。
    pass  # 预留：加载 Qwen → 生成 {"days":..., "season":...} → 填回 c
