"""POI 语义标注：全量 48,961 个 POI 打"适合人群 + 质量门控"标签.

背景（方法论审视结论）：
- 系统最大短板是检索层数据质量——"带父母慢节奏"路线里混进健身房/酒吧/烟酒超市
- 10K 上限是矩阵内存逼的（全量矩阵 9.6GB/张），但语义标注不需要矩阵
- 所以标注直接覆盖全量 48,961 个 POI

标签（新列）：
- is_tourism: bool    是否旅游 POI（排除健身房/台球/酒吧/烟酒超市/彩票等）
- suitable_elderly: bool  适合老人（公园/博物馆/安静景点/老年友好）
- suitable_family:  bool  适合亲子（乐园/动物园/亲子/儿童）
- suitable_youth:   bool  适合年轻人（夜生活/运动/极限/网红）
- unsuitable_tags:  str   排除原因（逗号分隔，空=无）

方法：
1. 规则关键词（name + category）——覆盖大部分，零成本
2. LLM 批量标注（Qwen）——只对"景点"类别（~2.3K）补语义，
   因为景点是最难用规则判断、也最影响路线的

用法:
    ./.venv/Scripts/python.exe scripts/label_poi_semantics.py [--llm-batch 500]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAW_PATH = "data/raw/merged_pois.csv"
OUT_PATH = "data/raw/merged_pois_labeled.csv"

# ==== 规则关键词 ====

# 非旅游 POI（质量门控）：这些设施不应出现在旅游路线里
NON_TOURISM_KW = [
    "健身", "台球", "拳", "搏击", "散打", "酒吧", "KTV", "ktv", "夜总会",
    "烟酒", "彩票", "网咖", "电竞", "麻将", "棋牌", "洗浴", "按摩", "足疗",
    "理发", "美发", "美甲", "洗车", "修车", "加油站", "汽配", "五金",
    "建材", "家具", "物流", "快递", "药店", "诊所", "医院", "牙科",
    "口腔", "宠物医院", "幼儿园", "小学", "中学", "驾校",
    "银行", "ATM", "保险", "证券", "彩票", "洗衣", "干洗", "家政",
    "房产", "中介", "会计", "律所", "办公", "写字楼", "园区",
    # 英文关键词（规则抓不到的中文名难判定，先补常见英文设施）
    "whisky", "whiskey", "bar", "pub", "club", "gym", "fitness",
    "sport", "ktv", "spa", "sauna", "massage", "salon", "casino",
    "bank", "mall", "store", "shop", "market", "school", "hospital",
]

# 适合老人的关键词
ELDERLY_KW = [
    "公园", "博物馆", "纪念馆", "旧址", "教堂", "寺庙", "道观", "塔",
    "湿地", "森林", "山庄", "景区", "风景", "文化", "历史", "展馆",
    "广场", "老街", "历史建筑", "步行街", "江", "湖", "山",
]

# 适合亲子的关键词
FAMILY_KW = [
    "儿童", "亲子", "乐园", "游乐园", "动物园", "海洋馆", "水族馆",
    "科技馆", "科学", "恐龙", "冰雪大世界", "雪雕", "冰灯", "滑雪",
    "主题公园", "方特", "万达乐园", "淘气堡",
]

# 适合年轻人的关键词
YOUTH_KW = [
    "夜", "livehouse", "Livehouse", "club", "CLUB", "电竞", "网咖",
    "健身", "攀岩", "滑板", "蹦床", "密室", "剧本杀", "射箭", "卡丁车",
    "桌游", "狼人杀", "酒吧", "精酿",
]

# 排除词（出现在名称里可能误判为景点，但实际是低质量设施）
SUSPICIOUS_KW = ["超市", "便利店", "烟酒", "五金", "药房", "快餐", "小吃",
                 "烧烤", "火锅", "面馆", "餐厅", "酒店", "宾馆", "旅馆",
                 "公寓", "民宿", "浴", "SPA", "会所"]


# 写死必须排除的设施词：任何类别都排除（名称含这些词几乎都是真设施）
GLOBAL_HARD = [
    "洗浴", "汤泉", "汗蒸", "按摩", "足疗", "电竞", "网咖", "夜总会",
    "烟酒", "彩票", "驾校", "诊所", "药店", "药房", "KTV",
]
# 仅对"景点/其他"类排除（住宿/餐饮/购物里的可能是地址后缀，如"XX中学店"）
ATTR_HARD = ["银行", "支行", "医院", "小学", "中学", "幼儿园", "加油站", "银行街"]


def rule_label(name: str, category: str) -> dict:
    """规则标注单个 POI.

    按类别分流：
    - 写死排除词（银行/医院/学校/洗浴/电竞/烟酒）任何类别都排除
    - 餐饮/住宿/购物：旅游配套，默认 is_tourism=True（酒吧/夜总会除外）
    - 景点类：重点，需语义标注（排除健身房/台球/酒吧等非旅游设施）
    - 交通：非旅游
    - 其他：排除词判断
    """
    name_l = name.lower()
    # 全局硬排除（任何类别）
    global_bad = [kw for kw in GLOBAL_HARD if kw.lower() in name_l]
    if global_bad:
        return {"is_tourism": False, "suitable_elderly": False,
                "suitable_family": False, "suitable_youth": False,
                "unsuitable_tags": ",".join(global_bad[:3])}

    if category in ("餐饮", "住宿", "购物"):
        # 旅游配套：酒吧/夜总会/whisky 明确排除（不适合"带父母/慢节奏"主线）
        bad = [kw for kw in NON_TOURISM_KW
               if kw.lower() in name_l and kw in ("酒吧", "bar", "pub", "whisky", "夜总会", "club")]
        is_tourism = not bad
        return {"is_tourism": is_tourism, "suitable_elderly": is_tourism,
                "suitable_family": is_tourism, "suitable_youth": False,
                "unsuitable_tags": ",".join(bad[:3])}
    if category == "交通":
        return {"is_tourism": False, "suitable_elderly": False,
                "suitable_family": False, "suitable_youth": False, "unsuitable_tags": "交通设施"}

    # 景点类 / 其他：属性硬排除 + 非旅游设施排除（is_facility 防地址尾部词误杀）
    attr_bad = [kw for kw in ATTR_HARD if kw.lower() in name_l]
    if attr_bad:
        return {"is_tourism": False, "suitable_elderly": False,
                "suitable_family": False, "suitable_youth": False,
                "unsuitable_tags": ",".join(attr_bad[:3])}

    def is_facility(kw: str) -> bool:
        pos = name_l.find(kw)
        if pos < 0:
            return False
        after = name_l[pos + len(kw):]
        if after and any(a in after for a in ["店", "广场", "大厦", "路", "街", "站", "地铁", "馆", "中心"]):
            return False
        return True

    non_tourism = [kw for kw in NON_TOURISM_KW if is_facility(kw)]
    if non_tourism:
        return {"is_tourism": False, "suitable_elderly": False,
                "suitable_family": False, "suitable_youth": False,
                "unsuitable_tags": ",".join(non_tourism[:3])}
    # 景点类语义：关键词判断
    elderly = any(kw in name for kw in ELDERLY_KW) if category == "景点" else True
    family = any(kw in name for kw in FAMILY_KW) if category == "景点" else True
    youth = any(kw in name for kw in YOUTH_KW) if category == "景点" else False
    return {"is_tourism": True, "suitable_elderly": elderly,
            "suitable_family": family, "suitable_youth": youth,
            "unsuitable_tags": ""}


def llm_batch_label(pois: pd.DataFrame, model, tokenizer, batch_size=100):
    """LLM 批量标注景点类的适合人群（Qwen 只补语义，规则已覆盖大部分）.

    输入候选名列表 → 输出 JSON [{name, elderly, family, youth}]
    """
    results = {}
    prompt_tpl = (
        "下面是哈尔滨的旅游景点名称列表。请为每个景点标注适合人群：\n"
        "elderly(适合老人:1/0), family(适合亲子:1/0), youth(适合年轻人:1/0)。\n"
        "只输出 JSON 数组 [{{\"name\":\"...\",\"elderly\":1,\"family\":0,\"youth\":0}},...]\n"
        "景点：{names}"
    )
    for start in range(0, len(pois), batch_size):
        batch = pois.iloc[start:start + batch_size]
        names = "、".join(batch["name"].tolist())
        prompt = prompt_tpl.format(names=names[:1500])  # 控制 token
        text = (
            "<|im_start|>system\n你是哈尔滨旅游 POI 语义标注助手，只输出 JSON。\n<|im_end|>\n"
            f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=600, do_sample=False)
        resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\[.*\]", resp, re.DOTALL)
        if m:
            try:
                for item in json.loads(m.group(0)):
                    results[item["name"]] = item
            except Exception:
                pass
        print(f"  LLM 批 {start//batch_size+1}: {len(batch)} 个，成功 {len(results)}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-batch", type=int, default=0,
                        help="用 Qwen 标注景点类（每批数量；0=纯规则，最快）")
    args = parser.parse_args()

    print(f"加载全量 POI: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, encoding="utf-8")
    print(f"总 POI: {len(df)}")

    # 规则标注
    labels = df.apply(lambda r: rule_label(str(r["name"]), str(r.get("category", ""))), axis=1)
    for col in ["is_tourism", "suitable_elderly", "suitable_family", "suitable_youth", "unsuitable_tags"]:
        df[col] = [l[col] for l in labels]

    # LLM 补景点类语义
    if args.llm_batch > 0:
        print("用 Qwen 批量标注景点类...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        MODEL = str(Path("data/external/modelscope_cache/models/Qwen--Qwen3-4B/snapshots/master"))
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.eval()

        att = df[df["category"] == "景点"].copy()
        llm_res = llm_batch_label(att, model, tok, args.llm_batch)
        # 合并 LLM 结果
        hit = 0
        for idx, row in att.iterrows():
            r = llm_res.get(row["name"])
            if r:
                df.at[idx, "suitable_elderly"] = bool(r.get("elderly", df.at[idx, "suitable_elderly"]))
                df.at[idx, "suitable_family"] = bool(r.get("family", df.at[idx, "suitable_family"]))
                df.at[idx, "suitable_youth"] = bool(r.get("youth", df.at[idx, "suitable_youth"]))
                hit += 1
        print(f"LLM 标注命中: {hit}/{len(att)}")

    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"保存至: {OUT_PATH}")

    # 统计
    print("\n=== 标注统计 ===")
    print(f"旅游 POI: {df['is_tourism'].sum()}/{len(df)} ({df['is_tourism'].mean()*100:.0f}%)")
    print(f"适合老人: {df['suitable_elderly'].sum()}")
    print(f"适合亲子: {df['suitable_family'].sum()}")
    print(f"适合年轻人: {df['suitable_youth'].sum()}")
    print(f"排除的非旅游 POI: {(~df['is_tourism']).sum()}")
    # 样例
    print("\n排除样例:", df[~df['is_tourism']]["name"].head(8).tolist())


if __name__ == "__main__":
    main()
