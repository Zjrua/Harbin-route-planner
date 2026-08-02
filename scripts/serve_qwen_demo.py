"""Qwen 路线生成模型 Web Demo.

用户输入自然语言指令（天数/季节/预算）→ Qwen 生成 POI 路线 →
展示路线 + v4 硬约束打分明细 + folium 地图。

支持 SFT / DPO 两个模型切换对比。

用法:
    ./.venv/Scripts/python.exe scripts/serve_qwen_demo.py [--port 8898] [--models sft,dpo]

接口:
    GET  /           交互页面
    GET  /health     状态检查
    POST /generate   {"instruction": "...", "model": "sft"|"dpo"}
"""

import argparse
import http.server
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 复用项目内的打分与地图模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.scoring import composite_score_v3, composite_score_v5
from src.visualize import plot_route_on_map

# ==== 路径常量 ====
MODEL_PATH = str(ROOT / "data/external/modelscope_cache/models/Qwen--Qwen3-4B/snapshots/master")
SFT_LORA = str(ROOT / "output/qwen_route_lora")
DPO_LORA = str(ROOT / "output/qwen_route_dpo")
GRPO_LORA = str(ROOT / "output/qwen_route_grpo")
DATA_DIR = ROOT / "data" / "processed"

SYSTEM_PROMPT = ("你是一位哈尔滨旅游规划专家，根据用户的需求生成合理的旅游路线。"
                 "路线用 POI 名称以 → 连接。路线不得重复景点，禁止中途折返。")

# 模型实例缓存（懒加载，避免服务启动就占 10GB 显存）
_MODELS = {}


# ==== 数据加载（单例） ====
_DATA = None


def load_data():
    """加载 POI 元数据与距离/时间矩阵（模块级单例）."""
    global _DATA
    if _DATA is not None:
        return _DATA
    pois = pd.read_csv(DATA_DIR / "poi_metadata.csv", encoding="utf-8")
    dist_matrix = np.load(DATA_DIR / "distance_matrix.npy")
    time_matrix = np.load(DATA_DIR / "time_matrix.npy")
    _DATA = {
        "pois": pois,
        "dist_matrix": dist_matrix,
        "time_matrix": time_matrix,
        "ratings": pois["rating"].values,
        "categories": pois["category"].values,
        "activity_types": np.load(DATA_DIR / "poi_activity_types.npy"),
    }
    return _DATA


# ==== 模型加载 ====
_MODEL_DIRS = {"sft": SFT_LORA, "dpo": DPO_LORA, "grpo": GRPO_LORA}


def get_model(name: str):
    """按名字加载模型（懒加载 + 缓存）：sft / dpo / grpo."""
    if name in _MODELS:
        return _MODELS[name]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    lora_dir = _MODEL_DIRS[name]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    _MODELS[name] = (model, tokenizer)
    return _MODELS[name]


def generate_route(model, tokenizer, instruction, max_new_tokens=250):
    """生成路线文本（采样解码，每次结果不同）."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    torch.manual_seed(int(time.time() * 1000) % (2 ** 31))  # 每次生成不同
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=0.4, top_p=0.9,
                             no_repeat_ngram_size=4)  # 采样解码 + 禁 4-gram 重复（低温保路线长度）
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_route(text):
    """从生成文本解析 POI 名称序列."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?assistant", "", text, flags=re.DOTALL)
    matches = re.findall(
        r"([\u4e00-\u9fa5A-Za-z0-9·（）()]+(?:\s*(?:→|->)\s*[\u4e00-\u9fa5A-Za-z0-9·（）()]+){2,})",
        text,
    )
    if matches:
        longest = max(matches, key=len)
        return [p.strip() for p in re.split(r"\s*(?:→|->)\s*", longest) if p.strip()]
    return []


def match_to_indices(names, pois):
    """POI 名称 → 索引（精确优先，contains 兜底），返回 (indices, 未匹配名)."""
    matched, unmatched = [], []
    for name in names:
        exact = pois[pois["name"] == name]
        if len(exact) > 0:
            matched.append(int(exact.index[0])); continue
        contains = pois[pois["name"].str.contains(re.escape(name), na=False, regex=True)]
        if len(contains) > 0:
            matched.append(int(contains.index[0]))
        else:
            unmatched.append(name)
    return matched, unmatched


def _build_plan_response(plan: dict, instruction: str, pois) -> dict:
    """把逐日规划结果转成前端响应格式."""
    days_out = []
    for day in plan["days"]:
        sd = day.get("score_detail") or {}
        days_out.append({
            "day": day["day"],
            "half": day["half"],
            "pois": day["pois"],
            "score_detail": {
                "score": round(float(sd.get("score", 0)), 4),
                "feasible": bool(sd.get("feasible", False)),
                "reason": sd.get("reason"),
            },
        })
    overall = plan.get("overall") or {}
    return {
        "ok": True,
        "pois": [p for day in plan["days"] for p in day["pois"]],
        "score_detail": {
            "score": round(float(overall.get("score", 0)), 4),
            "feasible": bool(overall.get("feasible", False)),
            "reason": overall.get("reason"),
            "requirement_match": round(float(overall.get("requirement_match", 1.0)), 4),
            "total_dist_km": round(float(overall.get("metrics", {}).get("total_dist_km", 0)), 1),
            "n_pois": plan["total_pois"],
            "n_days": len(plan["days"]),
            "inferred_days": overall.get("inferred_days", len(plan["days"])),
        },
        "days": days_out,
        "warnings": plan.get("warnings", []),
        "raw": "",  # 逐日规划不提供单条原始输出
    }


def handle_generate(body: dict) -> dict:
    """核心：生成 + 匹配 + 去重 + 打分 + 地图."""
    instruction = (body.get("instruction") or "").strip()
    model_name = body.get("model", "grpo")
    if model_name not in _MODEL_DIRS:
        return {"ok": False, "error": f"未知模型 {model_name}，可选 {list(_MODEL_DIRS)}"}
    if not instruction:
        return {"ok": False, "error": "请输入规划指令"}

    d = load_data()
    pois = d["pois"]

    try:
        model, tokenizer = get_model(model_name)
    except Exception as e:
        return {"ok": False, "error": f"模型加载失败: {e}"}

    # === Phase 1：多日游拆分 + RAG 逐日规划（含半日需求） ===
    try:
        from src.itinerary_planner import plan_itinerary
        plan = plan_itinerary(model, tokenizer, instruction, d)
        if plan.get("ok") and plan["total_pois"] >= 3:
            return _build_plan_response(plan, instruction, pois)
    except Exception as e:
        print(f"逐日规划失败，回退单次生成: {e}")

    raw = generate_route(model, tokenizer, instruction)
    names = parse_route(raw)
    if len(names) < 3:
        return {"ok": True, "raw": raw,
                "warnings": ["模型输出的路线太短（少于 3 个景点），无法打分/画图。可换一条指令或调整预算后重试。"],
                "pois": [], "score_detail": None, "map_html": None}

    matched, unmatched = match_to_indices(names, pois)
    n_orig = len(matched)
    # 按索引去重（保留首次出现）
    seen, deduped = set(), []
    for idx in matched:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    n_dropped = n_orig - len(deduped)
    match_rate = n_orig / max(len(names), 1)
    warnings = []
    if unmatched:
        warnings.append(f"{len(unmatched)} 个景点未匹配到 POI 库（已跳过）: {unmatched[:3]}")
    if n_dropped:
        warnings.append(f"路线有 {n_dropped} 个重复景点（去重后保留首次出现）")

    if len(deduped) < 3:
        return {"ok": True, "raw": raw, "warnings": warnings + ["匹配后不足 3 个景点，无法评估。"],
                "pois": [], "score_detail": None, "map_html": None}

    n_days = 1 if len(deduped) <= 10 else (2 if len(deduped) <= 16 else 3)
    result = composite_score_v5(deduped, d["dist_matrix"], d["time_matrix"],
                                d["ratings"], d["categories"], n_days=n_days,
                                activity_types=d["activity_types"],
                                instruction=instruction, use_llm=True,
                                poi_names=pois["name"].tolist(),
                                avg_costs=pois["avg_cost"].values,
                                season_winter=pois["season_winter"].values,
                                season_summer=pois["season_summer"].values)

    # POI 序列展示（含距离标注）
    poi_names = [str(pois.iloc[i]["name"]) for i in deduped]
    metrics = result.get("metrics", {})
    score_detail = {
        "score": round(float(result["score"]), 4),
        "feasible": bool(result.get("feasible")),
        "reason": result.get("reason"),
        "n_days": n_days,
        "match_rate": round(float(match_rate), 2),
        "n_dropped": n_dropped,
        "n_pois": int(metrics.get("n_pois", len(deduped))),
        "total_dist_km": round(float(metrics.get("total_dist_km", 0)), 1),
        "total_time_min": int(metrics.get("total_time_min", 0)),
        "satisfaction": round(float(metrics.get("satisfaction", 0)), 3),
        "diversity": round(float(metrics.get("diversity", 0)), 3),
        "proximity": round(float(result.get("proximity", 0)), 3),
        "area_density": round(float(result.get("area_density", 0)), 3),
        "rhythm": round(float(result.get("rhythm", 0)), 3),
        "sub_scores": {k: round(float(v), 3) for k, v in result.items()
                       if k in ("proximity", "area_density", "rhythm", "satisfaction", "diversity")},
        "requirement_match": round(float(result.get("requirement_match", 1.0)), 4),
        "requirement_breakdown": result.get("requirement_breakdown", {}),
        "inferred_days": result.get("inferred_days", n_days),
    }

    # 地图
    try:
        title = f"{model_name.upper()} 路线（{n_days}日游）"
        m = plot_route_on_map(pois, deduped, output_path=None,
                              center_lat=45.80, center_lng=126.53, title=title)
        map_html = m.get_root().render()
    except Exception as e:
        map_html = None
        warnings.append(f"地图生成失败: {e}")

    return {"ok": True, "raw": raw, "pois": poi_names, "score_detail": score_detail,
            "map_html": map_html, "warnings": warnings, "model": model_name}


# ==== 前端页面 ====
PAGE_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>哈尔滨路线规划 Demo</title>
<style>
  body { background:#0f1720; color:#e2e8f0; font-family:"Segoe UI",sans-serif; margin:0; }
  .wrap { max-width:1000px; margin:0 auto; padding:24px 16px 60px; }
  h1 { font-size:22px; margin:0 0 4px; color:#fff; }
  .sub { color:#94a3b8; font-size:13px; margin-bottom:20px; }
  .card { background:#1a2534; border:1px solid #2a3a4f; border-radius:10px; padding:16px; margin-bottom:16px; }
  textarea { width:100%; box-sizing:border-box; background:#0d1520; color:#e2e8f0;
             border:1px solid #2a3a4f; border-radius:6px; padding:10px; font-size:14px; resize:vertical; }
  select, button, .model-btn { padding:8px 14px; border-radius:6px; border:1px solid #2a3a4f;
             background:#243447; color:#e2e8f0; font-size:14px; cursor:pointer; }
  select { margin-right:8px; }
  button.primary { background:#2f6feb; border-color:#2f6feb; color:#fff; font-weight:600; }
  button.primary:hover { background:#3d7cf5; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .model-switch { display:inline-flex; margin-right:10px; }
  .model-switch label { padding:8px 16px; border:1px solid #2a3a4f; cursor:pointer; font-size:14px; }
  .model-switch label:first-child { border-radius:6px 0 0 6px; }
  .model-switch label:last-child { border-radius:0 6px 6px 0; }
  .model-switch input { display:none; }
  .model-switch input:checked + span { color:#7bb0ff; font-weight:600; }
  .bar { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:12px; }
  #status { margin-top:10px; font-size:13px; color:#7bb0ff; min-height:18px; }
  .err { color:#f87171; }
  .score-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:10px; margin-top:10px; }
  .score-item { background:#0d1520; border-radius:8px; padding:10px; text-align:center; }
  .score-item .v { font-size:20px; font-weight:700; color:#7bb0ff; }
  .score-item .k { font-size:11px; color:#94a3b8; margin-top:2px; }
  .score-item.main .v { color:#fbbf24; font-size:26px; }
  .badge { display:inline-block; padding:2px 10px; border-radius:10px; font-size:12px; margin-left:8px; }
  .badge.ok { background:#14532d; color:#86efac; }
  .badge.no { background:#7f1d1d; color:#fca5a5; }
  .route-list { margin-top:12px; }
  .poi-row { display:flex; align-items:center; padding:8px 10px; border-bottom:1px solid #223142; }
  .poi-row .n { min-width:26px; height:26px; border-radius:50%; background:#2f6feb; color:#fff;
                display:flex; align-items:center; justify-content:center; font-size:13px; margin-right:12px; flex-shrink:0; }
  .poi-row .nm { flex:1; font-size:14px; }
  .poi-row .idx { color:#64748b; font-size:11px; }
  #map-wrap { margin-top:16px; }
  #map-frame { width:100%; height:480px; border:1px solid #2a3a4f; border-radius:10px; background:#0d1520; }
  .warn { color:#fbbf24; font-size:13px; margin-top:8px; }
  details { margin-top:12px; font-size:13px; }
  details pre { background:#0d1520; padding:10px; border-radius:6px; color:#94a3b8; white-space:pre-wrap; }
  .loading { display:inline-block; width:14px; height:14px; border:2px solid #7bb0ff; border-top-color:transparent;
             border-radius:50%; animation:sp 1s linear infinite; vertical-align:middle; margin-right:6px; }
  @keyframes sp { to { transform:rotate(360deg); } }
</style></head><body><div class="wrap">
  <h1>❄️ 哈尔滨路线规划 · Qwen 演示</h1>
  <div class="sub">Qwen3-4B LoRA 微调（SFT / DPO）· 生成路线 + v4 硬约束打分 + 地图</div>

  <div class="card">
    <textarea id="instr" rows="3" placeholder="例如：帮我规划一条哈尔滨两日游路线，冬季出行，预算约1500元，喜欢美食和购物，节奏不要太赶。"></textarea>
    <div class="bar">
      <select id="examples" onchange="fillExample()">
        <option value="">— 示例指令 —</option>
        <option>帮我规划一条哈尔滨一日游路线，冰雪季出行，总预算约500元，从中央大街出发，希望多去经典景点。</option>
        <option>帮我规划一条哈尔滨两日游路线，夏季出行，预算约1500元，喜欢美食和购物，节奏不要太赶。</option>
        <option>帮我规划一条哈尔滨三日游路线，冬季出行，预算约3000元，以冰雪大世界和中央大街为核心。</option>
        <option>我在哈尔滨只有半天时间，想从圣索菲亚教堂附近出发，看看冰雪大世界，怎么安排？</option>
        <option>带父母去哈尔滨玩三天，预算充足，节奏要慢，适合老年人的景点优先。</option>
      </select>
      <div class="model-switch">
        <label><input type="radio" name="model" value="grpo" checked onchange="switchModel(this)"><span>GRPO</span></label>
        <label><input type="radio" name="model" value="dpo" onchange="switchModel(this)"><span>DPO</span></label>
        <label><input type="radio" name="model" value="sft" onchange="switchModel(this)"><span>SFT</span></label>
      </div>
      <button class="primary" id="genBtn" onclick="generate()">生成路线</button>
    </div>
    <div id="status">就绪。</div>
  </div>

  <div id="result"></div>
</div>
<script>
function fillExample(){ document.getElementById('instr').value = document.getElementById('examples').value; }
function switchModel(el){}
function setStatus(html){ document.getElementById('status').innerHTML = html; }

async function generate(){
  const instr = document.getElementById('instr').value.trim();
  const model = document.querySelector('input[name="model"]:checked').value;
  if(!instr){ setStatus('<span class="err">请输入规划指令</span>'); return; }
  const btn = document.getElementById('genBtn');
  btn.disabled = true;
  setStatus('<span class="loading"></span> 模型生成路线 + 打分中（首次约 1 分钟，之后 5-10 秒）...');
  document.getElementById('result').innerHTML = '';
  try{
    const r = await fetch('/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({instruction: instr, model: model})
    });
    const d = await r.json();
    if(!d.ok){ setStatus('<span class="err">' + (d.error||'生成失败') + '</span>'); return; }
    render(d);
    setStatus('完成（模型：' + (d.model||model).toUpperCase() + '）。');
  }catch(e){ setStatus('<span class="err">请求失败: '+e+'</span>'); }
  btn.disabled = false;
}

function render(d){
  const box = document.getElementById('result');
  let html = '';
  if(d.warnings && d.warnings.length){
    html += d.warnings.map(w => '<div class="warn">⚠️ '+w+'</div>').join('');
  }
  const s = d.score_detail;
  if(s){
    const badge = s.feasible ? '<span class="badge ok">可行</span>' : '<span class="badge no">不可行('+ (s.reason||'') +')</span>';
    html += '<div class="card"><b>v4 综合评分</b>' + badge + '<div class="score-grid">'
      + '<div class="score-item main"><div class="v">'+s.score+'</div><div class="k">综合分</div></div>'
      + '<div class="score-item"><div class="v">'+s.n_days+'日</div><div class="k">推断天数</div></div>'
      + '<div class="score-item"><div class="v">'+s.n_pois+'</div><div class="k">景点数</div></div>'
      + '<div class="score-item"><div class="v">'+s.total_dist_km+'km</div><div class="k">总距离</div></div>'
      + '<div class="score-item"><div class="v">'+s.total_time_min+'min</div><div class="k">总耗时</div></div>'
      + '<div class="score-item"><div class="v">'+s.satisfaction+'</div><div class="k">满意度</div></div>'
      + '<div class="score-item"><div class="v">'+s.diversity+'</div><div class="k">多样性</div></div>'
      + (('<div class="score-item"><div class="v">'+s.requirement_match+'</div><div class="k">需求匹配</div></div>')
         if s.requirement_match != null && s.requirement_match < 1 else '')
      + '</div><div class="score-grid" style="grid-template-columns:repeat(5,1fr)">'
      + ['proximity','area_density','rhythm','satisfaction','diversity'].map(k => {
          const v = s.sub_scores && s.sub_scores[k] != null ? s.sub_scores[k] : '—';
          const nm = {proximity:'就近性',area_density:'区域密度',rhythm:'节奏',satisfaction:'满意度',diversity:'多样性'}[k];
          return '<div class="score-item"><div class="v" style="font-size:15px">'+v+'</div><div class="k">'+nm+'</div></div>';
        }).join('')
      + '</div></div>';
  }
  if(d.days && d.days.length){
    html += '<div class="card"><b>逐日行程</b>';
    d.days.forEach(day => {
      const tag = day.half ? '<span class="badge" style="background:#3f3f46;color:#a1a1aa">半日</span>' : '';
      const sd = day.score_detail || {};
      const ok = sd.feasible ? '' : ' <span style="color:#f87171;font-size:12px">✗('+(sd.reason||'')+')</span>';
      html += '<div style="margin-top:10px"><b>第'+day.day+'天</b> '+tag+' <span style="color:#94a3b8;font-size:12px">v5='+sd.score+'</span>'+ok
        + '<div class="route-list">'
        + (day.pois||[]).map((p,i)=>'<div class="poi-row"><div class="n">'+(i+1)+'</div><div class="nm">'+p+'</div></div>').join('')
        + '</div></div>';
    });
    html += '</div>';
  }
  if(d.pois && d.pois.length){
    html += '<div class="card"><b>路线</b><div class="route-list">'
      + d.pois.map((p,i)=>'<div class="poi-row"><div class="n">'+(i+1)+'</div><div class="nm">'+p+'</div><div class="idx">'+i+'</div></div>').join('')
      + '</div></div>';
  }
  if(d.map_html){
    html += '<div class="card" id="map-wrap"><b>地图</b>'
      + '<iframe id="map-frame" srcdoc="'+esc(d.map_html)+'"></iframe></div>';
  }
  if(d.raw){
    html += '<details><summary>查看模型原始输出</summary><pre>'+esc(d.raw)+'</pre></details>';
  }
  box.innerHTML = html;
}
function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
</script>
</body></html>"""


# ==== HTTP 服务 ====
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # 静默访问日志

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, json.dumps({"ok": True, "models": list(_MODELS.keys()),
                                        "time": time.strftime("%H:%M:%S")},
                                       ensure_ascii=False), "application/json; charset=utf-8")
        else:
            self._send(200, PAGE_HTML)

    def do_POST(self):
        if not self.path.startswith("/generate"):
            self._send(404, json.dumps({"ok": False, "error": "not found"}),
                       "application/json; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as e:
            self._send(400, json.dumps({"ok": False, "error": f"请求解析失败: {e}"}),
                       "application/json; charset=utf-8")
            return
        result = handle_generate(body)
        self._send(200, json.dumps(result, ensure_ascii=False),
                   "application/json; charset=utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8898)
    parser.add_argument("--host", default="")
    args = parser.parse_args()

    # 预热：加载数据 + 主模型（GRPO）+ DPO（提升首个请求响应；SFT 懒加载）
    print("加载数据矩阵（POI/距离/时间）...")
    load_data()
    for name in ("grpo", "dpo"):
        print(f"加载模型 {name}（约 1 分钟）...")
        get_model(name)
    print("模型就绪，预热一次生成...")
    m, tok = get_model("grpo")
    _ = generate_route(m, tok, "帮我规划一条哈尔滨一日游路线，从中央大街出发。")
    print(f"预热完成。访问 http://localhost:{args.port}")

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
