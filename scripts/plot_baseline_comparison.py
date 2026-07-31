"""生成基线对比图表（方向C）+ 路线样例对比.

从 output/baseline_comparison.json 读取结果，生成：
1. 5 方法的指标对比柱状图
2. 路线样例对比（展示 NN 的"连续住宿"问题 vs Transformer 的合理节奏）

用法:
    ./.venv/Scripts/python.exe scripts/plot_baseline_comparison.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_metrics(json_path: str, out_dir: str):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    results = data["results"]
    methods = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {"random": "#999999", "NN": "#2E86AB", "2-opt": "#5DADE2",
              "OR-Tools": "#A569BD", "Transformer": "#E63946"}

    # === 左图：距离对比（对数尺度，因差距巨大）===
    ax = axes[0]
    dists = [results[m]["avg_distance_km"] for m in methods]
    bars = ax.bar(methods, dists, color=[colors.get(m, "gray") for m in methods], alpha=0.85)
    ax.set_ylabel("平均路线距离 (km)", fontsize=12)
    ax.set_title("路线距离对比（对数尺度）", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, d in zip(bars, dists):
        ax.annotate(f"{d:.1f}", xy=(bar.get_x() + bar.get_width()/2, d),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10)

    # === 右图：多样性 + 满意度 + composite ===
    ax2 = axes[1]
    metrics = ["avg_diversity", "avg_satisfaction", "composite"]
    labels = ["多样性", "满意度", "composite"]
    x = np.arange(len(metrics))
    width = 0.15
    for i, m in enumerate(methods):
        vals = [results[m][met] for met in metrics]
        ax2.bar(x + i*width - width*2, vals, width, label=m,
                color=colors.get(m, "gray"), alpha=0.85)
    ax2.set_ylabel("指标值", fontsize=12)
    ax2.set_title("多样性 / 满意度 / composite 对比", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 5.5)

    plt.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"指标对比图已保存: {out_path}")


if __name__ == "__main__":
    plot_metrics("output/baseline_comparison.json", "paper")
    plot_metrics("output/baseline_comparison.json", "output")
