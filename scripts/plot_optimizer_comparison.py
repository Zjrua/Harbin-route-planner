"""生成优化器对比图表（方向A）.

从 output/optimizer_comparison.json 读取两组训练曲线，
生成 publication-ready 的对比图，保存到 paper/ 和 output/。

用法:
    ./.venv/Scripts/python.exe scripts/plot_optimizer_comparison.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_comparison(json_path: str, out_dir: str):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    runs = data["runs"]
    if len(runs) < 2:
        print("不足两组，无法对比")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"adamw": "#2E86AB", "muon": "#E63946"}
    labels = {"adamw": "AdamW", "muon": "Muon"}

    # === 左图：train/val loss 曲线 ===
    ax = axes[0]
    for r in runs:
        name = r["optimizer"]
        epochs = [h["epoch"] for h in r["history"]]
        train = [h["train_loss"] for h in r["history"]]
        val = [h["val_loss"] for h in r["history"]]
        color = colors.get(name, "gray")
        ax.plot(epochs, train, "--", color=color, alpha=0.5, linewidth=1.2)
        ax.plot(epochs, val, "-", color=color, linewidth=2,
                label=f"{labels.get(name, name)} (val, best={r['best_val_loss']}@ep{r['best_epoch']})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("优化器对比：训练/验证损失曲线", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # 标注最佳点
    for r in runs:
        name = r["optimizer"]
        best_ep = r["best_epoch"]
        best_val = r["best_val_loss"]
        ax.scatter([best_ep], [best_val], color=colors.get(name, "gray"),
                   s=80, zorder=5, edgecolors="white", linewidths=1.5)

    # === 右图：关键指标对比柱状图 ===
    ax2 = axes[1]
    metrics = ["best_val_loss", "final_train_loss", "gap (val-train)"]
    adamw = next(r for r in runs if r["optimizer"] == "adamw")
    muon = next(r for r in runs if r["optimizer"] == "muon")
    adamw_vals = [adamw["best_val_loss"], adamw["final_train_loss"],
                  adamw["best_val_loss"] - adamw["final_train_loss"]]
    muon_vals = [muon["best_val_loss"], muon["final_train_loss"],
                 muon["best_val_loss"] - muon["final_train_loss"]]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax2.bar(x - width/2, adamw_vals, width, label="AdamW",
                    color=colors["adamw"], alpha=0.85)
    bars2 = ax2.bar(x + width/2, muon_vals, width, label="Muon",
                    color=colors["muon"], alpha=0.85)

    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("关键指标对比", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["best val_loss\n(越低越好)", "final train_loss", "过拟合 gap\n(val-train)"],
                        fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    # 数值标注
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=9)

    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "optimizer_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {out_path}")

    # 总结
    c = data.get("comparison", {})
    print(f"\n结论: {c.get('winner', '?')} 更优，"
          f"Muon 比 AdamW 差 {c.get('muon_best_val_loss', 0) - c.get('adamw_best_val_loss', 0):+.4f}")


if __name__ == "__main__":
    plot_comparison("output/optimizer_comparison.json", "paper")
    plot_comparison("output/optimizer_comparison.json", "output")
