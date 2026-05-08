"""生成论文用图：训练曲线 + 消融实验对比."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import json
from pathlib import Path

# 中文支持
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ema_smooth(data: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """指数移动平均平滑，保持趋势方向。"""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def generate_training_curve():
    """从 training_loss.csv 生成训练收敛曲线."""
    df = pd.read_csv(PROJECT_ROOT / "output" / "training_loss.csv")
    epochs = df["epoch"].values
    train_raw = df["train_loss"].values
    val_raw = df["val_loss"].values
    best_val_loss = df["best_val_loss"].values
    best_idx = np.argmin(best_val_loss)
    best_epoch = epochs[best_idx]
    best_val = best_val_loss[best_idx]

    # EMA 平滑
    train_smooth = ema_smooth(train_raw, alpha=0.25)
    val_smooth = ema_smooth(val_raw, alpha=0.25)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 原始数据：浅色背景
    ax.plot(epochs, train_raw, color="#2E86AB", linewidth=0.7, alpha=0.25)
    ax.plot(epochs, val_raw, color="#A23B72", linewidth=0.7, alpha=0.25)

    # 平滑曲线：主视觉
    ax.plot(epochs, train_smooth, color="#2E86AB", linewidth=2.2, label="训练损失 (Train Loss)")
    ax.plot(epochs, val_smooth, color="#A23B72", linewidth=2.2, label="验证损失 (Val Loss)")

    # 标注最佳 epoch
    ax.axvline(x=best_epoch, color="#F18F01", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.annotate(
        f"最佳 epoch {best_epoch}\nval_loss={best_val:.4f}",
        xy=(best_epoch, val_smooth[best_idx]),
        xytext=(best_epoch + 12, val_smooth[best_idx] + 0.6),
        fontsize=11,
        color="#D62828",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#D62828", lw=1.4, connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#F18F01", alpha=0.9),
    )

    # 标注早停点
    ax.axvline(x=121, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(121 + 0.5, 8.3, "早停 (epoch 121)", fontsize=10, color="gray", rotation=90, va="top")

    ax.set_xlabel("训练轮次 (Epoch)", fontsize=14, fontweight="bold")
    ax.set_ylabel("损失 (Loss)", fontsize=14, fontweight="bold")
    ax.set_title("RouteTransformer 训练损失收敛曲线", fontsize=15, fontweight="bold", pad=14)
    ax.legend(fontsize=12, loc="upper right", framealpha=0.9, edgecolor="gray")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 10)
    ax.tick_params(labelsize=11)

    # 关键统计
    stats_text = (
        f"初始 val_loss: {val_raw[0]:.2f}\n"
        f"最佳 val_loss: {best_val:.4f} (epoch {best_epoch})\n"
        f"最终 train_loss: {train_raw[-1]:.2f}\n"
        f"最终 val_loss: {val_raw[-1]:.4f}\n"
        f"train-val gap: {val_raw[-1] - train_raw[-1]:.2f}"
    )
    ax.text(
        0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#CCCCCC", alpha=0.85),
    )

    plt.tight_layout()
    out_path = PROJECT_ROOT / "output" / "training_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] training_curve.png saved to {out_path}")


def generate_ablation_comparison():
    """从 ablation_results.json 生成消融实验对比图."""
    with open(PROJECT_ROOT / "output" / "ablation_results.json", encoding="utf-8") as f:
        results = json.load(f)

    # 核心指标
    labels = [r["name_cn"] for r in results]
    scores = [r["composite_score"] for r in results]
    distances = [r["avg_distance_km"] for r in results]

    # 按综合得分排序
    sorted_idx = np.argsort(scores)[::-1]
    labels = [labels[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    distances = [distances[i] for i in sorted_idx]

    # 颜色方案：最优绿色，基线灰色，其他蓝色
    colors = []
    for label in labels:
        if "K=3" in label or "最优" in label:
            colors.append("#06A77D")  # 深绿 — 最优
        elif "基线" in label or "Transformer" in label:
            colors.append("#B0B0B0")  # 灰 — 基线
        elif "完整" in label:
            colors.append("#4A90D9")  # 蓝 — 完整模型
        else:
            colors.append("#6C8EBF")  # 浅蓝 — 消融组

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- 左图：综合得分 ----
    bars1 = ax1.barh(labels, scores, color=colors, edgecolor="white", height=0.6, linewidth=0.8)
    ax1.set_xlabel("综合得分 (Composite Score)", fontsize=13, fontweight="bold")
    ax1.set_title("消融实验综合得分对比", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlim(min(scores) - 0.002, max(scores) + 0.002)
    ax1.grid(True, alpha=0.2, axis="x", linestyle="--")
    ax1.tick_params(labelsize=11)

    # 在柱状图上标注数值
    for bar, val in zip(bars1, scores):
        ax1.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9, fontweight="bold", color="#333333")

    # ---- 右图：路线距离 ----
    bars2 = ax2.barh(labels, distances, color=colors, edgecolor="white", height=0.6, linewidth=0.8)
    ax2.set_xlabel("平均路线距离 (km)", fontsize=13, fontweight="bold")
    ax2.set_title("消融实验路线距离对比", fontsize=14, fontweight="bold", pad=12)
    ax2.grid(True, alpha=0.2, axis="x", linestyle="--")
    ax2.tick_params(labelsize=11)

    for bar, val in zip(bars2, distances):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=9, fontweight="bold", color="#333333")

    plt.tight_layout()
    out_path = PROJECT_ROOT / "output" / "ablation_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] ablation_comparison.png saved to {out_path}")


if __name__ == "__main__":
    generate_training_curve()
    generate_ablation_comparison()
    print("\nDone. Copy these to paper/:")
    print("  cp output/training_curve.png paper/training_curve.png")
    print("  cp output/ablation_comparison.png paper/ablation_comparison.png")
