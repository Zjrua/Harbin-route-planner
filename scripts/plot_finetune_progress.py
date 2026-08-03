"""Qwen 微调可视化：拆成多个独立小图.

输出：
- train_loss.png:   训练损失曲线
- val_loss.png:     验证损失曲线（每100步评估一次）
- grad_norm.png:    梯度范数曲线
- learning_rate.png: 学习率调度曲线

实时可重复运行（幂等），配合 Web 服务或 watch 使用。
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LOG_PATH = Path(r"C:\Users\Administrator\.zcode\cli\exec\sess_c3355536-7ef6-4966-a19a-5acb783c87a2\call_00_7LyRMemRHkbX7bUuvzEa8300-stdout.log")
OUT_DIR = Path("output/qwen_progress")


def parse_log(log_path: Path):
    """解析日志中的 train loss / eval loss / grad norm / lr / step."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    # train: {'loss': '1.039', 'grad_norm': '2.113', 'learning_rate': '0.0001122', 'epoch': '1.504'}
    train_pattern = re.compile(
        r"\{'loss': '([\d.]+)', 'grad_norm': '([\d.]+)', 'learning_rate': '([\d.e+-]+)', 'epoch': '([\d.]+)'\}"
    )
    train_steps, train_loss, grad_norm, lr, epochs = [], [], [], [], []
    for m in train_pattern.finditer(text):
        train_steps.append(len(train_steps) + 1)
        train_loss.append(float(m.group(1)))
        grad_norm.append(float(m.group(2)))
        lr.append(float(m.group(3)))
        epochs.append(float(m.group(4)))

    # eval: {'eval_loss': '1.329', 'eval_runtime': ..., 'epoch': '...'}
    eval_pattern = re.compile(r"\{'eval_loss': '([\d.]+)'")
    eval_steps, eval_loss = [], []
    for i, m in enumerate(eval_pattern.finditer(text)):
        eval_steps.append(i + 1)
        eval_loss.append(float(m.group(1)))

    return {
        "train_steps": train_steps, "train_loss": train_loss,
        "eval_steps": eval_steps, "eval_loss": eval_loss,
        "grad_norm": grad_norm, "lr": lr, "epochs": epochs,
    }


def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def plot_single(data, ylabel, title, color, name, latest_fn=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data, color=color, linewidth=2)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if latest_fn:
        ax.annotate(f"最新: {latest_fn():.4f}", xy=(len(data)-1, data[-1]),
                    xytext=(-80, 20), textcoords="offset points", fontsize=10,
                    arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout()
    save_fig(fig, name)


def main():
    d = parse_log(LOG_PATH)
    if len(d["train_loss"]) < 2:
        print("数据点不足")
        return

    print("生成微调进度分图...")
    n_train = len(d["train_loss"])

    # 1. train loss
    plot_single(d["train_loss"], "Train Loss", "Qwen 训练损失曲线",
                "#2E86AB", "train_loss.png",
                latest_fn=lambda: d["train_loss"][-1])

    # 2. val loss（独立图）
    if d["eval_loss"]:
        plot_single(d["eval_loss"], "Val Loss", "Qwen 验证损失曲线",
                    "#E67E22", "val_loss.png",
                    latest_fn=lambda: d["eval_loss"][-1])
    else:
        print("  (暂无 val loss 数据)")

    # 3. grad norm
    plot_single(d["grad_norm"], "Grad Norm", "Qwen 梯度范数曲线",
                "#E63946", "grad_norm.png",
                latest_fn=lambda: d["grad_norm"][-1])

    # 4. learning rate
    plot_single(d["lr"], "Learning Rate", "Qwen 学习率调度",
                "#8E44AD", "learning_rate.png",
                latest_fn=lambda: d["lr"][-1])

    print(f"\n完成。输出目录: {OUT_DIR}")
    print(f"  训练点: {n_train}, 验证点: {len(d['eval_loss'])}")


if __name__ == "__main__":
    main()
