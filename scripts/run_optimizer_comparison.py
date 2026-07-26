"""优化器对比实验：AdamW vs Muon（方向 A）.

目的：堵论文最大漏洞——Muon 作为"三大核心创新之一"此前零实测。
在严格可比条件下（相同 config/数据/seed/超参/早停，唯一变量=优化器），
分别训练 AdamW 和 Muon 两组，记录完整 loss 曲线与最佳 val_loss。

实验设计：
- 两组从零训练（不用现有 best_model.pt，保证 seed/初始权重一致）
- 复用 src.train.build_optimizer（与主训练同一套优化器构造逻辑）
- 输出：output/optimizer_comparison.json（两组完整曲线 + 最佳值）
      paper/optimizer_comparison.png（loss 曲线对比图，可选）

用法:
    ./.venv/Scripts/python.exe -m scripts.run_optimizer_comparison
    ./.venv/Scripts/python.exe -m scripts.run_optimizer_comparison --max-epochs 30  # 快速测试
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_dataloaders
from src.models.transformer import ItineraryTransformer
from src.models.losses import RouteLoss
from src.train import build_optimizer, set_seed


def train_one_run(config, optimizer_name, device, max_epochs=None):
    """训练单组（指定优化器），返回完整 loss 曲线 + 最佳值.

    与 run_ablation.py/train.py 的训练流程保持一致：
    - 共享数据加载、encoder 预计算、AMP、scheduled sampling、早停
    - 唯一区别：通过 config['optimizer']['name'] 切换优化器
    """
    # 强制设置优化器名（覆盖 config，确保对比的单一变量）
    config = {**config, "optimizer": {**config["optimizer"], "name": optimizer_name}}

    set_seed(config["training"]["seed"])  # 每组训练前重置 seed，保证初始权重一致

    model = ItineraryTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_loader, val_loader, _ = create_dataloaders("data/processed", config)
    shared = train_loader.dataset.get_shared_data(device)

    # encoder 预计算（共享）
    model.eval()
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        encoder_output = model.encode(
            shared["poi_features"].unsqueeze(0),
            shared["adjacency"].unsqueeze(0),
        )

    criterion = RouteLoss(
        ce_weight=config["loss"]["ce_weight"],
        distance_weight=config["loss"]["distance_weight"],
        mhc_weight=config["loss"]["mhc_weight"],
    )
    optimizer = build_optimizer(model, config)  # 根据 config['optimizer']['name'] 构造 AdamW 或 Muon

    total_epochs = max_epochs or config["training"]["epochs"]
    patience_limit = config["training"]["patience"]
    grad_clip = config["optimizer"].get("grad_clip", 1.0)
    base_tf = config["training"]["teacher_forcing_ratio"]
    use_prob = config.get("data", {}).get("use_probabilistic_edges", False)
    dataset = train_loader.dataset
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []  # [{epoch, train_loss, val_loss}]

    print(f"  [{optimizer_name}] 参数={n_params:,}, 优化器={type(optimizer).__name__}, "
          f"epochs≤{total_epochs}, patience={patience_limit}")

    t0 = time.time()
    for epoch in range(total_epochs):
        # === Train ===
        model.train()
        model.encoder.eval()  # encoder 共享，关闭 dropout
        tf_ratio = base_tf * max(0.0, 1.0 - epoch / total_epochs)
        train_loss_sum, n_b = 0.0, 0
        for route_seq, scores, route_activity in train_loader:
            if use_prob:
                distances = dataset.sample_noisy_distances(device)
            else:
                distances = shared["distances"]
            batch_device = {
                "poi_features": shared["poi_features"].unsqueeze(0),
                "adjacency": shared["adjacency"].unsqueeze(0),
                "route_sequence": route_seq.to(device, non_blocking=True),
                "distances": distances,
                "scores": scores.to(device, non_blocking=True),
                "activity_types": route_activity.to(device, non_blocking=True),
                "_encoder_output": encoder_output,
            }
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(batch_device)
                logits = output["logits"]
                target = batch_device["route_sequence"][:, 1:]
                min_len = min(logits.size(1), target.size(1))
                loss = criterion(logits[:, :min_len], target[:, :min_len],
                                 batch_device["distances"], output.get("embeddings"))
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            train_loss_sum += loss.item()
            n_b += 1
        train_loss = train_loss_sum / max(n_b, 1)

        # === Validate ===
        model.eval()
        val_loss_sum, n_v = 0.0, 0
        with torch.no_grad():
            for route_seq, scores, route_activity in val_loader:
                batch_device = {
                    "poi_features": shared["poi_features"].unsqueeze(0),
                    "adjacency": shared["adjacency"].unsqueeze(0),
                    "route_sequence": route_seq.to(device, non_blocking=True),
                    "distances": shared["distances"],
                    "scores": scores.to(device, non_blocking=True),
                    "activity_types": route_activity.to(device, non_blocking=True),
                    "_encoder_output": encoder_output,
                }
                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = model(batch_device)
                    logits = output["logits"]
                    target = batch_device["route_sequence"][:, 1:]
                    min_len = min(logits.size(1), target.size(1))
                    loss = criterion(logits[:, :min_len], target[:, :min_len],
                                     batch_device["distances"], output.get("embeddings"))
                val_loss_sum += loss.item()
                n_v += 1
        val_loss = val_loss_sum / max(n_v, 1)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4)})

        if (epoch + 1) % 10 == 0 or improved or patience_counter >= patience_limit:
            elapsed = time.time() - t0
            print(f"    [{optimizer_name}] ep{epoch+1:>3d} train={train_loss:.4f} "
                  f"val={val_loss:.4f} best={best_val_loss:.4f}@{best_epoch} "
                  f"pat={patience_counter}/{patience_limit} ({elapsed:.0f}s)")

        if patience_counter >= patience_limit:
            print(f"    [{optimizer_name}] 早停于 epoch {epoch+1}")
            break

    return {
        "optimizer": optimizer_name,
        "optimizer_class": type(optimizer).__name__,
        "n_params": n_params,
        "best_val_loss": round(best_val_loss, 4),
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history),
        "train_time_sec": round(time.time() - t0, 1),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="优化器对比实验：AdamW vs Muon")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="限制最大 epoch（快速测试用，默认用 config 的 epochs）")
    parser.add_argument("--only", choices=["adamw", "muon"], default=None,
                        help="只跑某一组（默认两组都跑）")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print("  优化器对比实验：AdamW vs Muon（方向 A）")
    print(f"  Device: {device}  |  数据: data/processed  |  config: {args.config}")
    if args.max_epochs:
        print(f"  ⚠️  限制 max_epochs={args.max_epochs}（快速测试模式）")
    print("=" * 64)

    runs = []
    targets = [args.only] if args.only else ["adamw", "muon"]
    for name in targets:
        print(f"\n=== 训练组：{name} ===")
        result = train_one_run(config, name, device, max_epochs=args.max_epochs)
        runs.append(result)
        print(f"  → {name}: best_val_loss={result['best_val_loss']} @ ep{result['best_epoch']}, "
              f"耗时 {result['train_time_sec']}s")

    # === 对比汇总 ===
    print("\n" + "=" * 64)
    print("  对比结果")
    print("=" * 64)
    print(f"{'优化器':<10} {'best_val_loss':<16} {'best_epoch':<12} {'train_loss':<12} {'耗时(s)':<10}")
    for r in runs:
        print(f"{r['optimizer']:<10} {r['best_val_loss']:<16} {r['best_epoch']:<12} "
              f"{r['final_train_loss']:<12} {r['train_time_sec']:<10}")

    if len(runs) == 2:
        adamw_loss = next(r["best_val_loss"] for r in runs if r["optimizer"] == "adamw")
        muon_loss = next(r["best_val_loss"] for r in runs if r["optimizer"] == "muon")
        diff = muon_loss - adamw_loss
        pct = diff / adamw_loss * 100
        winner = "Muon" if muon_loss < adamw_loss else "AdamW"
        print(f"\n  差值 (Muon - AdamW) = {diff:+.4f} ({pct:+.2f}%)")
        print(f"  → {winner} 更优")
        if abs(pct) < 1.0:
            print(f"  → 两者差异 <1%，统计上无显著区别")

    # === 保存结果 ===
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "optimizer_comparison.json"
    summary = {
        "config": str(args.config),
        "device": str(device),
        "max_epochs_override": args.max_epochs,
        "runs": runs,
    }
    if len(runs) == 2:
        summary["comparison"] = {
            "adamw_best_val_loss": next(r["best_val_loss"] for r in runs if r["optimizer"] == "adamw"),
            "muon_best_val_loss": next(r["best_val_loss"] for r in runs if r["optimizer"] == "muon"),
            "winner": winner,
        }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
