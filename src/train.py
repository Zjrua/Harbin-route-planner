"""训练主脚本.

支持:
- Teacher Forcing + Scheduled Sampling
- 早停策略
- Checkpoint 保存与加载
- 终端 tqdm 实时输出 + TensorBoard 日志记录
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from src.models.transformer import RouteTransformer
from src.models.losses import RouteLoss
from src.data.dataset import create_dataloaders
from src.optim.muon import MuonOptimizer


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """设置全局随机种子."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_optimizer(model: RouteTransformer, config: dict) -> torch.optim.Optimizer:
    """根据配置构建优化器，对注意力层和 FFN 层分组设置学习率."""
    opt_cfg = config["optimizer"]
    name = opt_cfg["name"]

    attention_params = []
    ffn_params = []
    other_params = []

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if "attn" in module_name or "attn" in param_name:
                attention_params.append(param)
            elif "ffn" in module_name or "ffn" in param_name:
                ffn_params.append(param)
            else:
                other_params.append(param)

    if name == "muon":
        param_groups = [
            {"params": attention_params, "group_type": "attention"},
            {"params": ffn_params, "group_type": "ffn"},
            {"params": other_params, "group_type": "other"},
        ]
        return MuonOptimizer(
            param_groups,
            lr_attn=opt_cfg["lr_attn"],
            lr_ffn=opt_cfg["lr_ffn"],
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            weight_decay=opt_cfg["weight_decay"],
        )
    else:
        grouped = [
            {"params": attention_params, "lr": opt_cfg["lr_attn"]},
            {"params": ffn_params, "lr": opt_cfg["lr_ffn"]},
            {"params": other_params, "lr": (opt_cfg["lr_attn"] + opt_cfg["lr_ffn"]) / 2},
        ]
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return cls(grouped, weight_decay=opt_cfg["weight_decay"])


def train_one_epoch(model: RouteTransformer, dataloader, optimizer,
                    criterion: RouteLoss, device: torch.device,
                    epoch: int, config: dict) -> dict:
    """训练一个 epoch，含 Teacher Forcing + Scheduled Sampling."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    grad_clip = config["optimizer"].get("grad_clip", 1.0)

    # Scheduled Sampling：随训练进行线性衰减 teacher forcing ratio
    base_ratio = config["training"]["teacher_forcing_ratio"]
    total_epochs = config["training"]["epochs"]
    tf_ratio = base_ratio * max(0.0, 1.0 - epoch / total_epochs)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False,
                ncols=120, unit="batch")
    for batch in pbar:
        poi_feat, adj, route_seq, dist, scores = batch
        batch_device = {
            "poi_features": poi_feat.to(device),
            "adjacency": adj.to(device),
            "route_sequence": route_seq.to(device),
            "distances": dist.to(device),
            "scores": scores.to(device),
        }

        optimizer.zero_grad()
        output = model(batch_device)

        logits = output["logits"]
        target = batch_device["route_sequence"][:, 1:]
        min_len = min(logits.size(1), target.size(1))
        logits = logits[:, :min_len]
        target = target[:, :min_len]

        distances = batch_device["distances"]
        if distances.dim() == 2:
            distances = distances.unsqueeze(0).expand(logits.size(0), -1, -1)
        elif distances.dim() == 3 and distances.size(0) == 1:
            distances = distances.expand(logits.size(0), -1, -1)

        loss = criterion(logits, target, distances, output.get("embeddings"))
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "tf": f"{tf_ratio:.2f}"})

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(model: RouteTransformer, dataloader, criterion: RouteLoss,
             device: torch.device, epoch: int = 0) -> dict:
    """验证集评估."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]  ", leave=False,
                ncols=120, unit="batch")
    for batch in pbar:
        poi_feat, adj, route_seq, dist, scores = batch
        batch_device = {
            "poi_features": poi_feat.to(device),
            "adjacency": adj.to(device),
            "route_sequence": route_seq.to(device),
            "distances": dist.to(device),
            "scores": scores.to(device),
        }
        output = model(batch_device)

        logits = output["logits"]
        target = batch_device["route_sequence"][:, 1:]
        min_len = min(logits.size(1), target.size(1))
        logits = logits[:, :min_len]
        target = target[:, :min_len]

        distances = batch_device["distances"]
        if distances.dim() == 2:
            distances = distances.unsqueeze(0).expand(logits.size(0), -1, -1)
        elif distances.dim() == 3 and distances.size(0) == 1:
            distances = distances.expand(logits.size(0), -1, -1)

        loss = criterion(logits, target, distances, output.get("embeddings"))
        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    return {"val_loss": total_loss / max(n_batches, 1)}


def main():
    parser = argparse.ArgumentParser(description="训练哈尔滨文旅路线模型")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备 (cuda/cpu)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(config["training"]["seed"])

    total_epochs = config["training"]["epochs"]
    patience_limit = config["training"]["patience"]

    print("=" * 60)
    print("  哈尔滨文旅线路优化模型 — 训练")
    print(f"  Device: {device}")
    print(f"  Epochs: {total_epochs}  |  Patience: {patience_limit}")
    print(f"  Optimizer: {config['optimizer']['name']}")
    print("=" * 60)

    # 数据加载
    train_loader, val_loader, test_loader = create_dataloaders("data/processed", config)
    print(f"  数据: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    # 模型初始化
    model = RouteTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {n_params:,}")

    criterion = RouteLoss(
        ce_weight=config["loss"]["ce_weight"],
        distance_weight=config["loss"]["distance_weight"],
        mhc_weight=config["loss"]["mhc_weight"],
    )
    optimizer = build_optimizer(model, config)

    # 恢复训练
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  从 epoch {start_epoch} 恢复训练")

    # 日志
    save_dir = Path(config["experiment"]["save_dir"])
    log_dir = Path(config["experiment"]["log_dir"])
    save_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir)

    print("=" * 60)
    print()

    # 训练循环
    patience_counter = 0

    for epoch in range(start_epoch, total_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        criterion, device, epoch, config)
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        # TensorBoard 记录
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        train_loss = train_metrics["loss"]
        val_loss = val_metrics["val_loss"]

        # Checkpoint 保存 & 早停
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, save_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        # 终端日志
        marker = " *" if improved else ""
        print(f"[Epoch {epoch+1:>3d}/{total_epochs}]  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"best={best_val_loss:.4f}{marker}  "
              f"patience={patience_counter}/{patience_limit}")

        if patience_counter >= patience_limit:
            print(f"\n早停于 epoch {epoch+1}，最佳 val_loss={best_val_loss:.4f}")
            break

    writer.close()
    print(f"\n训练完成。最佳模型已保存至 {save_dir / 'best_model.pt'}")
    print(f"TensorBoard 日志: {log_dir}")


if __name__ == "__main__":
    main()
