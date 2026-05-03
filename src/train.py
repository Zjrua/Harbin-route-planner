"""训练主脚本.

支持:
- Teacher Forcing + Scheduled Sampling
- 早停策略
- Checkpoint 保存与加载
- TensorBoard / 日志记录
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from models.transformer import RouteTransformer
from models.losses import RouteLoss
from data.dataset import create_dataloaders
from optim.muon import MuonOptimizer


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """设置全局随机种子."""
    raise NotImplementedError("需实现：torch / numpy / random seed 设置")


def build_optimizer(model: RouteTransformer, config: dict) -> torch.optim.Optimizer:
    """根据配置构建优化器.

    支持 Muon / Adam / AdamW，对注意力层和 FFN 层分组设置学习率。
    """
    raise NotImplementedError("需实现：参数分组 + 优化器实例化")


def train_one_epoch(model: RouteTransformer, dataloader, optimizer,
                    criterion: RouteLoss, device: torch.device,
                    epoch: int, config: dict) -> dict:
    """训练一个 epoch.

    包含 Teacher Forcing + Scheduled Sampling 策略。

    Args:
        model: 路线生成模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备
        epoch: 当前 epoch 数
        config: 配置字典

    Returns:
        包含 loss 等指标的字典
    """
    raise NotImplementedError("需实现：训练循环")


@torch.no_grad()
def validate(model: RouteTransformer, dataloader, criterion: RouteLoss,
             device: torch.device) -> dict:
    """验证集评估.

    Args:
        model: 路线生成模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备

    Returns:
        包含 val_loss 等指标的字典
    """
    raise NotImplementedError("需实现：验证循环")


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

    # 数据加载
    train_loader, val_loader, test_loader = create_dataloaders("data/processed", config)

    # 模型初始化
    model = RouteTransformer(config).to(device)
    criterion = RouteLoss(
        ce_weight=config["loss"]["ce_weight"],
        distance_weight=config["loss"]["distance_weight"],
        mhc_weight=config["loss"]["mhc_weight"],
    )
    optimizer = build_optimizer(model, config)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        raise NotImplementedError("需实现：加载 checkpoint")

    # 日志
    save_dir = Path(config["experiment"]["save_dir"])
    log_dir = Path(config["experiment"]["log_dir"])
    save_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 训练循环
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, config["training"]["epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        criterion, device, epoch, config)
        val_metrics = validate(model, val_loader, criterion, device)

        # TensorBoard 记录
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        # Checkpoint 保存 & 早停
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["patience"]:
                print(f"早停于 epoch {epoch}")
                break

    writer.close()


if __name__ == "__main__":
    main()
