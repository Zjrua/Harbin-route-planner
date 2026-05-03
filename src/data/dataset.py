"""PyTorch Dataset & DataLoader for Harbin tourism route data."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import numpy as np


class HarbinRouteDataset(Dataset):
    """哈尔滨旅游路线数据集.

    每条样本包含:
    - poi_features: 所有 POI 的特征矩阵 [n_pois, feature_dim]
    - adjacency: POI 间路网邻接矩阵 [n_pois, n_pois]
    - route_sequence: 一条历史路线的 POI 索引序列 [route_len]
    - scores: 路线评分 [1]

    支持训练/验证/测试划分。
    """

    def __init__(self, data_dir: str, split: str = "train",
                 train_ratio: float = 0.8, val_ratio: float = 0.1,
                 max_route_len: int = 20, transform=None):
        """
        Args:
            data_dir: 处理后数据目录路径
            split: 数据集划分，"train" / "val" / "test"
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            max_route_len: 路线最大长度（截断或填充）
            transform: 可选的数据变换
        """
        self.data_dir = data_dir
        self.split = split
        self.max_route_len = max_route_len
        self.transform = transform

        # 需要加载的数据
        self.poi_features = None    # [n_pois, feature_dim]
        self.adjacency = None       # [n_pois, n_pois]
        self.distances = None       # [n_pois, n_pois]
        self.routes = []            # List of route index arrays
        self.scores = []            # List of route scores

        self._load_data(train_ratio, val_ratio)

    def _load_data(self, train_ratio: float, val_ratio: float) -> None:
        """加载并划分数据.

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        raise NotImplementedError("需实现：加载 processed/ 目录下的数据文件")

    def __len__(self) -> int:
        return len(self.routes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            (poi_features, adjacency, route_sequence, scores) 元组
        """
        raise NotImplementedError("需实现：返回单个样本的张量元组")


def create_dataloaders(data_dir: str, config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试 DataLoader.

    Args:
        data_dir: 数据目录
        config: 配置字典

    Returns:
        (train_loader, val_loader, test_loader)
    """
    exp_cfg = config["experiment"]
    train_ds = HarbinRouteDataset(data_dir, "train",
                                  train_ratio=exp_cfg["train_ratio"],
                                  val_ratio=exp_cfg["val_ratio"],
                                  max_route_len=config["model"]["max_route_len"])
    val_ds = HarbinRouteDataset(data_dir, "val",
                                train_ratio=exp_cfg["train_ratio"],
                                val_ratio=exp_cfg["val_ratio"],
                                max_route_len=config["model"]["max_route_len"])
    test_ds = HarbinRouteDataset(data_dir, "test",
                                 train_ratio=exp_cfg["train_ratio"],
                                 val_ratio=exp_cfg["val_ratio"],
                                 max_route_len=config["model"]["max_route_len"])

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=exp_cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=exp_cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=exp_cfg["num_workers"])

    return train_loader, val_loader, test_loader
