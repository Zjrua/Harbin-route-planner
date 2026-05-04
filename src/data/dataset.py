"""PyTorch Dataset & DataLoader for Harbin tourism route data."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import numpy as np
from pathlib import Path


class HarbinRouteDataset(Dataset):
    """哈尔滨旅游路线数据集.

    每条样本包含:
    - poi_features: 所有 POI 的特征矩阵 [n_pois, feature_dim]
    - adjacency: POI 间路网邻接矩阵 [n_pois, n_pois]
    - route_sequence: 一条历史路线的 POI 索引序列 [route_len]
    - distances: 距离矩阵 [n_pois, n_pois]（训练时可加噪声）
    - score: 路线评分 [1]

    支持概率分布边权采样：训练时从 N(mean, std) 采样距离/耗时。
    """

    def __init__(self, data_dir: str, split: str = "train",
                 train_ratio: float = 0.8, val_ratio: float = 0.1,
                 max_route_len: int = 20, transform=None,
                 use_probabilistic_edges: bool = False,
                 noise_scale: float = 0.1):
        self.data_dir = data_dir
        self.split = split
        self.max_route_len = max_route_len
        self.transform = transform
        self.use_probabilistic_edges = use_probabilistic_edges
        self.noise_scale = noise_scale
        self.is_train = (split == "train")

        self.poi_features = None
        self.adjacency = None
        self.distances = None
        self.distances_std = None
        self.routes = []
        self.route_scores = []

        self._load_data(train_ratio, val_ratio)

    def _load_data(self, train_ratio: float, val_ratio: float) -> None:
        """加载并划分数据."""
        data_path = Path(self.data_dir)

        self.poi_features = np.load(data_path / "poi_features.npy")
        self.adjacency = np.load(data_path / "adjacency.npy")
        self.distances = np.load(data_path / "distance_matrix.npy")

        # 加载距离方差（可选，用于概率采样）
        std_path = data_path / "distance_std.npy"
        if std_path.exists():
            self.distances_std = np.load(std_path)
        elif self.use_probabilistic_edges:
            # 无显式方差时，用均值的 noise_scale 比例作为标准差
            self.distances_std = self.distances * self.noise_scale

        routes_data = np.load(data_path / "routes.npy", allow_pickle=True)

        # 读取评分
        import pandas as pd
        metadata_path = data_path / "poi_metadata.csv"
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            poi_scores = metadata.get("rating",
                        metadata.get("score", np.ones(len(metadata)))).values
        else:
            poi_scores = np.ones(self.poi_features.shape[0])

        # 划分数据
        n_routes = len(routes_data)
        indices = np.arange(n_routes)
        np.random.seed(42)
        np.random.shuffle(indices)

        train_end = int(n_routes * train_ratio)
        val_end = int(n_routes * (train_ratio + val_ratio))

        if self.split == "train":
            split_indices = indices[:train_end]
        elif self.split == "val":
            split_indices = indices[train_end:val_end]
        else:
            split_indices = indices[val_end:]

        self.routes = [routes_data[i] for i in split_indices]
        # 路线评分 = 路线中各 POI 评分的均值
        self.route_scores = []
        for idx in split_indices:
            route = routes_data[idx]
            if len(route) > 0 and route.max() < len(poi_scores):
                self.route_scores.append(float(poi_scores[route].mean()))
            else:
                self.route_scores.append(1.0)

    def _sample_distances(self) -> np.ndarray:
        """训练时从 N(mean, std) 采样距离矩阵，推理时返回均值."""
        if self.is_train and self.use_probabilistic_edges and self.distances_std is not None:
            noise = np.random.randn(*self.distances.shape).astype(np.float32)
            sampled = self.distances + noise * self.distances_std
            return np.clip(sampled, 0, None).astype(np.float32)
        return self.distances

    def __len__(self) -> int:
        return len(self.routes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """返回单个样本的张量元组."""
        route = self.routes[idx]
        route_len = len(route)
        if route_len >= self.max_route_len:
            route = route[:self.max_route_len]
        else:
            route = np.pad(route, (0, self.max_route_len - route_len), constant_values=0)

        distances = self._sample_distances()
        score = self.route_scores[idx] if idx < len(self.route_scores) else 1.0

        return (
            torch.tensor(self.poi_features, dtype=torch.float32),
            torch.tensor(self.adjacency, dtype=torch.float32),
            torch.tensor(route, dtype=torch.long),
            torch.tensor(distances, dtype=torch.float32),
            torch.tensor(score, dtype=torch.float32),
        )


def create_dataloaders(data_dir: str, config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试 DataLoader."""
    exp_cfg = config["experiment"]
    data_cfg = config.get("data", {})
    use_prob = data_cfg.get("use_probabilistic_edges", False)
    noise_scale = data_cfg.get("distance_noise_scale", 0.1)

    common = dict(
        data_dir=data_dir,
        train_ratio=exp_cfg["train_ratio"],
        val_ratio=exp_cfg["val_ratio"],
        max_route_len=config["model"]["max_route_len"],
        use_probabilistic_edges=use_prob,
        noise_scale=noise_scale,
    )

    train_ds = HarbinRouteDataset(**common, split="train")
    val_ds = HarbinRouteDataset(**common, split="val")
    test_ds = HarbinRouteDataset(**common, split="test")

    num_workers = exp_cfg.get("num_workers", 0)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
