"""PyTorch Dataset & DataLoader for tourism route data.

Optimized for large POI sets: shared matrices (features, adjacency, distances)
are loaded once and accessed via get_shared_data(), NOT returned per sample.
This avoids duplicating O(n^2) matrices across the batch.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import numpy as np
from pathlib import Path


class ItineraryDataset(Dataset):
    """旅游路线数据集.

    __getitem__ 只返回per-sample数据（路线序列、评分、活动类型）。
    共享矩阵（poi_features, adjacency, distances等）通过
    get_shared_data() 一次性获取，避免DataLoader每样本复制大矩阵。
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
        self.poi_activity_types = None
        self.routes = []
        self.route_scores = []

        self._load_data(train_ratio, val_ratio)

    def _load_data(self, train_ratio: float, val_ratio: float) -> None:
        data_path = Path(self.data_dir)

        self.poi_features = np.load(data_path / "poi_features.npy")
        self.adjacency = np.load(data_path / "adjacency.npy")
        self.distances = np.load(data_path / "distance_matrix.npy")

        activity_path = data_path / "poi_activity_types.npy"
        if activity_path.exists():
            self.poi_activity_types = np.load(activity_path)
        else:
            self.poi_activity_types = np.zeros(self.poi_features.shape[0], dtype=np.int64)

        std_path = data_path / "distance_std.npy"
        if std_path.exists():
            self.distances_std = np.load(std_path)
        elif self.use_probabilistic_edges:
            self.distances_std = self.distances * self.noise_scale

        routes_data = np.load(data_path / "routes.npy", allow_pickle=True)

        import pandas as pd
        metadata_path = data_path / "poi_metadata.csv"
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            poi_scores = metadata.get("rating",
                        metadata.get("score", np.ones(len(metadata)))).values
        else:
            poi_scores = np.ones(self.poi_features.shape[0])

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
        self.route_scores = []
        for idx in split_indices:
            route = routes_data[idx]
            if len(route) > 0 and route.max() < len(poi_scores):
                self.route_scores.append(float(poi_scores[route].mean()))
            else:
                self.route_scores.append(1.0)

    def get_shared_data(self, device: torch.device = None) -> dict:
        """Return shared tensors (features, adjacency, distances, activity types).

        Load these to GPU once before training to avoid per-sample duplication.
        """
        pf = torch.tensor(self.poi_features, dtype=torch.float32)
        adj = torch.tensor(self.adjacency, dtype=torch.float32)
        dist = torch.tensor(self.distances, dtype=torch.float32)
        dist_std = torch.tensor(self.distances_std, dtype=torch.float32) if self.distances_std is not None else None
        pat = torch.tensor(self.poi_activity_types, dtype=torch.long)

        if device is not None:
            pf = pf.to(device)
            adj = adj.to(device)
            dist = dist.to(device)
            if dist_std is not None:
                dist_std = dist_std.to(device)
            pat = pat.to(device)

        result = {
            "poi_features": pf,
            "adjacency": adj,
            "distances": dist,
            "poi_activity_types": pat,
        }
        if dist_std is not None:
            result["distance_std"] = dist_std
        return result

    def sample_noisy_distances(self, device: torch.device = None) -> torch.Tensor:
        """Sample noisy distance matrix for training (probabilistic edges)."""
        if self.is_train and self.use_probabilistic_edges and self.distances_std is not None:
            noise = np.random.randn(*self.distances.shape).astype(np.float32)
            sampled = np.clip(self.distances + noise * self.distances_std, 0, None)
            t = torch.tensor(sampled, dtype=torch.float32)
        else:
            t = torch.tensor(self.distances, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t

    def __len__(self) -> int:
        return len(self.routes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Return per-sample data only: route, score, route_activity_types."""
        route = self.routes[idx]
        route_len = len(route)
        if route_len >= self.max_route_len:
            route = route[:self.max_route_len]
        else:
            route = np.pad(route, (0, self.max_route_len - route_len), constant_values=0)

        score = self.route_scores[idx] if idx < len(self.route_scores) else 1.0

        route_activity_types = np.zeros(self.max_route_len, dtype=np.int64)
        for i, poi_idx in enumerate(route):
            if poi_idx < len(self.poi_activity_types):
                route_activity_types[i] = self.poi_activity_types[poi_idx]

        return (
            torch.tensor(route, dtype=torch.long),
            torch.tensor(score, dtype=torch.float32),
            torch.tensor(route_activity_types, dtype=torch.long),
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

    train_ds = ItineraryDataset(**common, split="train")
    val_ds = ItineraryDataset(**common, split="val")
    test_ds = ItineraryDataset(**common, split="test")

    num_workers = exp_cfg.get("num_workers", 0)
    pw = num_workers > 0

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=pw)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, persistent_workers=pw)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True, persistent_workers=pw)

    return train_loader, val_loader, test_loader
