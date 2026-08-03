"""测试数据集和数据加载功能.

用 tmp_path 构造最小数据集，不依赖真实的 data/processed 大文件。
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from src.data.dataset import ItineraryDataset, create_dataloaders


@pytest.fixture
def fake_data_dir(tmp_path):
    """在临时目录构造最小数据集（20 个 POI，10 条路线）."""
    n_pois = 20
    # 共享矩阵
    np.save(tmp_path / "poi_features.npy", np.random.rand(n_pois, 8).astype(np.float32))
    np.save(tmp_path / "adjacency.npy", (np.random.rand(n_pois, n_pois) < 0.3).astype(np.float32))
    np.save(tmp_path / "distance_matrix.npy",
            (np.random.rand(n_pois, n_pois) * 10 + 0.1).astype(np.float32))
    np.save(tmp_path / "distance_std.npy",
            (np.random.rand(n_pois, n_pois)).astype(np.float32))
    np.save(tmp_path / "poi_activity_types.npy",
            np.random.randint(0, 6, n_pois).astype(np.int64))
    # POI metadata（评分）
    pd.DataFrame({
        "name": [f"POI{i}" for i in range(n_pois)],
        "rating": np.linspace(3.5, 5.0, n_pois),
        "category": ["景点"] * n_pois,
    }).to_csv(tmp_path / "poi_metadata.csv", index=False, encoding="utf-8")
    # 10 条变长路线（每条 3-5 个 POI 索引，1-indexed，0 为 padding）
    routes = [np.array([1, 2, 3, 4, 5]),
              np.array([1, 6, 7]),
              np.array([2, 8, 9, 10]),
              np.array([3, 11, 12, 13, 14]),
              np.array([4, 15, 16]),
              np.array([5, 17, 18]),
              np.array([6, 19, 1, 2]),
              np.array([7, 8, 9]),
              np.array([10, 11, 12]),
              np.array([13, 14, 15, 16])]
    np.save(tmp_path / "routes.npy", np.array(routes, dtype=object))
    return str(tmp_path)


class TestItineraryDataset:
    """ItineraryDataset 单元测试."""

    def test_dataset_length_per_split(self, fake_data_dir):
        """train/val/test 长度符合 0.8/0.1/0.1 划分."""
        train = ItineraryDataset(fake_data_dir, split="train")
        val = ItineraryDataset(fake_data_dir, split="val")
        test = ItineraryDataset(fake_data_dir, split="test")
        assert len(train) == 8   # 10 * 0.8
        assert len(val) == 1     # 10 * 0.1
        assert len(test) == 1

    def test_no_overlap_between_splits(self, fake_data_dir):
        """train/val/test 三划分无样本重叠（基于固定 seed=42 的 shuffle）."""
        train = ItineraryDataset(fake_data_dir, split="train")
        val = ItineraryDataset(fake_data_dir, split="val")
        test = ItineraryDataset(fake_data_dir, split="test")
        # 用路线首 POI 作为弱指纹检查无重叠（10 条路线首 POI 各不相同）
        train_heads = {r[0] for r in train.routes}
        val_heads = {r[0] for r in val.routes}
        test_heads = {r[0] for r in test.routes}
        assert train_heads.isdisjoint(val_heads)
        assert train_heads.isdisjoint(test_heads)
        assert val_heads.isdisjoint(test_heads)

    def test_getitem_returns_per_sample_tensors(self, fake_data_dir):
        """__getitem__ 返回 (route, score, activity_types) 三个 per-sample 数据."""
        train = ItineraryDataset(fake_data_dir, split="train")
        route, score, activity = train[0]
        assert isinstance(route, torch.Tensor)
        assert route.dtype == torch.long
        assert route.dim() == 1
        assert isinstance(score, torch.Tensor)   # score 被包成 tensor
        assert score.dim() == 0                   # 标量
        assert isinstance(activity, torch.Tensor)
        assert activity.dtype == torch.long

    def test_get_shared_data_shapes(self, fake_data_dir):
        """get_shared_data 返回正确形状的共享张量."""
        train = ItineraryDataset(fake_data_dir, split="train")
        shared = train.get_shared_data()
        assert shared["poi_features"].shape == (20, 8)
        assert shared["adjacency"].shape == (20, 20)
        assert shared["distances"].shape == (20, 20)
        assert shared["poi_activity_types"].shape == (20,)

    def test_create_dataloaders(self, fake_data_dir):
        """create_dataloaders 返回三个 loader，train 可迭代."""
        config = {
            "data": {"use_probabilistic_edges": False},
            "model": {"max_route_len": 10},
            "experiment": {"train_ratio": 0.8, "val_ratio": 0.1, "num_workers": 0},
            "training": {"batch_size": 4},
        }
        train_loader, val_loader, test_loader = create_dataloaders(fake_data_dir, config)
        batch = next(iter(train_loader))
        route_seq, scores, activity = batch
        assert route_seq.shape[0] <= 4   # batch_size
        assert route_seq.dtype == torch.long

    def test_holdout_split_used_for_test(self, fake_data_dir):
        """方向B：当 routes_xhs_holdout.npy 存在时，test split 优先用它（不参与 shuffle）."""
        # 构造一个 holdout 文件（3 条路线，与主 routes 不重叠）
        holdout_path = Path(fake_data_dir) / "routes_xhs_holdout.npy"
        holdout_routes = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10])]
        np.save(holdout_path, np.array(holdout_routes, dtype=object))

        # test split 应使用 holdout（3 条），而非主 routes 的 10%
        test = ItineraryDataset(fake_data_dir, split="test")
        assert len(test) == 3, f"test 应用 holdout（3条），实际 {len(test)}"

        # train/val 不受影响，仍从主 routes 切（8 + 1）
        train = ItineraryDataset(fake_data_dir, split="train")
        val = ItineraryDataset(fake_data_dir, split="val")
        assert len(train) == 8
        assert len(val) == 1
