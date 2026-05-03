"""测试数据集和数据加载功能."""

import pytest
import torch
import numpy as np


class TestHarbinRouteDataset:
    """HarbinRouteDataset 单元测试."""

    def test_dataset_length(self):
        """测试数据集长度返回正确."""
        # TODO: 使用 mock 数据测试
        pass

    def testgetitem_returns_correct_shapes(self):
        """测试 __getitem__ 返回正确形状的张量."""
        # TODO: 验证 (poi_features, adjacency, route, scores) 形状
        pass

    def test_train_val_test_split(self):
        """测试数据集划分比例正确."""
        # TODO: 验证 train/val/test 划分无重叠
        pass
