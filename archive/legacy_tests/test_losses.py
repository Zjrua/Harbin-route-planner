"""测试 RouteLoss（CE + 距离惩罚 + MHC 正则）.

RouteLoss 驱动所有训练，此前零覆盖。本测试验证：
1. CE 损失忽略 padding（index=0）
2. distance_loss 支持 2D（共享）和 3D（per-batch）距离矩阵
3. MHC 正则项非负
4. forward 的加权求和正确
"""

import pytest
import torch

from src.models.losses import RouteLoss


class TestRouteLoss:
    """RouteLoss 单元测试."""

    @pytest.fixture
    def criterion(self):
        return RouteLoss(ce_weight=1.0, distance_weight=0.01, mhc_weight=0.05)

    def test_ce_ignores_padding(self, criterion):
        """CE 损失应忽略 padding（ignore_index=0）.

        策略：把 padding 位置的预测改成极端值，loss 应不变（因为被忽略）。
        """
        torch.manual_seed(0)
        pred = torch.randn(2, 5, 100)
        target = torch.tensor([[5, 10, 15, 0, 0],   # 后两位是 padding
                               [20, 25, 0, 0, 0]])  # 后三位是 padding
        ce_before = criterion.route_cross_entropy(pred, target)

        # 把 padding 位置的预测改成完全不同的值，loss 应保持不变
        pred_modified = pred.clone()
        pred_modified[0, 3:] = torch.randn(2, 100) * 1000  # padding 位置大改
        pred_modified[1, 2:] = torch.randn(3, 100) * 1000
        ce_after = criterion.route_cross_entropy(pred_modified, target)

        assert torch.allclose(ce_before, ce_after, atol=1e-5), (
            f"CE 应忽略 padding，但改了 padding 位置后 loss 变化: {ce_before} → {ce_after}")

    def test_distance_loss_2d_shared(self, criterion):
        """distance_loss 支持 2D 共享距离矩阵."""
        n_pois = 100
        dist = torch.rand(n_pois, n_pois) * 10
        pred_route = torch.tensor([[5, 10, 15, 20], [30, 40, 50, 60]])
        loss = criterion.distance_loss(pred_route, dist)
        # 应是正值且有限
        assert loss.item() > 0
        assert torch.isfinite(loss)
        # 归一化后应 <= 1（除以 max_dist * route_len）
        assert loss.item() <= 1.0 + 1e-5

    def test_distance_loss_3d_per_batch(self, criterion):
        """distance_loss 支持 3D per-batch 距离矩阵."""
        n_pois = 100
        dist_3d = torch.rand(2, n_pois, n_pois) * 10
        pred_route = torch.tensor([[5, 10, 15, 20], [30, 40, 50, 60]])
        loss = criterion.distance_loss(pred_route, dist_3d)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_mhc_regularization_non_negative(self, criterion):
        """MHC 正则项（嵌入范数平方均值）应非负."""
        embeddings = torch.randn(100, 32)
        reg = criterion.mhc_regularization(embeddings)
        assert reg.item() >= 0
        # 对零嵌入应为 0
        zero_reg = criterion.mhc_regularization(torch.zeros(10, 32))
        assert torch.allclose(zero_reg, torch.tensor(0.0))

    def test_forward_weighted_sum(self, criterion):
        """forward 应正确加权求和：ce*1.0 + dist*0.01 + mhc*0.05."""
        pred = torch.randn(2, 5, 100)
        target = torch.tensor([[5, 10, 15, 20, 25], [30, 40, 50, 60, 70]])
        dist = torch.rand(100, 100) * 10
        embeddings = torch.randn(100, 32)

        loss = criterion(pred, target, dist, embeddings)

        # 手动复算
        ce = criterion.route_cross_entropy(pred, target)
        pred_route = pred.argmax(dim=-1)
        dl = criterion.distance_loss(pred_route, dist)
        mhc = criterion.mhc_regularization(embeddings)
        expected = 1.0 * ce + 0.01 * dl + 0.05 * mhc

        assert torch.allclose(loss, expected, atol=1e-5)

    def test_forward_without_embeddings(self, criterion):
        """forward 不传 embeddings 时应跳过 MHC 项（MHC 关闭场景）."""
        pred = torch.randn(2, 5, 100)
        target = torch.tensor([[5, 10, 15, 20, 25], [30, 40, 50, 60, 70]])
        dist = torch.rand(100, 100) * 10

        loss = criterion(pred, target, dist, embeddings=None)

        ce = criterion.route_cross_entropy(pred, target)
        pred_route = pred.argmax(dim=-1)
        dl = criterion.distance_loss(pred_route, dist)
        expected = 1.0 * ce + 0.01 * dl  # 无 MHC 项

        assert torch.allclose(loss, expected, atol=1e-5)

    def test_loss_is_differentiable(self, criterion):
        """损失应可反向传播（梯度流回 pred）."""
        pred = torch.randn(2, 5, 100, requires_grad=True)
        target = torch.tensor([[5, 10, 15, 20, 25], [30, 40, 50, 60, 70]])
        dist = torch.rand(100, 100) * 10
        loss = criterion(pred, target, dist, None)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
