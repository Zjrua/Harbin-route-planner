"""损失函数：交叉熵 + 距离惩罚 + MHC 正则项.

总损失 = ce_weight * CE + distance_weight * DistLoss + mhc_weight * MHCReg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RouteLoss(nn.Module):
    """路线生成多目标损失函数.

    三个损失分量:
    1. route_cross_entropy: 标准交叉熵，衡量 POI 预测准确性
    2. distance_loss: 路线总距离惩罚，鼓励生成短距离路线
    3. mhc_regularization: 双曲空间正则，保持嵌入的流形约束
    """

    def __init__(self, ce_weight: float = 1.0, distance_weight: float = 0.5,
                 mhc_weight: float = 0.05):
        super().__init__()
        self.ce_weight = ce_weight
        self.distance_weight = distance_weight
        self.mhc_weight = mhc_weight

    def route_cross_entropy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """路线交叉熵损失，忽略 padding (index=0)."""
        batch_size, route_len, n_pois = pred.shape
        return F.cross_entropy(pred.reshape(-1, n_pois), target.reshape(-1), ignore_index=0)

    def distance_loss(self, pred_route: torch.Tensor,
                      distances: torch.Tensor) -> torch.Tensor:
        """路线总距离惩罚：沿路线累加相邻 POI 距离，归一化到 [0, 1]."""
        batch_size, route_len = pred_route.shape
        n_pois = distances.size(-1)
        total_dist = torch.zeros(batch_size, device=pred_route.device)
        for t in range(route_len - 1):
            src = pred_route[:, t].clamp(0, n_pois - 1)
            dst = pred_route[:, t + 1].clamp(0, n_pois - 1)
            total_dist = total_dist + distances[torch.arange(batch_size, device=pred_route.device), src, dst]
        # 归一化：除以最大距离 * 路线长度，避免远郊景点导致 loss 爆炸
        max_dist = distances.max() + 1e-8
        return (total_dist / (max_dist * route_len)).mean()

    def mhc_regularization(self, embeddings: torch.Tensor) -> torch.Tensor:
        """MHC 正则化：惩罚偏离庞加莱球的嵌入."""
        norms_sq = (embeddings ** 2).sum(dim=-1)
        return norms_sq.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                distances: torch.Tensor,
                embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """加权总损失."""
        loss = self.ce_weight * self.route_cross_entropy(pred, target)

        # 获取预测路线（greedy）
        pred_route = pred.argmax(dim=-1)
        loss = loss + self.distance_weight * self.distance_loss(pred_route, distances)

        if embeddings is not None:
            loss = loss + self.mhc_weight * self.mhc_regularization(embeddings)

        return loss
