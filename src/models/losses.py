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
        """路线交叉熵损失.

        Args:
            pred: 预测 logits, shape [batch, route_len, n_pois]
            target: 目标 POI 索引, shape [batch, route_len]

        Returns:
            标量损失值
        """
        raise NotImplementedError("需实现：reshape 后调用 F.cross_entropy")

    def distance_loss(self, pred_route: torch.Tensor,
                      distances: torch.Tensor) -> torch.Tensor:
        """路线总距离惩罚.

        鼓励模型生成距离较短的路线，避免不走回头路。

        Args:
            pred_route: 预测的路线 POI 索引, shape [batch, route_len]
            distances: POI 间距离矩阵, shape [batch, n_pois, n_pois]

        Returns:
            标量损失值（归一化后的路线总距离）
        """
        raise NotImplementedError("需实现：沿路线累加相邻 POI 距离")

    def mhc_regularization(self, embeddings: torch.Tensor) -> torch.Tensor:
        """MHC 双曲空间正则化损失.

        惩罚偏离庞加莱球的嵌入，保持双曲流形约束。

        Args:
            embeddings: POI 嵌入, shape [n_pois, dim]

        Returns:
            标量正则化损失
        """
        raise NotImplementedError("需实现：惩罚 ||x|| >= 1/sqrt(c) 的嵌入")

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                distances: torch.Tensor,
                embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """加权总损失.

        Args:
            pred: 预测 logits, shape [batch, route_len, n_pois]
            target: 目标 POI 索引, shape [batch, route_len]
            distances: POI 距离矩阵, shape [batch, n_pois, n_pois]
            embeddings: MHC 嵌入（可选）, shape [n_pois, dim]

        Returns:
            总损失标量
        """
        raise NotImplementedError("需实现：加权求和三个损失分量")
