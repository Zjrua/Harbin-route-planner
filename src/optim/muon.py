"""Muon 优化器实现.

灵感来自 DeepSeek 论文，Muon 优化器通过矩阵正交化更新梯度，
在注意力层和 FFN 层使用不同的学习率，提升训练稳定性。

核心思想:
- 对梯度进行 Newton-Schulze 迭代近似正交化
- 注意力层使用较低学习率，FFN 层使用较高学习率
- 比标准 Adam 在 Transformer 训练中收敛更快
"""

import torch
from torch.optim import Optimizer
from typing import Iterable


class MuonOptimizer(Optimizer):
    """Muon 优化器：矩阵正交化梯度更新.

    支持三种参数分组：
    - attention_params: 注意力层参数，使用 lr_attn
    - ffn_params: FFN 层参数，使用 lr_ffn
    - other_params: 其他参数，使用 (lr_attn + lr_ffn) / 2
    """

    def __init__(self, param_groups: Iterable[dict],
                 lr_attn: float = 3e-4, lr_ffn: float = 1e-3,
                 momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 1e-4):
        """
        Args:
            param_groups: 参数分组列表，每组需包含 'params' 和 'group_type' 键
            lr_attn: 注意力层学习率
            lr_ffn: FFN 层学习率
            momentum: 动量系数
            nesterov: 是否使用 Nesterov 动量
            ns_steps: Newton-Schulze 正交化迭代步数
            weight_decay: 权重衰减
        """
        defaults = dict(
            lr_attn=lr_attn, lr_ffn=lr_ffn,
            momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def _newton_schulze_orthogonalize(self, G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """Newton-Schulze 迭代近似正交化.

        将梯度矩阵 G 近似正交化，避免梯度方向退化。

        Args:
            G: 梯度矩阵
            steps: 迭代步数

        Returns:
            正交化后的梯度矩阵
        """
        raise NotImplementedError("需实现 Newton-Schulze 迭代公式")

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步参数更新.

        对每个参数组:
        1. 应用权重衰减
        2. 对矩阵参数（>=2D）进行 Newton-Schulze 正交化
        3. 根据参数组类型选择学习率
        4. 应用动量更新
        """
        raise NotImplementedError("需实现：分组学习率 + 正交化 + 动量更新")
