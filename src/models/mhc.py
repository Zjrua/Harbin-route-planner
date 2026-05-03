"""MHC (Manifold Hyperbolic Constraint) 双曲流形约束层.

基于庞加莱球模型(Poincare Ball Model)实现双曲空间中的嵌入，
使地理位置相近的 POI 在双曲空间中也保持较小的测地距离。
灵感来自 DeepSeek 论文中的双曲表征学习技术。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PoincareEmbedding(nn.Module):
    """庞加莱球 Embedding：在双曲空间中学习 POI 表征.

    核心思想：
    - POI 的地理距离关系在双曲空间中被更好地保持
    - 通过指数/对数映射在欧氏切空间和双曲流形间转换
    - 测地距离作为相似度度量，比欧氏距离更符合空间层次结构
    """

    def __init__(self, num_pois: int, dim: int = 64, curvature: float = -1.0):
        """
        Args:
            num_pois: POI 总数
            dim: 嵌入维度
            curvature: 双曲空间曲率（负值），绝对值越大曲率越大
        """
        super().__init__()
        self.num_pois = num_pois
        self.dim = dim
        self.curvature = curvature
        self.c = abs(curvature)  # 内部使用 c = |K|

        # 初始化嵌入，保证在庞加莱球内 (||x|| < 1/sqrt(c))
        self.embedding = nn.Parameter(
            torch.randn(num_pois, dim) * 1e-3
        )

    def expmap(self, v: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """指数映射：从切空间映射到庞加莱球.

        将欧氏切空间中的向量 v 映射到庞加莱球上 x 处的切空间结果。

        Args:
            v: 切空间向量, shape [..., dim]
            x: 流形上的基点, shape [..., dim]，默认为原点

        Returns:
            流形上的点, shape [..., dim]
        """
        raise NotImplementedError("需实现庞加莱球指数映射公式")

    def logmap(self, y: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """对数映射：从庞加莱球映射到切空间.

        将流形上的点 y 映射到 x 处的切空间向量。

        Args:
            y: 流形上的点, shape [..., dim]
            x: 流形上的基点, shape [..., dim]，默认为原点

        Returns:
            切空间向量, shape [..., dim]
        """
        raise NotImplementedError("需实现庞加莱球对数映射公式")

    def geodesic_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """计算两点间的测地距离.

        庞加莱球模型中两点的测地距离公式：
        d(u,v) = (1/sqrt(c)) * arcosh(1 + 2c * ||u-v||^2 / ((1-c||u||^2)(1-c||v||^2)))

        Args:
            u: 第一个点, shape [..., dim]
            v: 第二个点, shape [..., dim]

        Returns:
            测地距离, shape [...]
        """
        raise NotImplementedError("需实现测地距离公式")

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """投影到庞加莱球内.

        确保所有嵌入点严格在庞加莱球内：||x|| < 1/sqrt(c)。
        对超出边界的点进行径向投影。

        Args:
            x: 待投影的向量, shape [..., dim]

        Returns:
            投影后的向量, shape [..., dim]
        """
        raise NotImplementedError("需实现：clamp 到庞加莱球内")

    def forward(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播：获取 POI 的双曲嵌入.

        Args:
            indices: POI 索引, shape [...]，为 None 时返回全部嵌入

        Returns:
            双曲嵌入, shape [..., dim]，已投影到庞加莱球内
        """
        raise NotImplementedError("需实现：索引查找 + 投影")
