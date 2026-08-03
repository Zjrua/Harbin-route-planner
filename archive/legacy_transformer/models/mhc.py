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
        """指数映射：从切空间映射到庞加莱球."""
        if x is None:
            # 原点处的指数映射：exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
            v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=1e-10)
            return torch.tanh(torch.sqrt(torch.tensor(self.c, device=v.device)) * v_norm) * v / (
                torch.sqrt(torch.tensor(self.c, device=v.device)) * v_norm
            )
        # 非原点处的指数映射
        c = self.c
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True).clamp(min=1e-10)
        lambda_x = 2.0 / (1.0 - c * (x * x).sum(dim=-1, keepdim=True)).clamp(min=1e-10)
        inner = (x * v).sum(dim=-1, keepdim=True)
        sqrt_c = torch.sqrt(torch.tensor(c, device=v.device))
        direction = v / v_norm_sq.sqrt().clamp(min=1e-10) * torch.tanh(
            sqrt_c * lambda_x * v_norm_sq.sqrt() / 2.0
        )
        return self.project(
            x + direction * (2.0 / (lambda_x.clamp(min=1e-10)))
            * (1.0 / (1.0 + c * inner / lambda_x.clamp(min=1e-10)).clamp(min=1e-10))
        )

    def logmap(self, y: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """对数映射：从庞加莱球映射到切空间."""
        if x is None:
            # 原点处的对数映射：log_0(y) = arctanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
            y_norm = torch.clamp(y.norm(dim=-1, keepdim=True), min=1e-10)
            sqrt_c = torch.sqrt(torch.tensor(self.c, device=y.device))
            return torch.atanh(torch.clamp(sqrt_c * y_norm, max=1.0 - 1e-7)) * y / (sqrt_c * y_norm)
        # 非原点处的对数映射
        diff = y - x
        diff_norm = torch.clamp(diff.norm(dim=-1, keepdim=True), min=1e-10)
        return diff / diff_norm * self.geodesic_distance(x, y).unsqueeze(-1)

    def geodesic_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """计算两点间的测地距离."""
        c = self.c
        diff_sq = ((u - v) ** 2).sum(dim=-1)
        u_sq = (u ** 2).sum(dim=-1)
        v_sq = (v ** 2).sum(dim=-1)
        denom = (1 - c * u_sq) * (1 - c * v_sq)
        inner = 1.0 + 2.0 * c * diff_sq / denom.clamp(min=1e-10)
        return torch.acosh(inner.clamp(min=1.0 + 1e-7)) / torch.sqrt(torch.tensor(c, device=u.device))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """投影到庞加莱球内，确保 ||x|| < 1/sqrt(c)."""
        max_norm = 1.0 / torch.sqrt(torch.tensor(self.c, device=x.device)) - 1e-5
        x_norm = x.norm(dim=-1, keepdim=True)
        scale = torch.where(x_norm > max_norm, max_norm / x_norm.clamp(min=1e-10), torch.ones_like(x_norm))
        return x * scale

    def forward(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播：获取 POI 的双曲嵌入，已投影到庞加莱球内."""
        if indices is None:
            return self.project(self.embedding)
        return self.project(self.embedding[indices])
