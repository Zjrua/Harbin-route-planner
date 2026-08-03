"""Graph-aware Transformer Encoder.

在标准 Transformer Encoder 基础上，融合路网邻接信息和活动类型相似性作为图结构偏置，
使编码器能感知 POI 之间的空间拓扑关系和活动类型转换模式。
"""

import torch
import torch.nn as nn
from typing import Optional


class GraphAttentionLayer(nn.Module):
    """图注意力层：在 Self-Attention 中注入邻接矩阵偏置 + 活动类型偏置."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mask: Optional[torch.Tensor] = None,
                activity_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Self-Attention + 邻接偏置 + 活动类型偏置 + 残差连接."""
        residual = x
        x_norm = self.norm(x)

        if adj_mask is not None:
            attn_out, _ = self.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=False,
            )
            # 将邻接信息作为残差融合
            attn_out = attn_out + adj_mask @ x_norm
        else:
            attn_out, _ = self.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=False,
            )

        # 活动类型偏置：同类型 POI 之间增强注意力
        if activity_bias is not None:
            attn_out = attn_out + activity_bias @ x_norm

        return self.dropout(attn_out) + residual


class GraphAwareEncoder(nn.Module):
    """Graph-aware Transformer Encoder.

    将路网邻接矩阵和活动类型相似性编码为注意力偏置，
    叠加在标准 Self-Attention 上，使模型能够感知：
    1. POI 之间的连通关系和地理距离
    2. 活动类型转换模式（景点→餐饮→住宿的合理顺序）
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, poi_features: torch.Tensor,
                adjacency: torch.Tensor,
                activity_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """多层 GraphAttention + FFN."""
        x = poi_features
        for layer in self.layers:
            x = layer(x, adjacency, activity_bias)
        x = self.norm(x)
        x = x + self.ffn(x)
        return x
