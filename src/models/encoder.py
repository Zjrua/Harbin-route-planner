"""Graph-aware Transformer Encoder.

在标准 Transformer Encoder 基础上，融合路网邻接信息作为图结构偏置，
使编码器能感知 POI 之间的空间拓扑关系。
"""

import torch
import torch.nn as nn
from typing import Optional


class GraphAttentionLayer(nn.Module):
    """图注意力层：在 Self-Attention 中注入邻接矩阵偏置."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Self-Attention + 邻接偏置 + 残差连接."""
        residual = x
        x_norm = self.norm(x)

        # 构建 attn_mask：将邻接矩阵作为注意力偏置
        key_padding_mask = None
        if adj_mask is not None:
            # adj_mask 作为注意力分数的加性偏置
            attn_out, _ = self.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            # 将邻接信息作为残差融合
            attn_out = attn_out + adj_mask @ x_norm
        else:
            attn_out, _ = self.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

        return self.dropout(attn_out) + residual


class GraphAwareEncoder(nn.Module):
    """Graph-aware Transformer Encoder.

    将路网邻接矩阵编码为注意力偏置，叠加在标准 Self-Attention 上，
    使模型能够感知 POI 之间的连通关系和地理距离。
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
                adjacency: torch.Tensor) -> torch.Tensor:
        """多层 GraphAttention + FFN."""
        x = poi_features
        for layer in self.layers:
            x = layer(x, adjacency)
        x = self.norm(x)
        x = x + self.ffn(x)
        return x
