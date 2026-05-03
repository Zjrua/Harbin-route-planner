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
        """
        Args:
            x: [batch, seq_len, d_model]
            adj_mask: [batch, seq_len, seq_len] 邻接矩阵偏置
        """
        raise NotImplementedError("需实现：Self-Attention + 邻接偏置 + 残差")


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
        """
        Args:
            poi_features: [batch, n_pois, d_model]
            adjacency: [batch, n_pois, n_pois]

        Returns:
            [batch, n_pois, d_model]
        """
        raise NotImplementedError("需实现：多层 GraphAttention + FFN")
