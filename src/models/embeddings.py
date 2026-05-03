"""POI Embedding + 正弦位置编码.

将离散的 POI 标识和路线位置信息映射为连续的向量表征。
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """正弦余弦位置编码."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class POIEmbedding(nn.Module):
    """POI 嵌入层：将 POI 索引映射为向量，叠加正弦位置编码.

    支持多种 POI 特征：
    - POI ID 嵌入
    - 类别嵌入（景点/餐饮/住宿/交通）
    - 评分嵌入
    """

    def __init__(self, num_pois: int, d_model: int, max_route_len: int = 20):
        super().__init__()
        self.poi_embed = nn.Embedding(num_pois, d_model)
        self.category_embed = nn.Embedding(10, d_model)  # 10 个 POI 类别
        self.score_proj = nn.Linear(1, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_route_len)

    def forward(self, poi_ids: torch.Tensor, categories: torch.Tensor = None,
                scores: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            poi_ids: [batch, seq_len]
            categories: [batch, seq_len]（可选）
            scores: [batch, seq_len, 1]（可选）

        Returns:
            [batch, seq_len, d_model]
        """
        x = self.poi_embed(poi_ids)
        if categories is not None:
            x = x + self.category_embed(categories)
        if scores is not None:
            x = x + self.score_proj(scores.unsqueeze(-1))
        return self.pos_encoding(x)
