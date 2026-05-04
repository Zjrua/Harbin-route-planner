"""POI Embedding + 正弦位置编码 + 活动类型嵌入.

将离散的 POI 标识、活动类型和路线位置信息映射为连续的向量表征。
"""

import math
import torch
import torch.nn as nn


# 活动类型常量
ACTIVITY_TYPES = {
    "景点": 0,
    "餐饮": 1,
    "住宿": 2,
    "交通": 3,
    "购物": 4,
    "出发点": 5,
}
NUM_ACTIVITY_TYPES = len(ACTIVITY_TYPES)


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
    - 活动类型嵌入（景点/餐饮/住宿/交通/购物/出发点）
    - 评分嵌入
    """

    def __init__(self, num_pois: int, d_model: int, max_route_len: int = 20):
        super().__init__()
        self.poi_embed = nn.Embedding(num_pois, d_model)
        self.category_embed = nn.Embedding(10, d_model)  # 10 个 POI 类别
        self.activity_type_embed = nn.Embedding(NUM_ACTIVITY_TYPES, d_model)  # 活动类型嵌入
        self.score_proj = nn.Linear(1, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_route_len)

    def forward(self, poi_ids: torch.Tensor, categories: torch.Tensor = None,
                scores: torch.Tensor = None, activity_types: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            poi_ids: [batch, seq_len]
            categories: [batch, seq_len]（可选）
            scores: [batch, seq_len, 1]（可选）
            activity_types: [batch, seq_len]（可选）活动类型标签

        Returns:
            [batch, seq_len, d_model]
        """
        x = self.poi_embed(poi_ids)
        if categories is not None:
            x = x + self.category_embed(categories)
        if activity_types is not None:
            x = x + self.activity_type_embed(activity_types)
        if scores is not None:
            x = x + self.score_proj(scores.unsqueeze(-1))
        return self.pos_encoding(x)
