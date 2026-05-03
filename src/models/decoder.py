"""Masked Transformer Decoder + Cross-Attention + Engram 记忆增强.

解码器负责自回归地逐步生成路线中的下一个 POI，
通过 Cross-Attention 获取编码器上下文，通过 Engram 检索历史优质路线。
"""

import torch
import torch.nn as nn
from typing import Optional


class EngramDecoder(nn.Module):
    """带 Engram 记忆增强的 Transformer Decoder.

    每个解码层包含:
    1. Masked Self-Attention（因果掩码）
    2. Cross-Attention（关注编码器输出）
    3. Engram Attention（关注检索到的历史路线记忆，可选）
    4. Feed-Forward Network
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1, use_engram: bool = True):
        super().__init__()
        self.use_engram = use_engram
        self.d_model = d_model

        # 嵌入目标路线序列
        self.target_embedding = nn.Linear(d_model, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # 解码层
        self.layers = nn.ModuleList([
            self._build_decoder_layer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _build_decoder_layer(self, d_model, n_heads, d_ff, dropout):
        return nn.ModuleDict({
            "self_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            "cross_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            "engram_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) if self.use_engram else None,
            "ffn": nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout),
            ),
            "norm1": nn.LayerNorm(d_model),
            "norm2": nn.LayerNorm(d_model),
            "norm3": nn.LayerNorm(d_model),
            "norm4": nn.LayerNorm(d_model),
        })

    def forward(self, engram_memory: Optional[torch.Tensor],
                encoder_output: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            engram_memory: Engram 检索结果, shape [batch, top_k, d_model]
            encoder_output: 编码器输出, shape [batch, n_pois, d_model]
            target: 目标路线序列, shape [batch, route_len, d_model]
            causal_mask: 因果掩码, shape [route_len, route_len]

        Returns:
            decoder_output: [batch, route_len, d_model]
        """
        raise NotImplementedError("需实现：Masked Self-Attn -> Cross-Attn -> Engram-Attn -> FFN")
