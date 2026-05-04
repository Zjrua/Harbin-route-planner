"""Masked Transformer Decoder + Cross-Attention + Engram 记忆增强 + 活动类型条件生成.

解码器负责自回归地逐步生成路线中的下一个 POI，
通过 Cross-Attention 获取编码器上下文，通过 Engram 检索历史优质路线，
支持活动类型条件生成和约束解码。
"""

import torch
import torch.nn as nn
from typing import Optional


# 活动类型转换约束矩阵（软约束：-inf 表示禁止，0 表示允许，正值表示鼓励）
# 行=当前类型，列=下一类型
ACTIVITY_TRANSITION_CONSTRAINTS = torch.tensor([
    # 景点  餐饮  住宿  交通  购物  出发点
    [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # 景点后：都可以
    [  0.0, -1e9,  0.0,  0.0,  0.0, -1e9],  # 餐饮后：不能连续餐饮或出发点
    [  0.0,  0.0, -1e9,  0.0,  0.0,  0.0],  # 住宿后：不能连续住宿
    [  0.0,  0.0,  0.0, -1e9,  0.0,  0.0],  # 交通后：不能连续交通
    [  0.0,  0.0,  0.0,  0.0, -1e9,  0.0],  # 购物后：不能连续购物
    [  0.0,  0.0,  0.0,  0.0,  0.0, -1e9],  # 出发点后：不能连续出发点
], dtype=torch.float32)


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

        # 活动类型条件嵌入：用于在解码时注入"当前应该生成什么类型"的信息
        self.activity_condition_embed = nn.Linear(d_model, d_model)

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
                causal_mask: Optional[torch.Tensor] = None,
                activity_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Masked Self-Attn -> Cross-Attn -> Engram-Attn -> FFN.

        Args:
            engram_memory: [batch, memory_size, d_model] Engram 记忆
            encoder_output: [batch, n_pois, d_model] 编码器输出
            target: [batch, seq_len, d_model] 目标路线嵌入（训练时）
            causal_mask: [seq_len, seq_len] 因果掩码
            activity_condition: [batch, seq_len, d_model] 活动类型条件嵌入（可选）
        """
        if target is not None:
            x = self.target_embedding(target) + self.pos_encoding[:, :target.size(1)]
            # 如果有活动类型条件，添加到嵌入中
            if activity_condition is not None:
                x = x + self.activity_condition_embed(activity_condition)
        else:
            x = self.pos_encoding[:, :encoder_output.size(1)].expand(encoder_output.size(0), -1, -1)

        for layer in self.layers:
            # 1. Masked Self-Attention
            residual = x
            x = layer["norm1"](x)
            x, _ = layer["self_attn"](
                x, x, x, attn_mask=causal_mask, need_weights=False,
            )
            x = residual + x

            # 2. Cross-Attention
            residual = x
            x = layer["norm2"](x)
            x, _ = layer["cross_attn"](
                x, encoder_output, encoder_output, need_weights=False,
            )
            x = residual + x

            # 3. Engram Attention（可选）
            if self.use_engram and layer["engram_attn"] is not None and engram_memory is not None:
                residual = x
                x = layer["norm3"](x)
                x, _ = layer["engram_attn"](
                    x, engram_memory, engram_memory, need_weights=False,
                )
                x = residual + x

            # 4. FFN
            residual = x
            x = layer["norm4"](x)
            x = residual + layer["ffn"](x)

        return self.norm(x)
