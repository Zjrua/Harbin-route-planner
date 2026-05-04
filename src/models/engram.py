"""Engram 内容寻址记忆模块.

灵感来自 DeepSeek 论文的外部记忆机制，用于从历史优质路线中
检索相似路线片段，辅助当前解码决策。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EngramMemory(nn.Module):
    """外部内容寻址记忆库，存储历史优质路线的表征.

    工作流程:
    1. build_memory(): 将历史路线及其评分编码为记忆键值对
    2. retrieve(): 根据当前解码状态检索 top-k 相似路线
    3. forward(): 通过注意力机制融合检索结果与当前状态
    """

    def __init__(self, memory_size: int, d_model: int, top_k: int = 5,
                 gate_type: str = "learned"):
        """
        Args:
            memory_size: 记忆槽位数量
            d_model: 模型隐藏维度
            top_k: 检索时取最相似的 k 条路线
            gate_type: 门控类型，"learned" 为可学习门控，"fixed" 为固定权重
        """
        super().__init__()
        self.memory_size = memory_size
        self.d_model = d_model
        self.top_k = top_k
        self.gate_type = gate_type

        # 记忆键和值的存储
        self.register_buffer("memory_keys", torch.zeros(memory_size, d_model))
        self.register_buffer("memory_values", torch.zeros(memory_size, d_model))
        self.register_buffer("memory_scores", torch.zeros(memory_size))
        self.register_buffer("memory_mask", torch.zeros(memory_size, dtype=torch.bool))

        # 查询投影层
        self.query_proj = nn.Linear(d_model, d_model)
        # 输出投影层
        self.out_proj = nn.Linear(d_model, d_model)

        # 门控参数
        if gate_type == "learned":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        else:
            self.gate = None

        # 季节性权重（冬/夏两季）
        self.season_weights = nn.Parameter(torch.ones(2, memory_size))

    def build_memory(self, routes: torch.Tensor, scores: torch.Tensor) -> None:
        """从历史路线构建记忆库，将路线平均池化为记忆键."""
        n = min(routes.size(0), self.memory_size)
        # 平均池化：[n, route_len, d_model] -> [n, d_model]
        keys = routes[:n].mean(dim=1)
        self.memory_keys[:n] = keys
        self.memory_values[:n] = keys.clone()
        self.memory_scores[:n] = scores[:n]
        self.memory_mask[:n] = True

    def retrieve(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """检索与查询最相似的 top-k 条记忆（余弦相似度）."""
        # RMSNorm + signed_sqrt 计算 Engram 相似度（受 TileKernels 启发）
        q_norm = torch.sqrt(query.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(self.memory_keys.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        q_normalized = query / q_norm
        k_normalized = self.memory_keys / k_norm

        # 余弦相似度: [batch, memory_size]
        sim = torch.matmul(q_normalized, k_normalized.T)
        # signed_sqrt 缩放
        sim = torch.sign(sim) * torch.sqrt(sim.abs() + 1e-8)

        # 掩盖空槽位
        sim = sim.masked_fill(~self.memory_mask.unsqueeze(0), float("-inf"))

        # top-k
        topk_vals, topk_idx = sim.topk(self.top_k, dim=-1)
        attn_weights = F.softmax(topk_vals, dim=-1)

        # 收集对应的值
        batch_size = query.size(0)
        retrieved = self.memory_values[topk_idx]  # [batch, top_k, d_model]

        return retrieved, attn_weights

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """注意力检索 + 门控融合（TileKernels 风格 RMSNorm+signed_sqrt+sigmoid 门控）."""
        projected_query = self.query_proj(query)

        # 检索 top-k 记忆
        retrieved, attn_weights = self.retrieve(projected_query)

        # 注意力加权聚合
        context = (attn_weights.unsqueeze(-1) * retrieved).sum(dim=1)  # [batch, d_model]
        context = self.out_proj(context)

        # 门控融合：sigmoid(gate(cat(query, context))) * context + query
        if self.gate is not None:
            gate_input = torch.cat([query, context], dim=-1)
            gate_score = self.gate(gate_input)
            fused = query + gate_score * context
        else:
            fused = query + 0.5 * context

        return fused

    def update_season(self, season: str) -> None:
        """季节性记忆权重调整：winter=0, summer=1."""
        season_idx = 0 if season == "winter" else 1
        weights = self.season_weights[season_idx]  # [memory_size]
        self.memory_scores = weights * self.memory_scores
