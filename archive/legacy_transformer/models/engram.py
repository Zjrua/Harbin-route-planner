"""Engram 内容寻址记忆模块（已修复语义 bug）.

灵感来自 DeepSeek 论文的外部记忆机制，用于从历史优质路线中
检索相似路线片段，辅助当前解码决策。

修复说明（原版的三个缺陷）：
1. 原版 build_memory 喂的是 encoder_output（graph 表征，batch 间 broadcast 相同），
   导致 256 个 memory_keys 全是同一向量。新版喂路线表征（每条路线 mean-pooled embedding）。
2. 原版每个 batch 覆盖前 N 槽，无历史累积。新版用环形缓冲区（write_ptr）累积跨 batch 历史。
3. 原版 forward（门控融合）从未被调用，gate/out_proj 是死参数。新版 forward 被 decoder 调用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EngramMemory(nn.Module):
    """外部内容寻址记忆库，环形缓冲区累积历史路线表征.

    工作流程:
    1. build_memory(): 将当前 batch 的路线表征写入环形缓冲区（不覆盖历史）
    2. retrieve(): 根据当前解码状态检索 top-k 相似路线
    3. forward(): 通过门控融合检索结果与当前状态（被 decoder 调用）
    """

    def __init__(self, memory_size: int, d_model: int, top_k: int = 5,
                 gate_type: str = "learned"):
        """
        Args:
            memory_size: 记忆槽位数量（环形缓冲区大小）
            d_model: 模型隐藏维度
            top_k: 检索时取最相似的 k 条路线
            gate_type: 门控类型，"learned" 为可学习门控，"fixed" 为固定权重
        """
        super().__init__()
        self.memory_size = memory_size
        self.d_model = d_model
        self.top_k = top_k
        self.gate_type = gate_type

        # 记忆键和值的存储（环形缓冲区）
        self.register_buffer("memory_keys", torch.zeros(memory_size, d_model))
        self.register_buffer("memory_values", torch.zeros(memory_size, d_model))
        self.register_buffer("memory_scores", torch.zeros(memory_size))
        self.register_buffer("memory_mask", torch.zeros(memory_size, dtype=torch.bool))
        # 环形缓冲区写指针
        self.register_buffer("write_ptr", torch.zeros(1, dtype=torch.long))
        # 已写入总数（用于判断缓冲区是否填满）
        self.register_buffer("n_written", torch.zeros(1, dtype=torch.long))

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

        # 季节性权重（保留接口，当前未在训练循环中调用 update_season）
        self.season_weights = nn.Parameter(torch.ones(2, memory_size))

    def build_memory(self, route_embeddings: torch.Tensor,
                     scores: torch.Tensor) -> None:
        """将当前 batch 的路线表征写入环形缓冲区（累积历史，不覆盖）.

        Args:
            route_embeddings: [batch, d_model] 每条路线的表征（如 target_emb 的 mean-pool）
            scores: [batch] 每条路线的评分
        """
        n = min(route_embeddings.size(0), self.memory_size)
        keys = route_embeddings[:n]            # [n, d_model]
        with torch.no_grad():
            ptr = int(self.write_ptr.item())
            total = int(self.n_written.item())
            # 环形写入：到末尾后回绕
            for i in range(n):
                idx = (ptr + i) % self.memory_size
                self.memory_keys[idx] = keys[i].detach()
                self.memory_values[idx] = keys[i].detach()
                self.memory_scores[idx] = float(scores[i].item()) if i < scores.size(0) else 0.0
                self.memory_mask[idx] = True
            self.write_ptr[0] = (ptr + n) % self.memory_size
            self.n_written[0] = total + n

    def retrieve(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """检索与查询最相似的 top-k 条记忆（余弦相似度）."""
        # 只在已填充的槽位上检索
        n_valid = min(int(self.n_written.item()), self.memory_size)
        if n_valid == 0:
            # 缓冲区为空（首个 batch 前），返回零向量
            return (torch.zeros(query.size(0), self.top_k, self.d_model,
                                device=query.device, dtype=query.dtype),
                    torch.ones(query.size(0), self.top_k,
                               device=query.device, dtype=query.dtype) / self.top_k)

        # RMSNorm + signed_sqrt 计算 Engram 相似度（受 TileKernels 启发）
        q_norm = torch.sqrt(query.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        k_valid = self.memory_keys[:n_valid]
        k_norm = torch.sqrt(k_valid.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        q_normalized = query / q_norm
        k_normalized = k_valid / k_norm

        # 余弦相似度: [batch, n_valid]
        sim = torch.matmul(q_normalized, k_normalized.T)
        sim = torch.sign(sim) * torch.sqrt(sim.abs() + 1e-8)

        # top-k（k 可能大于 n_valid，此时取全部）
        k_actual = min(self.top_k, n_valid)
        topk_vals, topk_idx = sim.topk(k_actual, dim=-1)
        attn_weights = F.softmax(topk_vals, dim=-1)

        # 收集对应的值
        v_valid = self.memory_values[:n_valid]
        retrieved = v_valid[topk_idx]  # [batch, k_actual, d_model]

        # 如果 k_actual < top_k，用零填充到 top_k（保持下游形状稳定）
        if k_actual < self.top_k:
            pad = torch.zeros(query.size(0), self.top_k - k_actual, self.d_model,
                              device=query.device, dtype=query.dtype)
            retrieved = torch.cat([retrieved, pad], dim=1)
            pad_w = torch.zeros(query.size(0), self.top_k - k_actual,
                                device=query.device, dtype=query.dtype)
            attn_weights = torch.cat([attn_weights, pad_w], dim=1)

        return retrieved, attn_weights

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """注意力检索 + 门控融合（TileKernels 风格 RMSNorm+signed_sqrt+sigmoid 门控）.

        被 decoder 的 engram-attention 调用，让 gate/out_proj 真正参与训练。
        """
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
