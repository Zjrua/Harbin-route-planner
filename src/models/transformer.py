"""RouteTransformer: 基于 Transformer 的旅游路线生成模型.

完整模型架构:
1. Encoder: Graph-aware Transformer Encoder，融合 POI 特征与路网拓扑
2. Decoder: Masked Transformer Decoder + Cross-Attention + Engram 记忆增强
3. MHC: 双曲空间 POI 嵌入，提供结构化的距离先验
"""

import torch
import torch.nn as nn
from typing import Optional

from .engram import EngramMemory
from .mhc import PoincareEmbedding
from .embeddings import POIEmbedding
from .encoder import GraphAwareEncoder
from .decoder import EngramDecoder


class RouteTransformer(nn.Module):
    """基于 Transformer 的哈尔滨文旅路线生成模型.

    工作流程:
    1. encode(): 将 POI 特征 + 路网邻接矩阵编码为上下文表征
    2. decode(): 自回归解码，每步选择下一个 POI，融合 Engram 记忆
    3. forward(): 训练时前向传播（Teacher Forcing）
    4. generate(): 推理时 Beam Search 生成路线
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 完整配置字典，包含 model / engram / mhc 等子配置
        """
        super().__init__()
        self.config = config
        model_cfg = config["model"]

        # POI Embedding 层
        self.poi_embedding = POIEmbedding(
            num_pois=config["data"]["max_pois"],
            d_model=model_cfg["d_model"],
            max_route_len=model_cfg["max_route_len"],
        )

        # MHC 双曲嵌入（可选）
        mhc_cfg = config.get("mhc", {})
        self.use_mhc = mhc_cfg.get("enabled", False)
        if self.use_mhc:
            self.mhc_embedding = PoincareEmbedding(
                num_pois=config["data"]["max_pois"],
                dim=mhc_cfg["dim"],
                curvature=mhc_cfg["curvature"],
            )

        # Graph-aware Encoder
        self.encoder = GraphAwareEncoder(
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_encoder_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
        )

        # Engram 记忆增强 Decoder
        engram_cfg = config.get("engram", {})
        self.use_engram = engram_cfg.get("enabled", False)
        self.decoder = EngramDecoder(
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_decoder_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
            use_engram=self.use_engram,
        )

        # Engram 记忆模块（可选）
        if self.use_engram:
            self.engram = EngramMemory(
                memory_size=engram_cfg["memory_size"],
                d_model=model_cfg["d_model"],
                top_k=engram_cfg["top_k"],
                gate_type=engram_cfg["gate_type"],
            )

        # 输出层：映射到 POI 词表
        self.output_proj = nn.Linear(model_cfg["d_model"], config["data"]["max_pois"])

    def encode(self, poi_features: torch.Tensor,
               adjacency: torch.Tensor) -> torch.Tensor:
        """Graph-aware 编码."""
        return self.encoder(poi_features, adjacency)

    def decode(self, memory: torch.Tensor,
               encoder_output: torch.Tensor,
               target: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """带 Engram 记忆增强的自回归解码."""
        return self.decoder(memory, encoder_output, target, mask)

    def forward(self, batch: dict) -> dict:
        """完整前向传播（训练模式）."""
        poi_features = batch["poi_features"]
        adjacency = batch["adjacency"]
        route_seq = batch["route_sequence"]

        # 1. 编码
        encoder_output = self.encode(poi_features, adjacency)

        # 2. 构建 Engram 记忆（可选）
        engram_memory = None
        if self.use_engram:
            scores = batch.get("scores", None)
            if scores is not None:
                self.engram.build_memory(encoder_output.detach(), scores)
            # 使用编码器输出的均值作为查询检索记忆
            query = encoder_output.mean(dim=1)
            retrieved, _ = self.engram.retrieve(query)
            engram_memory = retrieved

        # 3. 目标路线嵌入
        max_route_len = self.config["model"]["max_route_len"]
        target_emb = self.poi_embedding(route_seq[:, :-1] if route_seq.size(1) > 1 else route_seq)

        # 4. 构建因果掩码
        seq_len = target_emb.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=target_emb.device), diagonal=1
        ).bool()

        # 5. 解码
        decoder_output = self.decode(engram_memory, encoder_output, target_emb, causal_mask)

        # 6. 输出投影
        logits = self.output_proj(decoder_output)

        result = {"logits": logits}

        # 7. MHC 嵌入（可选）
        if self.use_mhc:
            result["embeddings"] = self.mhc_embedding()

        return result

    def generate(self, encoder_output: torch.Tensor,
                 beam_size: int = 5) -> torch.Tensor:
        """Beam Search 推理生成路线."""
        batch_size = encoder_output.size(0)
        max_len = self.config["model"]["max_route_len"]
        device = encoder_output.device

        # 每个 batch 独立做 beam search
        best_routes = []
        for b in range(batch_size):
            enc_single = encoder_output[b:b+1]  # [1, n_pois, d_model]

            # Engram 检索（只做一次，所有 beam 共享）
            engram_memory = None
            if self.use_engram:
                query = enc_single.mean(dim=1)  # [1, d_model]
                retrieved, _ = self.engram.retrieve(query)
                engram_memory = retrieved  # [1, d_model]

            beams = [(0.0, [0])]  # (累计 log_prob, 路线索引列表)

            for step in range(max_len - 1):
                candidates = []
                for score, route in beams:
                    route_t = torch.tensor([route], dtype=torch.long, device=device)
                    target_emb = self.poi_embedding(route_t)  # [1, seq_len, d_model]
                    seq_len = target_emb.size(1)
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=device), diagonal=1
                    ).bool()

                    dec_out = self.decode(engram_memory, enc_single, target_emb, causal_mask)
                    logits = self.output_proj(dec_out)[:, -1, :]  # [1, n_pois]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_probs, topk_idx = log_probs[0].topk(beam_size)

                    for i in range(beam_size):
                        new_score = score + topk_probs[i].item()
                        candidates.append((new_score, route + [topk_idx[i].item()]))

                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_size]

            best_routes.append(beams[0][1])

        # 填充到 max_len
        routes = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for b, route in enumerate(best_routes):
            length = min(len(route), max_len)
            routes[b, :length] = torch.tensor(route[:length], dtype=torch.long)

        return routes
