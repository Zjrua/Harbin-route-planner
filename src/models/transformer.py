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
        """Graph-aware 编码.

        Args:
            poi_features: POI 特征矩阵, shape [batch, n_pois, d_model]
            adjacency: 路网邻接矩阵, shape [batch, n_pois, n_pois]

        Returns:
            encoder_output: 编码器输出, shape [batch, n_pois, d_model]
        """
        raise NotImplementedError("需实现：调用 self.encoder(poi_features, adjacency)")

    def decode(self, memory: torch.Tensor,
               encoder_output: torch.Tensor,
               target: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """带 Engram 记忆增强的自回归解码.

        Args:
            memory: Engram 检索结果（若启用）, shape [batch, top_k, d_model]
            encoder_output: 编码器输出, shape [batch, n_pois, d_model]
            target: 目标路线序列（训练时）, shape [batch, route_len]
            mask: 因果掩码, shape [route_len, route_len]

        Returns:
            logits: POI 预测分布, shape [batch, route_len, n_pois]
        """
        raise NotImplementedError("需实现：调用 self.decoder(memory, encoder_output, target, mask)")

    def forward(self, batch: dict) -> dict:
        """完整前向传播（训练模式）.

        Args:
            batch: 包含以下键的字典:
                - poi_features: [batch, n_pois, feature_dim]
                - adjacency: [batch, n_pois, n_pois]
                - route_sequence: [batch, route_len]
                - scores: [batch]（用于 Engram 构建）

        Returns:
            包含 logits, embeddings（MHC）, engram_memory 的字典
        """
        raise NotImplementedError("需实现：encode -> decode -> output_proj 的完整流程")

    def generate(self, encoder_output: torch.Tensor,
                 beam_size: int = 5) -> torch.Tensor:
        """Beam Search 推理生成路线.

        Args:
            encoder_output: 编码器输出, shape [batch, n_pois, d_model]
            beam_size: Beam 宽度

        Returns:
            best_routes: 最优路线, shape [batch, route_len]
        """
        raise NotImplementedError("需实现：Beam Search 解码逻辑")
