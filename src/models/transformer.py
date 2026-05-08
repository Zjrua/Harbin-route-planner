"""RouteTransformer: 基于 Transformer 的旅游路线生成模型.

完整模型架构:
1. Encoder: Graph-aware Transformer Encoder，融合 POI 特征、路网拓扑和活动类型相似性
2. Decoder: Masked Transformer Decoder + Cross-Attention + Engram 记忆增强
3. MHC: 双曲空间 POI 嵌入，提供结构化的距离先验
4. 活动类型条件生成：支持约束解码，生成符合旅游节奏的路线
"""

import torch
import torch.nn as nn
from typing import Optional

from .engram import EngramMemory
from .mhc import PoincareEmbedding
from .embeddings import POIEmbedding, ACTIVITY_TYPES
from .encoder import GraphAwareEncoder
from .decoder import EngramDecoder, ACTIVITY_TRANSITION_CONSTRAINTS


class RouteTransformer(nn.Module):
    """基于 Transformer 的哈尔滨文旅路线生成模型.

    工作流程:
    1. encode(): 将 POI 特征 + 路网邻接矩阵 + 活动类型偏置编码为上下文表征
    2. decode(): 自回归解码，每步选择下一个 POI，融合 Engram 记忆和活动类型条件
    3. forward(): 训练时前向传播（Teacher Forcing）
    4. generate(): 推理时约束 Beam Search 生成路线
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

        # 活动类型转换约束矩阵（注册为 buffer，随设备移动）
        self.register_buffer("activity_constraints", ACTIVITY_TRANSITION_CONSTRAINTS)

    def encode(self, poi_features: torch.Tensor,
               adjacency: torch.Tensor,
               activity_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Graph-aware 编码，支持活动类型偏置."""
        return self.encoder(poi_features, adjacency, activity_bias)

    def decode(self, memory: torch.Tensor,
               encoder_output: torch.Tensor,
               target: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None,
               activity_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """带 Engram 记忆增强的自回归解码，支持活动类型条件."""
        return self.decoder(memory, encoder_output, target, mask, activity_condition)

    def forward(self, batch: dict) -> dict:
        """完整前向传播（训练模式）.

        Args:
            batch: 包含以下字段的字典:
                - poi_features: [batch, n_pois, d_model] POI 特征
                - adjacency: [batch, n_pois, n_pois] 邻接矩阵
                - route_sequence: [batch, max_route_len] 路线序列
                - activity_types: [batch, max_route_len] 活动类型序列（可选）
                - scores: [batch, n_pois] POI 评分（可选）
        """
        route_seq = batch["route_sequence"]
        activity_types = batch.get("activity_types", None)

        # 1. 编码（支持预计算encoder_output以节省显存）
        if "_encoder_output" in batch:
            encoder_output = batch["_encoder_output"]
        else:
            poi_features = batch["poi_features"]
            adjacency = batch["adjacency"]
            activity_bias = batch.get("activity_bias", None)
            encoder_output = self.encode(poi_features, adjacency, activity_bias)

        # Expand encoder output to match batch size for cross-attention
        if encoder_output.size(0) != route_seq.size(0):
            encoder_output = encoder_output.expand(route_seq.size(0), -1, -1).contiguous()

        # 2. 构建 Engram 记忆（可选）
        engram_memory = None
        if self.use_engram:
            scores = batch.get("scores", None)
            if scores is not None:
                self.engram.build_memory(encoder_output.detach(), scores)
            query = encoder_output.mean(dim=1)
            retrieved, _ = self.engram.retrieve(query)
            engram_memory = retrieved

        # 3. 目标路线嵌入（包含活动类型嵌入）
        max_route_len = self.config["model"]["max_route_len"]
        target_seq = route_seq[:, :-1] if route_seq.size(1) > 1 else route_seq
        target_activity = activity_types[:, :-1] if activity_types is not None and activity_types.size(1) > 1 else activity_types

        target_emb = self.poi_embedding(
            target_seq,
            activity_types=target_activity
        )

        # 4. 构建因果掩码
        seq_len = target_emb.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=target_emb.device), diagonal=1
        ).bool()

        # 5. 解码（如果有活动类型条件，可以注入）
        activity_condition = batch.get("activity_condition", None)
        decoder_output = self.decode(engram_memory, encoder_output, target_emb, causal_mask, activity_condition)

        # 6. 输出投影
        logits = self.output_proj(decoder_output)

        result = {"logits": logits}

        # 7. MHC 嵌入（可选）
        if self.use_mhc:
            result["embeddings"] = self.mhc_embedding()

        return result

    def generate(self, encoder_output: torch.Tensor,
                 beam_size: int = 5,
                 poi_activity_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """约束 Beam Search 推理生成路线.

        增加位置感知逻辑：
        - 前面步骤禁止住宿（住宿只能是终点）
        - 连续 N 个同类型 POI 后鼓励转换（避免全是景点）
        - 倒数几步才允许/鼓励住宿

        Args:
            encoder_output: [1, n_pois, d_model] 编码器输出
            beam_size: Beam Search 宽度
            poi_activity_types: [n_pois] 每个 POI 的活动类型标签

        Returns:
            routes: [1, max_len] 生成的路线
        """
        batch_size = encoder_output.size(0)
        max_len = self.config["model"]["max_route_len"]
        device = encoder_output.device

        # 加载活动类型约束矩阵
        constraints = self.activity_constraints  # [6, 6]

        # 活动类型常量
        ATTR_SCENIC = 0   # 景点
        ATTR_DINING = 1   # 餐饮
        ATTR_HOTEL = 2    # 住宿

        # 连续同类型阈值：超过此值后强烈鼓励切换
        CONSECUTIVE_THRESHOLD = 3

        best_routes = []
        for b in range(batch_size):
            enc_single = encoder_output[b:b+1]  # [1, n_pois, d_model]

            # Engram 检索
            engram_memory = None
            if self.use_engram:
                query = enc_single.mean(dim=1)
                retrieved, _ = self.engram.retrieve(query)
                engram_memory = retrieved

            beams = [(0.0, [0])]  # (累计 log_prob, 路线索引列表)

            for step in range(max_len - 1):
                candidates = []
                for score, route in beams:
                    route_t = torch.tensor([route], dtype=torch.long, device=device)

                    # 获取活动类型嵌入
                    activity_type_ids = None
                    if poi_activity_types is not None:
                        activity_type_ids = poi_activity_types[route_t]

                    target_emb = self.poi_embedding(route_t, activity_types=activity_type_ids)
                    seq_len = target_emb.size(1)
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=device), diagonal=1
                    ).bool()

                    dec_out = self.decode(engram_memory, enc_single, target_emb, causal_mask)
                    logits = self.output_proj(dec_out)[:, -1, :]  # [1, n_pois]

                    # 应用活动类型约束
                    if poi_activity_types is not None and len(route) > 0:
                        last_poi = route[-1]
                        last_activity = poi_activity_types[last_poi].item()
                        current_activities = poi_activity_types
                        constraint_mask = constraints[last_activity]  # [6]

                        for poi_idx in range(logits.size(-1)):
                            activity = current_activities[poi_idx].item()

                            # 基本约束：应用转换矩阵
                            base_bias = constraint_mask[activity]

                            # 住宿只能在最后 3 步（选完即终止）
                            if activity == ATTR_HOTEL:
                                steps_remaining = max_len - (step + 1)
                                if steps_remaining > 3:
                                    base_bias = -1e9  # 太早，禁止
                                elif steps_remaining > 0:
                                    base_bias = 2.0   # 合适，鼓励

                            # 连续同类型检测：如果连续太多同类型，鼓励切换
                            if activity == last_activity:
                                consecutive = 1
                                for prev_idx in reversed(route):
                                    if prev_idx > 0 and poi_activity_types[prev_idx].item() == last_activity:
                                        consecutive += 1
                                    else:
                                        break
                                # 连续景点太多 → 降低再选景点的概率，提升餐饮
                                if consecutive >= CONSECUTIVE_THRESHOLD:
                                    if activity == ATTR_SCENIC:
                                        base_bias -= 3.0  # 不想再选景点
                                    if last_activity == ATTR_SCENIC and activity == ATTR_DINING:
                                        base_bias += 5.0  # 强烈鼓励去吃饭

                            logits[0, poi_idx] += base_bias

                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_probs, topk_idx = log_probs[0].topk(beam_size)

                    # 检查是否已选住宿：住宿后立即终止路线
                    route_has_hotel = False
                    if poi_activity_types is not None and len(route) > 0:
                        route_has_hotel = any(
                            poi_activity_types[pidx].item() == ATTR_HOTEL
                            for pidx in route if pidx > 0
                        )

                    if route_has_hotel:
                        # 住宿后不再扩展，直接加入候选
                        candidates.append((score, route))
                    else:
                        for i in range(beam_size):
                            new_score = score + topk_probs[i].item()
                            new_route = route + [topk_idx[i].item()]
                            # 如果刚选了住宿，标记路线已完成
                            if poi_activity_types is not None and poi_activity_types[topk_idx[i].item()].item() == ATTR_HOTEL:
                                candidates.append((new_score, new_route))
                            else:
                                candidates.append((new_score, new_route))

                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_size]

            best_routes.append(beams[0][1])

        # 填充到 max_len
        routes = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for b, route in enumerate(best_routes):
            length = min(len(route), max_len)
            routes[b, :length] = torch.tensor(route[:length], dtype=torch.long)

        return routes
