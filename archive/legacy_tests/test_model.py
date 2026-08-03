"""测试 ItineraryTransformer 模型."""

import pytest
import torch

from src.models.transformer import ItineraryTransformer


def _make_config(max_pois=50, d_model=64, use_engram=True, use_mhc=True):
    """构造一个最小可用 config（测试用，参数量小、跑得快）."""
    return {
        "data": {"max_pois": max_pois},
        "model": {
            "d_model": d_model, "n_heads": 4, "n_encoder_layers": 2,
            "n_decoder_layers": 2, "d_ff": 256, "dropout": 0.1,
            "max_route_len": 10,
        },
        "engram": {
            "enabled": use_engram, "memory_size": 100, "top_k": 3,
            "gate_type": "learned",
        },
        "mhc": {"enabled": use_mhc, "dim": 32, "curvature": -1.0},
    }


class TestItineraryTransformer:
    """ItineraryTransformer 模型单元测试."""

    @pytest.fixture
    def config(self):
        return _make_config()

    @pytest.fixture
    def model(self, config):
        return ItineraryTransformer(config)

    def test_model_instantiation(self, model):
        """模型能正确实例化，且可训练参数量为正."""
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0
        # 输出层应映射到 POI 词表大小
        assert model.output_proj.out_features == 50
        # 关键子模块都存在
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "poi_embedding")
        assert hasattr(model, "output_proj")
        # MHC / Engram 在 config 启用时应存在
        assert hasattr(model, "mhc_embedding")
        assert hasattr(model, "engram")

    def test_forward_pass(self, model, config):
        """前向传播返回正确形状的 logits（Teacher Forcing 模式）."""
        max_pois = config["data"]["max_pois"]
        d_model = config["model"]["d_model"]
        batch_size = 4
        route_len = config["model"]["max_route_len"]

        batch = {
            "poi_features": torch.randn(1, max_pois, d_model),
            "adjacency": torch.ones(1, max_pois, max_pois) * 0.5,
            "route_sequence": torch.randint(1, max_pois, (batch_size, route_len)),
            # scores 形状匹配真实 dataset 输出：[batch_size] 的每路线评分标量
            "scores": torch.rand(batch_size),
            "activity_types": torch.randint(0, 6, (batch_size, route_len)),
        }
        out = model(batch)
        assert "logits" in out
        # logits: [batch, route_len-1, max_pois]（target = route[:, 1:]，长度 route_len-1）
        assert out["logits"].shape[0] == batch_size
        assert out["logits"].shape[-1] == max_pois
        # MHC 启用时返回 embeddings
        assert "embeddings" in out

    def test_forward_without_optional_modules(self):
        """禁用 Engram 和 MHC 时仍能前向传播."""
        config = _make_config(use_engram=False, use_mhc=False)
        model = ItineraryTransformer(config)
        max_pois = config["data"]["max_pois"]
        d_model = config["model"]["d_model"]
        batch = {
            "poi_features": torch.randn(1, max_pois, d_model),
            "adjacency": torch.ones(1, max_pois, max_pois) * 0.5,
            "route_sequence": torch.randint(1, max_pois, (2, config["model"]["max_route_len"])),
            "scores": torch.rand(2),
            "activity_types": torch.randint(0, 6, (2, config["model"]["max_route_len"])),
        }
        out = model(batch)
        assert out["logits"].shape[-1] == max_pois
        assert "embeddings" not in out  # MHC 关闭
        assert not hasattr(model, "engram")

    def test_generate_with_beam_search(self, model, config):
        """Beam Search 生成返回合法路线（长度 ≤ max_route_len，POI 索引在词表内）."""
        max_pois = config["data"]["max_pois"]
        d_model = config["model"]["d_model"]
        encoder_output = torch.randn(1, max_pois, d_model)
        poi_activity_types = torch.randint(0, 6, (max_pois,))

        routes = model.generate(encoder_output, beam_size=3, poi_activity_types=poi_activity_types)
        assert routes.shape == (1, config["model"]["max_route_len"])
        # 所有非零索引应在 [0, max_pois) 内
        nonzero = routes[routes > 0]
        assert (nonzero < max_pois).all()

    def test_backward_runs(self, model, config):
        """损失能反向传播，梯度流入各子模块（验证计算图连通）."""
        max_pois = config["data"]["max_pois"]
        d_model = config["model"]["d_model"]
        batch = {
            "poi_features": torch.randn(1, max_pois, d_model),
            "adjacency": torch.ones(1, max_pois, max_pois) * 0.5,
            "route_sequence": torch.randint(1, max_pois, (2, config["model"]["max_route_len"])),
            "scores": torch.rand(2),
            "activity_types": torch.randint(0, 6, (2, config["model"]["max_route_len"])),
        }
        out = model(batch)
        loss = out["logits"].sum()
        loss.backward()
        # encoder/decoder/output_proj 都应有梯度
        assert model.output_proj.weight.grad is not None
        assert model.encoder.layers[0].self_attn.in_proj_weight.grad is not None
