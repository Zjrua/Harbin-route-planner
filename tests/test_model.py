"""测试 RouteTransformer 模型."""

import pytest
import torch


class TestRouteTransformer:
    """RouteTransformer 模型单元测试."""

    @pytest.fixture
    def config(self):
        return {
            "data": {"max_pois": 50},
            "model": {"d_model": 64, "n_heads": 4, "n_encoder_layers": 2,
                      "n_decoder_layers": 2, "d_ff": 256, "dropout": 0.1,
                      "max_route_len": 10},
            "engram": {"enabled": True, "memory_size": 100, "top_k": 3,
                       "gate_type": "learned"},
            "mhc": {"enabled": True, "dim": 32, "curvature": -1.0},
        }

    def test_model_instantiation(self, config):
        """测试模型能正确实例化."""
        # TODO: 验证模型参数
        pass

    def test_forward_pass(self, config):
        """测试前向传播不报错."""
        # TODO: 构造 mock batch，验证输出形状
        pass

    def test_generate_with_beam_search(self, config):
        """测试 Beam Search 生成."""
        # TODO: 验证生成路线合法
        pass
