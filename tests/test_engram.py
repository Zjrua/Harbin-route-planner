"""测试 Engram 内容寻址记忆模块."""

import pytest
import torch

from src.models.engram import EngramMemory


class TestEngramMemory:
    """EngramMemory 单元测试."""

    @pytest.fixture
    def engram(self):
        return EngramMemory(memory_size=50, d_model=64, top_k=3, gate_type="learned")

    def test_build_memory(self, engram):
        """build_memory 后，memory_keys/values/mask 正确填充.

        注：build_memory 把 routes 第 0 维当作"路线数"，scores 也按第 0 维对齐。
        实际训练调用 (transformer.py) 传 encoder_output[1, n_pois, d_model] + scores[batch]，
        因此 n=1，只存 1 条记忆——这是已知的设计局限（见 DIAGNOSTIC_REPORT 问题 5）。
        本测试用更一般的多路线输入验证核心逻辑。
        """
        routes = torch.randn(20, 10, 64)   # 20 条路线，每条 10 步，d_model=64
        scores = torch.rand(20)
        engram.build_memory(routes, scores)

        # 前 20 个槽位应被标记为有效
        assert engram.memory_mask[:20].all()
        assert not engram.memory_mask[20:].any()  # 剩余槽位无效
        # keys 应是 routes 在 route_len 维度的均值池化
        expected_keys = routes[:20].mean(dim=1)
        assert torch.allclose(engram.memory_keys[:20], expected_keys, atol=1e-6)
        # scores 写入正确
        assert torch.allclose(engram.memory_scores[:20], scores[:20])

    def test_build_memory_handles_overflow(self, engram):
        """传入超过 memory_size 的路线时，只取前 memory_size 条."""
        routes = torch.randn(100, 5, 64)  # 超过 memory_size=50
        scores = torch.rand(100)
        engram.build_memory(routes, scores)
        assert engram.memory_mask.sum() == 50

    def test_retrieve_top_k(self, engram):
        """retrieve 返回形状 [batch, top_k, d_model] 的检索结果."""
        routes = torch.randn(20, 10, 64)
        scores = torch.rand(20)
        engram.build_memory(routes, scores)

        query = torch.randn(4, 64)  # batch=4
        retrieved, attn_weights = engram.retrieve(query)

        assert retrieved.shape == (4, 3, 64)   # [batch, top_k, d_model]
        assert attn_weights.shape == (4, 3)    # [batch, top_k]
        # attn_weights 是 softmax 输出，每行应和为 1
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_gate_fusion(self, engram):
        """forward（门控融合）输出形状 [batch, d_model].

        注：forward() 在 transformer.py 中实际从未被调用（见 DIAGNOSTIC_REPORT 问题 5.1），
        门控参数是死参数。本测试验证它本身能正常工作。
        """
        routes = torch.randn(20, 10, 64)
        scores = torch.rand(20)
        engram.build_memory(routes, scores)

        query = torch.randn(4, 64)
        out = engram.forward(query, engram.memory_values[:20])
        assert out.shape == (4, 64)

    def test_season_update(self, engram):
        """update_season 行为验证.

        season_weights 初始化为全 1（torch.ones），因此初始时 update_season 不会改变 scores。
        手动设置非均匀权重后，update_season 才会产生差异。
        注：update_season 在训练/inference 中从未被调用（死参数）。
        """
        routes = torch.randn(20, 10, 64)
        scores = torch.ones(20) * 10.0
        engram.build_memory(routes, scores)

        # 初始权重全 1 → update_season 无变化
        original = engram.memory_scores[:20].clone()
        engram.update_season("winter")
        assert torch.allclose(engram.memory_scores[:20], original)

        # 手动设非均匀权重 → update_season 产生变化
        with torch.no_grad():
            engram.season_weights[0] = torch.linspace(0.5, 1.5, engram.memory_size)
        engram.memory_scores[:20] = scores  # 重置
        engram.update_season("winter")
        assert not torch.allclose(engram.memory_scores[:20], scores)
