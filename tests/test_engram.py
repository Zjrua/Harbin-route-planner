"""测试 Engram 记忆模块."""

import pytest
import torch


class TestEngramMemory:
    """EngramMemory 单元测试."""

    @pytest.fixture
    def engram(self):
        from src.models.engram import EngramMemory
        return EngramMemory(memory_size=50, d_model=64, top_k=3, gate_type="learned")

    def test_build_memory(self, engram):
        """测试记忆构建."""
        # TODO: 传入 mock 路线和评分，验证 memory_keys 已更新
        pass

    def test_retrieve_top_k(self, engram):
        """测试 top-k 检索返回正确数量."""
        # TODO: 验证检索结果形状 [batch, top_k, d_model]
        pass

    def test_gate_fusion(self, engram):
        """测试门控融合输出形状."""
        # TODO: 验证 forward 输出 [batch, d_model]
        pass

    def test_season_update(self, engram):
        """测试季节权重更新."""
        # TODO: 验证冬季/夏季权重不同
        pass
