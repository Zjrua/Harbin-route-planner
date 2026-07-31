"""测试 Engram 内容寻址记忆模块（修复版）.

修复后的 Engram：
- build_memory 接收路线表征 [batch, d_model]（非 graph encoder output）
- 环形缓冲区累积跨 batch 历史（不覆盖）
- forward 门控融合被 decoder 调用
"""

import pytest
import torch

from src.models.engram import EngramMemory


class TestEngramMemory:
    """EngramMemory 单元测试（适配修复后的 2D 接口）."""

    @pytest.fixture
    def engram(self):
        return EngramMemory(memory_size=50, d_model=64, top_k=3, gate_type="learned")

    def test_build_memory_2d_route_embeddings(self, engram):
        """build_memory 接收 [batch, d_model] 路线表征，写入环形缓冲区."""
        # 20 条路线，每条表征为 [d_model]（mean-pooled embedding）
        route_repr = torch.randn(20, 64)
        scores = torch.rand(20)
        engram.build_memory(route_repr, scores)

        # 前 20 个槽位应被标记为有效
        assert engram.memory_mask[:20].all()
        assert not engram.memory_mask[20:].any()
        # keys 应等于输入（不 mean-pool，直接存）
        assert torch.allclose(engram.memory_keys[:20], route_repr, atol=1e-6)
        assert torch.allclose(engram.memory_scores[:20], scores, atol=1e-6)
        # 写指针应前进 20
        assert engram.write_ptr.item() == 20
        assert engram.n_written.item() == 20

    def test_ring_buffer_accumulates_across_batches(self, engram):
        """环形缓冲区：跨 batch 累积，不覆盖历史（修复原版的核心 bug）."""
        # 第一批 20 条
        batch1 = torch.randn(20, 64)
        engram.build_memory(batch1, torch.rand(20))
        assert engram.n_written.item() == 20
        # 此时 20 个槽有效
        assert engram.memory_mask[:20].sum() == 20

        # 第二批 20 条（应写到 20-39 槽，不覆盖 0-19）
        batch2 = torch.randn(20, 64)
        engram.build_memory(batch2, torch.rand(20))
        assert engram.n_written.item() == 40
        # 40 个槽全有效
        assert engram.memory_mask[:40].sum() == 40
        # 前 20 个槽仍是 batch1（未被覆盖）
        assert torch.allclose(engram.memory_keys[:20], batch1, atol=1e-6)
        # 20-39 槽是 batch2
        assert torch.allclose(engram.memory_keys[20:40], batch2, atol=1e-6)

    def test_ring_buffer_wraps_around(self, engram):
        """缓冲区满后回绕（memory_size=50，写 60 条应覆盖前 10 条）."""
        # 写满 50 条
        batch1 = torch.randn(50, 64)
        engram.build_memory(batch1, torch.rand(50))
        # 再写 10 条，应回绕覆盖 0-9 槽
        batch2 = torch.randn(10, 64)
        engram.build_memory(batch2, torch.rand(10))
        assert engram.n_written.item() == 60
        assert engram.write_ptr.item() == 10  # 回绕后指针
        # 0-9 槽应是 batch2
        assert torch.allclose(engram.memory_keys[:10], batch2, atol=1e-6)
        # 10-49 槽仍是 batch1
        assert torch.allclose(engram.memory_keys[10:50], batch1[10:50], atol=1e-6)

    def test_retrieve_top_k(self, engram):
        """retrieve 返回形状 [batch, top_k, d_model] 的检索结果."""
        route_repr = torch.randn(20, 64)
        engram.build_memory(route_repr, torch.rand(20))

        query = torch.randn(4, 64)  # batch=4
        retrieved, attn_weights = engram.retrieve(query)

        assert retrieved.shape == (4, 3, 64)   # [batch, top_k, d_model]
        assert attn_weights.shape == (4, 3)    # [batch, top_k]
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_retrieve_empty_buffer(self, engram):
        """缓冲区为空时（首个 batch 前）retrieve 返回零向量，不报错."""
        query = torch.randn(4, 64)
        retrieved, attn_weights = engram.retrieve(query)
        assert retrieved.shape == (4, 3, 64)
        assert torch.allclose(retrieved, torch.zeros(4, 3, 64))

    def test_gate_fusion_forward(self, engram):
        """forward（门控融合）输出形状 [batch, d_model]，且 gate/out_proj 参与计算图."""
        route_repr = torch.randn(20, 64)
        engram.build_memory(route_repr, torch.rand(20))

        query = torch.randn(4, 64, requires_grad=True)
        out = engram.forward(query, None)
        assert out.shape == (4, 64)
        # 验证梯度能回传到 gate/out_proj
        loss = out.sum()
        loss.backward()
        assert engram.gate[0].weight.grad is not None
        assert engram.out_proj.weight.grad is not None
        assert query.grad is not None

    def test_season_update(self, engram):
        """update_season 行为：初始全1权重无变化，非均匀权重后产生变化."""
        route_repr = torch.randn(20, 64)
        scores = torch.ones(20) * 10.0
        engram.build_memory(route_repr, scores)

        original = engram.memory_scores[:20].clone()
        engram.update_season("winter")
        assert torch.allclose(engram.memory_scores[:20], original)

        with torch.no_grad():
            engram.season_weights[0] = torch.linspace(0.5, 1.5, engram.memory_size)
        engram.memory_scores[:20] = scores
        engram.update_season("winter")
        assert not torch.allclose(engram.memory_scores[:20], scores)
