"""测试 Muon 优化器（Newton-Schulze 正交化 + AdamW 混合）.

这是论文声称的"三大创新之一"但此前零测试覆盖。本测试验证：
1. 优化器能正确构造并 step
2. >=2D 参数走 Muon 正交化分支（state 含 momentum_buffer）
3. 1D 参数走 AdamW 分支（state 含 exp_avg / exp_avg_sq）
4. 一次 step 后损失下降（在简单二次函数上）
"""

import torch

from src.optim.muon import MuonOptimizer, zeropower_via_newtonschulz5


def _build_param_groups(weight_2d, bias_1d):
    """构造 attention/ffn/other 三组参数（模拟 build_optimizer 的输出）."""
    return [
        {"params": [weight_2d], "group_type": "attention"},
        {"params": [bias_1d], "group_type": "other"},
    ]


class TestMuonOptimizer:
    """MuonOptimizer 单元测试."""

    def test_zeropower_via_newtonschulz5_runs_and_is_bounded(self):
        """NS5 迭代能正常运行，输出有界且非平凡.

        NS5 在 bfloat16 下对小矩阵的精确正交化效果受精度限制（原版设计面向大矩阵），
        因此本测试验证的是工程正确性：函数能跑、形状对、输出范数合理（既非0也不爆炸），
        且对接近正交的输入能将其谱范数收敛到 O(1) 量级。
        """
        torch.manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(32, 32))  # 用较大矩阵，更接近实际使用场景
        G = Q + torch.randn(32, 32) * 0.05
        out = zeropower_via_newtonschulz5(G).float()
        assert out.shape == G.shape
        # 输出不应为 0 或 NaN
        assert torch.isfinite(out).all()
        assert out.abs().sum() > 0
        # 谱范数（最大奇异值）应在 O(1) 量级（正交矩阵的谱范数=1）
        sv = torch.linalg.svdvals(out)
        assert 0.5 < sv.max().item() < 2.0, f"NS5 输出谱范数异常: max_sv={sv.max().item()}"

    def test_optimizer_construction(self):
        """优化器能正确构造，参数分组 lr 正确."""
        w = torch.nn.Parameter(torch.randn(16, 8))
        b = torch.nn.Parameter(torch.randn(8))
        groups = _build_param_groups(w, b)
        opt = MuonOptimizer(groups, lr_attn=1e-4, lr_ffn=3e-4)
        # 三种 group_type 的 lr 应符合预期
        lrs = {g.get("group_type"): g["lr"] for g in opt.param_groups}
        assert abs(lrs["attention"] - 1e-4) < 1e-12
        assert abs(lrs["other"] - (1e-4 + 3e-4) / 2) < 1e-12

    def test_step_2d_uses_muon_1d_uses_adamw(self):
        """2D 参数走 Muon（momentum_buffer），1D 参数走 AdamW（exp_avg/exp_avg_sq）."""
        w = torch.nn.Parameter(torch.randn(16, 8))
        b = torch.nn.Parameter(torch.randn(8))
        groups = _build_param_groups(w, b)
        opt = MuonOptimizer(groups, lr_attn=1e-3, lr_ffn=3e-3)

        # 模拟梯度
        w.grad = torch.randn_like(w)
        b.grad = torch.randn_like(b)
        opt.step()

        # 2D 参数（attention 组）应有 momentum_buffer（Muon 分支）
        w_state = opt.state[w]
        assert "momentum_buffer" in w_state, "2D 参数应走 Muon 分支，含 momentum_buffer"

        # 1D 参数（other 组）应有 exp_avg / exp_avg_sq（AdamW 分支）
        b_state = opt.state[b]
        assert "exp_avg" in b_state and "exp_avg_sq" in b_state, "1D 参数应走 AdamW 分支"

    def test_loss_decreases_on_quadratic(self):
        """在简单二次函数 min ||W - W*||^2 上，Muon 应让损失下降."""
        torch.manual_seed(42)
        W_target = torch.randn(32, 16)
        W = torch.nn.Parameter(torch.randn(32, 16))
        opt = MuonOptimizer(
            [{"params": [W], "group_type": "attention"}],
            lr_attn=0.1, weight_decay=0.0,
        )

        loss0 = ((W - W_target) ** 2).sum()
        for _ in range(20):
            opt.zero_grad()
            loss = ((W - W_target) ** 2).sum()
            loss.backward()
            opt.step()
        loss_final = ((W - W_target) ** 2).sum().item()
        assert loss_final < loss0.item() * 0.5, f"Muon 未有效降低损失: {loss0.item():.4f} -> {loss_final:.4f}"
