"""Muon 优化器：矩阵正交化梯度更新.

参考:
- KellerJordan/Muon 官方实现: https://github.com/KellerJordan/Muon
- Kimi 论文 (Scalable Matryoshka): https://arxiv.org/abs/2502.16982

核心思想:
- Newton-Schulz quintic 迭代近似矩阵正交化 (a=3.4445, b=-4.7750, c=2.0315)
- Nesterov 动量: momentum.lerp_(grad, 1-beta), update = grad.lerp(momentum, beta)
- >=2D 参数使用 Muon 正交化更新，1D 参数回退到 AdamW
- Kimi 缩放: update *= 0.2 * sqrt(max(A,B)) 以匹配 AdamW 的更新量级
- 权重衰减: p.mul_(1 - lr * wd) (乘性衰减)
- NS 迭代在 bfloat16 下运行以提升 GPU 效率
"""

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz quintic 迭代近似矩阵正交化.

    将梯度矩阵 G 映射到最近的正交矩阵 (G @ G^T ≈ I).
    使用五次多项式迭代: X = a*X + (b*A + c*A^2) @ X, 其中 A = X @ X^T.

    Args:
        G: 输入梯度矩阵 (2D or higher, 最后两维视为矩阵)
        steps: NS 迭代步数

    Returns:
        正交化后的矩阵，与 G 同形状
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    # 确保行数 <= 列数，迭代在较小的矩阵上进行
    if X.size(-2) > X.size(-1):
        X = X.mT

    # 按谱范数归一化 (使用最后两维的 Frobenius 范数)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    # 恢复原始转置
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor,
                beta: float = 0.95, ns_steps: int = 5,
                nesterov: bool = True) -> torch.Tensor:
    """计算 Muon 更新量: Nesterov 动量 + NS 正交化.

    Args:
        grad: 当前梯度
        momentum: 动量缓冲区
        beta: 动量系数
        ns_steps: NS 迭代步数
        nesterov: 是否使用 Nesterov 动量

    Returns:
        正交化后的更新量
    """
    # Nesterov 动量: 先更新动量，再用当前梯度和动量的混合
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum.clone()

    # 将 4D 张量展平为 2D 进行正交化
    if update.ndim == 4:
        update = update.view(len(update), -1)

    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    # 缩放: 乘以 sqrt(max(1, rows/cols)) 以补偿正交化的缩放效果
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

    return update


class MuonOptimizer(Optimizer):
    """Muon + AdamW 混合优化器.

    >=2D 参数 (权重矩阵) 使用 Muon 正交化更新:
        update = NS_orthogonalize(nesterov_momentum(grad))
        p = p * (1 - lr * wd) - lr * 0.2 * sqrt(max(A,B)) * update

    1D 参数 (偏置、LayerNorm、嵌入) 使用 AdamW:
        betas=(0.9, 0.95), eps=1e-10

    支持三种参数分组:
    - attention: 注意力层, lr=lr_attn
    - ffn: FFN 层, lr=lr_ffn
    - other: 其他参数, lr=(lr_attn + lr_ffn) / 2
    """

    def __init__(self, param_groups, lr_attn: float = 3e-4, lr_ffn: float = 1e-3,
                 momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 1e-4,
                 adam_betas: tuple = (0.9, 0.95), adam_eps: float = 1e-10):
        for group in param_groups:
            group.setdefault("lr", lr_attn if group.get("group_type") == "attention"
                             else lr_ffn if group.get("group_type") == "ffn"
                             else (lr_attn + lr_ffn) / 2)
            group.setdefault("weight_decay", weight_decay)
            group.setdefault("momentum", momentum)
            group.setdefault("nesterov", nesterov)
            group.setdefault("ns_steps", ns_steps)
            group.setdefault("adam_betas", adam_betas)
            group.setdefault("adam_eps", adam_eps)

        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta_m = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            adam_betas = group["adam_betas"]
            adam_eps = group["adam_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0

                state["step"] += 1

                # 权重衰减 (乘性)
                if wd > 0:
                    p.mul_(1 - lr * wd)

                if p.dim() >= 2:
                    # >=2D: Muon 正交化更新
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)

                    buf = state["momentum_buffer"]
                    update = muon_update(grad, buf, beta=beta_m,
                                         ns_steps=ns_steps, nesterov=nesterov)

                    # Kimi 缩放: 匹配 AdamW 的更新量级
                    scale = 0.2 * max(p.shape[-2], p.shape[-1]) ** 0.5
                    update = update * scale

                    p.add_(update.reshape(p.shape), alpha=-lr)
                else:
                    # 1D: AdamW 回退
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.lerp_(grad, 1 - adam_betas[0])
                    exp_avg_sq.lerp_(grad.square(), 1 - adam_betas[1])

                    bias_correction1 = 1 - adam_betas[0] ** state["step"]
                    bias_correction2 = 1 - adam_betas[1] ** state["step"]

                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / bias_correction2 ** 0.5).add_(adam_eps)

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
