"""多平台加速设备选择（Mac MPS / Windows CUDA / Windows Intel XPU / CPU）.

用法:
    from src.device import get_device
    device = get_device()
"""

import torch


def get_device():
    """自动选择可用加速设备（跨平台安全）.

    检测顺序：CUDA（NVIDIA）→ MPS（Apple Silicon）→ XPU（Intel Arc）→ CPU。

    注意：CPU 版 torch 可能**没有** mps / xpu 属性（而非仅返回 False），
    必须用 hasattr 保护，否则直接 AttributeError。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
