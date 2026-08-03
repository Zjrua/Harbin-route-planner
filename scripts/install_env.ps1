# Windows 环境安装脚本：自动探测 NVIDIA GPU 选择 CUDA 版 PyTorch
# 用法（PowerShell）：./scripts/install_env.ps1
# - NVIDIA 独显 → uv sync --index pytorch-cu128（CUDA 版 torch）
# - Intel Arc 核显 / 无独显 → uv sync（CPU 版 torch，≥2.5 内置 XPU 运行时）

$hasNvidia = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" }

if ($hasNvidia) {
    Write-Host "检测到 NVIDIA 显卡，安装 CUDA 版 PyTorch..."
    uv sync --index pytorch-cu128
} else {
    Write-Host "未检测到 NVIDIA，安装 CPU 版 PyTorch（Intel XPU 内置）..."
    uv sync
}

Write-Host "环境就绪。运行 demo: uv run python scripts/serve_qwen_demo.py"
