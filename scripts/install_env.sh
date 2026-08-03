#!/bin/bash
# macOS (Apple Silicon) 环境安装脚本
# PyPI 官方 torch 包自带 MPS 后端，无需额外 index。
# 用法：./scripts/install_env.sh

uv sync
echo "macOS MPS 后端随 torch 官方包内置。运行 demo: uv run python scripts/serve_qwen_demo.py"
